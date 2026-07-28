"""Run profile assembly and manifest construction for milestone-2 eval runs.

Turns the already-built per-case runner (`eval/runner.py`), tunable agent
(`agent/tunable.py`), and Temporal harness (`eval/harness.py`) into complete runs: the
right task-queue topology and reviewer-policy sequence for each of the four run
profiles from plan.md's "Run profiles" section, plus the `RunManifest` recording how a
run was configured and what code/dataset produced it. `run_profile()` returns its
artifacts in memory; persisting them under `evals/runs/<run_id>/` is the CLI's job.
"""

import asyncio
import hashlib
import importlib.metadata
import platform
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, NamedTuple

from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment

from ticketflow.agent.base import Agent
from ticketflow.agent.ollama import OllamaAgent
from ticketflow.agent.prompts import CLASSIFICATION_SPEC, DRAFT_SPEC
from ticketflow.agent.tunable import TunableAgentProfile, TunableMockAgent
from ticketflow.eval.cache import ResponseCache
from ticketflow.eval.dataset import EvalCase, ExpectedOutcome
from ticketflow.eval.harness import (
    CombinedWorker,
    WorkflowEvalConfig,
    current_workflow_eval_config,
    make_agent_worker,
    make_run_workers,
    make_workflow_worker,
    patched_workflow_constants,
    time_skipping_environment,
)
from ticketflow.eval.preflight import PreflightResult
from ticketflow.eval.records import (
    CallEvent,
    CaseRecord,
    GenerationSettings,
    PreflightMeasurement,
    RunManifest,
    TimeoutAdjustment,
)
from ticketflow.eval.reviewers import oracle, rubber_stamp
from ticketflow.eval.runner import CaseRunner, Reviewer
from ticketflow.eval.telemetry import RuntimeIdentityMap, TelemetrySink

RunProfile = Literal[
    "primary-quality", "fallback-quality", "fallback-routing", "reliability"
]

_REVIEWERS: dict[str, Reviewer] = {"oracle": oracle, "rubber_stamp": rubber_stamp}
_DEPENDENCY_NAMES = ("temporalio", "pydantic", "httpx")

# Recorded in every manifest so an old run stays reproducible if derive_generation_seed
# ever changes. Bump the version suffix whenever that function's output changes.
GENERATION_SEED_RULE = "sha256(run_seed \\x1f repeat_index)[:8] big-endian/v1"


class ProfilesError(Exception):
    """Base class for run-profile configuration and assembly failures."""


class ProfileConfigError(ProfilesError):
    """Raised for an invalid RunOptions or a run precondition that failed."""


class GitInfo(NamedTuple):
    """The commit and dirty state of a git worktree at run time."""

    commit: str
    dirty: bool


def _run_git(*args: str, cwd: Path) -> str:
    """Run one git subcommand in cwd and return its stdout."""
    completed = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return completed.stdout


def git_info(repo_dir: Path | None = None) -> GitInfo:
    """Return the current commit hash and whether the worktree is dirty."""
    cwd = repo_dir or Path(__file__).resolve().parent
    try:
        commit = _run_git("rev-parse", "HEAD", cwd=cwd).strip()
        porcelain = _run_git("status", "--porcelain", cwd=cwd)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProfileConfigError(f"failed to read git state: {exc}") from exc
    return GitInfo(commit=commit, dirty=bool(porcelain.strip()))


def dataset_sha256(dataset_path: str | Path) -> str:
    """Hash a dataset file or shard directory by sorted filename and content."""
    root = Path(dataset_path)
    try:
        shard_paths = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
        hasher = hashlib.sha256()
        for shard_path in shard_paths:
            hasher.update(shard_path.name.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(shard_path.read_bytes())
            hasher.update(b"\x00")
    except OSError as exc:
        raise ProfileConfigError(
            f"failed to hash dataset {dataset_path!r}: {exc}"
        ) from exc
    return hasher.hexdigest()


def derive_generation_seed(run_seed: int, repeat_index: int) -> int:
    """Derive one repeat's TunableMockAgent seed from the run seed.

    Byte-slicing matches agent.tunable._stream_rng's convention. TunableMockAgent
    further remixes case_key/operation/stream on top of this seed, so this two-stage
    hash is identical across reviewer policies for the same case+repeat, distinct
    across repeats, and fully reproducible for a fixed run seed.
    """
    digest = hashlib.sha256(f"{run_seed}\x1f{repeat_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def validate_repeats_cache(repeats: int, cache_enabled: bool) -> None:
    """Reject repeats > 1 with the cache enabled.

    With the cache on, every repeat after the first would be a cache hit by
    construction, defeating the purpose of repeating a case.
    """
    if repeats > 1 and cache_enabled:
        raise ProfileConfigError(
            f"--repeats={repeats} requires the cache disabled; every repeat "
            "after the first would be a cache hit by construction"
        )


def _dependency_versions() -> dict[str, str]:
    """Best-effort package versions for the manifest's provenance record."""
    versions: dict[str, str] = {}
    for name in _DEPENDENCY_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


ReviewerPolicy = Literal["oracle", "rubber_stamp"]


def _reviewer_policies_for(profile: RunProfile) -> list[ReviewerPolicy]:
    """Return the reviewer policies a profile may execute, in run order.

    Quality profiles pair both policies so reviewer effect can be separated from
    model quality. Routing and reliability measure availability, not review, so
    plan.md restricts them to the oracle.
    """
    if profile in ("primary-quality", "fallback-quality"):
        return ["oracle", "rubber_stamp"]
    return ["oracle"]


def resolve_reviewer_policies(
    profile: RunProfile, requested: Sequence[ReviewerPolicy] | None
) -> list[ReviewerPolicy]:
    """Resolve an explicit reviewer selection against what the profile allows.

    `None` means "run the profile's full set". An explicit selection must be a
    non-empty subset of that set, and is returned in the profile's own order so the
    reviewer sequence never depends on how the flags were typed.
    """
    allowed = _reviewer_policies_for(profile)
    if requested is None:
        return allowed
    if not requested:
        raise ProfileConfigError("reviewer_policies must not be empty")
    unsupported = [policy for policy in requested if policy not in allowed]
    if unsupported:
        raise ProfileConfigError(
            f"profile {profile!r} does not support reviewer policy "
            f"{unsupported[0]!r}; it allows {', '.join(allowed)}"
        )
    return [policy for policy in allowed if policy in set(requested)]


def _effective_cache_enabled(profile: RunProfile, cache_enabled: bool) -> bool:
    """Force the cache off for reliability; pass the setting through otherwise."""
    if profile == "reliability":
        return False
    return cache_enabled


@dataclass(frozen=True)
class RunOptions:
    """Immutable configuration for one call to run_profile()."""

    profile: RunProfile
    dataset_path: str | Path
    cases: list[EvalCase]
    primary_agent_profile: TunableAgentProfile
    fallback_agent_profile: TunableAgentProfile | None = None
    agent_backend: Literal["tunable", "mock", "ollama"] = "tunable"
    primary_model: str = "tunable-primary"
    fallback_model: str | None = "tunable-fallback"
    ollama_endpoint: str | None = None
    preflight_result: PreflightResult | None = None
    response_cache: ResponseCache | None = None
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex}")
    seed: int = 0
    bootstrap_seed: int = 0
    concurrency: int = 8
    repeats: int = 1
    cache_enabled: bool = True
    case_deadline: timedelta = timedelta(seconds=60)
    reviewer_policies: tuple[ReviewerPolicy, ...] | None = None
    # None keeps ticketflow.workflows' own AGENT_SCHEDULE_TO_START_S. fallback-routing
    # has to actually wait this out in wall-clock time (the test server does not skip
    # schedule-to-start), so a routing run wants it well under case_deadline.
    schedule_to_start_s: float | None = None
    db_path: str | None = None
    poll_interval_s: float = 0.01
    environment_factory: Callable[
        [], AbstractAsyncContextManager[WorkflowEnvironment]
    ] = time_skipping_environment

    def __post_init__(self) -> None:
        """Validate the option set before any I/O or workflow starts."""
        if self.repeats < 1:
            raise ProfileConfigError(f"repeats must be >= 1, got {self.repeats}")
        if self.concurrency < 1:
            raise ProfileConfigError(
                f"concurrency must be >= 1, got {self.concurrency}"
            )
        if not self.cases:
            raise ProfileConfigError("cases must not be empty")
        if self.agent_backend not in ("tunable", "ollama"):
            raise ProfileConfigError(
                f"agent_backend {self.agent_backend!r} is not supported by "
                "run_profile(); supported backends are 'tunable' and 'ollama'"
            )
        if (
            self.agent_backend == "tunable"
            and
            self.profile in ("fallback-quality", "fallback-routing")
            and self.fallback_agent_profile is None
        ):
            raise ProfileConfigError(
                f"profile {self.profile!r} requires fallback_agent_profile"
            )
        if self.schedule_to_start_s is not None and self.schedule_to_start_s <= 0:
            raise ProfileConfigError(
                f"schedule_to_start_s must be > 0, got {self.schedule_to_start_s}"
            )
        if self.agent_backend == "ollama":
            self._validate_ollama_preflight()
        effective_cache_enabled = _effective_cache_enabled(
            self.profile, self.cache_enabled
        )
        if self.agent_backend == "ollama":
            effective_cache_enabled = (
                effective_cache_enabled and self.response_cache is not None
            )
        validate_repeats_cache(self.repeats, effective_cache_enabled)
        # Raises for a selection the profile does not allow, before any I/O.
        resolve_reviewer_policies(self.profile, self.reviewer_policies)

    def _validate_ollama_preflight(self) -> None:
        """Require preflight provenance that matches this real-model profile."""
        if self.preflight_result is None:
            raise ProfileConfigError("ollama runs require a completed preflight_result")
        if self.ollama_endpoint is None:
            raise ProfileConfigError("ollama runs require an ollama_endpoint")

        models = {model.role: model for model in self.preflight_result.models}
        primary = models.get("primary")
        if primary is None or primary.name != self.primary_model:
            raise ProfileConfigError(
                "ollama preflight primary model does not match primary_model"
            )
        if self.fallback_model is None:
            raise ProfileConfigError(
                "ollama runs require fallback_model for complete manifest provenance"
            )
        fallback = models.get("fallback")
        if fallback is None or fallback.name != self.fallback_model:
            raise ProfileConfigError(
                "ollama preflight fallback model does not match fallback_model"
            )


def _tunable_agent(
    profile: TunableAgentProfile,
    *,
    role: Literal["primary", "fallback"],
    identity_map: RuntimeIdentityMap,
    telemetry_sink: TelemetrySink,
    expected_outcomes: Mapping[str, ExpectedOutcome],
    generation_seed: int,
) -> TunableMockAgent:
    """Build one repeat's tunable agent, forcing its role label."""
    return TunableMockAgent(
        identity_map=identity_map,
        telemetry_sink=telemetry_sink,
        expected_outcomes=expected_outcomes,
        profile=profile.model_copy(update={"role": role}),
        generation_seed=generation_seed,
    )


def _ollama_agent(
    options: RunOptions,
    *,
    role: Literal["primary", "fallback"],
    identity_map: RuntimeIdentityMap,
    telemetry_sink: TelemetrySink,
    generation_seed: int,
    cache_enabled: bool,
) -> OllamaAgent:
    """Build one real-model agent from preflight-confirmed run provenance."""
    assert options.preflight_result is not None
    assert options.ollama_endpoint is not None
    model_name = options.primary_model if role == "primary" else options.fallback_model
    assert model_name is not None
    model_info = next(
        model
        for model in options.preflight_result.models
        if model.role == role and model.name == model_name
    )
    return OllamaAgent(
        endpoint=options.ollama_endpoint,
        model=model_name,
        timeout_s=options.preflight_result.timeout_adjustment.http_timeout_s,
        seed=generation_seed,
        role=role,
        response_cache=options.response_cache if cache_enabled else None,
        identity_map=identity_map if cache_enabled else None,
        telemetry_sink=telemetry_sink,
        model_digest=model_info.digest,
        ollama_version=options.preflight_result.ollama_version,
    )


def _build_workers_for_profile(
    *,
    profile: RunProfile,
    client: Client,
    config: WorkflowEvalConfig,
    workflow_queue: str,
    identity_map: RuntimeIdentityMap,
    telemetry_sink: TelemetrySink,
    expected_outcomes: Mapping[str, ExpectedOutcome],
    primary_agent_profile: TunableAgentProfile,
    fallback_agent_profile: TunableAgentProfile | None,
    generation_seed: int,
    db_path: str | None,
    options: RunOptions,
    cache_enabled: bool,
) -> tuple[CombinedWorker, tuple[OllamaAgent, ...]]:
    """Build one repeat's agent(s) and task-queue topology for profile."""
    if profile == "fallback-routing":
        if options.agent_backend == "tunable" and fallback_agent_profile is None:
            raise ProfileConfigError(
                f"profile {profile!r} requires fallback_agent_profile"
            )
        if options.agent_backend == "ollama":
            fallback_agent: Agent = _ollama_agent(
                options,
                role="fallback",
                identity_map=identity_map,
                telemetry_sink=telemetry_sink,
                generation_seed=generation_seed,
                cache_enabled=cache_enabled,
            )
            managed_agents = (fallback_agent,)
        else:
            assert fallback_agent_profile is not None
            fallback_agent = _tunable_agent(
                fallback_agent_profile,
                role="fallback",
                identity_map=identity_map,
                telemetry_sink=telemetry_sink,
                expected_outcomes=expected_outcomes,
                generation_seed=generation_seed,
            )
            managed_agents = ()
        workflow_worker = make_workflow_worker(
            client, workflow_queue, fallback_agent, db_path=db_path
        )
        fallback_worker = make_agent_worker(
            client, fallback_agent, config.fallback_task_queue
        )
        return CombinedWorker(workflow_worker, fallback_worker), managed_agents

    if profile == "fallback-quality":
        if options.agent_backend == "tunable" and fallback_agent_profile is None:
            raise ProfileConfigError(
                f"profile {profile!r} requires fallback_agent_profile"
            )
        if options.agent_backend == "ollama":
            agent: Agent = _ollama_agent(
                options,
                role="fallback",
                identity_map=identity_map,
                telemetry_sink=telemetry_sink,
                generation_seed=generation_seed,
                cache_enabled=cache_enabled,
            )
            managed_agents = (agent,)
        else:
            assert fallback_agent_profile is not None
            agent = _tunable_agent(
                fallback_agent_profile,
                role="fallback",
                identity_map=identity_map,
                telemetry_sink=telemetry_sink,
                expected_outcomes=expected_outcomes,
                generation_seed=generation_seed,
            )
            managed_agents = ()
    else:
        if options.agent_backend == "ollama":
            agent = _ollama_agent(
                options,
                role="primary",
                identity_map=identity_map,
                telemetry_sink=telemetry_sink,
                generation_seed=generation_seed,
                cache_enabled=cache_enabled,
            )
            managed_agents = (agent,)
        else:
            agent = _tunable_agent(
                primary_agent_profile,
                role="primary",
                identity_map=identity_map,
                telemetry_sink=telemetry_sink,
                expected_outcomes=expected_outcomes,
                generation_seed=generation_seed,
            )
            managed_agents = ()

    return (
        make_run_workers(
            client,
            workflow_eval_config=config,
            workflow_task_queue=workflow_queue,
            primary_agent=agent,
            db_path=db_path,
        ),
        managed_agents,
    )


async def _run_policy(
    runner: CaseRunner,
    cases: list[EvalCase],
    *,
    policy: str,
    reviewer: Reviewer,
    repeat_index: int,
    concurrency: int,
) -> tuple[list[CaseRecord], list[CallEvent]]:
    """Run every case for one (repeat, policy) pair with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(case: EvalCase) -> tuple[CaseRecord, list[CallEvent]]:
        async with semaphore:
            return await runner.run_case(
                case, policy=policy, reviewer=reviewer, repeat_index=repeat_index
            )

    results = await asyncio.gather(*(_run_one(case) for case in cases))
    records = [record for record, _ in results]
    events = [event for _, case_events in results for event in case_events]
    return records, events


async def run_profile(
    options: RunOptions,
) -> tuple[RunManifest, list[CaseRecord], list[CallEvent]]:
    """Run every repeat and reviewer policy for a profile and assemble its manifest."""
    started_at = datetime.now(timezone.utc)
    git = git_info()
    dataset_hash = dataset_sha256(options.dataset_path)

    if options.agent_backend == "ollama":
        assert options.preflight_result is not None
        base_config = options.preflight_result.workflow_eval_config
    else:
        base_config = current_workflow_eval_config()
    config_update: dict[str, object] = {
        "agent_task_queue": f"eval-{options.run_id}-agent",
        "fallback_task_queue": f"eval-{options.run_id}-fallback",
    }
    if options.schedule_to_start_s is not None:
        config_update["agent_schedule_to_start_s"] = options.schedule_to_start_s
    config = base_config.model_copy(update=config_update)
    workflow_queue = f"eval-{options.run_id}-workflow"
    reviewer_policies = resolve_reviewer_policies(
        options.profile, options.reviewer_policies
    )
    effective_cache_enabled = _effective_cache_enabled(
        options.profile, options.cache_enabled
    )
    if options.agent_backend == "ollama":
        effective_cache_enabled = (
            effective_cache_enabled and options.response_cache is not None
        )
    expected_outcomes = {case.id: case.expected for case in options.cases}

    identity_map = RuntimeIdentityMap()
    telemetry_sink = TelemetrySink()
    all_records: list[CaseRecord] = []
    all_events: list[CallEvent] = []

    async with options.environment_factory() as env:
        runner = CaseRunner(
            env.client,
            run_id=options.run_id,
            workflow_task_queue=workflow_queue,
            db_path=options.db_path,
            case_deadline=options.case_deadline,
            identity_map=identity_map,
            telemetry_sink=telemetry_sink,
            confidence_threshold=config.confidence_threshold,
            poll_interval_s=options.poll_interval_s,
        )
        # The time-skipping test server's auto-unlock-on-result() isn't safe under
        # concurrency: multiple in-flight WorkflowHandle.result() calls race its
        # global unlock/lock counter and the server throws. Cases never rely on
        # skipped workflow timers for correctness (schedule-to-start isn't a
        # skippable timer either -- it's enforced by the matching service, not
        # visible to the workflow), so disabling auto time skipping for the whole
        # run trades away nothing and makes bounded concurrency safe.
        with patched_workflow_constants(config), env.auto_time_skipping_disabled():
            for repeat_index in range(options.repeats):
                generation_seed = derive_generation_seed(options.seed, repeat_index)
                workers, managed_agents = _build_workers_for_profile(
                    profile=options.profile,
                    client=env.client,
                    config=config,
                    workflow_queue=workflow_queue,
                    identity_map=identity_map,
                    telemetry_sink=telemetry_sink,
                    expected_outcomes=expected_outcomes,
                    primary_agent_profile=options.primary_agent_profile,
                    fallback_agent_profile=options.fallback_agent_profile,
                    generation_seed=generation_seed,
                    db_path=options.db_path,
                    options=options,
                    cache_enabled=effective_cache_enabled,
                )
                async with AsyncExitStack() as stack:
                    for agent in managed_agents:
                        await stack.enter_async_context(agent)
                    async with workers:
                        for policy in reviewer_policies:
                            records, events = await _run_policy(
                                runner,
                                options.cases,
                                policy=policy,
                                reviewer=_REVIEWERS[policy],
                                repeat_index=repeat_index,
                                concurrency=options.concurrency,
                            )
                            all_records.extend(records)
                            all_events.extend(events)

    finished_at = datetime.now(timezone.utc)

    manifest_data: dict[str, object] = {
        "run_id": options.run_id,
        "git_commit": git.commit,
        "git_dirty": git.dirty,
        "dataset_path": str(options.dataset_path),
        "dataset_sha256": dataset_hash,
        "agent_backend": options.agent_backend,
        "run_profile": options.profile,
        "primary_model": options.primary_model,
        "fallback_model": options.fallback_model,
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(),
        "reviewer_policies": reviewer_policies,
        "cache_enabled": effective_cache_enabled,
        "confidence_threshold": config.confidence_threshold,
        "agent_task_queue": config.agent_task_queue,
        "fallback_task_queue": config.fallback_task_queue,
        "agent_schedule_to_start_s": config.agent_schedule_to_start_s,
        "agent_activity_timeout_s": config.agent_activity_timeout_s,
        "agent_heartbeat_timeout_s": config.agent_heartbeat_timeout_s,
        "seed": options.seed,
        "bootstrap_seed": options.bootstrap_seed,
        "generation_seed_rule": GENERATION_SEED_RULE,
        "concurrency": options.concurrency,
        "repeats": options.repeats,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if options.agent_backend == "ollama":
        assert options.preflight_result is not None
        model_digests = {
            model.role: model.digest for model in options.preflight_result.models
        }
        adjustment = options.preflight_result.timeout_adjustment
        manifest_data.update(
            {
            "primary_model_digest": model_digests["primary"],
            "fallback_model_digest": model_digests.get("fallback"),
            "ollama_version": options.preflight_result.ollama_version,
            "prompt_hashes": {
                CLASSIFICATION_SPEC.operation: CLASSIFICATION_SPEC.prompt_hash,
                DRAFT_SPEC.operation: DRAFT_SPEC.prompt_hash,
            },
            "schema_hashes": {
                CLASSIFICATION_SPEC.operation: CLASSIFICATION_SPEC.schema_hash,
                DRAFT_SPEC.operation: DRAFT_SPEC.schema_hash,
            },
            "generation_settings": GenerationSettings(
                stream=False,
                think=False,
                temperature=0.0,
            ),
            "preflight_measurements": tuple(
                PreflightMeasurement(
                    operation=measurement.operation,
                    ticket_id=measurement.ticket_id,
                    wall_latency_s=measurement.wall_latency_s,
                    load_duration_s=measurement.load_duration_s,
                    generation_duration_s=measurement.generation_duration_s,
                )
                for measurement in options.preflight_result.measurements
            ),
            "timeout_adjustment": TimeoutAdjustment(
                configured_activity_timeout_s=adjustment.configured_activity_timeout_s,
                slowest_observed_stage_s=adjustment.slowest_observed_stage_s,
                effective_activity_timeout_s=adjustment.effective_activity_timeout_s,
                safety_margin_s=adjustment.safety_margin_s,
                http_timeout_s=adjustment.http_timeout_s,
            ),
            }
        )

    manifest = RunManifest.model_validate(manifest_data)
    return manifest, all_records, all_events
