import json
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, get_args

import httpx
import pytest

from ticketflow import workflows
from ticketflow.agent.ollama import OllamaAgent
from ticketflow.agent.prompts import CLASSIFICATION_SPEC
from ticketflow.agent.tunable import TunableAgentProfile
from ticketflow.eval import profiles
from ticketflow.eval.cache import FileResponseCache
from ticketflow.eval.dataset import EvalCase, ExpectedOutcome
from ticketflow.eval.harness import current_workflow_eval_config
from ticketflow.eval.invariants import check_all_invariants
from ticketflow.eval.preflight import (
    ConfidenceGateResult,
    ModelInfo,
    PreflightResult,
    StageMeasurement,
)
from ticketflow.eval.preflight import (
    TimeoutAdjustment as PreflightTimeoutAdjustment,
)
from ticketflow.eval.profiles import (
    GitInfo,
    ProfileConfigError,
    RunOptions,
    RunProfile,
    dataset_sha256,
    derive_generation_seed,
    run_profile,
    validate_repeats_cache,
)
from ticketflow.eval.progress import ProgressEvent
from ticketflow.eval.records import RunManifest
from ticketflow.models import ActionType, TicketCategory


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        customer_email="eval@example.com",
        subject="Help with my account",
        body="Please help me log in.",
        expected=ExpectedOutcome(
            acceptable_categories=frozenset({TicketCategory.BILLING}),
            reference_category=TicketCategory.BILLING,
            acceptable_actions=frozenset({ActionType.REPLY_ONLY}),
        ),
        difficulty="easy",
        source="handwritten",
        authored_by="tester",
        label_verified=True,
        verified_by="reviewer",
        verified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _cases(n: int) -> list[EvalCase]:
    return [_case(f"case-{i}") for i in range(n)]


def _write_dataset(tmp_path: Path, cases: list[EvalCase]) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(case.model_dump_json() for case in cases) + "\n")
    return path


def _options(
    tmp_path: Path, cases: list[EvalCase], profile: RunProfile, **overrides: Any
) -> RunOptions:
    dataset_path = overrides.pop("dataset_path", None) or _write_dataset(
        tmp_path, cases
    )
    defaults: dict[str, Any] = dict(
        profile=profile,
        dataset_path=dataset_path,
        cases=cases,
        primary_agent_profile=TunableAgentProfile(),
        db_path=str(tmp_path / "readmodel.db"),
        case_deadline=timedelta(seconds=10),
    )
    defaults.update(overrides)
    return RunOptions(**defaults)


def _preflight_result(
    *, primary_model: str = "primary-model", fallback_model: str = "fallback-model"
) -> PreflightResult:
    adjustment = PreflightTimeoutAdjustment(
        configured_activity_timeout_s=30.0,
        slowest_observed_stage_s=20.0,
        effective_activity_timeout_s=60.0,
        safety_margin_s=6.0,
        http_timeout_s=54.0,
    )
    return PreflightResult(
        ollama_version="0.6.2",
        models=(
            ModelInfo(role="primary", name=primary_model, digest="sha256:primary"),
            ModelInfo(role="fallback", name=fallback_model, digest="sha256:fallback"),
        ),
        measurements=(
            StageMeasurement(
                operation="classify",
                ticket_id="probe-1",
                wall_latency_s=1.0,
                load_duration_s=0.25,
                generation_duration_s=0.75,
            ),
            StageMeasurement(
                operation="draft",
                ticket_id="probe-1",
                wall_latency_s=2.0,
                load_duration_s=0.5,
                generation_duration_s=1.5,
            ),
        ),
        timeout_adjustment=adjustment,
        confidence_gate=ConfidenceGateResult(
            samples=(0.5, 0.6, 0.7, 0.8, 0.9),
            std_dev=0.1,
            distinct_count=5,
            passes_std_dev_gate=True,
            passes_distinctness_gate=True,
        ),
        workflow_eval_config=current_workflow_eval_config().model_copy(
            update={"agent_activity_timeout_s": adjustment.effective_activity_timeout_s}
        ),
    )


def _fake_git(commit: str, porcelain: str):
    """Build a _run_git replacement returning fixed rev-parse/status output."""

    def _run(*args: str, cwd: Path) -> str:
        if args == ("rev-parse", "HEAD"):
            return f"{commit}\n"
        if args == ("status", "--porcelain"):
            return porcelain
        raise AssertionError(f"unexpected git args: {args}")

    return _run


# -- RunProfile / RunManifest literal agreement --------------------------------


def test_run_profile_literal_matches_run_manifest_literal():
    manifest_literal = RunManifest.model_fields["run_profile"].annotation
    assert set(get_args(RunProfile)) == set(get_args(manifest_literal))


# -- Seed derivation -------------------------------------------------------------


def test_derive_generation_seed_is_deterministic_and_repeat_sensitive():
    same_again = derive_generation_seed(0, 0)
    assert derive_generation_seed(0, 0) == same_again
    assert derive_generation_seed(0, 0) != derive_generation_seed(0, 1)
    assert derive_generation_seed(0, 0) != derive_generation_seed(1, 0)


# -- validate_repeats_cache / RunOptions validation -------------------------------


def test_validate_repeats_cache_rejects_repeats_greater_than_one_with_cache_enabled():
    with pytest.raises(ProfileConfigError):
        validate_repeats_cache(2, True)
    validate_repeats_cache(2, False)
    validate_repeats_cache(1, True)


def test_run_options_rejects_repeats_gt_one_with_cache_enabled_at_construction():
    with pytest.raises(ProfileConfigError):
        RunOptions(
            profile="primary-quality",
            dataset_path="unused",
            cases=[_case("case-1")],
            primary_agent_profile=TunableAgentProfile(),
            repeats=2,
            cache_enabled=True,
        )


@pytest.mark.parametrize("profile", ["fallback-quality", "fallback-routing"])
def test_run_options_requires_fallback_profile_for_fallback_profiles(profile):
    with pytest.raises(ProfileConfigError):
        RunOptions(
            profile=profile,
            dataset_path="unused",
            cases=[_case("case-1")],
            primary_agent_profile=TunableAgentProfile(),
        )


def test_run_options_rejects_non_tunable_agent_backend():
    with pytest.raises(ProfileConfigError):
        RunOptions(
            profile="primary-quality",
            dataset_path="unused",
            cases=[_case("case-1")],
            primary_agent_profile=TunableAgentProfile(),
            agent_backend="mock",
        )


def test_run_options_requires_completed_preflight_for_ollama():
    with pytest.raises(ProfileConfigError, match="preflight"):
        RunOptions(
            profile="primary-quality",
            dataset_path="unused",
            cases=[_case("case-1")],
            primary_agent_profile=TunableAgentProfile(),
            agent_backend="ollama",
        )


def test_run_options_rejects_ollama_preflight_for_a_different_primary_model():
    with pytest.raises(ProfileConfigError, match="primary model"):
        RunOptions(
            profile="primary-quality",
            dataset_path="unused",
            cases=[_case("case-1")],
            primary_agent_profile=TunableAgentProfile(),
            agent_backend="ollama",
            primary_model="configured-primary",
            fallback_model="fallback-model",
            ollama_endpoint="http://ollama.test",
            preflight_result=_preflight_result(primary_model="preflight-primary"),
        )


def test_run_options_requires_fallback_provenance_for_every_ollama_run():
    with pytest.raises(ProfileConfigError, match="fallback_model"):
        RunOptions(
            profile="primary-quality",
            dataset_path="unused",
            cases=[_case("case-1")],
            primary_agent_profile=TunableAgentProfile(),
            agent_backend="ollama",
            primary_model="primary-model",
            fallback_model=None,
            ollama_endpoint="http://ollama.test",
            preflight_result=_preflight_result(),
        )


def test_ollama_repeats_without_an_optional_cache_are_allowed():
    options = RunOptions(
        profile="primary-quality",
        dataset_path="unused",
        cases=[_case("case-1")],
        primary_agent_profile=TunableAgentProfile(),
        agent_backend="ollama",
        primary_model="primary-model",
        fallback_model="fallback-model",
        ollama_endpoint="http://ollama.test",
        preflight_result=_preflight_result(),
        repeats=2,
    )

    assert options.response_cache is None


def test_run_options_rejects_empty_cases():
    with pytest.raises(ProfileConfigError):
        RunOptions(
            profile="primary-quality",
            dataset_path="unused",
            cases=[],
            primary_agent_profile=TunableAgentProfile(),
        )


# -- git_info ----------------------------------------------------------------------


def test_git_info_reports_clean_tree(monkeypatch):
    monkeypatch.setattr(profiles, "_run_git", _fake_git("abc123", ""))
    assert profiles.git_info() == GitInfo(commit="abc123", dirty=False)


def test_git_info_reports_dirty_tree_truthfully(monkeypatch):
    monkeypatch.setattr(
        profiles,
        "_run_git",
        _fake_git("abc123", " M src/ticketflow/eval/profiles.py\n"),
    )
    assert profiles.git_info() == GitInfo(commit="abc123", dirty=True)


# -- dataset_sha256 ------------------------------------------------------------------


def test_dataset_sha256_is_deterministic(tmp_path):
    shard = tmp_path / "shard.jsonl"
    shard.write_text('{"id": "case-1"}\n')
    assert dataset_sha256(tmp_path) == dataset_sha256(tmp_path)


def test_dataset_sha256_changes_when_shard_content_changes(tmp_path):
    shard = tmp_path / "shard.jsonl"
    shard.write_text('{"id": "case-1"}\n')
    before = dataset_sha256(tmp_path)
    shard.write_text('{"id": "case-2"}\n')
    assert dataset_sha256(tmp_path) != before


def test_dataset_sha256_changes_when_shard_renamed(tmp_path):
    shard_a = tmp_path / "a.jsonl"
    shard_a.write_text('{"id": "case-1"}\n')
    before = dataset_sha256(tmp_path)
    shard_a.unlink()
    shard_b = tmp_path / "b.jsonl"
    shard_b.write_text('{"id": "case-1"}\n')
    assert dataset_sha256(tmp_path) != before


# -- Per-profile behavior -------------------------------------------------------------


async def test_primary_quality_profile_labels_records_primary(tmp_path):
    cases = _cases(2)
    options = _options(tmp_path, cases, "primary-quality")

    manifest, records, _events = await run_profile(options)

    assert manifest.reviewer_policies == ["oracle", "rubber_stamp"]
    assert len(records) == 2 * len(cases)
    assert all(record.model_path == "primary/primary" for record in records)
    assert all(record.prediction_available for record in records)


async def test_fallback_quality_profile_labels_records_fallback_and_pairs_by_case_id(
    tmp_path,
):
    cases = _cases(2)
    primary_options = _options(tmp_path, cases, "primary-quality")
    fallback_options = _options(
        tmp_path,
        cases,
        "fallback-quality",
        fallback_agent_profile=TunableAgentProfile(),
    )

    _primary_manifest, primary_records, _ = await run_profile(primary_options)
    fallback_manifest, fallback_records, _ = await run_profile(fallback_options)

    assert fallback_manifest.reviewer_policies == ["oracle", "rubber_stamp"]
    assert all(record.model_path == "fallback/fallback" for record in fallback_records)

    primary_case_keys = {r.case_key for r in primary_records if r.policy == "oracle"}
    fallback_case_keys = {r.case_key for r in fallback_records if r.policy == "oracle"}
    assert primary_case_keys == fallback_case_keys == {c.id for c in cases}


async def test_reliability_profile_uses_oracle_only_and_forces_cache_disabled(tmp_path):
    cases = _cases(2)
    options = _options(tmp_path, cases, "reliability", cache_enabled=True)

    manifest, records, _events = await run_profile(options)

    assert manifest.reviewer_policies == ["oracle"]
    assert manifest.cache_enabled is False
    assert len(records) == len(cases)


async def test_fallback_routing_completes_far_below_schedule_to_start_wall_clock(
    tmp_path, monkeypatch
):
    # Temporal's time-skipping test server does not auto-advance schedule-to-start
    # timers (they're enforced by the matching service, not visible to the
    # workflow as a timer/command the client can skip to) -- tests/test_workflow.py's
    # own schedule-to-start test proves this same mechanism the same way: by
    # configuring a small timeout rather than relying on time-skip to fast-forward
    # the real 30s default. profiles.py itself never shrinks this value (asserted
    # below); this monkeypatch only sizes the *test's* configured timeout.
    monkeypatch.setattr(workflows, "AGENT_SCHEDULE_TO_START_S", 1.0)
    cases = _cases(1)
    options = _options(
        tmp_path,
        cases,
        "fallback-routing",
        fallback_agent_profile=TunableAgentProfile(),
        case_deadline=timedelta(seconds=10),
    )

    started = time.perf_counter()
    manifest, records, _events = await run_profile(options)
    elapsed = time.perf_counter() - started

    assert manifest.agent_schedule_to_start_s == 1.0
    assert elapsed < 5.0
    assert all(record.model_path == "fallback/fallback" for record in records)


async def test_fallback_routing_results_excluded_from_quality_comparisons(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(workflows, "AGENT_SCHEDULE_TO_START_S", 1.0)
    cases = _cases(2)
    quality_options = _options(tmp_path, cases, "primary-quality")
    routing_options = _options(
        tmp_path,
        cases,
        "fallback-routing",
        fallback_agent_profile=TunableAgentProfile(),
        case_deadline=timedelta(seconds=10),
    )

    quality_manifest, quality_records, _ = await run_profile(quality_options)
    routing_manifest, routing_records, routing_events = await run_profile(
        routing_options
    )

    assert quality_manifest.run_id != routing_manifest.run_id
    assert {r.run_id for r in quality_records}.isdisjoint(
        {r.run_id for r in routing_records}
    )
    assert quality_manifest.run_profile == "primary-quality"
    assert routing_manifest.run_profile == "fallback-routing"

    report = check_all_invariants(
        routing_records,
        routing_events,
        confidence_threshold=routing_manifest.confidence_threshold,
    )
    assert report.ok


# -- Seed rule end to end -------------------------------------------------------------


async def test_seed_identical_across_reviewer_policies_for_same_case_and_repeat(
    tmp_path,
):
    cases = _cases(3)
    options = _options(tmp_path, cases, "primary-quality", seed=42)

    _manifest, records, _events = await run_profile(options)

    oracle_records = {r.case_key: r for r in records if r.policy == "oracle"}
    rubber_stamp_records = {
        r.case_key: r for r in records if r.policy == "rubber_stamp"
    }

    for case_key, oracle_record in oracle_records.items():
        rubber_stamp_record = rubber_stamp_records[case_key]
        assert (
            oracle_record.predicted_category == rubber_stamp_record.predicted_category
        )
        assert oracle_record.predicted_action == rubber_stamp_record.predicted_action
        assert (
            oracle_record.predicted_refund_amount
            == rubber_stamp_record.predicted_refund_amount
        )
        assert (
            oracle_record.classification_confidence
            == rubber_stamp_record.classification_confidence
        )
        assert oracle_record.draft_confidence == rubber_stamp_record.draft_confidence
        assert oracle_record.reply_text == rubber_stamp_record.reply_text


async def test_seed_differs_across_repeats(tmp_path):
    cases = _cases(1)
    options = _options(
        tmp_path, cases, "primary-quality", seed=7, repeats=2, cache_enabled=False
    )

    _manifest, records, _events = await run_profile(options)

    by_repeat = {r.repeat_index: r for r in records if r.policy == "oracle"}
    assert (
        by_repeat[0].classification_confidence != by_repeat[1].classification_confidence
    )


# -- Manifest assembly ----------------------------------------------------------------


async def test_run_profile_manifest_records_dirty_state_truthfully_end_to_end(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        profiles,
        "_run_git",
        _fake_git("deadbeef", " M src/ticketflow/eval/profiles.py\n"),
    )
    cases = _cases(1)
    options = _options(tmp_path, cases, "primary-quality")

    manifest, _records, _events = await run_profile(options)

    assert manifest.git_commit == "deadbeef"
    assert manifest.git_dirty is True


async def test_run_profile_concurrency_does_not_change_results(tmp_path):
    cases = _cases(5)

    async def _run(concurrency: int) -> set:
        options = _options(tmp_path, cases, "primary-quality", concurrency=concurrency)
        _manifest, records, _events = await run_profile(options)
        return {
            (
                r.policy,
                r.case_key,
                r.repeat_index,
                r.predicted_category,
                r.predicted_action,
                r.predicted_refund_amount,
                r.classification_confidence,
                r.draft_confidence,
                r.terminal_outcome,
                r.was_gated,
            )
            for r in records
        }

    serial = await _run(1)
    parallel = await _run(4)
    assert serial == parallel


async def test_run_profile_manifest_assembly_correctness(tmp_path):
    cases = _cases(2)
    dataset_path = _write_dataset(tmp_path, cases)
    options = _options(
        tmp_path,
        cases,
        "primary-quality",
        dataset_path=dataset_path,
        seed=5,
        bootstrap_seed=9,
        concurrency=3,
        repeats=1,
    )

    manifest, _records, _events = await run_profile(options)

    expected_config = current_workflow_eval_config()
    assert manifest.seed == 5
    assert manifest.bootstrap_seed == 9
    assert manifest.concurrency == 3
    assert manifest.repeats == 1
    assert manifest.confidence_threshold == expected_config.confidence_threshold
    assert (
        manifest.agent_schedule_to_start_s == expected_config.agent_schedule_to_start_s
    )
    assert manifest.dataset_path == str(dataset_path)
    assert manifest.dataset_sha256 == dataset_sha256(dataset_path)
    assert manifest.python_version == platform.python_version()


# -- Ollama profile integration -------------------------------------------------------


def _ollama_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    system_prompt = payload["messages"][0]["content"]
    if system_prompt == CLASSIFICATION_SPEC.system_prompt:
        content = {"category": "billing", "confidence": 0.9}
    else:
        content = {
            "reply_text": "Thanks for reaching out.",
            "action": {"type": "reply_only"},
            "confidence": 0.8,
        }
    return httpx.Response(
        200,
        json={
            "message": {"content": json.dumps(content)},
            "total_duration": 100_000_000,
            "load_duration": 20_000_000,
        },
        request=request,
    )


async def test_ollama_primary_profile_uses_preflight_cache_and_manifest_provenance(
    tmp_path, monkeypatch
):
    captured_kwargs: list[dict[str, object]] = []
    created_agents: list[OllamaAgent] = []

    class CapturingOllamaAgent(OllamaAgent):
        def __init__(self, *args, **kwargs):
            captured_kwargs.append(kwargs.copy())
            super().__init__(
                *args,
                transport=httpx.MockTransport(_ollama_response),
                **kwargs,
            )
            created_agents.append(self)

    monkeypatch.setattr(profiles, "OllamaAgent", CapturingOllamaAgent)
    cases = _cases(1)
    cache = FileResponseCache(tmp_path / "cache")
    options = _options(
        tmp_path,
        cases,
        "primary-quality",
        agent_backend="ollama",
        primary_model="primary-model",
        fallback_model="fallback-model",
        ollama_endpoint="http://ollama.test",
        preflight_result=_preflight_result(),
        response_cache=cache,
    )

    manifest, _records, events = await run_profile(options)

    assert manifest.agent_activity_timeout_s == 60.0
    assert manifest.primary_model_digest == "sha256:primary"
    assert manifest.fallback_model_digest == "sha256:fallback"
    assert manifest.ollama_version == "0.6.2"
    assert manifest.prompt_hashes is not None
    assert set(manifest.prompt_hashes) == {"classify", "draft"}
    assert manifest.schema_hashes is not None
    assert set(manifest.schema_hashes) == {"classify", "draft"}
    assert manifest.generation_settings is not None
    assert manifest.generation_settings.stream is False
    assert manifest.timeout_adjustment is not None
    assert manifest.timeout_adjustment.http_timeout_s == 54.0
    assert manifest.dependency_versions["httpx"]
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["role"] == "primary"
    assert captured_kwargs[0]["model"] == "primary-model"
    assert captured_kwargs[0]["model_digest"] == "sha256:primary"
    assert captured_kwargs[0]["ollama_version"] == "0.6.2"
    assert captured_kwargs[0]["timeout_s"] == 54.0
    assert captured_kwargs[0]["response_cache"] is cache
    assert captured_kwargs[0]["identity_map"] is not None
    assert captured_kwargs[0]["telemetry_sink"] is not None
    assert len([event for event in events if event.cache_hit]) == 2
    assert all(agent._client.is_closed for agent in created_agents)


async def test_ollama_fallback_profile_uses_confirmed_fallback_provenance(
    tmp_path, monkeypatch
):
    captured_kwargs: list[dict[str, object]] = []

    class CapturingOllamaAgent(OllamaAgent):
        def __init__(self, *args, **kwargs):
            captured_kwargs.append(kwargs.copy())
            super().__init__(
                *args,
                transport=httpx.MockTransport(_ollama_response),
                **kwargs,
            )

    monkeypatch.setattr(profiles, "OllamaAgent", CapturingOllamaAgent)
    cases = _cases(1)
    options = _options(
        tmp_path,
        cases,
        "fallback-quality",
        fallback_agent_profile=TunableAgentProfile(role="fallback"),
        agent_backend="ollama",
        primary_model="primary-model",
        fallback_model="fallback-model",
        ollama_endpoint="http://ollama.test",
        preflight_result=_preflight_result(),
        cache_enabled=False,
    )

    manifest, records, _events = await run_profile(options)

    assert manifest.agent_activity_timeout_s == 60.0
    assert all(record.model_path == "fallback/fallback" for record in records)
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["role"] == "fallback"
    assert captured_kwargs[0]["model"] == "fallback-model"
    assert captured_kwargs[0]["model_digest"] == "sha256:fallback"


# -- Progress reporting ---------------------------------------------------------------


async def test_progress_callback_reports_every_case_across_policies(tmp_path):
    cases = _cases(3)
    events: list[ProgressEvent] = []
    options = _options(
        tmp_path, cases, "primary-quality", progress=events.append, concurrency=2
    )

    _manifest, records, _events = await run_profile(options)

    case_events = [event for event in events if event.phase == "case"]
    assert len(case_events) == len(records) == 6
    # The counter spans both reviewer policies rather than restarting per policy, and
    # every case reports the same total.
    assert [event.completed for event in case_events] == [1, 2, 3, 4, 5, 6]
    assert all(event.total == 6 for event in case_events)
    assert {event.policy for event in case_events} == {"oracle", "rubber_stamp"}
    assert {event.case_key for event in case_events} == {case.id for case in cases}
    assert all(event.message == "resolved" for event in case_events)
    assert all(event.elapsed_s is not None for event in case_events)

    run_events = [event for event in events if event.phase == "run"]
    assert run_events[0].total == 6
    assert [event.policy for event in run_events[1:]] == ["oracle", "rubber_stamp"]


async def test_run_profile_emits_nothing_when_no_progress_callback_is_installed(
    tmp_path, capsys
):
    options = _options(tmp_path, _cases(2), "primary-quality")

    await run_profile(options)

    assert options.progress is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
