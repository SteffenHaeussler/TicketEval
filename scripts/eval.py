"""Command-line tools for validating and running ticket evaluations."""

import argparse
import asyncio
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import timedelta
from itertools import zip_longest
from pathlib import Path
from typing import cast

from ticketflow import config
from ticketflow.agent.tunable import TunableAgentProfile
from ticketflow.eval.cache import FileResponseCache
from ticketflow.eval.dataset import DatasetError, EvalCase, load_cases, validate_dataset
from ticketflow.eval.harness import (
    current_workflow_eval_config,
    local_environment,
    time_skipping_environment,
)
from ticketflow.eval.invariants import check_all_invariants
from ticketflow.eval.preflight import (
    MIN_PROBE_CASES,
    PreflightError,
    PreflightResult,
    run_preflight,
)
from ticketflow.eval.profiles import (
    ProfileConfigError,
    ReviewerPolicy,
    RunOptions,
    run_profile,
)
from ticketflow.eval.records import (
    RecordsError,
    write_call_events,
    write_case_records,
    write_json_artifact,
    write_run_manifest,
)
from ticketflow.models import Ticket

DEFAULT_DATASET_DIR = Path("evals/data/tickets")
DEFAULT_CACHE_DIR = Path("evals/cache")
RUNS_DIR = Path("evals/runs")
DIFFICULTIES = ("easy", "ambiguous", "adversarial")
SOURCES = ("handwritten", "generated")
REFERENCE_CATEGORIES = ("billing", "technical", "account", "general")
_REVIEWER_SELECTIONS: dict[str, tuple[ReviewerPolicy, ...]] = {
    "oracle": ("oracle",),
    "rubber_stamp": ("rubber_stamp",),
    "both": ("oracle", "rubber_stamp"),
}
# What each profile allows; mirrors profiles._reviewer_policies_for. A selection must
# be a subset, so --reviewer oracle is a legal narrowing of a quality profile.
_PROFILE_REVIEWERS: dict[str, tuple[ReviewerPolicy, ...]] = {
    "primary-quality": ("oracle", "rubber_stamp"),
    "fallback-quality": ("oracle", "rubber_stamp"),
    "fallback-routing": ("oracle",),
    "reliability": ("oracle",),
}


def dataset_check(args: argparse.Namespace) -> int:
    """Load, validate, and report the requested evaluation dataset."""
    try:
        cases = load_cases(
            args.shard if args.shard is not None else DEFAULT_DATASET_DIR,
            require_verified=not args.allow_unverified,
        )
        if args.shard is None:
            validate_dataset(cases)
    except DatasetError as exc:
        print(f"dataset-check failed: {exc}", file=sys.stderr)
        return 1

    difficulties = Counter(case.difficulty for case in cases)
    sources = Counter(case.source for case in cases)
    categories = Counter(case.expected.reference_category.value for case in cases)

    print(f"valid cases: {len(cases)}")
    print()
    print("difficulty:")
    for label in DIFFICULTIES:
        print(f"  {label}: {difficulties[label]}")
    print()
    print("source:")
    for label in SOURCES:
        print(f"  {label}: {sources[label]}")
    print()
    print("reference_category:")
    for label in REFERENCE_CATEGORIES:
        print(f"  {label}: {categories[label]}")
    return 0


def _validate_run_args(args: argparse.Namespace) -> str | None:
    """Return a user-facing validation error before any workflow is started."""
    if args.agent == "mock":
        return (
            "--agent mock is not available in scripts/eval.py; supported eval "
            "agents are 'tunable' and 'ollama'"
        )
    if args.limit is not None and args.limit < 1:
        return "--limit must be >= 1"
    if args.repeats < 1:
        return "--repeats must be >= 1"
    if args.concurrency < 1:
        return "--concurrency must be >= 1"
    if args.repeats > 1 and not args.no_cache:
        return f"--repeats={args.repeats} requires --no-cache"
    if args.profile == "reliability" and not args.no_cache:
        return "--profile reliability requires --no-cache"
    if args.case_deadline <= 0:
        return "--case-deadline must be > 0"
    if args.schedule_to_start is not None and args.schedule_to_start <= 0:
        return "--schedule-to-start must be > 0"
    if (
        args.schedule_to_start is not None
        and args.schedule_to_start >= args.case_deadline
    ):
        return (
            f"--schedule-to-start={args.schedule_to_start} must be below "
            f"--case-deadline={args.case_deadline}; every case would otherwise "
            "hit its deadline before the fallback activity is dispatched"
        )

    allowed = _PROFILE_REVIEWERS[args.profile]
    if args.reviewer is not None:
        unsupported = [
            policy
            for policy in _REVIEWER_SELECTIONS[args.reviewer]
            if policy not in allowed
        ]
        if unsupported:
            return (
                f"--profile {args.profile} does not support reviewer "
                f"{unsupported[0]!r}; it allows {', '.join(allowed)}"
            )
    return None


def _limited_cases(cases: list[EvalCase], limit: int) -> list[EvalCase]:
    """Take `limit` cases spread across difficulties, in dataset order.

    Shards load alphabetically (adversarial, ambiguous, easy), so a plain head slice
    would hand back only the hardest cases. Round-robin across difficulty first, then
    restore dataset order so artifacts stay easy to diff.
    """
    by_difficulty: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        by_difficulty[case.difficulty].append(case)

    groups = [by_difficulty[key] for key in DIFFICULTIES if by_difficulty[key]]
    picked: set[str] = set()
    for row in zip_longest(*groups):
        for case in row:
            if case is not None and len(picked) < limit:
                picked.add(case.id)
    return [case for case in cases if case.id in picked]


async def _run_ollama_preflight(
    probe_cases: list[EvalCase], args: argparse.Namespace
) -> PreflightResult:
    """Confirm Ollama is ready and size timeouts before any case is scored.

    Probes from the full, unlimited dataset regardless of `--limit`, so a small ad hoc
    run still gets a valid preflight sample.
    """
    probe_tickets = [
        Ticket(
            id=f"preflight-{case.id}",
            customer_email=case.customer_email,
            subject=case.subject,
            body=case.body,
        )
        for case in probe_cases[:MIN_PROBE_CASES]
    ]
    return await run_preflight(
        endpoint=args.ollama_endpoint,
        required_models={
            "primary": args.primary_model,
            "fallback": args.fallback_model,
        },
        probe_tickets=probe_tickets,
        workflow_eval_config=current_workflow_eval_config(),
        probe_http_timeout_s=config.OLLAMA_TIMEOUT_S,
        seed=args.seed,
    )


def run(args: argparse.Namespace) -> int:
    """Run one supported profile and persist its immutable raw artifacts."""
    validation_error = _validate_run_args(args)
    if validation_error is not None:
        print(f"run failed: {validation_error}", file=sys.stderr)
        return 1

    try:
        all_cases = load_cases(
            DEFAULT_DATASET_DIR, require_verified=not args.allow_unverified
        )
    except DatasetError as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1

    cases = all_cases
    if args.limit is not None:
        cases = _limited_cases(cases, args.limit)
    if not cases:
        print("run failed: no cases selected", file=sys.stderr)
        return 1

    preflight_result: PreflightResult | None = None
    response_cache = None
    if args.agent == "ollama":
        try:
            preflight_result = asyncio.run(_run_ollama_preflight(all_cases, args))
        except PreflightError as exc:
            print(f"run failed: preflight failed: {exc}", file=sys.stderr)
            return 1
        if not args.no_cache:
            response_cache = FileResponseCache(DEFAULT_CACHE_DIR)

    try:
        options = RunOptions(
            profile=args.profile,
            dataset_path=DEFAULT_DATASET_DIR,
            cases=cases,
            primary_agent_profile=TunableAgentProfile(),
            fallback_agent_profile=TunableAgentProfile(role="fallback"),
            agent_backend=args.agent,
            primary_model=(
                args.primary_model if args.agent == "ollama" else "tunable-primary"
            ),
            fallback_model=(
                args.fallback_model if args.agent == "ollama" else "tunable-fallback"
            ),
            ollama_endpoint=args.ollama_endpoint if args.agent == "ollama" else None,
            preflight_result=preflight_result,
            response_cache=response_cache,
            seed=args.seed,
            bootstrap_seed=args.bootstrap_seed,
            concurrency=args.concurrency,
            repeats=args.repeats,
            cache_enabled=not args.no_cache,
            case_deadline=timedelta(seconds=args.case_deadline),
            reviewer_policies=(
                None if args.reviewer is None else _REVIEWER_SELECTIONS[args.reviewer]
            ),
            schedule_to_start_s=args.schedule_to_start,
            environment_factory=(
                local_environment
                if args.agent == "ollama"
                else time_skipping_environment
            ),
        )
        manifest, records, events = asyncio.run(run_profile(options))
    except ProfileConfigError as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1

    invariants = check_all_invariants(
        records, events, confidence_threshold=manifest.confidence_threshold
    )

    run_dir = RUNS_DIR / manifest.run_id
    staging_dir: Path | None = None
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(dir=RUNS_DIR, prefix=f".{manifest.run_id}.")
        )
        write_run_manifest(staging_dir / "manifest.json", manifest)
        write_case_records(staging_dir / "records.jsonl", records)
        write_call_events(staging_dir / "calls.jsonl", events)
        write_json_artifact(staging_dir / "invariants.json", invariants)
        if run_dir.exists():
            raise FileExistsError(f"{run_dir}: refusing to overwrite existing run")
        staging_dir.rename(run_dir)
    except (FileExistsError, OSError, RecordsError) as exc:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"run failed: could not write artifacts: {exc}", file=sys.stderr)
        return 1

    print(f"run_id: {manifest.run_id}")
    print(f"artifacts: {run_dir}")
    print(f"cases: {len(records)} records, {len(events)} call events")
    # A violated invariant is a finding, not a run failure, so this never changes the
    # exit status -- it only makes the finding impossible to miss.
    if invariants.ok:
        print(f"invariants: ok ({invariants.total_checked} records checked)")
    else:
        print(
            f"invariants: {len(invariants.violations)} violation(s) across "
            f"{invariants.total_checked} records"
        )
        for violation in invariants.violations:
            print(f"  {violation.invariant} [{violation.case_key}]: {violation.detail}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser and register implemented commands."""
    parser = argparse.ArgumentParser(prog="eval.py")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_check_parser = subparsers.add_parser("dataset-check")
    dataset_check_parser.add_argument("--shard", type=Path)
    dataset_check_parser.add_argument("--allow-unverified", action="store_true")
    dataset_check_parser.set_defaults(handler=dataset_check)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--profile",
        choices=(
            "primary-quality",
            "fallback-quality",
            "fallback-routing",
            "reliability",
        ),
        required=True,
    )
    run_parser.add_argument(
        "--agent", choices=("tunable", "mock", "ollama"), default="tunable"
    )
    run_parser.add_argument("--primary-model", default=config.PRIMARY_MODEL)
    run_parser.add_argument("--fallback-model", default=config.FALLBACK_MODEL)
    run_parser.add_argument("--ollama-endpoint", default=config.OLLAMA_ENDPOINT)
    run_parser.add_argument(
        "--reviewer",
        choices=("oracle", "rubber_stamp", "both"),
        default=None,
        help="reviewer policies to run; defaults to everything the profile allows",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        help="cap the case count, sampled across difficulties rather than head-sliced",
    )
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--concurrency", type=int, default=8)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--bootstrap-seed", type=int, default=0)
    run_parser.add_argument("--no-cache", action="store_true")
    run_parser.add_argument("--allow-unverified", action="store_true")
    run_parser.add_argument(
        "--case-deadline",
        type=float,
        default=60.0,
        help="per-case wall-clock deadline in seconds (default: 60)",
    )
    run_parser.add_argument(
        "--schedule-to-start",
        type=float,
        default=None,
        help=(
            "override the agent schedule-to-start timeout in seconds. "
            "fallback-routing waits this out in real time, so set it well below "
            "--case-deadline (default: ticketflow.workflows' own constant)"
        ),
    )
    run_parser.set_defaults(handler=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected evaluation CLI subcommand."""
    args = build_parser().parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
