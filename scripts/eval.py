"""Command-line tools for validating and running ticket evaluations."""

import argparse
import asyncio
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ticketflow.agent.tunable import TunableAgentProfile
from ticketflow.eval.dataset import DatasetError, load_cases, validate_dataset
from ticketflow.eval.profiles import ProfileConfigError, RunOptions, run_profile
from ticketflow.eval.records import (
    RecordsError,
    write_call_events,
    write_case_records,
    write_run_manifest,
)

DEFAULT_DATASET_DIR = Path("evals/data/tickets")
RUNS_DIR = Path("evals/runs")
DIFFICULTIES = ("easy", "ambiguous", "adversarial")
SOURCES = ("handwritten", "generated")
REFERENCE_CATEGORIES = ("billing", "technical", "account", "general")
_PROFILE_REVIEWERS = {
    "primary-quality": "both",
    "fallback-quality": "both",
    "fallback-routing": "oracle",
    "reliability": "oracle",
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
    if args.agent != "tunable":
        return (
            f"--agent {args.agent!r} is not available yet; milestone 2 supports "
            "only --agent tunable"
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

    expected_reviewer = _PROFILE_REVIEWERS[args.profile]
    if args.reviewer is not None and args.reviewer != expected_reviewer:
        return f"{args.profile} requires --reviewer {expected_reviewer}"
    return None


def run(args: argparse.Namespace) -> int:
    """Run one supported profile and persist its immutable raw artifacts."""
    validation_error = _validate_run_args(args)
    if validation_error is not None:
        print(f"run failed: {validation_error}", file=sys.stderr)
        return 1

    try:
        cases = load_cases(
            DEFAULT_DATASET_DIR, require_verified=not args.allow_unverified
        )
    except DatasetError as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print("run failed: no cases selected", file=sys.stderr)
        return 1

    try:
        options = RunOptions(
            profile=args.profile,
            dataset_path=DEFAULT_DATASET_DIR,
            cases=cases,
            primary_agent_profile=TunableAgentProfile(),
            fallback_agent_profile=TunableAgentProfile(role="fallback"),
            agent_backend=args.agent,
            seed=args.seed,
            bootstrap_seed=args.bootstrap_seed,
            concurrency=args.concurrency,
            repeats=args.repeats,
            cache_enabled=not args.no_cache,
        )
        manifest, records, events = asyncio.run(run_profile(options))
    except ProfileConfigError as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1

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
    run_parser.add_argument(
        "--reviewer", choices=("oracle", "rubber_stamp", "both"), default=None
    )
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--concurrency", type=int, default=8)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--bootstrap-seed", type=int, default=0)
    run_parser.add_argument("--no-cache", action="store_true")
    run_parser.add_argument("--allow-unverified", action="store_true")
    run_parser.set_defaults(handler=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected evaluation CLI subcommand."""
    args = build_parser().parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
