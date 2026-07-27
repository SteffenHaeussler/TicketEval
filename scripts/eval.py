"""Command-line tools for validating the ticket evaluation dataset."""

import argparse
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ticketflow.eval.dataset import DatasetError, load_cases, validate_dataset

DEFAULT_DATASET_DIR = Path("evals/data/tickets")
DIFFICULTIES = ("easy", "ambiguous", "adversarial")
SOURCES = ("handwritten", "generated")
REFERENCE_CATEGORIES = ("billing", "technical", "account", "general")


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


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser and register implemented commands."""
    parser = argparse.ArgumentParser(prog="eval.py")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_check_parser = subparsers.add_parser("dataset-check")
    dataset_check_parser.add_argument("--shard", type=Path)
    dataset_check_parser.add_argument("--allow-unverified", action="store_true")
    dataset_check_parser.set_defaults(handler=dataset_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected evaluation CLI subcommand."""
    args = build_parser().parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
