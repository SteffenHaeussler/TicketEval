"""Eval dataset models, shard loading, and labelling/distribution validation."""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from ticketflow.models import ActionType, TicketCategory


class DatasetError(Exception):
    """Base class for eval-dataset structural and distribution failures."""


class DatasetLoadError(DatasetError):
    """Raised by load_cases for a structural violation in one or more shards."""


class DatasetValidationError(DatasetError):
    """Raised by validate_dataset for a release-level distribution violation."""


class ExpectedOutcome(BaseModel):
    """Acceptable and reference outcome labels for one eval case."""

    acceptable_categories: set[TicketCategory]
    reference_category: TicketCategory
    acceptable_actions: set[ActionType]
    expected_refund_amount: float | None = None
    refund_tolerance: float = 0.01


class EvalCase(BaseModel):
    """One labelled ticket-support case with provenance and verification metadata."""

    id: str
    subject: str
    body: str
    customer_email: str = "eval@example.com"
    expected: ExpectedOutcome
    difficulty: Literal["easy", "ambiguous", "adversarial"]
    source: Literal["handwritten", "generated"]
    authored_by: str
    generated_by: str | None = None
    label_verified: bool
    verified_by: str | None = None
    verified_at: datetime | None = None
    notes: str | None = None


class DatasetValidationPolicy(BaseModel):
    """Overridable release-level distribution thresholds for validate_dataset."""

    easy_range: tuple[float, float] = (0.30, 0.50)
    ambiguous_range: tuple[float, float] = (0.30, 0.50)
    adversarial_range: tuple[float, float] = (0.15, 0.35)
    category_range: tuple[float, float] = (0.15, 0.35)


DEFAULT_DATASET_VALIDATION_POLICY = DatasetValidationPolicy()

_AMOUNT_TOKEN_RE = re.compile(
    r"(?<![\w.,])"
    r"(?P<prefix>[$€£])?"
    r"(?P<int>\d+)"
    r"(?:(?P<sep>[.,])(?P<frac>\d{1,2}))?"
    r"(?!\d)"
    r"(?:\s?(?P<currency>USD|EUR|GBP))?"
    r"(?![\w])(?![.,]\d)"
)


def _iter_amount_tokens(text: str):
    """Yield every standalone currency-like numeric token found in text."""
    for m in _AMOUNT_TOKEN_RE.finditer(text):
        frac = m.group("frac")
        has_currency = bool(m.group("prefix")) or bool(m.group("currency"))
        if frac is None and not has_currency:
            continue
        value = f"{m.group('int')}.{frac}" if frac else m.group("int")
        yield float(value)


def refund_amount_in_text(amount: float, tolerance: float, *texts: str) -> bool:
    """Return True if a standalone currency token in texts matches amount."""
    return any(
        abs(token - amount) <= tolerance
        for text in texts
        for token in _iter_amount_tokens(text)
    )


def _shard_paths(root: Path) -> list[Path]:
    """Return the ordered list of shard files backing a load_cases path."""
    if root.is_dir():
        paths = sorted(root.glob("*.jsonl"))
        if not paths:
            raise DatasetLoadError(f"{root}: no .jsonl shards found")
        return paths
    return [root]


def _read_shard(shard_path: Path) -> list[tuple[EvalCase, Path]]:
    """Parse one shard file into (case, shard_path) pairs."""
    out: list[tuple[EvalCase, Path]] = []
    with shard_path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetLoadError(
                    f"{shard_path}: line {lineno}: invalid JSON ({exc})"
                ) from exc
            try:
                case = EvalCase.model_validate(raw)
            except ValidationError as exc:
                case_id = raw.get("id", "<unknown>")
                raise DatasetLoadError(
                    f"{shard_path}: case {case_id!r}: invalid case ({exc})"
                ) from exc
            out.append((case, shard_path))
    return out


def _check_duplicate_ids(loaded: list[tuple[EvalCase, Path]]) -> None:
    """Reject a case ID that appears in more than one shard."""
    shards_by_id: dict[str, list[Path]] = defaultdict(list)
    for case, shard_path in loaded:
        shards_by_id[case.id].append(shard_path)

    for case_id, shard_paths in shards_by_id.items():
        if len(shard_paths) > 1:
            shard_names = ", ".join(str(p) for p in shard_paths)
            raise DatasetLoadError(
                f"duplicate case id {case_id!r} found in shards: {shard_names}"
            )


def _check_case(case: EvalCase, shard_path: Path, *, require_verified: bool) -> None:
    """Apply every structural and labelling rule to one loaded case."""
    expected = case.expected

    if not expected.acceptable_categories:
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: acceptable_categories must not be empty"
        )
    if not expected.acceptable_actions:
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: acceptable_actions must not be empty"
        )
    if expected.reference_category not in expected.acceptable_categories:
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: reference_category "
            f"{expected.reference_category!r} is not in acceptable_categories"
        )
    if (
        ActionType.REFUND in expected.acceptable_actions
        and expected.expected_refund_amount is None
    ):
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: refund is acceptable but "
            "expected_refund_amount is not set"
        )
    if expected.expected_refund_amount is not None and not refund_amount_in_text(
        expected.expected_refund_amount,
        expected.refund_tolerance,
        case.subject,
        case.body,
    ):
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: expected_refund_amount "
            f"{expected.expected_refund_amount} does not appear in the ticket text"
        )
    if not case.authored_by:
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: authored_by is required"
        )
    if case.source == "generated" and not case.generated_by:
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: generated cases require generated_by"
        )
    if case.source == "handwritten" and case.generated_by:
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: handwritten cases must not set "
            "generated_by"
        )

    if not require_verified:
        return

    if not case.label_verified:
        raise DatasetLoadError(f"{shard_path}: case {case.id!r}: label is not verified")
    if not case.verified_by:
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: verified label requires verified_by"
        )
    if case.verified_at is None:
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: verified label requires verified_at"
        )
    if case.verified_by == case.authored_by:
        raise DatasetLoadError(
            f"{shard_path}: case {case.id!r}: verified_by must not equal authored_by"
        )


def load_cases(path: str | Path, *, require_verified: bool = True) -> list[EvalCase]:
    """Load and structurally validate eval cases from a file or shard directory."""
    root = Path(path)
    loaded: list[tuple[EvalCase, Path]] = []
    for shard_path in _shard_paths(root):
        loaded.extend(_read_shard(shard_path))

    _check_duplicate_ids(loaded)
    for case, shard_path in loaded:
        _check_case(case, shard_path, require_verified=require_verified)
    return [case for case, _ in loaded]


def _check_share(
    cases: list[EvalCase],
    label: str,
    predicate,
    bounds: tuple[float, float],
    total: int,
) -> None:
    """Raise if the share of cases matching predicate falls outside bounds."""
    low, high = bounds
    share = sum(1 for c in cases if predicate(c)) / total
    if not (low <= share <= high):
        raise DatasetValidationError(
            f"{label} is {share:.1%} of {total} cases; expected {low:.0%}-{high:.0%}"
        )


def validate_dataset(
    cases: list[EvalCase], policy: DatasetValidationPolicy | None = None
) -> None:
    """Apply release-level difficulty and reference-category distribution checks."""
    policy = policy or DEFAULT_DATASET_VALIDATION_POLICY
    total = len(cases)
    if total == 0:
        raise DatasetValidationError("dataset is empty")

    _check_share(
        cases, "easy", lambda c: c.difficulty == "easy", policy.easy_range, total
    )
    _check_share(
        cases,
        "ambiguous",
        lambda c: c.difficulty == "ambiguous",
        policy.ambiguous_range,
        total,
    )
    _check_share(
        cases,
        "adversarial",
        lambda c: c.difficulty == "adversarial",
        policy.adversarial_range,
        total,
    )
    for category in TicketCategory:
        _check_share(
            cases,
            f"reference_category={category}",
            lambda c, cat=category: c.expected.reference_category == cat,
            policy.category_range,
            total,
        )
