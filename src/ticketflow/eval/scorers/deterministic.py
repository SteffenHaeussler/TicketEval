"""Deterministic correctness predicates and metrics, per plan.md's correctness rules."""

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ticketflow.eval.records import CallEvent, CaseRecord
from ticketflow.models import ActionType, TicketCategory


class ScorerError(Exception):
    """Base class for deterministic-scorer failures."""


class UnscoredCaseError(ScorerError):
    """Raised when a correctness predicate is evaluated on a case with no draft."""


class MissingPredictionFieldError(ScorerError):
    """Raised when a scored case lacks a predicted field a predicate requires."""


class Rate(BaseModel):
    """A proportion that always names its denominator population."""

    model_config = ConfigDict(frozen=True)

    denominator_label: str
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @model_validator(mode="after")
    def _numerator_within_denominator(self) -> "Rate":
        """Reject a numerator that exceeds its own denominator."""
        if self.numerator > self.denominator:
            raise ValueError(
                f"Rate({self.denominator_label!r}): numerator {self.numerator} "
                f"exceeds denominator {self.denominator}"
            )
        return self

    @property
    def value(self) -> float | None:
        """Return numerator / denominator, or None when the denominator is 0."""
        return self.numerator / self.denominator if self.denominator else None


class ConfusionMatrix(BaseModel):
    """Reference-category-by-predicted-category counts over one population."""

    model_config = ConfigDict(frozen=True)

    denominator_label: str
    cells: dict[TicketCategory, dict[TicketCategory, int]]
    total: int = Field(ge=0)


class CategoryClassMetrics(BaseModel):
    """Per-class precision, recall, and F1 derived from a ConfusionMatrix."""

    model_config = ConfigDict(frozen=True)

    category: TicketCategory
    precision: Rate
    recall: Rate
    f1: float | None


class RefundAmountError(BaseModel):
    """Mean absolute error between predicted and expected refund amounts."""

    model_config = ConfigDict(frozen=True)

    denominator_label: str
    denominator: int = Field(ge=0)
    mean_absolute_error: float | None


class DeterministicMetrics(BaseModel):
    """Every M1-T5 deterministic metric for one run, computed at once."""

    model_config = ConfigDict(frozen=True)

    scored_population_size: int
    total_population_size: int

    unreviewed_structured_error_rate: Rate
    unreviewed_category_error_rate: Rate
    unreviewed_action_error_rate: Rate
    review_load: Rate
    gate_recall: Rate
    gate_precision: Rate
    category_confusion_matrix: ConfusionMatrix
    category_class_metrics: dict[TicketCategory, CategoryClassMetrics]
    action_accuracy: Rate
    refund_amount_error: RefundAmountError
    escalation_rate: Rate
    invalid_output_rate: Rate
    fallback_usage_rate: Rate
    unhelpful_outcome_rate: Rate


def _require_scored(record: CaseRecord) -> None:
    """Raise UnscoredCaseError if record has no draft."""
    if not record.prediction_available:
        raise UnscoredCaseError(
            f"case {record.case_key!r} (run {record.run_id!r}): "
            "predicate evaluated on an unscored case (no draft)"
        )


def _predicted_category_or_raise(record: CaseRecord) -> TicketCategory:
    """Return predicted_category or raise if a scored case is missing it."""
    if record.predicted_category is None:
        raise MissingPredictionFieldError(
            f"case {record.case_key!r} (run {record.run_id!r}): "
            "prediction_available is True but predicted_category is None"
        )
    return record.predicted_category


def _predicted_action_or_raise(record: CaseRecord) -> ActionType:
    """Return predicted_action or raise if a scored case is missing it."""
    if record.predicted_action is None:
        raise MissingPredictionFieldError(
            f"case {record.case_key!r} (run {record.run_id!r}): "
            "prediction_available is True but predicted_action is None"
        )
    return record.predicted_action


def category_correct(record: CaseRecord) -> bool:
    """Return True if the predicted category is in the acceptable set."""
    _require_scored(record)
    predicted = _predicted_category_or_raise(record)
    return predicted in record.expected.acceptable_categories


def action_correct(record: CaseRecord) -> bool:
    """Return True if the predicted action is in the acceptable set."""
    _require_scored(record)
    predicted = _predicted_action_or_raise(record)
    return predicted in record.expected.acceptable_actions


def refund_correct(record: CaseRecord) -> bool:
    """Return True unless a wrong or unverifiable refund amount was proposed."""
    _require_scored(record)
    predicted_action = _predicted_action_or_raise(record)
    if predicted_action is not ActionType.REFUND:
        return True
    expected_amount = record.expected.expected_refund_amount
    predicted_amount = record.predicted_refund_amount
    if expected_amount is None or predicted_amount is None:
        return False
    return abs(predicted_amount - expected_amount) <= record.expected.refund_tolerance


def structured_correct(record: CaseRecord) -> bool:
    """Return True iff category, action, and refund correctness all hold."""
    return (
        category_correct(record) and action_correct(record) and refund_correct(record)
    )


def _scored_population(records: Sequence[CaseRecord]) -> list[CaseRecord]:
    """Return only the cases that produced a draft."""
    return [r for r in records if r.prediction_available]


def _case_identity(obj: CaseRecord | CallEvent) -> tuple[str, str, str, int, str]:
    """Return the (run_id, policy, case_key, repeat_index, ticket_id) join key."""
    return (obj.run_id, obj.policy, obj.case_key, obj.repeat_index, obj.ticket_id)


def unreviewed_structured_error_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored, ungated cases with an incorrect outcome."""
    scored = _scored_population(records)
    n = sum(1 for r in scored if not r.was_gated and not structured_correct(r))
    return Rate(
        numerator=n, denominator=len(scored), denominator_label="scored_population"
    )


def unreviewed_category_error_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored, ungated cases with an incorrect category."""
    scored = _scored_population(records)
    n = sum(1 for r in scored if not r.was_gated and not category_correct(r))
    return Rate(
        numerator=n, denominator=len(scored), denominator_label="scored_population"
    )


def unreviewed_action_error_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored, ungated cases with an incorrect action."""
    scored = _scored_population(records)
    n = sum(1 for r in scored if not r.was_gated and not action_correct(r))
    return Rate(
        numerator=n, denominator=len(scored), denominator_label="scored_population"
    )


def review_load(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored cases gated for approval."""
    scored = _scored_population(records)
    n = sum(1 for r in scored if r.was_gated)
    return Rate(
        numerator=n, denominator=len(scored), denominator_label="scored_population"
    )


def gate_recall(records: Sequence[CaseRecord]) -> Rate:
    """Return P(gated | not structured_correct) over the scored population."""
    scored = _scored_population(records)
    incorrect = [r for r in scored if not structured_correct(r)]
    n = sum(1 for r in incorrect if r.was_gated)
    return Rate(
        numerator=n,
        denominator=len(incorrect),
        denominator_label="incorrect_scored_cases",
    )


def gate_precision(records: Sequence[CaseRecord]) -> Rate:
    """Return P(not structured_correct | gated) over the scored population."""
    scored = _scored_population(records)
    gated = [r for r in scored if r.was_gated]
    n = sum(1 for r in gated if not structured_correct(r))
    return Rate(
        numerator=n, denominator=len(gated), denominator_label="gated_scored_cases"
    )


def action_accuracy(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored cases with a correct action, regardless of gating."""
    scored = _scored_population(records)
    n = sum(1 for r in scored if action_correct(r))
    return Rate(
        numerator=n, denominator=len(scored), denominator_label="scored_population"
    )


def escalation_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of all cases that escalated."""
    n = sum(1 for r in records if r.terminal_outcome == "escalated")
    return Rate(numerator=n, denominator=len(records), denominator_label="all_cases")


def invalid_output_rate(
    records: Sequence[CaseRecord], events: Sequence[CallEvent]
) -> Rate:
    """Return the share of all cases with at least one invalid-output call event."""
    flagged = {_case_identity(e) for e in events if e.outcome == "invalid_output"}
    n = sum(1 for r in records if _case_identity(r) in flagged)
    return Rate(numerator=n, denominator=len(records), denominator_label="all_cases")


def fallback_usage_rate(
    records: Sequence[CaseRecord], events: Sequence[CallEvent]
) -> Rate:
    """Return the share of all cases with at least one fallback-role call event."""
    flagged = {_case_identity(e) for e in events if e.role == "fallback"}
    n = sum(1 for r in records if _case_identity(r) in flagged)
    return Rate(numerator=n, denominator=len(records), denominator_label="all_cases")


def unhelpful_outcome_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the combined quality-and-availability failure rate over all cases."""
    n = sum(
        1 for r in records if not r.prediction_available or not structured_correct(r)
    )
    return Rate(
        numerator=n,
        denominator=len(records),
        denominator_label="all_cases_combining_quality_and_availability",
    )


def category_confusion_matrix(records: Sequence[CaseRecord]) -> ConfusionMatrix:
    """Return reference-by-predicted category counts over the scored population."""
    scored = _scored_population(records)
    cells: dict[TicketCategory, dict[TicketCategory, int]] = {
        ref: {pred: 0 for pred in TicketCategory} for ref in TicketCategory
    }
    for r in scored:
        predicted = _predicted_category_or_raise(r)
        cells[r.expected.reference_category][predicted] += 1
    return ConfusionMatrix(
        denominator_label="scored_population", cells=cells, total=len(scored)
    )


def category_class_metrics(
    matrix: ConfusionMatrix,
) -> dict[TicketCategory, CategoryClassMetrics]:
    """Return per-category precision, recall, and F1 derived from a confusion matrix."""
    result: dict[TicketCategory, CategoryClassMetrics] = {}
    for cat in TicketCategory:
        tp = matrix.cells[cat][cat]
        fp = sum(matrix.cells[ref][cat] for ref in TicketCategory if ref != cat)
        fn = sum(matrix.cells[cat][pred] for pred in TicketCategory if pred != cat)
        precision = Rate(
            numerator=tp,
            denominator=tp + fp,
            denominator_label=f"predicted_{cat.value}",
        )
        recall = Rate(
            numerator=tp,
            denominator=tp + fn,
            denominator_label=f"reference_{cat.value}",
        )
        p, r = precision.value, recall.value
        if p is None or r is None:
            f1 = None
        elif p == 0.0 and r == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * p * r / (p + r)
        result[cat] = CategoryClassMetrics(
            category=cat, precision=precision, recall=recall, f1=f1
        )
    return result


def refund_amount_error(records: Sequence[CaseRecord]) -> RefundAmountError:
    """Return the mean absolute error between predicted and expected refund amounts."""
    scored = _scored_population(records)
    diffs = [
        abs(r.predicted_refund_amount - r.expected.expected_refund_amount)
        for r in scored
        if r.expected.expected_refund_amount is not None
        and r.predicted_refund_amount is not None
    ]
    mae = sum(diffs) / len(diffs) if diffs else None
    return RefundAmountError(
        denominator_label="cases_with_expected_and_predicted_refund_amount",
        denominator=len(diffs),
        mean_absolute_error=mae,
    )


def score_run(
    records: Sequence[CaseRecord], events: Sequence[CallEvent]
) -> DeterministicMetrics:
    """Compute every M1-T5 deterministic metric for one run's records and events."""
    scored = _scored_population(records)
    matrix = category_confusion_matrix(records)
    return DeterministicMetrics(
        scored_population_size=len(scored),
        total_population_size=len(records),
        unreviewed_structured_error_rate=unreviewed_structured_error_rate(records),
        unreviewed_category_error_rate=unreviewed_category_error_rate(records),
        unreviewed_action_error_rate=unreviewed_action_error_rate(records),
        review_load=review_load(records),
        gate_recall=gate_recall(records),
        gate_precision=gate_precision(records),
        category_confusion_matrix=matrix,
        category_class_metrics=category_class_metrics(matrix),
        action_accuracy=action_accuracy(records),
        refund_amount_error=refund_amount_error(records),
        escalation_rate=escalation_rate(records),
        invalid_output_rate=invalid_output_rate(records, events),
        fallback_usage_rate=fallback_usage_rate(records, events),
        unhelpful_outcome_rate=unhelpful_outcome_rate(records),
    )


def scored_case_predicate_pairs(
    records: Sequence[CaseRecord], predicate: Callable[[CaseRecord], bool]
) -> list[tuple[str, float]]:
    """Return (case_key, 1.0/0.0) pairs over the scored population for M1-T6."""
    return [
        (r.case_key, float(predicate(r))) for r in records if r.prediction_available
    ]
