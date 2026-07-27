"""Deterministic correctness predicates and metrics, per plan.md's correctness rules."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

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


class RateSeries(BaseModel):
    """A denominator-bearing rate and its case-clustered binary observations."""

    model_config = ConfigDict(frozen=True)

    rate: Rate
    clustered_values: tuple[tuple[str, float], ...]


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
    f1: Rate


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


def _rate_series(
    pairs: Sequence[tuple[str, float]], denominator_label: str
) -> RateSeries:
    """Build a rate and preserve the exact observations behind its denominator."""
    clustered_values = tuple(pairs)
    rate = Rate(
        numerator=sum(int(value) for _, value in clustered_values),
        denominator=len(clustered_values),
        denominator_label=denominator_label,
    )
    return RateSeries(rate=rate, clustered_values=clustered_values)


def _scored_pairs(
    records: Sequence[CaseRecord], predicate: Callable[[CaseRecord], bool]
) -> list[tuple[str, float]]:
    """Return binary observations over the scored population."""
    return [
        (record.case_key, float(predicate(record)))
        for record in records
        if record.prediction_available
    ]


def _incorrect_scored_pairs(
    records: Sequence[CaseRecord], predicate: Callable[[CaseRecord], bool]
) -> list[tuple[str, float]]:
    """Return observations over scored records with an incorrect outcome."""
    return [
        (record.case_key, float(predicate(record)))
        for record in records
        if record.prediction_available and not structured_correct(record)
    ]


def _gated_scored_pairs(
    records: Sequence[CaseRecord], predicate: Callable[[CaseRecord], bool]
) -> list[tuple[str, float]]:
    """Return observations over gated records in the scored population."""
    return [
        (record.case_key, float(predicate(record)))
        for record in records
        if record.prediction_available and record.was_gated
    ]


def _event_flag_pairs(
    records: Sequence[CaseRecord],
    events: Sequence[CallEvent],
    predicate: Callable[[CallEvent], bool],
) -> list[tuple[str, float]]:
    """Flag every record whose joined call events satisfy predicate."""
    flagged = {_case_identity(event) for event in events if predicate(event)}
    return [
        (record.case_key, float(_case_identity(record) in flagged))
        for record in records
    ]


RateValuesBuilder = Callable[
    [Sequence[CaseRecord], Sequence[CallEvent]], list[tuple[str, float]]
]


@dataclass(frozen=True)
class _RateSpec:
    """One headline rate's denominator and observation builder."""

    denominator_label: str
    build_values: RateValuesBuilder


_RATE_SPECS = {
    "unreviewed_structured_error_rate": _RateSpec(
        "scored_population",
        lambda records, _: _scored_pairs(
            records,
            lambda record: not record.was_gated and not structured_correct(record),
        ),
    ),
    "unreviewed_category_error_rate": _RateSpec(
        "scored_population",
        lambda records, _: _scored_pairs(
            records,
            lambda record: not record.was_gated and not category_correct(record),
        ),
    ),
    "unreviewed_action_error_rate": _RateSpec(
        "scored_population",
        lambda records, _: _scored_pairs(
            records,
            lambda record: not record.was_gated and not action_correct(record),
        ),
    ),
    "review_load": _RateSpec(
        "scored_population",
        lambda records, _: _scored_pairs(records, lambda record: record.was_gated),
    ),
    "gate_recall": _RateSpec(
        "incorrect_scored_cases",
        lambda records, _: _incorrect_scored_pairs(
            records, lambda record: record.was_gated
        ),
    ),
    "gate_precision": _RateSpec(
        "gated_scored_cases",
        lambda records, _: _gated_scored_pairs(
            records, lambda record: not structured_correct(record)
        ),
    ),
    "action_accuracy": _RateSpec(
        "scored_population",
        lambda records, _: _scored_pairs(records, action_correct),
    ),
    "escalation_rate": _RateSpec(
        "all_cases",
        lambda records, _: [
            (record.case_key, float(record.terminal_outcome == "escalated"))
            for record in records
        ],
    ),
    "invalid_output_rate": _RateSpec(
        "all_cases",
        lambda records, events: _event_flag_pairs(
            records, events, lambda event: event.outcome == "invalid_output"
        ),
    ),
    "fallback_usage_rate": _RateSpec(
        "all_cases",
        lambda records, events: _event_flag_pairs(
            records, events, lambda event: event.role == "fallback"
        ),
    ),
    "unhelpful_outcome_rate": _RateSpec(
        "all_cases_combining_quality_and_availability",
        lambda records, _: [
            (
                record.case_key,
                float(
                    not record.prediction_available or not structured_correct(record)
                ),
            )
            for record in records
        ],
    ),
}


def _build_named_rate_series(
    name: str,
    records: Sequence[CaseRecord],
    events: Sequence[CallEvent] = (),
) -> RateSeries:
    """Build one named rate from the central rate specification."""
    spec = _RATE_SPECS[name]
    return _rate_series(spec.build_values(records, events), spec.denominator_label)


def unreviewed_structured_error_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored, ungated cases with an incorrect outcome."""
    return _build_named_rate_series("unreviewed_structured_error_rate", records).rate


def unreviewed_category_error_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored, ungated cases with an incorrect category."""
    return _build_named_rate_series("unreviewed_category_error_rate", records).rate


def unreviewed_action_error_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored, ungated cases with an incorrect action."""
    return _build_named_rate_series("unreviewed_action_error_rate", records).rate


def review_load(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored cases gated for approval."""
    return _build_named_rate_series("review_load", records).rate


def gate_recall(records: Sequence[CaseRecord]) -> Rate:
    """Return P(gated | not structured_correct) over the scored population."""
    return _build_named_rate_series("gate_recall", records).rate


def gate_precision(records: Sequence[CaseRecord]) -> Rate:
    """Return P(not structured_correct | gated) over the scored population."""
    return _build_named_rate_series("gate_precision", records).rate


def action_accuracy(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of scored cases with a correct action, regardless of gating."""
    return _build_named_rate_series("action_accuracy", records).rate


def escalation_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the share of all cases that escalated."""
    return _build_named_rate_series("escalation_rate", records).rate


def invalid_output_rate(
    records: Sequence[CaseRecord], events: Sequence[CallEvent]
) -> Rate:
    """Return the share of all cases with at least one invalid-output call event."""
    return _build_named_rate_series("invalid_output_rate", records, events).rate


def fallback_usage_rate(
    records: Sequence[CaseRecord], events: Sequence[CallEvent]
) -> Rate:
    """Return the share of all cases with at least one fallback-role call event."""
    return _build_named_rate_series("fallback_usage_rate", records, events).rate


def unhelpful_outcome_rate(records: Sequence[CaseRecord]) -> Rate:
    """Return the combined quality-and-availability failure rate over all cases."""
    return _build_named_rate_series("unhelpful_outcome_rate", records).rate


def build_rate_series(
    records: Sequence[CaseRecord], events: Sequence[CallEvent]
) -> dict[str, RateSeries]:
    """Build every headline rate and the exact observations behind each rate."""
    return {
        name: _build_named_rate_series(name, records, events) for name in _RATE_SPECS
    }


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
        f1 = Rate(
            numerator=2 * tp,
            denominator=2 * tp + fp + fn,
            denominator_label=f"f1_2tp_plus_fp_plus_fn_{cat.value}",
        )
        result[cat] = CategoryClassMetrics(
            category=cat, precision=precision, recall=recall, f1=f1
        )
    return result


def category_f1_clustered_values(
    records: Sequence[CaseRecord], category: TicketCategory
) -> list[tuple[str, tuple[bool, bool]]]:
    """Return case-clustered (is_reference, is_predicted) values for class F1."""
    values = []
    for record in records:
        if not record.prediction_available:
            continue
        predicted = _predicted_category_or_raise(record)
        is_reference = record.expected.reference_category == category
        is_predicted = predicted == category
        if is_reference or is_predicted:
            values.append((record.case_key, (is_reference, is_predicted)))
    return values


def category_precision_clustered_values(
    records: Sequence[CaseRecord], category: TicketCategory
) -> list[tuple[str, float]]:
    """Return correctness values over records predicted as category."""
    return [
        (
            record.case_key,
            float(record.expected.reference_category == category),
        )
        for record in records
        if record.prediction_available and record.predicted_category == category
    ]


def category_recall_clustered_values(
    records: Sequence[CaseRecord], category: TicketCategory
) -> list[tuple[str, float]]:
    """Return correctness values over records whose reference is category."""
    return [
        (record.case_key, float(record.predicted_category == category))
        for record in records
        if record.prediction_available
        and record.expected.reference_category == category
    ]


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
    series = build_rate_series(records, events)
    return DeterministicMetrics(
        scored_population_size=len(scored),
        total_population_size=len(records),
        unreviewed_structured_error_rate=series[
            "unreviewed_structured_error_rate"
        ].rate,
        unreviewed_category_error_rate=series["unreviewed_category_error_rate"].rate,
        unreviewed_action_error_rate=series["unreviewed_action_error_rate"].rate,
        review_load=series["review_load"].rate,
        gate_recall=series["gate_recall"].rate,
        gate_precision=series["gate_precision"].rate,
        category_confusion_matrix=matrix,
        category_class_metrics=category_class_metrics(matrix),
        action_accuracy=series["action_accuracy"].rate,
        refund_amount_error=refund_amount_error(records),
        escalation_rate=series["escalation_rate"].rate,
        invalid_output_rate=series["invalid_output_rate"].rate,
        fallback_usage_rate=series["fallback_usage_rate"].rate,
        unhelpful_outcome_rate=series["unhelpful_outcome_rate"].rate,
    )


def scored_case_predicate_pairs(
    records: Sequence[CaseRecord], predicate: Callable[[CaseRecord], bool]
) -> list[tuple[str, float]]:
    """Return (case_key, 1.0/0.0) pairs over the scored population for M1-T6."""
    return [
        (r.case_key, float(predicate(r))) for r in records if r.prediction_available
    ]
