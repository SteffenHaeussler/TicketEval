"""Markdown/console rendering of the M1 deterministic metrics (plan.md's Reporting)."""

from collections.abc import Callable, Sequence

from ticketflow.eval.records import CallEvent, CaseRecord
from ticketflow.eval.scorers.deterministic import (
    DeterministicMetrics,
    Rate,
    action_correct,
    category_correct,
    score_run,
    structured_correct,
)
from ticketflow.eval.statistics import Interval, bootstrap_ci
from ticketflow.models import TicketCategory

_DIRECTIONAL_ONLY_THRESHOLD = 100


def _scored_pairs(
    records: Sequence[CaseRecord], predicate: Callable[[CaseRecord], bool]
) -> list[tuple[str, float]]:
    """Return (case_key, 0/1) pairs over the scored population for predicate."""
    return [
        (r.case_key, float(predicate(r))) for r in records if r.prediction_available
    ]


def _incorrect_scored_pairs(
    records: Sequence[CaseRecord], predicate: Callable[[CaseRecord], bool]
) -> list[tuple[str, float]]:
    """Return (case_key, 0/1) pairs over scored cases with an incorrect outcome."""
    return [
        (r.case_key, float(predicate(r)))
        for r in records
        if r.prediction_available and not structured_correct(r)
    ]


def _gated_scored_pairs(
    records: Sequence[CaseRecord], predicate: Callable[[CaseRecord], bool]
) -> list[tuple[str, float]]:
    """Return (case_key, 0/1) pairs over gated scored cases."""
    return [
        (r.case_key, float(predicate(r)))
        for r in records
        if r.prediction_available and r.was_gated
    ]


def _class_precision_pairs(
    records: Sequence[CaseRecord], category: TicketCategory
) -> list[tuple[str, float]]:
    """Return (case_key, 0/1) pairs over scored cases predicted as category."""
    return [
        (r.case_key, float(r.expected.reference_category == category))
        for r in records
        if r.prediction_available and r.predicted_category == category
    ]


def _class_recall_pairs(
    records: Sequence[CaseRecord], category: TicketCategory
) -> list[tuple[str, float]]:
    """Return (case_key, 0/1) pairs over scored cases whose reference is category."""
    return [
        (r.case_key, float(r.predicted_category == category))
        for r in records
        if r.prediction_available and r.expected.reference_category == category
    ]


def _case_identity(obj: CaseRecord | CallEvent) -> tuple[str, str, str, int, str]:
    """Return the (run_id, policy, case_key, repeat_index, ticket_id) join key."""
    return (obj.run_id, obj.policy, obj.case_key, obj.repeat_index, obj.ticket_id)


def _event_flag_pairs(
    records: Sequence[CaseRecord],
    events: Sequence[CallEvent],
    predicate: Callable[[CallEvent], bool],
) -> list[tuple[str, float]]:
    """Return (case_key, 0/1) pairs flagging cases whose events match predicate."""
    flagged = {_case_identity(e) for e in events if predicate(e)}
    return [(r.case_key, float(_case_identity(r) in flagged)) for r in records]


def _unhelpful_outcome_pairs(records: Sequence[CaseRecord]) -> list[tuple[str, float]]:
    """Return (case_key, 0/1) pairs combining unavailability and structured errors."""
    return [
        (
            r.case_key,
            float((not r.prediction_available) or (not structured_correct(r))),
        )
        for r in records
    ]


def _refund_error_pairs(records: Sequence[CaseRecord]) -> list[tuple[str, float]]:
    """Return (case_key, abs error) pairs for scored cases with both refund amounts."""
    return [
        (r.case_key, abs(r.predicted_refund_amount - r.expected.expected_refund_amount))
        for r in records
        if r.prediction_available
        and r.expected.expected_refund_amount is not None
        and r.predicted_refund_amount is not None
    ]


def _interval_or_none(
    pairs: list[tuple[str, float]], seed: int, n_resamples: int
) -> Interval | None:
    """Return a bootstrap interval over pairs, or None when the population is empty."""
    if not pairs:
        return None
    return bootstrap_ci(pairs, seed=seed, n_resamples=n_resamples)


def _percent(value: float) -> str:
    """Format a fraction as a one-decimal percentage."""
    return f"{value * 100:.1f}%"


def _format_rate_value(rate: Rate, interval: Interval | None) -> str:
    """Render one Rate's value, interval, and denominator, without a label prefix."""
    if interval is None or rate.value is None:
        return f"n/a — {rate.numerator}/{rate.denominator} ({rate.denominator_label})"
    ci = f"CI [{_percent(interval.low)}, {_percent(interval.high)}]"
    return (
        f"{_percent(rate.value)} ({ci}) — "
        f"{rate.numerator}/{rate.denominator} ({rate.denominator_label})"
    )


def _format_rate_line(label: str, rate: Rate, interval: Interval | None) -> str:
    """Render one Rate as a Markdown bullet with its interval and denominator name."""
    return f"- **{label}**: {_format_rate_value(rate, interval)}"


def _format_refund_error_line(
    mae: float | None,
    denominator: int,
    denominator_label: str,
    interval: Interval | None,
) -> str:
    """Render the refund-amount MAE as a Markdown bullet with its interval."""
    denom = f"n={denominator} ({denominator_label})"
    if interval is None or mae is None:
        return f"- **Refund amount MAE**: n/a — {denom}"
    ci = f"CI [{interval.low:.2f}, {interval.high:.2f}]"
    return f"- **Refund amount MAE**: {mae:.2f} ({ci}) — {denom}"


def _render_quality_metrics(
    records: Sequence[CaseRecord],
    metrics: DeterministicMetrics,
    seed: int,
    n_resamples: int,
) -> list[str]:
    """Render Quality Metrics; every rate carries an interval and a denominator."""
    lines = ["## Quality Metrics", ""]

    rate_specs: list[tuple[str, Rate, list[tuple[str, float]]]] = [
        (
            "Unreviewed structured-error rate",
            metrics.unreviewed_structured_error_rate,
            _scored_pairs(
                records, lambda r: not r.was_gated and not structured_correct(r)
            ),
        ),
        (
            "Unreviewed category-error rate",
            metrics.unreviewed_category_error_rate,
            _scored_pairs(
                records, lambda r: not r.was_gated and not category_correct(r)
            ),
        ),
        (
            "Unreviewed action-error rate",
            metrics.unreviewed_action_error_rate,
            _scored_pairs(records, lambda r: not r.was_gated and not action_correct(r)),
        ),
        (
            "Review load",
            metrics.review_load,
            _scored_pairs(records, lambda r: r.was_gated),
        ),
        (
            "Gate recall",
            metrics.gate_recall,
            _incorrect_scored_pairs(records, lambda r: r.was_gated),
        ),
        (
            "Gate precision",
            metrics.gate_precision,
            _gated_scored_pairs(records, lambda r: not structured_correct(r)),
        ),
        (
            "Action accuracy",
            metrics.action_accuracy,
            _scored_pairs(records, action_correct),
        ),
    ]
    for label, rate, pairs in rate_specs:
        interval = _interval_or_none(pairs, seed, n_resamples)
        lines.append(_format_rate_line(label, rate, interval))

    refund_pairs = _refund_error_pairs(records)
    refund_interval = _interval_or_none(refund_pairs, seed, n_resamples)
    lines.append(
        _format_refund_error_line(
            metrics.refund_amount_error.mean_absolute_error,
            metrics.refund_amount_error.denominator,
            metrics.refund_amount_error.denominator_label,
            refund_interval,
        )
    )
    lines.append("")
    return lines


def _render_exclusion_breakdown(records: Sequence[CaseRecord]) -> list[str]:
    """Render the count and per-reason breakdown of unscored cases."""
    unscored = [r for r in records if not r.prediction_available]
    lines = [f"Excluded (no prediction): {len(unscored)}"]
    if unscored:
        by_reason: dict[str, int] = {}
        for r in unscored:
            reason = r.prediction_unavailable_reason or "unspecified"
            by_reason[reason] = by_reason.get(reason, 0) + 1
        for reason in sorted(by_reason):
            lines.append(f"- {reason}: {by_reason[reason]}")
    lines.append("")
    return lines


def _render_header(records: Sequence[CaseRecord], scored: int, total: int) -> list[str]:
    """Render the title, directional-only banner, and population/exclusion lines."""
    lines = ["# Deterministic Metrics Report", ""]
    if scored < _DIRECTIONAL_ONLY_THRESHOLD:
        lines.append(
            f"> **DIRECTIONAL ONLY** — scored population ({scored}) is below 100 "
            "cases; treat every metric below as directional, not decision-grade."
        )
        lines.append("")
    lines.append(f"Scored population: {scored} of {total} total cases")
    lines.append("")
    lines.extend(_render_exclusion_breakdown(records))
    return lines


def _render_confusion_matrix(metrics: DeterministicMetrics) -> list[str]:
    """Render the reference-by-predicted confusion matrix as a Markdown table."""
    matrix = metrics.category_confusion_matrix
    categories = list(TicketCategory)
    columns = " | ".join(c.value for c in categories)
    header = f"| Reference \\ Predicted | {columns} |"
    separator = "| --- | " + " | ".join("---" for _ in categories) + " |"
    lines = ["## Category Confusion Matrix", "", header, separator]
    for ref in categories:
        row = " | ".join(str(matrix.cells[ref][pred]) for pred in categories)
        lines.append(f"| {ref.value} | {row} |")
    lines.append("")
    lines.append(f"_scored population: {matrix.total} ({matrix.denominator_label})_")
    lines.append("")
    return lines


def _render_per_class_metrics(
    records: Sequence[CaseRecord],
    metrics: DeterministicMetrics,
    seed: int,
    n_resamples: int,
) -> list[str]:
    """Render per-category precision/recall/F1; precision and recall carry intervals."""
    lines = ["### Per-Category Precision / Recall / F1", ""]
    for category in TicketCategory:
        class_metrics = metrics.category_class_metrics[category]
        precision_interval = _interval_or_none(
            _class_precision_pairs(records, category), seed, n_resamples
        )
        recall_interval = _interval_or_none(
            _class_recall_pairs(records, category), seed, n_resamples
        )
        precision_text = _format_rate_value(class_metrics.precision, precision_interval)
        recall_text = _format_rate_value(class_metrics.recall, recall_interval)
        f1_text = "n/a" if class_metrics.f1 is None else f"{class_metrics.f1:.3f}"
        lines.append(f"- **{category.value}**")
        lines.append(f"  - Precision: {precision_text}")
        lines.append(f"  - Recall: {recall_text}")
        lines.append(f"  - F1: {f1_text}")
    lines.append("")
    return lines


def _render_escalation_and_availability(
    records: Sequence[CaseRecord],
    events: Sequence[CallEvent],
    metrics: DeterministicMetrics,
    seed: int,
    n_resamples: int,
) -> list[str]:
    """Render escalation/invalid-output/availability rates, apart from quality."""
    lines = [
        "## Escalation & Availability",
        "",
        "_Kept separate from quality metrics: availability failures are not scored "
        "as structured-quality errors._",
        "",
    ]
    rate_specs: list[tuple[str, Rate, list[tuple[str, float]]]] = [
        (
            "Escalation rate",
            metrics.escalation_rate,
            [(r.case_key, float(r.terminal_outcome == "escalated")) for r in records],
        ),
        (
            "Invalid-output rate",
            metrics.invalid_output_rate,
            _event_flag_pairs(records, events, lambda e: e.outcome == "invalid_output"),
        ),
        (
            "Fallback usage rate",
            metrics.fallback_usage_rate,
            _event_flag_pairs(records, events, lambda e: e.role == "fallback"),
        ),
        (
            "Unhelpful outcome rate",
            metrics.unhelpful_outcome_rate,
            _unhelpful_outcome_pairs(records),
        ),
    ]
    for label, rate, pairs in rate_specs:
        interval = _interval_or_none(pairs, seed, n_resamples)
        lines.append(_format_rate_line(label, rate, interval))
    lines.append("")
    return lines


def render_markdown(
    records: Sequence[CaseRecord],
    events: Sequence[CallEvent],
    *,
    bootstrap_seed: int,
    n_resamples: int = 5000,
) -> str:
    """Render the M1 deterministic metrics report as Markdown."""
    metrics = score_run(records, events)
    lines = _render_header(
        records, metrics.scored_population_size, metrics.total_population_size
    )
    lines.extend(_render_quality_metrics(records, metrics, bootstrap_seed, n_resamples))
    lines.extend(_render_confusion_matrix(metrics))
    lines.extend(
        _render_per_class_metrics(records, metrics, bootstrap_seed, n_resamples)
    )
    lines.extend(
        _render_escalation_and_availability(
            records, events, metrics, bootstrap_seed, n_resamples
        )
    )
    return "\n".join(lines)


def render_console(
    records: Sequence[CaseRecord],
    events: Sequence[CallEvent],
    *,
    bootstrap_seed: int,
    n_resamples: int = 5000,
) -> str:
    """Return the deterministic metrics report as plain console text."""
    return render_markdown(
        records, events, bootstrap_seed=bootstrap_seed, n_resamples=n_resamples
    )
