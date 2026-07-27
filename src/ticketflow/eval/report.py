"""Markdown rendering of the M1 deterministic metrics (plan.md's Reporting)."""

from collections import Counter
from collections.abc import Sequence

from ticketflow.eval.records import CallEvent, CaseRecord
from ticketflow.eval.scorers.deterministic import (
    DeterministicMetrics,
    Rate,
    RateSeries,
    build_rate_series,
    category_f1_clustered_values,
    category_precision_clustered_values,
    category_recall_clustered_values,
    score_run,
)
from ticketflow.eval.statistics import Interval, bootstrap_ci, clustered_statistic_ci
from ticketflow.models import TicketCategory

_DIRECTIONAL_ONLY_THRESHOLD = 100


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


def _f1_statistic(observations: Sequence[tuple[bool, bool]]) -> float:
    """Compute F1 from (is_reference, is_predicted) observations."""
    tp = sum(reference and predicted for reference, predicted in observations)
    fp = sum(not reference and predicted for reference, predicted in observations)
    fn = sum(reference and not predicted for reference, predicted in observations)
    return 2 * tp / (2 * tp + fp + fn)


def _f1_interval_or_none(
    records: Sequence[CaseRecord],
    category: TicketCategory,
    seed: int,
    n_resamples: int,
) -> Interval | None:
    """Return the category F1 interval, or None when its denominator is empty."""
    values = category_f1_clustered_values(records, category)
    if not values:
        return None
    return clustered_statistic_ci(
        values, statistic=_f1_statistic, seed=seed, n_resamples=n_resamples
    )


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
    series_by_name: dict[str, RateSeries],
    seed: int,
    n_resamples: int,
) -> list[str]:
    """Render Quality Metrics; every rate carries an interval and a denominator."""
    lines = ["## Quality Metrics", ""]

    rate_specs = [
        ("Unreviewed structured-error rate", "unreviewed_structured_error_rate"),
        ("Unreviewed category-error rate", "unreviewed_category_error_rate"),
        ("Unreviewed action-error rate", "unreviewed_action_error_rate"),
        ("Review load", "review_load"),
        ("Gate recall", "gate_recall"),
        ("Gate precision", "gate_precision"),
        ("Action accuracy", "action_accuracy"),
    ]
    for label, name in rate_specs:
        series = series_by_name[name]
        interval = _interval_or_none(list(series.clustered_values), seed, n_resamples)
        lines.append(_format_rate_line(label, series.rate, interval))

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


def _render_dataset_composition(records: Sequence[CaseRecord]) -> list[str]:
    """Render dataset composition once per unique case key, ignoring repeats."""
    unique_records = {record.case_key: record for record in records}
    difficulties = Counter(record.difficulty for record in unique_records.values())
    sources = Counter(record.source for record in unique_records.values())
    categories = Counter(
        record.expected.reference_category.value for record in unique_records.values()
    )
    return [
        "## Dataset Composition",
        "",
        f"Unique cases: {len(unique_records)}",
        (
            "Difficulty: "
            f"easy {difficulties['easy']}, "
            f"ambiguous {difficulties['ambiguous']}, "
            f"adversarial {difficulties['adversarial']}"
        ),
        (
            "Source: "
            f"handwritten {sources['handwritten']}, "
            f"generated {sources['generated']}"
        ),
        (
            "Reference category: "
            f"billing {categories['billing']}, "
            f"technical {categories['technical']}, "
            f"account {categories['account']}, "
            f"general {categories['general']}"
        ),
        "",
    ]


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
            category_precision_clustered_values(records, category), seed, n_resamples
        )
        recall_interval = _interval_or_none(
            category_recall_clustered_values(records, category), seed, n_resamples
        )
        precision_text = _format_rate_value(class_metrics.precision, precision_interval)
        recall_text = _format_rate_value(class_metrics.recall, recall_interval)
        f1_interval = _f1_interval_or_none(records, category, seed, n_resamples)
        f1_text = _format_rate_value(class_metrics.f1, f1_interval)
        lines.append(f"- **{category.value}**")
        lines.append(f"  - Precision: {precision_text}")
        lines.append(f"  - Recall: {recall_text}")
        lines.append(f"  - F1: {f1_text}")
    lines.append("")
    return lines


def _render_escalation_and_availability(
    series_by_name: dict[str, RateSeries],
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
    rate_specs = [
        ("Escalation rate", "escalation_rate"),
        ("Invalid-output rate", "invalid_output_rate"),
        ("Fallback usage rate", "fallback_usage_rate"),
        ("Unhelpful outcome rate", "unhelpful_outcome_rate"),
    ]
    for label, name in rate_specs:
        series = series_by_name[name]
        interval = _interval_or_none(list(series.clustered_values), seed, n_resamples)
        lines.append(_format_rate_line(label, series.rate, interval))
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
    series_by_name = build_rate_series(records, events)
    lines = _render_header(
        records, metrics.scored_population_size, metrics.total_population_size
    )
    lines.extend(_render_dataset_composition(records))
    lines.extend(
        _render_quality_metrics(
            records, metrics, series_by_name, bootstrap_seed, n_resamples
        )
    )
    lines.extend(_render_confusion_matrix(metrics))
    lines.extend(
        _render_per_class_metrics(records, metrics, bootstrap_seed, n_resamples)
    )
    lines.extend(
        _render_escalation_and_availability(series_by_name, bootstrap_seed, n_resamples)
    )
    return "\n".join(lines)
