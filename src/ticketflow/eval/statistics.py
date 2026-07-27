"""Clustered bootstrap intervals, exact McNemar, and the threshold sweep.

Per execution.md's structural refinement 4, this module operates only on primitive
sequences and dataclasses; it never imports the rest of ticketflow, so it stays testable
independently of the record models built on top of it.
"""

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

_THRESHOLD_STEPS = 21


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of values."""
    return float(np.mean(values))


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided confidence interval."""

    point: float
    low: float
    high: float
    confidence: float = 0.95


def _percentile_bounds(
    resample_stats: np.ndarray, confidence: float
) -> tuple[float, float]:
    """Return the lower/upper percentile bounds of resampled statistics."""
    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 - (1 - confidence) / 2) * 100
    low, high = np.percentile(resample_stats, [lower_pct, upper_pct])
    return float(low), float(high)


def bootstrap_ci(
    clustered_values: Sequence[tuple[Hashable, float]],
    *,
    seed: int,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    statistic: Callable[[Sequence[float]], float] = _mean,
) -> Interval:
    """Return a case-clustered percentile bootstrap interval over clustered_values.

    Each resample draws cluster ids with replacement and keeps every value
    belonging to a drawn cluster, so repeated observations within a cluster never
    add resampling variance beyond the cluster-level draw.
    """
    by_cluster: dict[Hashable, list[float]] = {}
    all_values: list[float] = []
    for cluster_id, value in clustered_values:
        by_cluster.setdefault(cluster_id, []).append(value)
        all_values.append(value)

    cluster_ids = list(by_cluster)
    rng = np.random.default_rng(seed)
    resample_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        pooled = [value for index in drawn for value in by_cluster[cluster_ids[index]]]
        resample_stats[i] = statistic(pooled)

    low, high = _percentile_bounds(resample_stats, confidence)
    return Interval(
        point=statistic(all_values), low=low, high=high, confidence=confidence
    )


def paired_bootstrap_ci(
    clustered_pairs: Sequence[tuple[Hashable, float, float]],
    *,
    seed: int,
    n_resamples: int = 5000,
    confidence: float = 0.95,
) -> Interval:
    """Return a case-clustered bootstrap interval over paired differences (a - b).

    Uses the same cluster-resampling scheme as bootstrap_ci: each resample draws
    cluster ids with replacement and pools all paired differences belonging to
    drawn clusters.
    """
    by_cluster: dict[Hashable, list[float]] = {}
    all_diffs: list[float] = []
    for cluster_id, value_a, value_b in clustered_pairs:
        diff = value_a - value_b
        by_cluster.setdefault(cluster_id, []).append(diff)
        all_diffs.append(diff)

    cluster_ids = list(by_cluster)
    rng = np.random.default_rng(seed)
    resample_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        pooled = [diff for index in drawn for diff in by_cluster[cluster_ids[index]]]
        resample_stats[i] = _mean(pooled)

    low, high = _percentile_bounds(resample_stats, confidence)
    return Interval(point=_mean(all_diffs), low=low, high=high, confidence=confidence)


@dataclass(frozen=True)
class McNemarResult:
    """The discordant-pair counts and exact two-sided McNemar p-value."""

    b: int
    c: int
    p_value: float


def mcnemar_exact(paired_outcomes: Sequence[tuple[bool, bool]]) -> McNemarResult:
    """Run the exact (binomial) McNemar test over paired binary outcomes."""
    b = sum(1 for a, b_ in paired_outcomes if a and not b_)
    c = sum(1 for a, b_ in paired_outcomes if b_ and not a)

    if b + c == 0:
        return McNemarResult(b=b, c=c, p_value=1.0)

    p_value = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    return McNemarResult(b=b, c=c, p_value=p_value)


@dataclass(frozen=True)
class ThresholdSweepCase:
    """One case's inputs to the review-load / unreviewed-error threshold sweep."""

    is_refund: bool
    draft_confidence: float | None
    structured_correct: bool


@dataclass(frozen=True)
class ThresholdSweepRow:
    """Review load and unreviewed structured-error rate at one threshold."""

    threshold: float
    review_load: float
    unreviewed_structured_error_rate: float


@dataclass(frozen=True)
class ThresholdSweepResult:
    """The full threshold sweep plus the count of cases excluded for no draft."""

    rows: list[ThresholdSweepRow]
    excluded_no_draft_count: int


def threshold_sweep(cases: Sequence[ThresholdSweepCase]) -> ThresholdSweepResult:
    """Sweep gating thresholds from 0.00 to 1.00 in 0.05 steps.

    Cases with draft_confidence is None have no confidence to gate on and are
    excluded from every threshold's population; their count is reported alongside
    the rows.
    """
    scored = [
        (case, case.draft_confidence)
        for case in cases
        if case.draft_confidence is not None
    ]
    excluded_no_draft_count = len(cases) - len(scored)

    rows = []
    for step in range(_THRESHOLD_STEPS):
        threshold = round(step * 0.05, 2)
        if not scored:
            rows.append(
                ThresholdSweepRow(
                    threshold=threshold,
                    review_load=0.0,
                    unreviewed_structured_error_rate=0.0,
                )
            )
            continue

        gated = [
            case.is_refund or confidence < threshold for case, confidence in scored
        ]
        unreviewed_errors = [
            (not gate) and (not case.structured_correct)
            for gate, (case, _) in zip(gated, scored, strict=True)
        ]
        rows.append(
            ThresholdSweepRow(
                threshold=threshold,
                review_load=_mean(gated),
                unreviewed_structured_error_rate=_mean(unreviewed_errors),
            )
        )

    return ThresholdSweepResult(
        rows=rows, excluded_no_draft_count=excluded_no_draft_count
    )
