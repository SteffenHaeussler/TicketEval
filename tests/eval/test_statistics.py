import ast
import statistics as pystats
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from ticketflow.eval import statistics as eval_statistics
from ticketflow.eval.statistics import (
    Interval,
    McNemarResult,
    ThresholdSweepCase,
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap_ci,
    threshold_sweep,
)


def _naive_bootstrap_ci(values, *, seed, n_resamples=5000, confidence=0.95):
    """Reference per-observation (non-clustered) bootstrap, for contrast in tests."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        draw = rng.choice(values, size=n, replace=True)
        stats[i] = draw.mean()
    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 - (1 - confidence) / 2) * 100
    low, high = np.percentile(stats, [lower_pct, upper_pct])
    return float(low), float(high)


class TestBootstrapCi:
    def test_same_seed_reproduces_bit_for_bit(self):
        clustered = [(f"case-{i}", float(i % 3)) for i in range(20)]

        first = bootstrap_ci(clustered, seed=42)
        second = bootstrap_ci(clustered, seed=42)

        assert first == second

    def test_repeats_do_not_narrow_interval(self):
        base_values = [float(i % 4) for i in range(10)]

        single_repeat = [(f"case-{i}", v) for i, v in enumerate(base_values)]
        ten_repeats = [
            (f"case-{i}", v) for i, v in enumerate(base_values) for _ in range(10)
        ]

        single_result = bootstrap_ci(single_repeat, seed=7)
        repeated_result = bootstrap_ci(ten_repeats, seed=7)

        assert single_result == repeated_result

    def test_clustered_interval_is_not_narrower_than_naive(self):
        base_values = [float(i % 4) for i in range(10)]
        ten_repeats = [
            (f"case-{i}", v) for i, v in enumerate(base_values) for _ in range(10)
        ]
        flat_values = [v for _, v in ten_repeats]

        clustered = bootstrap_ci(ten_repeats, seed=7)
        naive_low, naive_high = _naive_bootstrap_ci(flat_values, seed=7)

        clustered_width = clustered.high - clustered.low
        naive_width = naive_high - naive_low

        assert clustered_width > naive_width

    def test_point_estimate_is_plain_mean(self):
        clustered = [("a", 1.0), ("a", 1.0), ("b", 0.0), ("c", 1.0)]

        result = bootstrap_ci(clustered, seed=1)

        assert result.point == pystats.mean([1.0, 1.0, 0.0, 1.0])

    def test_returns_interval_instance(self):
        clustered = [("a", 1.0), ("b", 0.0)]

        result = bootstrap_ci(clustered, seed=1)

        assert isinstance(result, Interval)
        assert result.confidence == 0.95
        assert result.low <= result.point <= result.high


class TestClusteredStatisticCi:
    def test_supports_a_derived_statistic_over_structured_values(self):
        values = [
            ("a", (True, True)),
            ("b", (True, False)),
            ("c", (False, True)),
        ]

        def f1(observations):
            tp = sum(reference and predicted for reference, predicted in observations)
            fp = sum(
                not reference and predicted for reference, predicted in observations
            )
            fn = sum(
                reference and not predicted for reference, predicted in observations
            )
            return 2 * tp / (2 * tp + fp + fn)

        result = eval_statistics.clustered_statistic_ci(
            values, statistic=f1, seed=4, n_resamples=100
        )

        assert result.point == 0.5
        assert result.low <= result.point <= result.high


class TestPairedBootstrapCi:
    def test_same_seed_reproduces_bit_for_bit(self):
        pairs = [(f"case-{i}", float(i % 2), float((i + 1) % 2)) for i in range(20)]

        first = paired_bootstrap_ci(pairs, seed=3)
        second = paired_bootstrap_ci(pairs, seed=3)

        assert first == second

    def test_identical_arms_give_zero_point_and_zero_width(self):
        pairs = [(f"case-{i}", 1.0, 1.0) for i in range(10)]

        result = paired_bootstrap_ci(pairs, seed=5)

        assert result.point == 0.0
        assert result.low == 0.0
        assert result.high == 0.0

    def test_point_is_mean_of_differences(self):
        pairs = [("a", 1.0, 0.0), ("b", 1.0, 0.0), ("c", 0.0, 0.0), ("d", 0.0, 1.0)]

        result = paired_bootstrap_ci(pairs, seed=2)

        assert result.point == pystats.mean([1.0, 1.0, 0.0, -1.0])


class TestMcnemarExact:
    def test_matches_hand_computed_binomial(self):
        b, c = 1, 9
        outcomes = [(True, False)] * b + [(False, True)] * c
        expected_p = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue

        result = mcnemar_exact(outcomes)

        assert isinstance(result, McNemarResult)
        assert result.b == b
        assert result.c == c
        assert result.p_value == expected_p

    def test_no_discordant_pairs_gives_p_value_one(self):
        outcomes = [(True, True), (False, False), (True, True)]

        result = mcnemar_exact(outcomes)

        assert result.b == 0
        assert result.c == 0
        assert result.p_value == 1.0

    def test_symmetric_discordance_is_not_significant(self):
        outcomes = [(True, False)] * 5 + [(False, True)] * 5

        result = mcnemar_exact(outcomes)

        assert result.p_value == 1.0


class TestThresholdSweep:
    def test_hand_computed_review_load_and_error_rate(self):
        cases = [
            ThresholdSweepCase(
                is_refund=False, draft_confidence=0.9, structured_correct=True
            ),
            ThresholdSweepCase(
                is_refund=False, draft_confidence=0.4, structured_correct=False
            ),
            ThresholdSweepCase(
                is_refund=True, draft_confidence=0.9, structured_correct=True
            ),
            ThresholdSweepCase(
                is_refund=False, draft_confidence=None, structured_correct=True
            ),
        ]

        result = threshold_sweep(cases)

        assert result.excluded_no_draft_count == 1
        assert len(result.rows) == 21

        by_threshold = {round(row.threshold, 2): row for row in result.rows}

        # threshold 0.00: nothing gated by confidence; only the refund case is gated.
        row = by_threshold[0.00]
        assert row.review_load == pystats.mean([False, False, True])
        assert row.unreviewed_structured_error_rate == pystats.mean(
            [False, True, False]
        )

        # threshold 0.50: the 0.4-confidence case is now also gated on confidence.
        row = by_threshold[0.50]
        assert row.review_load == pystats.mean([False, True, True])
        assert row.unreviewed_structured_error_rate == pystats.mean(
            [False, False, False]
        )

        # threshold 1.00: every case is gated.
        row = by_threshold[1.00]
        assert row.review_load == 1.0
        assert row.unreviewed_structured_error_rate == 0.0

    def test_thresholds_span_zero_to_one_in_twentieths(self):
        cases = [
            ThresholdSweepCase(
                is_refund=False, draft_confidence=0.5, structured_correct=True
            )
        ]

        result = threshold_sweep(cases)

        thresholds = [round(row.threshold, 2) for row in result.rows]

        assert thresholds == [round(i * 0.05, 2) for i in range(21)]

    def test_all_no_draft_cases_are_excluded_and_rows_are_empty_population(self):
        cases = [
            ThresholdSweepCase(
                is_refund=False, draft_confidence=None, structured_correct=True
            ),
            ThresholdSweepCase(
                is_refund=True, draft_confidence=None, structured_correct=False
            ),
        ]

        result = threshold_sweep(cases)

        assert result.excluded_no_draft_count == 2
        assert all(row.review_load == 0.0 for row in result.rows)
        assert all(row.unreviewed_structured_error_rate == 0.0 for row in result.rows)


class TestNoRecordsImport:
    def test_statistics_module_does_not_import_ticketflow(self):
        source_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "ticketflow"
            / "eval"
            / "statistics.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        referenced_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                referenced_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                referenced_modules.append(node.module)

        assert not any(
            module == "ticketflow" or module.startswith("ticketflow.")
            for module in referenced_modules
        )
