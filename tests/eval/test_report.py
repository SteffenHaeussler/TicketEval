from tests.eval.test_records import make_call_event, make_case_record, make_expected
from ticketflow.eval.report import render_markdown
from ticketflow.models import ApprovalDecision


def _minimal_records(n_scored: int, n_unscored: int = 0, **unscored_overrides):
    records = []
    for i in range(n_scored):
        records.append(
            make_case_record(case_key=f"scored-{i}", ticket_id=f"ticket-{i}")
        )
    for i in range(n_unscored):
        overrides = dict(
            case_key=f"unscored-{i}",
            ticket_id=f"unscored-ticket-{i}",
            prediction_available=False,
            draft_confidence=None,
            classification_confidence=None,
            predicted_category=None,
            predicted_action=None,
            reply_text=None,
            terminal_outcome="escalated",
        )
        overrides.update(unscored_overrides)
        records.append(make_case_record(**overrides))
    return records


def test_header_reports_scored_and_total_population():
    records = _minimal_records(n_scored=2, n_unscored=1)
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "Scored population: 2 of 3 total cases" in text


def test_directional_only_banner_present_below_100_scored():
    records = _minimal_records(n_scored=99)
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "DIRECTIONAL ONLY" in text


def test_directional_only_banner_absent_at_100_scored():
    records = _minimal_records(n_scored=100)
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "DIRECTIONAL ONLY" not in text


def test_exclusion_breakdown_groups_by_reason_and_labels_none_as_unspecified():
    records = _minimal_records(n_scored=1)
    records.append(
        make_case_record(
            case_key="unscored-a",
            ticket_id="ticket-unscored-a",
            prediction_available=False,
            draft_confidence=None,
            classification_confidence=None,
            predicted_category=None,
            predicted_action=None,
            reply_text=None,
            terminal_outcome="escalated",
            prediction_unavailable_reason="agent exhausted repair budget",
        )
    )
    records.append(
        make_case_record(
            case_key="unscored-b",
            ticket_id="ticket-unscored-b",
            prediction_available=False,
            draft_confidence=None,
            classification_confidence=None,
            predicted_category=None,
            predicted_action=None,
            reply_text=None,
            terminal_outcome="escalated",
            prediction_unavailable_reason="agent exhausted repair budget",
        )
    )
    records.append(
        make_case_record(
            case_key="unscored-c",
            ticket_id="ticket-unscored-c",
            prediction_available=False,
            draft_confidence=None,
            classification_confidence=None,
            predicted_category=None,
            predicted_action=None,
            reply_text=None,
            terminal_outcome="escalated",
            prediction_unavailable_reason=None,
        )
    )
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "Excluded (no prediction): 3" in text
    assert "agent exhausted repair budget: 2" in text
    assert "unspecified: 1" in text


def test_exclusion_breakdown_omitted_line_reads_zero_when_nothing_excluded():
    records = _minimal_records(n_scored=2)
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "Excluded (no prediction): 0" in text


# --- quality metrics: rate + interval + denominator ---


def test_action_accuracy_line_carries_interval_and_denominator():
    records = [
        make_case_record(case_key="right-1", predicted_action="reply_only"),
        make_case_record(case_key="right-2", predicted_action="reply_only"),
        make_case_record(case_key="wrong-1", predicted_action="refund"),
    ]
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "Action accuracy" in text
    assert "CI [" in text
    assert "2/3 (scored_population)" in text


def test_zero_denominator_rate_renders_as_na_with_denominator_label():
    # nothing gated => gate_precision has denominator 0
    records = [
        make_case_record(case_key="c-1", was_gated=False),
        make_case_record(case_key="c-2", was_gated=False),
    ]
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "Gate precision" in text
    assert "n/a — 0/0 (gated_scored_cases)" in text


def test_refund_amount_mae_line_carries_interval_and_denominator():
    records = [
        make_case_record(
            case_key="r-1",
            expected=make_expected(
                acceptable_actions=["refund"], expected_refund_amount=50.0
            ),
            predicted_action="refund",
            predicted_refund_amount=45.0,
        ),
        make_case_record(
            case_key="r-2",
            expected=make_expected(
                acceptable_actions=["refund"], expected_refund_amount=100.0
            ),
            predicted_action="refund",
            predicted_refund_amount=90.0,
        ),
    ]
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "Refund amount MAE" in text
    assert "CI [" in text
    assert "n=2 (cases_with_expected_and_predicted_refund_amount)" in text


def test_refund_amount_mae_renders_na_when_no_qualifying_cases():
    records = _minimal_records(n_scored=2)
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "Refund amount MAE" in text
    assert "n/a — n=0 (cases_with_expected_and_predicted_refund_amount)" in text


# --- confusion matrix + per-class metrics ---


def test_confusion_matrix_and_per_class_metrics_render():
    records = [
        make_case_record(
            case_key="b-1",
            expected=make_expected(
                acceptable_categories=["billing"], reference_category="billing"
            ),
            predicted_category="billing",
        ),
        make_case_record(
            case_key="b-2",
            expected=make_expected(
                acceptable_categories=["billing"], reference_category="billing"
            ),
            predicted_category="technical",
        ),
    ]
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "Category Confusion Matrix" in text
    assert "billing" in text and "technical" in text
    assert "Precision: 100.0% (CI [100.0%, 100.0%]) — 1/1 (predicted_billing)" in text
    assert "Recall: 50.0% (CI [" in text and "(reference_billing)" in text
    assert "F1: 0.667" in text


# --- escalation & availability: visually separate block ---


def test_escalation_and_availability_section_present_with_intervals():
    records = _minimal_records(n_scored=2, n_unscored=1)
    events = [
        make_call_event(case_key="scored-0", ticket_id="ticket-0", role="fallback"),
    ]
    text = render_markdown(records, events, bootstrap_seed=1, n_resamples=50)
    assert "Escalation & Availability" in text
    assert "Escalation rate" in text
    assert "Invalid-output rate" in text
    assert "Fallback usage rate" in text
    assert "Unhelpful outcome rate" in text
    assert "(all_cases)" in text


def test_escalation_and_availability_section_appears_after_quality_metrics():
    records = _minimal_records(n_scored=2, n_unscored=1)
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    quality_idx = text.index("## Quality Metrics")
    escalation_idx = text.index("## Escalation & Availability")
    assert escalation_idx > quality_idx
    for label in [
        "Escalation rate",
        "Invalid-output rate",
        "Fallback usage rate",
        "Unhelpful outcome rate",
    ]:
        assert text.index(label) > quality_idx


# --- out-of-scope sections absent ---


def test_no_threshold_sweep_or_judge_sections():
    records = _minimal_records(n_scored=3, n_unscored=1)
    text = render_markdown(records, [], bootstrap_seed=1, n_resamples=50)
    assert "threshold" not in text.lower()
    assert "judge" not in text.lower()


# --- console alias ---


def test_render_console_matches_render_markdown():
    from ticketflow.eval.report import render_console

    records = _minimal_records(n_scored=3, n_unscored=1)
    events = [
        make_call_event(case_key="scored-0", ticket_id="ticket-0", role="fallback")
    ]
    assert render_console(
        records, events, bootstrap_seed=1, n_resamples=50
    ) == render_markdown(records, events, bootstrap_seed=1, n_resamples=50)


# --- golden file ---

_GOLDEN_MARKDOWN = (
    "# Deterministic Metrics Report\n"
    "\n"
    "> **DIRECTIONAL ONLY** — scored population (5) is below 100 cases; treat every "
    "metric below as directional, not decision-grade.\n"
    "\n"
    "Scored population: 5 of 6 total cases\n"
    "\n"
    "Excluded (no prediction): 1\n"
    "- agent exhausted repair budget: 1\n"
    "\n"
    "## Quality Metrics\n"
    "\n"
    "- **Unreviewed structured-error rate**: 20.0% (CI [0.0%, 60.0%]) — "
    "1/5 (scored_population)\n"
    "- **Unreviewed category-error rate**: 0.0% (CI [0.0%, 0.0%]) — "
    "0/5 (scored_population)\n"
    "- **Unreviewed action-error rate**: 0.0% (CI [0.0%, 0.0%]) — "
    "0/5 (scored_population)\n"
    "- **Review load**: 40.0% (CI [0.0%, 80.0%]) — 2/5 (scored_population)\n"
    "- **Gate recall**: 50.0% (CI [0.0%, 100.0%]) — 1/2 (incorrect_scored_cases)\n"
    "- **Gate precision**: 50.0% (CI [0.0%, 100.0%]) — 1/2 (gated_scored_cases)\n"
    "- **Action accuracy**: 100.0% (CI [100.0%, 100.0%]) — 5/5 (scored_population)\n"
    "- **Refund amount MAE**: 10.00 (CI [0.00, 20.00]) — "
    "n=2 (cases_with_expected_and_predicted_refund_amount)\n"
    "\n"
    "## Category Confusion Matrix\n"
    "\n"
    "| Reference \\ Predicted | billing | technical | account | general |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| billing | 3 | 0 | 0 | 0 |\n"
    "| technical | 0 | 0 | 0 | 1 |\n"
    "| account | 0 | 0 | 1 | 0 |\n"
    "| general | 0 | 0 | 0 | 0 |\n"
    "\n"
    "_scored population: 5 (scored_population)_\n"
    "\n"
    "### Per-Category Precision / Recall / F1\n"
    "\n"
    "- **billing**\n"
    "  - Precision: 100.0% (CI [100.0%, 100.0%]) — 3/3 (predicted_billing)\n"
    "  - Recall: 100.0% (CI [100.0%, 100.0%]) — 3/3 (reference_billing)\n"
    "  - F1: 1.000\n"
    "- **technical**\n"
    "  - Precision: n/a — 0/0 (predicted_technical)\n"
    "  - Recall: 0.0% (CI [0.0%, 0.0%]) — 0/1 (reference_technical)\n"
    "  - F1: n/a\n"
    "- **account**\n"
    "  - Precision: 100.0% (CI [100.0%, 100.0%]) — 1/1 (predicted_account)\n"
    "  - Recall: 100.0% (CI [100.0%, 100.0%]) — 1/1 (reference_account)\n"
    "  - F1: 1.000\n"
    "- **general**\n"
    "  - Precision: 0.0% (CI [0.0%, 0.0%]) — 0/1 (predicted_general)\n"
    "  - Recall: n/a — 0/0 (reference_general)\n"
    "  - F1: n/a\n"
    "\n"
    "## Escalation & Availability\n"
    "\n"
    "_Kept separate from quality metrics: availability failures are not scored "
    "as structured-quality errors._\n"
    "\n"
    "- **Escalation rate**: 16.7% (CI [0.0%, 50.0%]) — 1/6 (all_cases)\n"
    "- **Invalid-output rate**: 16.7% (CI [0.0%, 50.0%]) — 1/6 (all_cases)\n"
    "- **Fallback usage rate**: 16.7% (CI [0.0%, 50.0%]) — 1/6 (all_cases)\n"
    "- **Unhelpful outcome rate**: 50.0% (CI [16.7%, 83.3%]) — "
    "3/6 (all_cases_combining_quality_and_availability)\n"
)


def _build_golden_fixture():
    records = []
    events = []

    records.append(
        make_case_record(
            case_key="case-01",
            ticket_id="ticket-01",
            expected=make_expected(
                acceptable_categories=["billing"],
                reference_category="billing",
                acceptable_actions=["reply_only"],
            ),
            predicted_category="billing",
            predicted_action="reply_only",
            was_gated=False,
            terminal_outcome="resolved",
        )
    )
    events.append(
        make_call_event(
            case_key="case-01",
            ticket_id="ticket-01",
            operation="classify",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )
    events.append(
        make_call_event(
            case_key="case-01",
            ticket_id="ticket-01",
            operation="draft",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )

    records.append(
        make_case_record(
            case_key="case-02",
            ticket_id="ticket-02",
            expected=make_expected(
                acceptable_categories=["technical"],
                reference_category="technical",
                acceptable_actions=["reply_only"],
            ),
            predicted_category="general",
            predicted_action="reply_only",
            was_gated=True,
            terminal_outcome="rejected",
            decision=ApprovalDecision(approved=False, approver="oracle-reviewer"),
        )
    )
    events.append(
        make_call_event(
            case_key="case-02",
            ticket_id="ticket-02",
            operation="classify",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )
    events.append(
        make_call_event(
            case_key="case-02",
            ticket_id="ticket-02",
            operation="draft",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )

    records.append(
        make_case_record(
            case_key="case-03",
            ticket_id="ticket-03",
            expected=make_expected(
                acceptable_categories=["general"],
                reference_category="general",
                acceptable_actions=["reply_only"],
            ),
            terminal_outcome="escalated",
            draft_confidence=None,
            classification_confidence=None,
            predicted_category=None,
            predicted_action=None,
            reply_text=None,
            was_gated=False,
            prediction_available=False,
            prediction_unavailable_reason="agent exhausted repair budget",
        )
    )
    events.append(
        make_call_event(
            case_key="case-03",
            ticket_id="ticket-03",
            operation="classify",
            role="primary",
            attempt=1,
            outcome="invalid_output",
            error_type="bad_json",
        )
    )
    events.append(
        make_call_event(
            case_key="case-03",
            ticket_id="ticket-03",
            operation="classify",
            role="primary",
            attempt=2,
            outcome="invalid_output",
            error_type="bad_json",
        )
    )

    records.append(
        make_case_record(
            case_key="case-04",
            ticket_id="ticket-04",
            expected=make_expected(
                acceptable_categories=["account"],
                reference_category="account",
                acceptable_actions=["reply_only"],
            ),
            predicted_category="account",
            predicted_action="reply_only",
            was_gated=False,
            terminal_outcome="resolved",
        )
    )
    events.append(
        make_call_event(
            case_key="case-04",
            ticket_id="ticket-04",
            operation="classify",
            role="fallback",
            attempt=1,
            outcome="success",
        )
    )
    events.append(
        make_call_event(
            case_key="case-04",
            ticket_id="ticket-04",
            operation="draft",
            role="fallback",
            attempt=1,
            outcome="success",
        )
    )

    records.append(
        make_case_record(
            case_key="case-05",
            ticket_id="ticket-05",
            expected=make_expected(
                acceptable_categories=["billing"],
                reference_category="billing",
                acceptable_actions=["refund"],
                expected_refund_amount=50.0,
            ),
            predicted_category="billing",
            predicted_action="refund",
            predicted_refund_amount=50.0,
            was_gated=True,
            terminal_outcome="resolved",
            decision=ApprovalDecision(approved=True, approver="oracle-reviewer"),
        )
    )
    events.append(
        make_call_event(
            case_key="case-05",
            ticket_id="ticket-05",
            operation="classify",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )
    events.append(
        make_call_event(
            case_key="case-05",
            ticket_id="ticket-05",
            operation="draft",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )

    records.append(
        make_case_record(
            case_key="case-06",
            ticket_id="ticket-06",
            expected=make_expected(
                acceptable_categories=["billing"],
                reference_category="billing",
                acceptable_actions=["refund"],
                expected_refund_amount=100.0,
            ),
            predicted_category="billing",
            predicted_action="refund",
            predicted_refund_amount=80.0,
            was_gated=False,
            terminal_outcome="resolved",
        )
    )
    events.append(
        make_call_event(
            case_key="case-06",
            ticket_id="ticket-06",
            operation="classify",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )
    events.append(
        make_call_event(
            case_key="case-06",
            ticket_id="ticket-06",
            operation="draft",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )

    return records, events


def test_render_markdown_golden_file_matches_pinned_text():
    records, events = _build_golden_fixture()
    text = render_markdown(records, events, bootstrap_seed=20260727, n_resamples=300)
    assert text == _GOLDEN_MARKDOWN
