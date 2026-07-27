import pytest
from pydantic import ValidationError

from tests.eval.test_records import make_call_event, make_case_record, make_expected
from ticketflow.eval.scorers import deterministic
from ticketflow.eval.scorers.deterministic import (
    ConfusionMatrix,
    MissingPredictionFieldError,
    Rate,
    RefundAmountError,
    UnscoredCaseError,
    action_accuracy,
    action_correct,
    category_class_metrics,
    category_confusion_matrix,
    category_correct,
    escalation_rate,
    fallback_usage_rate,
    gate_precision,
    gate_recall,
    invalid_output_rate,
    refund_amount_error,
    refund_correct,
    review_load,
    score_run,
    scored_case_predicate_pairs,
    structured_correct,
    unhelpful_outcome_rate,
    unreviewed_action_error_rate,
    unreviewed_structured_error_rate,
)
from ticketflow.models import ApprovalDecision, TicketCategory, TicketStatus

# --- shared 14-case synthetic fixture ---


def _build_fixture():
    records = []
    events = []

    # case-01: correct, ungated, repaired (invalid_output then success on classify)
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
            outcome="invalid_output",
            error_type="bad_json",
        )
    )
    events.append(
        make_call_event(
            case_key="case-01",
            ticket_id="ticket-01",
            operation="classify",
            role="primary",
            attempt=2,
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

    # case-02: category wrong, ungated
    records.append(
        make_case_record(
            case_key="case-02",
            ticket_id="ticket-02",
            expected=make_expected(
                acceptable_categories=["billing"],
                reference_category="billing",
                acceptable_actions=["reply_only"],
            ),
            predicted_category="technical",
            predicted_action="reply_only",
            was_gated=False,
            terminal_outcome="resolved",
        )
    )

    # case-03: action wrong (refund not acceptable), gated, oracle-rejected
    records.append(
        make_case_record(
            case_key="case-03",
            ticket_id="ticket-03",
            expected=make_expected(
                acceptable_categories=["billing"],
                reference_category="billing",
                acceptable_actions=["reply_only"],
                expected_refund_amount=50.0,
            ),
            predicted_category="billing",
            predicted_action="refund",
            predicted_refund_amount=50.0,
            was_gated=True,
            terminal_outcome="rejected",
            decision=ApprovalDecision(approved=False, approver="oracle-reviewer"),
        )
    )

    # case-04: fully correct refund, gated, approved
    records.append(
        make_case_record(
            case_key="case-04",
            ticket_id="ticket-04",
            expected=make_expected(
                acceptable_categories=["billing"],
                reference_category="billing",
                acceptable_actions=["refund", "reply_only"],
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

    # case-05: refund amount outside tolerance, ungated
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
            predicted_refund_amount=40.0,
            was_gated=False,
            terminal_outcome="resolved",
        )
    )

    # case-06: fully correct, ungated, fallback role
    records.append(
        make_case_record(
            case_key="case-06",
            ticket_id="ticket-06",
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
            case_key="case-06",
            ticket_id="ticket-06",
            operation="classify",
            role="fallback",
            attempt=1,
            outcome="success",
        )
    )
    events.append(
        make_call_event(
            case_key="case-06",
            ticket_id="ticket-06",
            operation="draft",
            role="fallback",
            attempt=1,
            outcome="success",
        )
    )

    # case-07: category wrong, gated, oracle-rejected
    records.append(
        make_case_record(
            case_key="case-07",
            ticket_id="ticket-07",
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

    # case-08: category accepted by set but differs from reference_category
    records.append(
        make_case_record(
            case_key="case-08",
            ticket_id="ticket-08",
            expected=make_expected(
                acceptable_categories=["billing", "general"],
                reference_category="billing",
                acceptable_actions=["reply_only"],
            ),
            predicted_category="general",
            predicted_action="reply_only",
            was_gated=False,
            terminal_outcome="resolved",
        )
    )

    # case-09: refund amount outside tolerance, ungated
    records.append(
        make_case_record(
            case_key="case-09",
            ticket_id="ticket-09",
            expected=make_expected(
                acceptable_categories=["account"],
                reference_category="account",
                acceptable_actions=["refund"],
                expected_refund_amount=50.0,
            ),
            predicted_category="account",
            predicted_action="refund",
            predicted_refund_amount=45.0,
            was_gated=False,
            terminal_outcome="resolved",
        )
    )

    # case-10: action wrong (reply_only when only refund acceptable), ungated
    records.append(
        make_case_record(
            case_key="case-10",
            ticket_id="ticket-10",
            expected=make_expected(
                acceptable_categories=["general"],
                reference_category="general",
                acceptable_actions=["refund"],
            ),
            predicted_category="general",
            predicted_action="reply_only",
            was_gated=False,
            terminal_outcome="resolved",
        )
    )

    # case-11: escalated before a draft, exhausted repair (no prediction)
    records.append(
        make_case_record(
            case_key="case-11",
            ticket_id="ticket-11",
            expected=make_expected(
                acceptable_categories=["technical"],
                reference_category="technical",
                acceptable_actions=["reply_only"],
            ),
            terminal_status=TicketStatus.ESCALATED,
            terminal_outcome="escalated",
            draft_confidence=None,
            classification_confidence=None,
            predicted_category=None,
            predicted_action=None,
            predicted_refund_amount=None,
            reply_text=None,
            was_gated=False,
            prediction_available=False,
            prediction_unavailable_reason="agent exhausted repair budget",
        )
    )
    events.append(
        make_call_event(
            case_key="case-11",
            ticket_id="ticket-11",
            operation="classify",
            role="primary",
            attempt=1,
            outcome="invalid_output",
            error_type="bad_json",
        )
    )
    events.append(
        make_call_event(
            case_key="case-11",
            ticket_id="ticket-11",
            operation="classify",
            role="primary",
            attempt=2,
            outcome="invalid_output",
            error_type="bad_json",
        )
    )

    # case-12: escalated before a draft, agent failure (no prediction)
    records.append(
        make_case_record(
            case_key="case-12",
            ticket_id="ticket-12",
            expected=make_expected(
                acceptable_categories=["general"],
                reference_category="general",
                acceptable_actions=["reply_only"],
            ),
            terminal_status=TicketStatus.ESCALATED,
            terminal_outcome="escalated",
            draft_confidence=None,
            classification_confidence=None,
            predicted_category=None,
            predicted_action=None,
            predicted_refund_amount=None,
            reply_text=None,
            was_gated=False,
            prediction_available=False,
            prediction_unavailable_reason="agent permanent error",
        )
    )
    events.append(
        make_call_event(
            case_key="case-12",
            ticket_id="ticket-12",
            operation="classify",
            role="primary",
            attempt=1,
            outcome="permanent_error",
            error_type="agent_permanent_error",
        )
    )

    # case-13: fully correct, runner deadline exceeded after draft, remains scored
    records.append(
        make_case_record(
            case_key="case-13",
            ticket_id="ticket-13",
            expected=make_expected(
                acceptable_categories=["account"],
                reference_category="account",
                acceptable_actions=["reply_only"],
            ),
            predicted_category="account",
            predicted_action="reply_only",
            was_gated=False,
            terminal_status=TicketStatus.ESCALATED,
            terminal_outcome="runner_deadline_exceeded",
            cleanup_action="terminated",
        )
    )
    events.append(
        make_call_event(
            case_key="case-13",
            ticket_id="ticket-13",
            operation="classify",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )
    events.append(
        make_call_event(
            case_key="case-13",
            ticket_id="ticket-13",
            operation="draft",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )

    # case-14: fully correct, gated, but the workflow update was rejected post-draft
    records.append(
        make_case_record(
            case_key="case-14",
            ticket_id="ticket-14",
            expected=make_expected(
                acceptable_categories=["billing"],
                reference_category="billing",
                acceptable_actions=["reply_only", "refund"],
            ),
            predicted_category="billing",
            predicted_action="reply_only",
            was_gated=True,
            terminal_outcome="update_rejected",
            decision=ApprovalDecision(approved=True, approver="oracle-reviewer"),
        )
    )
    events.append(
        make_call_event(
            case_key="case-14",
            ticket_id="ticket-14",
            operation="classify",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )
    events.append(
        make_call_event(
            case_key="case-14",
            ticket_id="ticket-14",
            operation="draft",
            role="primary",
            attempt=1,
            outcome="success",
        )
    )

    # remaining plain classify+draft success events (cases 02,03,04,05,07,08,09,10)
    for case_key, ticket_id in [
        ("case-02", "ticket-02"),
        ("case-03", "ticket-03"),
        ("case-04", "ticket-04"),
        ("case-05", "ticket-05"),
        ("case-07", "ticket-07"),
        ("case-08", "ticket-08"),
        ("case-09", "ticket-09"),
        ("case-10", "ticket-10"),
    ]:
        events.append(
            make_call_event(
                case_key=case_key,
                ticket_id=ticket_id,
                operation="classify",
                role="primary",
                attempt=1,
                outcome="success",
            )
        )
        events.append(
            make_call_event(
                case_key=case_key,
                ticket_id=ticket_id,
                operation="draft",
                role="primary",
                attempt=1,
                outcome="success",
            )
        )

    return records, events


FIXTURE_RECORDS, FIXTURE_EVENTS = _build_fixture()


# --- predicates ---


def test_category_correct_true_for_reference_category_match():
    record = make_case_record(
        expected=make_expected(
            acceptable_categories=["billing"], reference_category="billing"
        ),
        predicted_category="billing",
    )
    assert category_correct(record) is True


def test_category_correct_true_when_prediction_differs_from_reference_but_in_set():
    record = next(r for r in FIXTURE_RECORDS if r.case_key == "case-08")
    assert record.predicted_category != record.expected.reference_category
    assert category_correct(record) is True


def test_category_correct_false_outside_acceptable_set():
    record = next(r for r in FIXTURE_RECORDS if r.case_key == "case-02")
    assert category_correct(record) is False


def test_action_correct_true_and_false():
    correct = next(r for r in FIXTURE_RECORDS if r.case_key == "case-01")
    wrong = next(r for r in FIXTURE_RECORDS if r.case_key == "case-10")
    assert action_correct(correct) is True
    assert action_correct(wrong) is False


def test_refund_correct_true_when_action_is_not_refund():
    record = next(r for r in FIXTURE_RECORDS if r.case_key == "case-01")
    assert refund_correct(record) is True


def test_refund_correct_true_within_tolerance():
    record = next(r for r in FIXTURE_RECORDS if r.case_key == "case-04")
    assert refund_correct(record) is True


def test_refund_correct_false_outside_tolerance():
    record = next(r for r in FIXTURE_RECORDS if r.case_key == "case-05")
    assert refund_correct(record) is False


def test_refund_correct_false_when_expected_amount_missing():
    record = make_case_record(
        expected=make_expected(
            acceptable_categories=["billing"],
            reference_category="billing",
            acceptable_actions=["refund"],
            expected_refund_amount=None,
        ),
        predicted_action="refund",
        predicted_refund_amount=50.0,
    )
    assert refund_correct(record) is False


def test_refund_correct_false_when_predicted_amount_missing():
    record = make_case_record(
        expected=make_expected(
            acceptable_categories=["billing"],
            reference_category="billing",
            acceptable_actions=["refund"],
            expected_refund_amount=50.0,
        ),
        predicted_action="refund",
        predicted_refund_amount=None,
    )
    assert refund_correct(record) is False


def test_structured_correct_requires_all_three():
    action_fails = next(r for r in FIXTURE_RECORDS if r.case_key == "case-03")
    refund_fails = next(r for r in FIXTURE_RECORDS if r.case_key == "case-05")
    all_pass = next(r for r in FIXTURE_RECORDS if r.case_key == "case-01")
    assert structured_correct(action_fails) is False
    assert structured_correct(refund_fails) is False
    assert structured_correct(all_pass) is True


def test_category_correct_raises_unscored_case_error_on_case_without_draft():
    record = next(r for r in FIXTURE_RECORDS if r.case_key == "case-11")
    with pytest.raises(UnscoredCaseError, match="case-11"):
        category_correct(record)


def test_action_correct_raises_unscored_case_error_on_case_without_draft():
    record = next(r for r in FIXTURE_RECORDS if r.case_key == "case-11")
    with pytest.raises(UnscoredCaseError, match="case-11"):
        action_correct(record)


def test_category_correct_raises_missing_field_error_when_predicted_category_none():
    record = make_case_record(
        draft_confidence=0.9, prediction_available=True, predicted_category=None
    )
    with pytest.raises(MissingPredictionFieldError, match="case-1"):
        category_correct(record)


# --- scored-population exclusion ---


def test_cases_without_draft_absent_from_quality_denominators():
    scored = make_case_record(
        case_key="scored-1", prediction_available=True, draft_confidence=0.9
    )
    unscored = make_case_record(
        case_key="unscored-1",
        prediction_available=False,
        draft_confidence=None,
        classification_confidence=None,
        predicted_category=None,
        predicted_action=None,
        reply_text=None,
        terminal_outcome="escalated",
    )
    records = [
        scored,
        scored.model_copy(update={"case_key": "scored-2"}),
        unscored,
        unscored.model_copy(update={"case_key": "unscored-2"}),
    ]
    assert unreviewed_structured_error_rate(records).denominator == 2
    assert review_load(records).denominator == 2


def test_repaired_case_with_draft_remains_scored():
    record = next(r for r in FIXTURE_RECORDS if r.case_key == "case-01")
    assert record.prediction_available is True
    assert structured_correct(record) is True


# --- invalid_output_rate semantics ---


def test_invalid_output_rate_counts_cases_not_attempts():
    rate = invalid_output_rate(FIXTURE_RECORDS, FIXTURE_EVENTS)
    case01_events = [e for e in FIXTURE_EVENTS if e.case_key == "case-01"]
    assert sum(1 for e in case01_events if e.outcome == "invalid_output") == 1
    assert rate.numerator == 2  # case-01 and case-11, each counted once


def test_invalid_output_rate_counts_exhausted_repair_without_draft():
    case11 = next(r for r in FIXTURE_RECORDS if r.case_key == "case-11")
    assert case11.prediction_available is False
    flagged_case_keys = {
        e.case_key for e in FIXTURE_EVENTS if e.outcome == "invalid_output"
    }
    assert "case-11" in flagged_case_keys


def test_invalid_output_rate_denominator_is_all_cases_not_scored():
    rate = invalid_output_rate(FIXTURE_RECORDS, FIXTURE_EVENTS)
    assert rate.denominator == len(FIXTURE_RECORDS) == 14
    assert rate.denominator != sum(1 for r in FIXTURE_RECORDS if r.prediction_available)


# --- gate_recall / gate_precision policy invariance ---


def _make_gate_invariance_records(policy: str):
    expected_correct = make_expected(
        acceptable_categories=["billing"],
        reference_category="billing",
        acceptable_actions=["reply_only"],
    )
    expected_wrong = make_expected(
        acceptable_categories=["billing"],
        reference_category="billing",
        acceptable_actions=["reply_only"],
    )
    reviewer = "oracle-reviewer" if policy == "oracle" else "rubber-stamp"
    return [
        make_case_record(
            case_key="case-a",
            ticket_id="ticket-a",
            policy=policy,
            expected=expected_wrong,
            predicted_category="technical",
            predicted_action="reply_only",
            was_gated=True,
            terminal_outcome="rejected" if policy == "oracle" else "resolved",
            decision=ApprovalDecision(approved=(policy != "oracle"), approver=reviewer),
        ),
        make_case_record(
            case_key="case-b",
            ticket_id="ticket-b",
            policy=policy,
            expected=expected_correct,
            predicted_category="billing",
            predicted_action="reply_only",
            was_gated=True,
            terminal_outcome="resolved",
            decision=ApprovalDecision(approved=True, approver=reviewer),
        ),
        make_case_record(
            case_key="case-c",
            ticket_id="ticket-c",
            policy=policy,
            expected=expected_wrong,
            predicted_category="technical",
            predicted_action="reply_only",
            was_gated=False,
            terminal_outcome="resolved",
            decision=None,
        ),
        make_case_record(
            case_key="case-d",
            ticket_id="ticket-d",
            policy=policy,
            expected=expected_correct,
            predicted_category="billing",
            predicted_action="reply_only",
            was_gated=False,
            terminal_outcome="resolved",
            decision=None,
        ),
    ]


def test_gate_recall_identical_between_oracle_and_rubber_stamp_records():
    oracle_records = _make_gate_invariance_records("oracle")
    rubber_stamp_records = _make_gate_invariance_records("rubber_stamp")
    assert gate_recall(oracle_records) == gate_recall(rubber_stamp_records)


def test_gate_precision_identical_between_oracle_and_rubber_stamp_records():
    oracle_records = _make_gate_invariance_records("oracle")
    rubber_stamp_records = _make_gate_invariance_records("rubber_stamp")
    assert gate_precision(oracle_records) == gate_precision(rubber_stamp_records)


# --- confusion matrix / per-class metrics ---


def test_category_confusion_matrix_cells_match_hand_computation():
    matrix = category_confusion_matrix(FIXTURE_RECORDS)
    assert matrix.cells[TicketCategory.BILLING][TicketCategory.BILLING] == 5
    assert matrix.cells[TicketCategory.BILLING][TicketCategory.TECHNICAL] == 1
    assert matrix.cells[TicketCategory.BILLING][TicketCategory.GENERAL] == 1
    assert matrix.cells[TicketCategory.ACCOUNT][TicketCategory.ACCOUNT] == 3
    assert matrix.cells[TicketCategory.TECHNICAL][TicketCategory.GENERAL] == 1
    assert matrix.cells[TicketCategory.GENERAL][TicketCategory.GENERAL] == 1


def test_category_confusion_matrix_excludes_unscored_cases():
    matrix = category_confusion_matrix(FIXTURE_RECORDS)
    assert matrix.total == 12


def test_category_class_metrics_billing_precision_recall_f1():
    matrix = category_confusion_matrix(FIXTURE_RECORDS)
    metrics = category_class_metrics(matrix)[TicketCategory.BILLING]
    assert metrics.precision == Rate(
        numerator=5, denominator=5, denominator_label="predicted_billing"
    )
    assert metrics.recall == Rate(
        numerator=5, denominator=7, denominator_label="reference_billing"
    )
    assert metrics.f1 == Rate(
        numerator=10,
        denominator=12,
        denominator_label="f1_2tp_plus_fp_plus_fn_billing",
    )


def test_category_class_metrics_zero_precision_and_recall_gives_f1_zero_not_none():
    matrix = category_confusion_matrix(FIXTURE_RECORDS)
    metrics = category_class_metrics(matrix)[TicketCategory.TECHNICAL]
    assert metrics.precision.value == 0.0
    assert metrics.recall.value == 0.0
    assert metrics.f1.value == 0.0


def test_category_class_metrics_perfect_class_has_f1_one():
    matrix = category_confusion_matrix(FIXTURE_RECORDS)
    metrics = category_class_metrics(matrix)[TicketCategory.ACCOUNT]
    assert metrics.precision.value == 1.0
    assert metrics.recall.value == 1.0
    assert metrics.f1.value == 1.0


def test_category_class_metrics_general_precision_recall_f1():
    matrix = category_confusion_matrix(FIXTURE_RECORDS)
    metrics = category_class_metrics(matrix)[TicketCategory.GENERAL]
    assert metrics.precision.value == pytest.approx(1 / 3)
    assert metrics.recall.value == 1.0
    assert metrics.f1.value == pytest.approx(0.5)


# --- refund amount error ---


def test_refund_amount_error_mean_absolute_error():
    result = refund_amount_error(FIXTURE_RECORDS)
    assert result.denominator == 4
    assert result.mean_absolute_error == pytest.approx(3.75)


def test_refund_amount_error_excludes_cases_missing_either_amount():
    case01 = next(r for r in FIXTURE_RECORDS if r.case_key == "case-01")
    case10 = next(r for r in FIXTURE_RECORDS if r.case_key == "case-10")
    assert case01.predicted_refund_amount is None
    assert case10.predicted_refund_amount is None


def test_refund_amount_error_none_when_no_qualifying_cases():
    records = [
        make_case_record(
            expected=make_expected(expected_refund_amount=None),
            predicted_refund_amount=None,
        )
    ]
    result = refund_amount_error(records)
    assert result.denominator == 0
    assert result.mean_absolute_error is None


# --- action accuracy / escalation / fallback usage ---


def test_action_accuracy_over_full_scored_population_ignores_gating():
    accuracy = action_accuracy(FIXTURE_RECORDS)
    unreviewed_error = unreviewed_action_error_rate(FIXTURE_RECORDS)
    assert accuracy == Rate(
        numerator=10, denominator=12, denominator_label="scored_population"
    )
    assert unreviewed_error == Rate(
        numerator=1, denominator=12, denominator_label="scored_population"
    )
    # not complements: different populations, not simply 1 - x
    assert accuracy.numerator + unreviewed_error.numerator != accuracy.denominator


def test_escalation_rate_over_all_cases():
    rate = escalation_rate(FIXTURE_RECORDS)
    assert rate == Rate(numerator=2, denominator=14, denominator_label="all_cases")


def test_fallback_usage_rate_flags_case_with_any_fallback_role_event():
    rate = fallback_usage_rate(FIXTURE_RECORDS, FIXTURE_EVENTS)
    assert rate == Rate(numerator=1, denominator=14, denominator_label="all_cases")


def test_unhelpful_outcome_rate_combines_escalated_and_scored_incorrect():
    rate = unhelpful_outcome_rate(FIXTURE_RECORDS)
    assert rate.numerator == 8
    assert rate.denominator == 14


# --- Rate / denominator-label discipline ---


def test_rate_requires_denominator_label():
    with pytest.raises(ValidationError):
        Rate.model_validate({"numerator": 1, "denominator": 2})


def test_confusion_matrix_requires_denominator_label():
    cells = {ref: {pred: 0 for pred in TicketCategory} for ref in TicketCategory}
    with pytest.raises(ValidationError):
        ConfusionMatrix.model_validate({"cells": cells, "total": 0})


def test_refund_amount_error_requires_denominator_label():
    with pytest.raises(ValidationError):
        RefundAmountError.model_validate(
            {"denominator": 0, "mean_absolute_error": None}
        )


def test_rate_rejects_numerator_exceeding_denominator():
    with pytest.raises(ValidationError):
        Rate(numerator=3, denominator=2, denominator_label="x")


def test_rate_value_is_none_when_denominator_is_zero():
    assert Rate(numerator=0, denominator=0, denominator_label="x").value is None


def test_rate_value_computes_normally():
    assert Rate(numerator=1, denominator=4, denominator_label="x").value == 0.25


def test_gate_recall_denominator_zero_when_no_incorrect_cases():
    records = _make_gate_invariance_records("oracle")
    all_correct = [r for r in records if r.case_key in ("case-b", "case-d")]
    rate = gate_recall(all_correct)
    assert rate.denominator == 0
    assert rate.value is None


def test_gate_precision_denominator_zero_when_nothing_gated():
    records = _make_gate_invariance_records("oracle")
    all_ungated = [r for r in records if r.case_key in ("case-c", "case-d")]
    rate = gate_precision(all_ungated)
    assert rate.denominator == 0
    assert rate.value is None


# --- full-run integration ---


def test_score_run_reproduces_every_hand_computed_metric():
    metrics = score_run(FIXTURE_RECORDS, FIXTURE_EVENTS)

    assert metrics.scored_population_size == 12
    assert metrics.total_population_size == 14
    assert metrics.unreviewed_structured_error_rate == Rate(
        numerator=4, denominator=12, denominator_label="scored_population"
    )
    assert metrics.unreviewed_category_error_rate == Rate(
        numerator=1, denominator=12, denominator_label="scored_population"
    )
    assert metrics.unreviewed_action_error_rate == Rate(
        numerator=1, denominator=12, denominator_label="scored_population"
    )
    assert metrics.review_load == Rate(
        numerator=4, denominator=12, denominator_label="scored_population"
    )
    assert metrics.gate_recall == Rate(
        numerator=2, denominator=6, denominator_label="incorrect_scored_cases"
    )
    assert metrics.gate_precision == Rate(
        numerator=2, denominator=4, denominator_label="gated_scored_cases"
    )
    assert metrics.category_confusion_matrix.total == 12
    assert metrics.action_accuracy == Rate(
        numerator=10, denominator=12, denominator_label="scored_population"
    )
    assert metrics.refund_amount_error.denominator == 4
    assert metrics.refund_amount_error.mean_absolute_error == pytest.approx(3.75)
    assert metrics.escalation_rate == Rate(
        numerator=2, denominator=14, denominator_label="all_cases"
    )
    assert metrics.invalid_output_rate == Rate(
        numerator=2, denominator=14, denominator_label="all_cases"
    )
    assert metrics.fallback_usage_rate == Rate(
        numerator=1, denominator=14, denominator_label="all_cases"
    )
    assert metrics.unhelpful_outcome_rate.numerator == 8
    assert metrics.unhelpful_outcome_rate.denominator == 14


def test_rate_series_are_the_source_of_score_run_rate_fields():
    series_by_name = deterministic.build_rate_series(FIXTURE_RECORDS, FIXTURE_EVENTS)
    metrics = score_run(FIXTURE_RECORDS, FIXTURE_EVENTS)

    expected_names = {
        "unreviewed_structured_error_rate",
        "unreviewed_category_error_rate",
        "unreviewed_action_error_rate",
        "review_load",
        "gate_recall",
        "gate_precision",
        "action_accuracy",
        "escalation_rate",
        "invalid_output_rate",
        "fallback_usage_rate",
        "unhelpful_outcome_rate",
    }
    assert set(series_by_name) == expected_names
    for name, series in series_by_name.items():
        assert series.rate == getattr(metrics, name)
        assert len(series.clustered_values) == series.rate.denominator


def test_category_f1_clustered_values_use_only_records_in_f1_denominator():
    values = deterministic.category_f1_clustered_values(
        FIXTURE_RECORDS, TicketCategory.BILLING
    )

    assert len(values) == 7
    assert all(
        is_reference or is_predicted for _, (is_reference, is_predicted) in values
    )


def test_category_precision_clustered_values_match_predicted_population():
    values = deterministic.category_precision_clustered_values(
        FIXTURE_RECORDS, TicketCategory.BILLING
    )

    assert len(values) == 5
    assert sum(value for _, value in values) == 5


def test_category_recall_clustered_values_match_reference_population():
    values = deterministic.category_recall_clustered_values(
        FIXTURE_RECORDS, TicketCategory.BILLING
    )

    assert len(values) == 7
    assert sum(value for _, value in values) == 5


# --- T6-prep helper ---


def test_scored_case_predicate_pairs_excludes_unscored_and_maps_bool_to_float():
    pairs = scored_case_predicate_pairs(FIXTURE_RECORDS, structured_correct)
    assert len(pairs) == 12
    case_keys = {case_key for case_key, _ in pairs}
    assert "case-11" not in case_keys
    assert "case-12" not in case_keys
    values = dict(pairs)
    assert values["case-01"] == 1.0
    assert values["case-02"] == 0.0
