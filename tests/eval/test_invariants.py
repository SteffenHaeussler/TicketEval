from datetime import datetime, timezone

from ticketflow.eval.dataset import ExpectedOutcome
from ticketflow.eval.invariants import (
    check_all_invariants,
    check_at_most_one_refund_per_ticket,
    check_executed_refund_implies_approved,
    check_fallback_routing_identifies_fallback_path,
    check_gating_matches_threshold,
    check_refund_attempts_at_least_executed,
)
from ticketflow.eval.records import CallEvent, CaseRecord
from ticketflow.models import ApprovalDecision

THRESHOLD = 0.75


def make_expected(**overrides):
    base = {
        "acceptable_categories": ["billing"],
        "reference_category": "billing",
        "acceptable_actions": ["reply_only", "refund"],
        "expected_refund_amount": 42.0,
        "refund_tolerance": 0.01,
    }
    base.update(overrides)
    return ExpectedOutcome.model_validate(base)


def make_case_record(**overrides):
    base = dict(
        run_id="run-1",
        policy="oracle",
        case_key="case-1",
        repeat_index=0,
        ticket_id="ticket-1",
        difficulty="easy",
        source="handwritten",
        expected=make_expected(),
        predicted_category="billing",
        predicted_action="reply_only",
        predicted_refund_amount=None,
        classification_confidence=0.9,
        draft_confidence=0.9,
        reply_text="Thanks for reaching out.",
        model_path="primary/primary",
        terminal_status="resolved",
        was_gated=False,
        decision=None,
        refund_executed_count=0,
        refund_attempt_count=0,
        prediction_available=True,
        prediction_unavailable_reason=None,
        terminal_outcome="resolved",
        cleanup_action=None,
        end_to_end_latency_ms=123.4,
        terminal_error=None,
    )
    base.update(overrides)
    return CaseRecord.model_validate(base)


def make_call_event(**overrides):
    base = dict(
        run_id="run-1",
        case_key="case-1",
        ticket_id="ticket-1",
        policy="oracle",
        repeat_index=0,
        operation="classify",
        role="primary",
        attempt=1,
        cache_hit=False,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        wall_latency_ms=42.0,
        model_total_duration_ms=30.0,
        model_load_duration_ms=5.0,
        outcome="success",
        error_type=None,
    )
    base.update(overrides)
    return CallEvent.model_validate(base)


class TestGatingMatchesThreshold:
    def test_refund_not_gated_is_flagged(self):
        record = make_case_record(
            predicted_action="refund", predicted_refund_amount=42.0, was_gated=False
        )
        violations = check_gating_matches_threshold(
            [record], confidence_threshold=THRESHOLD
        )
        assert len(violations) == 1
        assert violations[0].invariant == "gating_matches_threshold"

    def test_low_confidence_not_gated_is_flagged(self):
        record = make_case_record(draft_confidence=0.5, was_gated=False)
        violations = check_gating_matches_threshold(
            [record], confidence_threshold=THRESHOLD
        )
        assert len(violations) == 1

    def test_correctly_gated_and_ungated_records_are_clean(self):
        gated = make_case_record(
            predicted_action="refund", predicted_refund_amount=42.0, was_gated=True
        )
        ungated = make_case_record(draft_confidence=0.9, was_gated=False)
        violations = check_gating_matches_threshold(
            [gated, ungated], confidence_threshold=THRESHOLD
        )
        assert violations == []

    def test_no_draft_expects_ungated(self):
        record = make_case_record(
            predicted_action=None,
            predicted_refund_amount=None,
            draft_confidence=None,
            reply_text=None,
            classification_confidence=None,
            was_gated=False,
            prediction_available=False,
            prediction_unavailable_reason="escalated before draft",
            terminal_outcome="escalated",
        )
        violations = check_gating_matches_threshold(
            [record], confidence_threshold=THRESHOLD
        )
        assert violations == []


class TestAtMostOneRefundPerTicket:
    def test_two_records_summing_to_two_refunds_is_flagged(self):
        records = [
            make_case_record(ticket_id="t1", repeat_index=0, refund_executed_count=1),
            make_case_record(ticket_id="t1", repeat_index=1, refund_executed_count=1),
        ]
        violations = check_at_most_one_refund_per_ticket(records)
        assert len(violations) == 1
        assert violations[0].invariant == "at_most_one_refund_per_ticket"

    def test_single_refund_is_clean(self):
        record = make_case_record(refund_executed_count=1)
        assert check_at_most_one_refund_per_ticket([record]) == []


class TestRefundAttemptsAtLeastExecuted:
    def test_fewer_attempts_than_executed_is_flagged(self):
        record = make_case_record(refund_attempt_count=0, refund_executed_count=1)
        violations = check_refund_attempts_at_least_executed([record])
        assert len(violations) == 1
        assert violations[0].invariant == "refund_attempts_at_least_executed"

    def test_attempts_matching_executed_is_clean(self):
        record = make_case_record(refund_attempt_count=1, refund_executed_count=1)
        assert check_refund_attempts_at_least_executed([record]) == []


class TestExecutedRefundImpliesApproved:
    def test_executed_refund_without_decision_is_flagged(self):
        record = make_case_record(refund_executed_count=1, decision=None)
        violations = check_executed_refund_implies_approved([record])
        assert len(violations) == 1

    def test_executed_refund_with_rejected_decision_is_flagged(self):
        record = make_case_record(
            refund_executed_count=1,
            decision=ApprovalDecision(approved=False, approver="oracle"),
        )
        violations = check_executed_refund_implies_approved([record])
        assert len(violations) == 1

    def test_executed_refund_with_approval_is_clean(self):
        record = make_case_record(
            refund_executed_count=1,
            decision=ApprovalDecision(approved=True, approver="oracle"),
        )
        assert check_executed_refund_implies_approved([record]) == []


class TestFallbackRoutingIdentifiesFallbackPath:
    def test_mismatched_draft_role_is_flagged(self):
        record = make_case_record(model_path="primary/primary")
        events = [
            make_call_event(operation="classify", role="primary", outcome="success"),
            make_call_event(operation="draft", role="fallback", outcome="success"),
        ]
        violations = check_fallback_routing_identifies_fallback_path([record], events)
        assert len(violations) == 1
        assert violations[0].invariant == "fallback_routing_identifies_path"

    def test_matching_roles_are_clean(self):
        record = make_case_record(model_path="primary/fallback")
        events = [
            make_call_event(operation="classify", role="primary", outcome="success"),
            make_call_event(operation="draft", role="fallback", outcome="success"),
        ]
        assert check_fallback_routing_identifies_fallback_path([record], events) == []

    def test_no_successful_event_is_skipped(self):
        record = make_case_record(model_path="primary/primary")
        events = [
            make_call_event(
                operation="draft", role="fallback", outcome="transient_error"
            )
        ]
        assert check_fallback_routing_identifies_fallback_path([record], events) == []


class TestCheckAllInvariants:
    def test_aggregates_multiple_violation_types(self):
        records = [
            make_case_record(
                predicted_action="refund", predicted_refund_amount=42.0, was_gated=False
            ),
            make_case_record(
                refund_attempt_count=0,
                refund_executed_count=1,
                decision=ApprovalDecision(approved=True, approver="oracle"),
            ),
        ]
        report = check_all_invariants(records, [], confidence_threshold=THRESHOLD)
        assert report.total_checked == 2
        assert report.ok is False
        assert len(report.violations) == 2

    def test_all_clean_records_produce_an_ok_report(self):
        record = make_case_record(was_gated=False, draft_confidence=0.9)
        report = check_all_invariants([record], [], confidence_threshold=THRESHOLD)
        assert report.ok is True
        assert report.violations == ()
