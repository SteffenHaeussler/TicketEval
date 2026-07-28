"""Tests for pure evaluation reviewer policies."""

import pytest

from tests.eval.test_records import make_case_record, make_expected
from ticketflow.eval.reviewers import oracle, rubber_stamp
from ticketflow.eval.scorers.deterministic import UnscoredCaseError
from ticketflow.models import ActionType


def test_oracle_approves_a_correct_reply_only_outcome():
    """Oracle approval identifies a fully correct structured outcome."""
    decision = oracle(make_case_record())

    assert decision.approved is True
    assert decision.approver == "oracle-reviewer"
    assert decision.note is None


@pytest.mark.parametrize(
    ("overrides", "description"),
    [
        ({"predicted_category": "technical"}, "incorrect category"),
        (
            {"predicted_action": "refund", "predicted_refund_amount": 12.0},
            "incorrect action",
        ),
    ],
)
def test_oracle_rejects_an_incorrect_structured_outcome(overrides, description):
    """Oracle rejects incorrect categories and actions."""
    decision = oracle(make_case_record(**overrides))

    assert decision.approved is False, description
    assert decision.approver == "oracle-reviewer"
    assert decision.note is None


def test_oracle_applies_the_refund_tolerance_boundary():
    """Oracle delegates refund validation, including the label tolerance."""
    expected = make_expected(
        acceptable_actions=["refund"],
        expected_refund_amount=50.0,
        refund_tolerance=0.01,
    )
    within_tolerance = make_case_record(
        expected=expected,
        predicted_action=ActionType.REFUND,
        predicted_refund_amount=50.005,
    )
    outside_tolerance = within_tolerance.model_copy(
        update={"predicted_refund_amount": 50.02}
    )

    assert oracle(within_tolerance).approved is True
    assert oracle(outside_tolerance).approved is False


def test_rubber_stamp_approves_an_outcome_the_oracle_rejects():
    """Rubber-stamp policy approves every supplied record."""
    incorrect = make_case_record(predicted_category="technical")

    assert oracle(incorrect).approved is False
    decision = rubber_stamp(incorrect)
    assert decision.approved is True
    assert decision.approver == "rubber-stamp"
    assert decision.note is None


def test_oracle_propagates_the_unscored_record_error():
    """Oracle does not silently approve a record with no draft."""
    unscored = make_case_record(
        predicted_category=None,
        predicted_action=None,
        draft_confidence=None,
        reply_text=None,
        prediction_available=False,
    )

    with pytest.raises(UnscoredCaseError, match="case-1"):
        oracle(unscored)
