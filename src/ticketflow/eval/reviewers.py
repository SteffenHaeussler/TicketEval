"""Pure reviewer policies for evaluation workflow drafts."""

from ticketflow.eval.records import CaseRecord
from ticketflow.eval.scorers.deterministic import structured_correct
from ticketflow.models import ApprovalDecision


def oracle(record: CaseRecord) -> ApprovalDecision:
    """Approve exactly the records with correct structured outcomes."""
    return ApprovalDecision(
        approved=structured_correct(record),
        approver="oracle-reviewer",
    )


def rubber_stamp(record: CaseRecord) -> ApprovalDecision:
    """Approve every supplied record regardless of structured correctness."""
    _ = record
    return ApprovalDecision(approved=True, approver="rubber-stamp")
