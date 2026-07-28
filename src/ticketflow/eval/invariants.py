"""System-invariant checks from plan.md's reporting section.

Each check inspects `CaseRecord`s (and, where needed, joined `CallEvent`s) for a
specific structural guarantee the workflow and runner are supposed to uphold. A
violated invariant is a finding, not a crash: every check returns data
(`InvariantViolation`s), never raises.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ticketflow.eval.records import CallEvent, CaseRecord
from ticketflow.models import ActionType

InvariantName = Literal[
    "gating_matches_threshold",
    "at_most_one_refund_per_ticket",
    "refund_attempts_at_least_executed",
    "executed_refund_implies_approved",
    "fallback_routing_identifies_path",
]

_CaseIdentity = tuple[str, str, str, int, str]


class InvariantViolation(BaseModel):
    """One instance of a system invariant not holding for a case record."""

    model_config = ConfigDict(frozen=True)

    invariant: InvariantName
    run_id: str
    policy: str
    case_key: str
    repeat_index: int
    ticket_id: str
    detail: str


class InvariantReport(BaseModel):
    """Every violation found across a run's records, plus the population size."""

    model_config = ConfigDict(frozen=True)

    total_checked: int
    violations: tuple[InvariantViolation, ...]

    @property
    def ok(self) -> bool:
        """Return True if no invariant violation was found."""
        return not self.violations


def _identity(record: CaseRecord) -> _CaseIdentity:
    """Return the (run_id, policy, case_key, repeat_index, ticket_id) join key."""
    return (
        record.run_id,
        record.policy,
        record.case_key,
        record.repeat_index,
        record.ticket_id,
    )


def _violation(
    record: CaseRecord, invariant: InvariantName, detail: str
) -> InvariantViolation:
    """Build a violation for record, filling in its join-key fields."""
    return InvariantViolation(
        invariant=invariant,
        run_id=record.run_id,
        policy=record.policy,
        case_key=record.case_key,
        repeat_index=record.repeat_index,
        ticket_id=record.ticket_id,
        detail=detail,
    )


def check_gating_matches_threshold(
    records: Sequence[CaseRecord], *, confidence_threshold: float
) -> list[InvariantViolation]:
    """Flag records whose was_gated disagrees with the threshold/refund rule."""
    violations = []
    for record in records:
        if record.predicted_action is None or record.draft_confidence is None:
            expected_gate = False
        else:
            expected_gate = (
                record.predicted_action == ActionType.REFUND
                or record.draft_confidence < confidence_threshold
            )
        if record.was_gated != expected_gate:
            violations.append(
                _violation(
                    record,
                    "gating_matches_threshold",
                    f"was_gated={record.was_gated} but expected {expected_gate} "
                    f"(predicted_action={record.predicted_action}, "
                    f"draft_confidence={record.draft_confidence}, "
                    f"threshold={confidence_threshold})",
                )
            )
    return violations


def check_at_most_one_refund_per_ticket(
    records: Sequence[CaseRecord],
) -> list[InvariantViolation]:
    """Flag any ticket whose records' refund_executed_count sums to more than one."""
    totals: dict[str, int] = {}
    representative: dict[str, CaseRecord] = {}
    for record in records:
        totals[record.ticket_id] = (
            totals.get(record.ticket_id, 0) + record.refund_executed_count
        )
        representative.setdefault(record.ticket_id, record)

    violations = []
    for ticket_id, total in totals.items():
        if total > 1:
            violations.append(
                _violation(
                    representative[ticket_id],
                    "at_most_one_refund_per_ticket",
                    f"ticket {ticket_id!r} has {total} executed refunds across "
                    "its records",
                )
            )
    return violations


def check_refund_attempts_at_least_executed(
    records: Sequence[CaseRecord],
) -> list[InvariantViolation]:
    """Flag records where refund_attempt_count is less than refund_executed_count."""
    violations = []
    for record in records:
        if record.refund_attempt_count < record.refund_executed_count:
            violations.append(
                _violation(
                    record,
                    "refund_attempts_at_least_executed",
                    f"refund_attempt_count={record.refund_attempt_count} < "
                    f"refund_executed_count={record.refund_executed_count}",
                )
            )
    return violations


def check_executed_refund_implies_approved(
    records: Sequence[CaseRecord],
) -> list[InvariantViolation]:
    """Flag any executed refund whose decision is missing or not approved."""
    violations = []
    for record in records:
        if record.refund_executed_count <= 0:
            continue
        if record.decision is None or not record.decision.approved:
            violations.append(
                _violation(
                    record,
                    "executed_refund_implies_approved",
                    f"refund_executed_count={record.refund_executed_count} but "
                    f"decision={record.decision!r}",
                )
            )
    return violations


def check_fallback_routing_identifies_fallback_path(
    records: Sequence[CaseRecord], events: Sequence[CallEvent]
) -> list[InvariantViolation]:
    """Flag a model_path role that disagrees with the successful joined CallEvent."""
    events_by_identity: dict[_CaseIdentity, list[CallEvent]] = {}
    for event in events:
        key = (
            event.run_id,
            event.policy,
            event.case_key,
            event.repeat_index,
            event.ticket_id,
        )
        events_by_identity.setdefault(key, []).append(event)

    violations = []
    for record in records:
        classification_role, _, draft_role = record.model_path.partition("/")
        case_events = events_by_identity.get(_identity(record), [])
        for operation, model_path_role in (
            ("classify", classification_role),
            ("draft", draft_role),
        ):
            successful = next(
                (
                    event
                    for event in case_events
                    if event.operation == operation and event.outcome == "success"
                ),
                None,
            )
            if successful is None:
                continue
            if successful.role != model_path_role:
                violations.append(
                    _violation(
                        record,
                        "fallback_routing_identifies_path",
                        f"model_path {operation} role {model_path_role!r} disagrees "
                        f"with successful CallEvent role {successful.role!r}",
                    )
                )
    return violations


def check_all_invariants(
    records: Sequence[CaseRecord],
    events: Sequence[CallEvent],
    *,
    confidence_threshold: float,
) -> InvariantReport:
    """Run every system-invariant check and aggregate the findings."""
    violations = [
        *check_gating_matches_threshold(
            records, confidence_threshold=confidence_threshold
        ),
        *check_at_most_one_refund_per_ticket(records),
        *check_refund_attempts_at_least_executed(records),
        *check_executed_refund_implies_approved(records),
        *check_fallback_routing_identifies_fallback_path(records, events),
    ]
    return InvariantReport(total_checked=len(records), violations=tuple(violations))
