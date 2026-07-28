"""Per-case workflow execution for evaluation runs."""

import asyncio
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from typing import Literal, cast

from temporalio.client import Client, WorkflowHandle, WorkflowUpdateFailedError

from ticketflow import readmodel, workflows
from ticketflow.eval.dataset import EvalCase
from ticketflow.eval.records import CallEvent, CaseRecord
from ticketflow.eval.telemetry import (
    RuntimeIdentityMap,
    TelemetrySink,
    drain_call_events,
)
from ticketflow.models import (
    ApprovalDecision,
    Ticket,
    TicketResult,
    TicketStatus,
    TicketStatusInfo,
)
from ticketflow.workflows import TicketWorkflow

Reviewer = Callable[[CaseRecord], ApprovalDecision]
TerminalOutcome = Literal[
    "resolved",
    "rejected",
    "escalated",
    "update_rejected",
    "runner_deadline_exceeded",
]
CleanupAction = Literal["cancelled", "terminated"]
_CANCEL_CONFIRMATION_TIMEOUT_S = 5.0


class CaseRunner:
    """Run individual evaluation cases with run-scoped collaborators."""

    def __init__(
        self,
        client: Client,
        *,
        run_id: str,
        workflow_task_queue: str,
        db_path: str | None,
        case_deadline: timedelta,
        identity_map: RuntimeIdentityMap,
        telemetry_sink: TelemetrySink,
        confidence_threshold: float | None = None,
        poll_interval_s: float = 0.01,
    ) -> None:
        """Store the client and immutable run context used by every case."""
        self._client = client
        self._run_id = run_id
        self._workflow_task_queue = workflow_task_queue
        self._db_path = db_path
        self._case_deadline = case_deadline
        self._identity_map = identity_map
        self._telemetry_sink = telemetry_sink
        self._confidence_threshold = (
            workflows.CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )
        self._poll_interval_s = poll_interval_s

    async def run_case(
        self,
        case: EvalCase,
        *,
        policy: str,
        reviewer: Reviewer,
        repeat_index: int,
    ) -> tuple[CaseRecord, list[CallEvent]]:
        """Execute one case and return its record with drained call telemetry."""
        started_at = time.perf_counter()
        ticket_id = self._ticket_id(case.id, policy, repeat_index)
        self._identity_map.register(ticket_id, case.id)
        ticket = Ticket(
            id=ticket_id,
            customer_email=case.customer_email,
            subject=case.subject,
            body=case.body,
        )
        handle = await self._client.start_workflow(
            TicketWorkflow.run,
            ticket,
            id=f"workflow-{ticket_id}",
            task_queue=self._workflow_task_queue,
        )

        try:
            result, status, terminal_outcome, terminal_error = await asyncio.wait_for(
                self._await_terminal(
                    handle,
                    case,
                    policy,
                    reviewer,
                    repeat_index,
                ),
                timeout=max(
                    0.0,
                    self._case_deadline.total_seconds()
                    - (time.perf_counter() - started_at),
                ),
            )
            cleanup_action = None
        except asyncio.TimeoutError:
            status = await self._best_effort_status(handle)
            cleanup_action, cleanup_error = await self._cleanup_deadline(handle)
            result = None
            terminal_outcome = "runner_deadline_exceeded"
            terminal_error = self._deadline_error(cleanup_error)

        record = self._record(
            case=case,
            policy=policy,
            repeat_index=repeat_index,
            ticket_id=ticket_id,
            status=status,
            result=result,
            terminal_outcome=terminal_outcome,
            cleanup_action=cleanup_action,
            terminal_error=terminal_error,
            end_to_end_latency_ms=(time.perf_counter() - started_at) * 1000,
        )
        events = drain_call_events(
            self._telemetry_sink,
            ticket_id,
            run_id=self._run_id,
            policy=policy,
            case_key=case.id,
            repeat_index=repeat_index,
        )
        return record, events

    async def _await_terminal(
        self,
        handle: WorkflowHandle,
        case: EvalCase,
        policy: str,
        reviewer: Reviewer,
        repeat_index: int,
    ) -> tuple[
        TicketResult | None,
        TicketStatusInfo | None,
        TerminalOutcome,
        str | None,
    ]:
        """Wait for result while submitting approval after the gated state appears."""
        terminal_statuses = {
            TicketStatus.RESOLVED,
            TicketStatus.REJECTED,
            TicketStatus.ESCALATED,
        }
        while True:
            status = await self._best_effort_status(handle)
            if status is not None and status.status == TicketStatus.AWAITING_APPROVAL:
                reviewer_input = self._record(
                    case=case,
                    policy=policy,
                    repeat_index=repeat_index,
                    ticket_id=status.ticket_id,
                    status=status,
                    result=None,
                    terminal_outcome="resolved",
                    cleanup_action=None,
                    terminal_error=None,
                    end_to_end_latency_ms=0.0,
                )
                try:
                    await handle.execute_update(
                        "submit_approval",
                        reviewer(reviewer_input),
                        result_type=TicketStatus,
                    )
                except WorkflowUpdateFailedError as exc:
                    return (
                        None,
                        await self._best_effort_status(handle) or status,
                        "update_rejected",
                        str(exc),
                    )
            elif status is not None and status.status in terminal_statuses:
                result = cast(TicketResult, await handle.result())
                completed_status = await self._best_effort_status(handle)
                return (
                    result,
                    completed_status or status,
                    self._terminal_outcome(result.status),
                    None,
                )
            await asyncio.sleep(self._poll_interval_s)

    async def _best_effort_status(
        self, handle: WorkflowHandle
    ) -> TicketStatusInfo | None:
        """Query status, returning None when a best-effort query cannot complete."""
        try:
            return cast(TicketStatusInfo, await handle.query(TicketWorkflow.status))
        except Exception:
            return None

    async def _cleanup_deadline(
        self, handle: WorkflowHandle
    ) -> tuple[CleanupAction, str | None]:
        """Cancel a timed-out workflow, terminating only if it stays open."""
        reason = "eval runner deadline exceeded"
        try:
            await handle.cancel(reason=reason)
            try:
                await asyncio.wait_for(
                    handle.result(), timeout=_CANCEL_CONFIRMATION_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                return await self._terminate_after_deadline(handle, reason)
            except Exception:
                pass
            return "cancelled", None
        except Exception as exc:
            cleanup_action, termination_error = await self._terminate_after_deadline(
                handle, reason
            )
            detail = f"cancellation request failed: {exc}"
            if termination_error:
                detail = f"{detail}; {termination_error}"
            return cleanup_action, detail

    async def _terminate_after_deadline(
        self, handle: WorkflowHandle, reason: str
    ) -> tuple[CleanupAction, str | None]:
        """Request termination without allowing cleanup RPC failures to escape."""
        try:
            await handle.terminate(reason=reason)
        except Exception as exc:
            return "terminated", f"termination request failed: {exc}"
        return "terminated", None

    @staticmethod
    def _deadline_error(cleanup_error: str | None) -> str:
        """Render the deadline finding and any cleanup failure as record data."""
        if cleanup_error is None:
            return "runner deadline exceeded"
        return f"runner deadline exceeded; {cleanup_error}"

    def _record(
        self,
        *,
        case: EvalCase,
        policy: str,
        repeat_index: int,
        ticket_id: str,
        status: TicketStatusInfo | None,
        result: TicketResult | None,
        terminal_outcome: TerminalOutcome,
        cleanup_action: CleanupAction | None,
        terminal_error: str | None,
        end_to_end_latency_ms: float,
    ) -> CaseRecord:
        """Flatten workflow state and public refund observation into a record."""
        classification = status.classification if status else None
        draft = status.draft if status else None
        observation = readmodel.get_refund_observation(ticket_id, self._db_path)
        model_path = (
            result.model_path
            if result is not None
            else f"{classification.model if classification else 'primary'}/"
            f"{draft.model if draft else 'primary'}"
        )
        was_gated = bool(
            draft
            and (
                draft.action.type.value == "refund"
                or draft.confidence < self._confidence_threshold
            )
        )
        return CaseRecord(
            run_id=self._run_id,
            policy=policy,
            case_key=case.id,
            repeat_index=repeat_index,
            ticket_id=ticket_id,
            difficulty=case.difficulty,
            source=case.source,
            expected=case.expected,
            predicted_category=classification.category if classification else None,
            predicted_action=draft.action.type if draft else None,
            predicted_refund_amount=(draft.action.refund_amount if draft else None),
            classification_confidence=(
                classification.confidence if classification else None
            ),
            draft_confidence=draft.confidence if draft else None,
            reply_text=draft.reply_text if draft else None,
            model_path=model_path,
            terminal_status=(
                result.status if result else (status.status if status else None)
            ),
            was_gated=was_gated,
            decision=status.decision if status else None,
            refund_executed_count=observation.executed_count,
            refund_attempt_count=observation.attempt_count,
            prediction_available=draft is not None,
            prediction_unavailable_reason=(
                None
                if draft is not None
                else terminal_error or "draft was not captured"
            ),
            terminal_outcome=terminal_outcome,
            cleanup_action=cleanup_action,
            end_to_end_latency_ms=end_to_end_latency_ms,
            terminal_error=terminal_error,
        )

    def _ticket_id(self, case_key: str, policy: str, repeat_index: int) -> str:
        """Create an opaque unique runtime ID without parsing metadata back out."""
        return (
            f"eval-{self._run_id}-{policy}-{case_key}-{repeat_index}-{uuid.uuid4().hex}"
        )

    @staticmethod
    def _terminal_outcome(status: TicketStatus) -> TerminalOutcome:
        """Map workflow terminal states to the case-record outcome literal."""
        match status:
            case TicketStatus.RESOLVED:
                return "resolved"
            case TicketStatus.REJECTED:
                return "rejected"
            case TicketStatus.ESCALATED:
                return "escalated"
        raise ValueError(f"workflow returned non-terminal status {status!r}")
