"""Activities wrap the agent and side effects."""

import asyncio
import contextlib
from collections.abc import Awaitable
from typing import TypeVar

from temporalio import activity
from temporalio.exceptions import ApplicationError

from ticketflow import readmodel
from ticketflow.agent.base import Agent, AgentPermanentError
from ticketflow.models import Classification, DraftReply, Ticket, TicketResult

_T = TypeVar("_T")

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0


async def _call_with_heartbeat(
    call: Awaitable[_T], *, interval_seconds: float, detail: str
) -> _T:
    """Await `call` while heartbeating `detail` every `interval_seconds`.

    A slow agent call can exceed the activity's heartbeat timeout, so a
    background loop heartbeats for the duration of the call. The loop is
    always cancelled and awaited before this function returns or raises.
    """

    async def _loop() -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            activity.heartbeat(detail)

    heartbeat_task = asyncio.create_task(_loop())
    try:
        return await call
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task


class TicketActivities:
    """Temporal activities for agent calls and external side effects."""

    def __init__(
        self,
        agent: Agent,
        db_path: str | None = None,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ):
        """Create activities backed by an agent and optional read-model path."""
        self._agent = agent
        self._db_path = db_path
        self._heartbeat_interval_seconds = heartbeat_interval_seconds

    @activity.defn
    async def classify_ticket(self, ticket: Ticket) -> Classification:
        """Classify a ticket through the configured agent."""
        activity.heartbeat("classifying ticket")
        try:
            result = await _call_with_heartbeat(
                self._agent.classify(ticket),
                interval_seconds=self._heartbeat_interval_seconds,
                detail="classifying ticket",
            )
        except AgentPermanentError as exc:
            raise ApplicationError(
                str(exc), type="AgentPermanentError", non_retryable=True
            ) from exc
        activity.heartbeat("classified ticket")
        return result

    @activity.defn
    async def draft_reply(
        self, ticket: Ticket, classification: Classification
    ) -> DraftReply:
        """Ask the configured agent to draft a response."""
        activity.heartbeat("drafting reply")
        try:
            result = await _call_with_heartbeat(
                self._agent.draft_reply(ticket, classification),
                interval_seconds=self._heartbeat_interval_seconds,
                detail="drafting reply",
            )
        except AgentPermanentError as exc:
            raise ApplicationError(
                str(exc), type="AgentPermanentError", non_retryable=True
            ) from exc
        activity.heartbeat("drafted reply")
        return result

    @activity.defn
    async def send_reply(self, ticket: Ticket, reply_text: str) -> None:
        """Send or log the customer reply."""
        activity.logger.info(
            "Sending reply to %s: %s", ticket.customer_email, reply_text
        )

    @activity.defn
    async def execute_refund(self, ticket_id: str, amount: float) -> None:
        """Execute a refund at most once per ticket id."""
        # Idempotent by ticket id: a real implementation would use ticket_id
        # as the payment provider's idempotency key.
        attempt = activity.info().attempt
        first = await asyncio.to_thread(
            readmodel.record_refund, ticket_id, amount, attempt, self._db_path
        )
        if first:
            activity.logger.info(
                "Refunding %.2f for ticket %s (attempt %d)", amount, ticket_id, attempt
            )
        else:
            activity.logger.info(
                "Refund for ticket %s already executed; attempt %d is a no-op",
                ticket_id,
                attempt,
            )

    @activity.defn
    async def record_result(self, result: TicketResult) -> None:
        """Persist the terminal workflow result to the read model."""
        await asyncio.to_thread(readmodel.save_result, result, self._db_path)
