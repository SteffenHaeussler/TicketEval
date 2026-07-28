"""Runtime identity resolution and process-local per-attempt telemetry.

Per execution.md's M2-T6, agents only ever know a runtime `ticket_id` and what they
themselves observed for one classify/draft attempt. They neither know nor need
`run_id`, `policy`, or `repeat_index`. `RuntimeIdentityMap` resolves a runtime ticket
ID back to the stable `case_key`, and `TelemetrySink` records the raw per-attempt data
agents can supply. The per-case runner (M2-T4), which does know `run_id`/`policy`/
`repeat_index`, drains and enriches raw attempts into full `CallEvent`s via
`drain_call_events`.
"""

import threading
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ticketflow.eval.records import CallEvent


class TelemetryError(Exception):
    """Base class for identity-map and telemetry-sink failures."""


class UnknownTicketIdError(TelemetryError):
    """Raised by resolve() for a ticket_id that was never registered."""


class DuplicateTicketIdError(TelemetryError):
    """Raised by register() when ticket_id already maps to a different case_key."""


class RuntimeIdentityMap:
    """Process-local, run-scoped mapping from runtime ticket ID to case_key."""

    def __init__(self) -> None:
        """Create an empty, concurrency-safe identity map."""
        self._lock = threading.Lock()
        self._case_keys_by_ticket_id: dict[str, str] = {}

    def register(self, ticket_id: str, case_key: str) -> None:
        """Register ticket_id -> case_key, idempotently for a repeated mapping."""
        with self._lock:
            existing = self._case_keys_by_ticket_id.get(ticket_id)
            if existing is not None and existing != case_key:
                raise DuplicateTicketIdError(
                    f"ticket_id {ticket_id!r} is already registered to case_key "
                    f"{existing!r}, cannot re-register to {case_key!r}"
                )
            self._case_keys_by_ticket_id[ticket_id] = case_key

    def resolve(self, ticket_id: str) -> str:
        """Return the case_key registered for ticket_id."""
        with self._lock:
            case_key = self._case_keys_by_ticket_id.get(ticket_id)
        if case_key is None:
            raise UnknownTicketIdError(f"ticket_id {ticket_id!r} was never registered")
        return case_key


class RawAttempt(BaseModel):
    """One classify/draft attempt as an agent can observe it.

    Identical to `CallEvent` except it omits the four fields only the per-case
    runner can supply: `run_id`, `policy`, `case_key`, and `repeat_index`.
    """

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    operation: Literal["classify", "draft"]
    role: Literal["primary", "fallback"]
    attempt: int
    cache_hit: bool
    started_at: datetime
    wall_latency_ms: float
    model_total_duration_ms: float | None
    model_load_duration_ms: float | None
    outcome: Literal["success", "invalid_output", "transient_error", "permanent_error"]
    error_type: str | None


class TelemetrySink:
    """Process-local, run-scoped sink for raw per-attempt telemetry.

    Attempts are keyed by `(ticket_id, operation)`. Agents call `record()`; the
    per-case runner calls `drain()` once a ticket's workflow has finished.
    """

    def __init__(self) -> None:
        """Create an empty, concurrency-safe telemetry sink."""
        self._lock = threading.Lock()
        self._attempts_by_ticket_id: dict[str, list[RawAttempt]] = {}
        self._counts_by_key: dict[tuple[str, str], int] = {}

    def record(
        self,
        *,
        ticket_id: str,
        operation: Literal["classify", "draft"],
        role: Literal["primary", "fallback"],
        cache_hit: bool,
        started_at: datetime,
        wall_latency_ms: float,
        model_total_duration_ms: float | None,
        model_load_duration_ms: float | None,
        outcome: Literal[
            "success", "invalid_output", "transient_error", "permanent_error"
        ],
        error_type: str | None,
    ) -> RawAttempt:
        """Record one attempt, auto-numbering it within its (ticket_id, operation)."""
        key = (ticket_id, operation)
        with self._lock:
            attempt_number = self._counts_by_key.get(key, 0) + 1
            self._counts_by_key[key] = attempt_number
            raw = RawAttempt(
                ticket_id=ticket_id,
                operation=operation,
                role=role,
                attempt=attempt_number,
                cache_hit=cache_hit,
                started_at=started_at,
                wall_latency_ms=wall_latency_ms,
                model_total_duration_ms=model_total_duration_ms,
                model_load_duration_ms=model_load_duration_ms,
                outcome=outcome,
                error_type=error_type,
            )
            self._attempts_by_ticket_id.setdefault(ticket_id, []).append(raw)
        return raw

    def drain(self, ticket_id: str) -> list[RawAttempt]:
        """Remove and return every attempt recorded for ticket_id, oldest first."""
        with self._lock:
            attempts = self._attempts_by_ticket_id.pop(ticket_id, [])
        return sorted(attempts, key=lambda attempt: attempt.started_at)


def build_call_event(
    attempt: RawAttempt,
    *,
    run_id: str,
    policy: str,
    case_key: str,
    repeat_index: int,
) -> CallEvent:
    """Enrich one raw attempt into a full CallEvent using runner-known join fields."""
    return CallEvent(
        run_id=run_id,
        case_key=case_key,
        ticket_id=attempt.ticket_id,
        policy=policy,
        repeat_index=repeat_index,
        operation=attempt.operation,
        role=attempt.role,
        attempt=attempt.attempt,
        cache_hit=attempt.cache_hit,
        started_at=attempt.started_at,
        wall_latency_ms=attempt.wall_latency_ms,
        model_total_duration_ms=attempt.model_total_duration_ms,
        model_load_duration_ms=attempt.model_load_duration_ms,
        outcome=attempt.outcome,
        error_type=attempt.error_type,
    )


def drain_call_events(
    sink: TelemetrySink,
    ticket_id: str,
    *,
    run_id: str,
    policy: str,
    case_key: str,
    repeat_index: int,
) -> list[CallEvent]:
    """Drain ticket_id's raw attempts and enrich each into a full CallEvent."""
    return [
        build_call_event(
            attempt,
            run_id=run_id,
            policy=policy,
            case_key=case_key,
            repeat_index=repeat_index,
        )
        for attempt in sink.drain(ticket_id)
    ]
