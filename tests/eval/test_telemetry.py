import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest

from ticketflow.eval.telemetry import (
    DuplicateTicketIdError,
    RawAttempt,
    RuntimeIdentityMap,
    TelemetrySink,
    UnknownTicketIdError,
    build_call_event,
    drain_call_events,
)

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def record_attempt(
    sink: TelemetrySink,
    ticket_id: str,
    operation: Literal["classify", "draft"] = "classify",
    *,
    role: Literal["primary", "fallback"] = "primary",
    cache_hit: bool = False,
    started_at: datetime = BASE_TIME,
    wall_latency_ms: float = 1.0,
    model_total_duration_ms: float | None = None,
    model_load_duration_ms: float | None = None,
    outcome: Literal[
        "success", "invalid_output", "transient_error", "permanent_error"
    ] = "success",
    error_type: str | None = None,
) -> RawAttempt:
    return sink.record(
        ticket_id=ticket_id,
        operation=operation,
        role=role,
        cache_hit=cache_hit,
        started_at=started_at,
        wall_latency_ms=wall_latency_ms,
        model_total_duration_ms=model_total_duration_ms,
        model_load_duration_ms=model_load_duration_ms,
        outcome=outcome,
        error_type=error_type,
    )


class TestRuntimeIdentityMap:
    def test_register_then_resolve_round_trips(self):
        identity_map = RuntimeIdentityMap()
        identity_map.register("ticket-1", "case-1")
        assert identity_map.resolve("ticket-1") == "case-1"

    def test_resolve_unknown_ticket_raises(self):
        identity_map = RuntimeIdentityMap()
        with pytest.raises(UnknownTicketIdError):
            identity_map.resolve("nope")

    def test_reregistering_same_pair_is_a_no_op(self):
        identity_map = RuntimeIdentityMap()
        identity_map.register("ticket-1", "case-1")
        identity_map.register("ticket-1", "case-1")
        assert identity_map.resolve("ticket-1") == "case-1"

    def test_reregistering_different_case_key_raises(self):
        identity_map = RuntimeIdentityMap()
        identity_map.register("ticket-1", "case-1")
        with pytest.raises(DuplicateTicketIdError):
            identity_map.register("ticket-1", "case-2")

    def test_different_tickets_resolve_to_same_case_key(self):
        identity_map = RuntimeIdentityMap()
        identity_map.register("oracle-ticket", "case-1")
        identity_map.register("rubber-stamp-ticket", "case-1")
        assert identity_map.resolve("oracle-ticket") == "case-1"
        assert identity_map.resolve("rubber-stamp-ticket") == "case-1"


class TestTelemetrySinkRecording:
    def test_attempt_numbers_increment_per_ticket_and_operation(self):
        sink = TelemetrySink()
        record_attempt(sink, "ticket-1", "classify")
        record_attempt(sink, "ticket-1", "classify")
        record_attempt(sink, "ticket-1", "draft")

        attempts = sink.drain("ticket-1")
        classify_attempts = [a for a in attempts if a.operation == "classify"]
        draft_attempts = [a for a in attempts if a.operation == "draft"]
        assert [a.attempt for a in classify_attempts] == [1, 2]
        assert [a.attempt for a in draft_attempts] == [1]

    def test_attempt_numbers_are_independent_across_tickets(self):
        sink = TelemetrySink()
        record_attempt(sink, "ticket-1")
        record_attempt(sink, "ticket-2")
        assert sink.drain("ticket-1")[0].attempt == 1
        assert sink.drain("ticket-2")[0].attempt == 1

    def test_drain_orders_by_started_at_across_operations(self):
        sink = TelemetrySink()
        record_attempt(
            sink, "ticket-1", "draft", started_at=BASE_TIME + timedelta(seconds=5)
        )
        record_attempt(sink, "ticket-1", "classify", started_at=BASE_TIME)

        attempts = sink.drain("ticket-1")
        assert [a.operation for a in attempts] == ["classify", "draft"]

    def test_drain_empties_the_ticket(self):
        sink = TelemetrySink()
        record_attempt(sink, "ticket-1")
        assert len(sink.drain("ticket-1")) == 1
        assert sink.drain("ticket-1") == []

    def test_drain_unknown_ticket_returns_empty_list(self):
        sink = TelemetrySink()
        assert sink.drain("nope") == []


class TestTelemetrySinkConcurrency:
    async def test_concurrent_coroutine_records_are_serialized(self):
        sink = TelemetrySink()
        n = 200

        async def do_record(i: int):
            record_attempt(
                sink, "ticket-1", started_at=BASE_TIME + timedelta(microseconds=i)
            )

        await asyncio.gather(*(do_record(i) for i in range(n)))

        attempts = sink.drain("ticket-1")
        assert sorted(a.attempt for a in attempts) == list(range(1, n + 1))

    def test_concurrent_thread_records_are_serialized(self):
        sink = TelemetrySink()
        n = 200

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(
                pool.map(
                    lambda i: record_attempt(
                        sink,
                        "ticket-1",
                        started_at=BASE_TIME + timedelta(microseconds=i),
                    ),
                    range(n),
                )
            )

        attempts = sink.drain("ticket-1")
        assert sorted(a.attempt for a in attempts) == list(range(1, n + 1))


class TestCallEventEnrichment:
    def test_build_call_event_carries_every_raw_field_and_adds_join_fields(self):
        sink = TelemetrySink()
        raw = record_attempt(
            sink,
            "ticket-1",
            "draft",
            role="fallback",
            cache_hit=True,
            wall_latency_ms=42.0,
            outcome="invalid_output",
            error_type="SchemaError",
        )

        event = build_call_event(
            raw, run_id="run-1", policy="oracle", case_key="case-1", repeat_index=2
        )

        assert event.run_id == "run-1"
        assert event.policy == "oracle"
        assert event.case_key == "case-1"
        assert event.repeat_index == 2
        assert event.ticket_id == "ticket-1"
        assert event.operation == "draft"
        assert event.role == "fallback"
        assert event.attempt == 1
        assert event.cache_hit is True
        assert event.wall_latency_ms == 42.0
        assert event.outcome == "invalid_output"
        assert event.error_type == "SchemaError"

    def test_drain_call_events_enriches_every_drained_attempt(self):
        sink = TelemetrySink()
        record_attempt(sink, "ticket-1", "classify")
        record_attempt(sink, "ticket-1", "draft")

        events = drain_call_events(
            sink,
            "ticket-1",
            run_id="run-1",
            policy="rubber_stamp",
            case_key="case-1",
            repeat_index=0,
        )

        assert len(events) == 2
        assert all(
            e.case_key == "case-1" and e.policy == "rubber_stamp" for e in events
        )
        assert sink.drain("ticket-1") == []
