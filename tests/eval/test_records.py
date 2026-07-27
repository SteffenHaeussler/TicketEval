import json
import os
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ticketflow.eval.dataset import ExpectedOutcome
from ticketflow.eval.records import (
    ArtifactExistsError,
    CallEvent,
    CaseRecord,
    RecordsReadError,
    RunManifest,
    read_call_events,
    read_case_records,
    read_run_manifest,
    write_call_events,
    write_case_records,
    write_run_manifest,
)
from ticketflow.models import ApprovalDecision, TicketStatus


def make_expected(**overrides):
    base = {
        "acceptable_categories": ["billing"],
        "reference_category": "billing",
        "acceptable_actions": ["reply_only"],
        "expected_refund_amount": None,
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
        terminal_status=TicketStatus.RESOLVED,
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


def make_manifest(**overrides):
    base = dict(
        run_id="run-1",
        git_commit="abc123",
        git_dirty=False,
        dataset_path="evals/data/tickets",
        dataset_sha256="deadbeef",
        agent_backend="mock",
        run_profile="primary-quality",
        primary_model="primary-model",
        fallback_model=None,
        python_version="3.12.0",
        dependency_versions={"pydantic": "2.9.0"},
        reviewer_policies=["oracle", "rubber_stamp"],
        cache_enabled=True,
        confidence_threshold=0.75,
        agent_task_queue="agent-queue",
        fallback_task_queue="fallback-queue",
        agent_schedule_to_start_s=30.0,
        agent_activity_timeout_s=60.0,
        agent_heartbeat_timeout_s=10.0,
        seed=0,
        bootstrap_seed=0,
        concurrency=4,
        repeats=1,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        finished_at=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return RunManifest.model_validate(base)


# --- overwrite refusal ---


def test_write_case_records_refuses_to_overwrite_existing_file(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("sentinel")

    with pytest.raises(ArtifactExistsError, match=str(path)):
        write_case_records(path, [make_case_record()])

    assert path.read_text() == "sentinel"
    assert list(tmp_path.iterdir()) == [path]


def test_write_call_events_refuses_to_overwrite_existing_file(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text("sentinel")

    with pytest.raises(ArtifactExistsError, match=str(path)):
        write_call_events(path, [make_call_event()])

    assert path.read_text() == "sentinel"
    assert list(tmp_path.iterdir()) == [path]


def test_write_run_manifest_refuses_to_overwrite_existing_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("sentinel")

    with pytest.raises(ArtifactExistsError, match=str(path)):
        write_run_manifest(path, make_manifest())

    assert path.read_text() == "sentinel"
    assert list(tmp_path.iterdir()) == [path]


# --- no partial file left behind ---


def test_write_case_records_touches_no_filesystem_when_serialization_raises(
    tmp_path, monkeypatch
):
    """Serialization happens fully in memory before any file is created."""
    path = tmp_path / "records.jsonl"

    def boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(CaseRecord, "model_dump_json", boom, raising=True)

    with pytest.raises(RuntimeError):
        write_case_records(path, [make_case_record()])

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_case_records_leaves_no_temp_file_when_fsync_raises(
    tmp_path, monkeypatch
):
    """Interrupts after content is written to the temp file but before commit."""
    path = tmp_path / "records.jsonl"

    def boom(fd):
        raise OSError("fsync boom")

    monkeypatch.setattr(os, "fsync", boom)

    with pytest.raises(OSError):
        write_case_records(path, [make_case_record()])

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_call_events_leaves_no_temp_file_when_fsync_raises(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"

    def boom(fd):
        raise OSError("fsync boom")

    monkeypatch.setattr(os, "fsync", boom)

    with pytest.raises(OSError):
        write_call_events(path, [make_call_event()])

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_run_manifest_leaves_no_temp_file_when_fsync_raises(
    tmp_path, monkeypatch
):
    path = tmp_path / "manifest.json"

    def boom(fd):
        raise OSError("fsync boom")

    monkeypatch.setattr(os, "fsync", boom)

    with pytest.raises(OSError):
        write_run_manifest(path, make_manifest())

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


# --- round trip ---


def test_case_record_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    record = make_case_record(
        predicted_category="technical",
        predicted_action="refund",
        predicted_refund_amount=12.5,
        terminal_status=TicketStatus.ESCALATED,
        was_gated=True,
        decision=ApprovalDecision(approved=True, approver="alice", note="looks fine"),
        terminal_outcome="runner_deadline_exceeded",
        cleanup_action="cancelled",
    )

    write_case_records(path, [record])
    [loaded] = read_case_records(path)

    assert loaded == record


def test_call_event_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "calls.jsonl"
    event = make_call_event(
        operation="draft",
        role="fallback",
        outcome="invalid_output",
        error_type="bad_json",
    )

    write_call_events(path, [event])
    [loaded] = read_call_events(path)

    assert loaded == event


def test_run_manifest_round_trips_through_json(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = make_manifest()

    write_run_manifest(path, manifest)
    loaded = read_run_manifest(path)

    assert loaded == manifest


def test_case_record_round_trip_preserves_expected_outcome_set_field(tmp_path):
    path = tmp_path / "records.jsonl"
    expected = make_expected(
        acceptable_categories=["billing", "technical"], reference_category="billing"
    )
    record = make_case_record(expected=expected)

    write_case_records(path, [record])
    [loaded] = read_case_records(path)

    assert loaded.expected.acceptable_categories == {"billing", "technical"}
    assert isinstance(loaded.expected.acceptable_categories, set)


# --- prediction_available invariant ---


def test_prediction_available_true_regardless_of_terminal_outcome():
    resolved = make_case_record(terminal_outcome="resolved", draft_confidence=0.5)
    update_rejected = make_case_record(
        terminal_outcome="update_rejected", draft_confidence=0.5
    )

    assert resolved.prediction_available is True
    assert update_rejected.prediction_available is True


def test_prediction_available_mismatch_with_draft_confidence_raises():
    with pytest.raises(ValidationError, match="case-1"):
        make_case_record(prediction_available=True, draft_confidence=None)


def test_prediction_available_false_with_draft_confidence_set_raises():
    with pytest.raises(ValidationError, match="case-1"):
        make_case_record(prediction_available=False, draft_confidence=0.9)


# --- cleanup_action invariant ---


def test_cleanup_action_required_for_runner_deadline_exceeded():
    with pytest.raises(ValidationError, match="case-1"):
        make_case_record(
            terminal_outcome="runner_deadline_exceeded", cleanup_action=None
        )


def test_cleanup_action_forbidden_outside_runner_deadline_exceeded():
    with pytest.raises(ValidationError, match="case-1"):
        make_case_record(terminal_outcome="resolved", cleanup_action="cancelled")


@pytest.mark.parametrize("cleanup_action", ["cancelled", "terminated"])
def test_cleanup_action_both_values_accepted_for_deadline_outcome(cleanup_action):
    record = make_case_record(
        terminal_outcome="runner_deadline_exceeded", cleanup_action=cleanup_action
    )
    assert record.cleanup_action == cleanup_action


# --- scored-population domain invariant (plan.md: escalation before a draft) ---


def test_escalation_before_a_draft_has_no_prediction_and_is_constructible():
    record = make_case_record(
        terminal_outcome="escalated",
        draft_confidence=None,
        classification_confidence=None,
        predicted_category=None,
        predicted_action=None,
        reply_text=None,
        prediction_available=False,
        prediction_unavailable_reason="agent exhausted repair budget",
    )

    assert record.prediction_available is False
    assert record.terminal_outcome == "escalated"


# --- frozen ---


def test_case_record_is_frozen():
    record = make_case_record()
    with pytest.raises(ValidationError):
        record.run_id = "other"


def test_call_event_is_frozen():
    event = make_call_event()
    with pytest.raises(ValidationError):
        event.run_id = "other"


def test_run_manifest_is_frozen():
    manifest = make_manifest()
    with pytest.raises(ValidationError):
        manifest.run_id = "other"


# --- RunManifest completeness ---


def test_run_manifest_defers_model_digests_and_preflight_fields_to_none_by_default():
    manifest = make_manifest()

    assert manifest.primary_model_digest is None
    assert manifest.fallback_model_digest is None
    assert manifest.ollama_version is None
    assert manifest.prompt_hash is None
    assert manifest.schema_hash is None
    assert manifest.generation_options is None
    assert manifest.preflight_measurements is None
    assert manifest.effective_activity_timeout_s is None
    assert manifest.safety_margin_s is None
    assert manifest.http_timeout_s is None


# --- reader error handling ---


def test_read_case_records_raises_records_read_error_on_invalid_json_line(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("not json\n")

    with pytest.raises(RecordsReadError, match=r"line 1"):
        read_case_records(path)


def test_read_case_records_raises_records_read_error_on_schema_violation(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"run_id": "run-1"}) + "\n")

    with pytest.raises(RecordsReadError):
        read_case_records(path)


def test_read_run_manifest_raises_records_read_error_on_invalid_json(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("not json")

    with pytest.raises(RecordsReadError, match=str(path)):
        read_run_manifest(path)


# --- empty list ---


def test_write_and_read_case_records_round_trips_empty_list(tmp_path):
    path = tmp_path / "records.jsonl"

    write_case_records(path, [])

    assert read_case_records(path) == []
