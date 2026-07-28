"""Cross-cutting, network-free workflow regression tests for the eval harness."""

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from temporalio import activity
from temporalio.client import Client, WorkflowUpdateFailedError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from tests.helpers import billing_classification, reply_only_draft
from ticketflow import workflows
from ticketflow.activities import TicketActivities
from ticketflow.agent.base import AgentOverloadedError
from ticketflow.agent.tunable import TunableAgentProfile, TunableMockAgent
from ticketflow.eval.dataset import EvalCase, ExpectedOutcome
from ticketflow.eval.harness import (
    CombinedWorker,
    current_workflow_eval_config,
    make_agent_worker,
)
from ticketflow.eval.profiles import RunOptions, RunProfile, run_profile
from ticketflow.eval.records import CallEvent, CaseRecord
from ticketflow.eval.runner import CaseRunner
from ticketflow.eval.telemetry import RuntimeIdentityMap, TelemetrySink
from ticketflow.models import (
    ActionType,
    ApprovalDecision,
    TicketCategory,
    TicketStatus,
    TicketStatusInfo,
)
from ticketflow.workflows import TicketWorkflow


def _case(case_id: str, *, refund: bool = False) -> EvalCase:
    """Build one labelled case with either a reply-only or refund outcome."""
    expected = ExpectedOutcome(
        acceptable_categories=frozenset({TicketCategory.BILLING}),
        reference_category=TicketCategory.BILLING,
        acceptable_actions=frozenset(
            {ActionType.REFUND if refund else ActionType.REPLY_ONLY}
        ),
        expected_refund_amount=42.0 if refund else None,
    )
    return EvalCase(
        id=case_id,
        customer_email="eval@example.com",
        subject="Account help",
        body="Please help with my account.",
        expected=expected,
        difficulty="easy",
        source="handwritten",
        authored_by="workflow-suite",
        label_verified=True,
        verified_by="reviewer",
        verified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _write_dataset(tmp_path: Path, cases: list[EvalCase], name: str) -> Path:
    """Persist cases so profile provenance hashes an actual dataset artifact."""
    path = tmp_path / name
    path.write_text("\n".join(case.model_dump_json() for case in cases) + "\n")
    return path


def _options(
    tmp_path: Path,
    cases: list[EvalCase],
    profile: RunProfile,
    **overrides: Any,
) -> RunOptions:
    """Build run options with isolated artifacts and a short case deadline."""
    dataset_path = overrides.pop("dataset_path", None) or _write_dataset(
        tmp_path, cases, f"{profile}-{uuid.uuid4().hex}.jsonl"
    )
    defaults: dict[str, Any] = {
        "profile": profile,
        "dataset_path": dataset_path,
        "cases": cases,
        "primary_agent_profile": TunableAgentProfile(),
        "db_path": str(tmp_path / f"{uuid.uuid4().hex}.db"),
        "case_deadline": timedelta(seconds=10),
        "run_id": f"workflow-suite-{uuid.uuid4().hex}",
    }
    defaults.update(overrides)
    return RunOptions(**defaults)


def _record(records: list[CaseRecord], policy: str, case_key: str) -> CaseRecord:
    """Return exactly one record for a policy/case pair."""
    matches = [
        record
        for record in records
        if record.policy == policy and record.case_key == case_key
    ]
    assert len(matches) == 1
    return matches[0]


def _normalized_records(records: list[CaseRecord]) -> list[tuple[object, ...]]:
    """Compare record semantics without run-specific IDs or timing fields."""
    return sorted(
        (
            record.policy,
            record.case_key,
            record.repeat_index,
            record.predicted_category,
            record.predicted_action,
            record.predicted_refund_amount,
            record.classification_confidence,
            record.draft_confidence,
            record.reply_text,
            record.model_path,
            record.terminal_status,
            record.was_gated,
            None if record.decision is None else record.decision.approved,
            record.refund_executed_count,
            record.refund_attempt_count,
            record.prediction_available,
            record.prediction_unavailable_reason,
            record.terminal_outcome,
            record.cleanup_action,
            record.terminal_error,
        )
        for record in records
    )


def _normalized_events(events: list[CallEvent]) -> list[tuple[object, ...]]:
    """Compare telemetry semantics without timestamps and runtime ticket IDs."""
    return sorted(
        (
            event.policy,
            event.case_key,
            event.repeat_index,
            event.operation,
            event.role,
            event.attempt,
            event.cache_hit,
            event.outcome,
            event.error_type,
        )
        for event in events
    )


async def test_profile_records_exact_errors_and_reviewer_outcomes(tmp_path):
    """Exact perturbation sets retain exact workflow-level reviewer outcomes."""
    cases = [
        _case("clean"),
        _case("category-error"),
        _case("action-error"),
        _case("expected-refund", refund=True),
    ]
    options = _options(
        tmp_path,
        cases,
        "primary-quality",
        primary_agent_profile=TunableAgentProfile(
            category_error_case_keys=frozenset({"category-error"}),
            action_error_case_keys=frozenset({"action-error"}),
        ),
    )

    _manifest, records, _events = await run_profile(options)

    for policy in ("oracle", "rubber_stamp"):
        policy_records = [record for record in records if record.policy == policy]
        assert len(policy_records) == len(cases)
        assert {
            record.case_key
            for record in policy_records
            if record.predicted_category != record.expected.reference_category
        } == {"category-error"}
        assert {
            record.case_key
            for record in policy_records
            if record.predicted_action not in record.expected.acceptable_actions
        } == {"action-error"}
        assert (
            sum(
                record.predicted_category != record.expected.reference_category
                for record in policy_records
            )
            / len(policy_records)
            == 1 / 4
        )
        assert (
            sum(
                record.predicted_action not in record.expected.acceptable_actions
                for record in policy_records
            )
            / len(policy_records)
            == 1 / 4
        )

    clean = _record(records, "oracle", "clean")
    assert clean.terminal_outcome == "resolved"
    assert clean.was_gated is False
    assert clean.decision is None

    oracle_error = _record(records, "oracle", "action-error")
    assert oracle_error.terminal_status == TicketStatus.REJECTED
    assert oracle_error.decision is not None and not oracle_error.decision.approved
    assert oracle_error.refund_executed_count == 0

    rubber_stamp_error = _record(records, "rubber_stamp", "action-error")
    assert rubber_stamp_error.terminal_status == TicketStatus.RESOLVED
    assert (
        rubber_stamp_error.decision is not None and rubber_stamp_error.decision.approved
    )
    assert rubber_stamp_error.refund_executed_count == 1


async def test_fallback_quality_and_routing_are_distinct_experiments(
    tmp_path, monkeypatch
):
    """Direct fallback quality and schedule-to-start routing stay separate."""
    monkeypatch.setattr(workflows, "AGENT_SCHEDULE_TO_START_S", 0.1)
    cases = [_case("fallback")]
    fallback_profile = TunableAgentProfile()

    quality_started = time.perf_counter()
    quality_manifest, quality_records, quality_events = await run_profile(
        _options(
            tmp_path,
            cases,
            "fallback-quality",
            fallback_agent_profile=fallback_profile,
        )
    )
    quality_elapsed = time.perf_counter() - quality_started
    routing_started = time.perf_counter()
    routing_manifest, routing_records, routing_events = await run_profile(
        _options(
            tmp_path,
            cases,
            "fallback-routing",
            fallback_agent_profile=fallback_profile,
        )
    )
    routing_elapsed = time.perf_counter() - routing_started

    assert quality_manifest.run_profile == "fallback-quality"
    assert routing_manifest.run_profile == "fallback-routing"
    assert quality_manifest.run_id != routing_manifest.run_id
    assert routing_manifest.agent_schedule_to_start_s == 0.1
    assert routing_elapsed >= quality_elapsed + 0.15
    assert all(record.model_path == "fallback/fallback" for record in quality_records)
    assert all(record.model_path == "fallback/fallback" for record in routing_records)
    assert all(event.role == "fallback" for event in quality_events)
    assert all(event.role == "fallback" for event in routing_events)


async def test_exact_transient_failures_emit_all_retry_events(tmp_path, monkeypatch):
    """An exact transient failure set produces all workflow retry attempts."""
    monkeypatch.setattr(
        workflows,
        "RETRY_POLICY",
        RetryPolicy(
            initial_interval=timedelta(milliseconds=1),
            backoff_coefficient=workflows.RETRY_POLICY.backoff_coefficient,
            maximum_attempts=workflows.RETRY_POLICY.maximum_attempts,
        ),
    )
    case = _case("flaky")
    _manifest, records, events = await run_profile(
        _options(
            tmp_path,
            [case],
            "reliability",
            primary_agent_profile=TunableAgentProfile(
                transient_failure_case_keys=frozenset({case.id})
            ),
        )
    )

    assert records[0].terminal_status == TicketStatus.ESCALATED
    assert records[0].prediction_available is False
    assert [event.operation for event in events] == ["classify"] * 5
    assert [event.attempt for event in events] == [1, 2, 3, 4, 5]
    assert all(event.outcome == "transient_error" for event in events)
    assert all(event.error_type == AgentOverloadedError.__name__ for event in events)


class _FailOnceRefundActivities(TicketActivities):
    """Persist a refund, then fail once so Temporal retries the activity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Track whether the first completed refund should trigger a retry."""
        super().__init__(*args, **kwargs)
        self._fail_once = True

    @activity.defn
    async def execute_refund(self, ticket_id: str, amount: float) -> None:
        """Record the refund before injecting a retryable first-attempt failure."""
        await super().execute_refund(ticket_id, amount)
        if self._fail_once:
            self._fail_once = False
            raise ApplicationError("retry refund activity")


async def test_refund_activity_retry_remains_idempotent(env, tmp_path):
    """A replayed refund activity creates one refund and two recorded attempts."""
    case = _case("refund", refund=True)
    identity_map = RuntimeIdentityMap()
    telemetry_sink = TelemetrySink()
    db_path = str(tmp_path / "refund-retry.db")
    agent = TunableMockAgent(
        identity_map=identity_map,
        telemetry_sink=telemetry_sink,
        expected_outcomes={case.id: case.expected},
        profile=TunableAgentProfile(),
        generation_seed=0,
    )
    config = current_workflow_eval_config()
    workflow_queue = f"workflow-suite-{uuid.uuid4().hex}"
    activities = _FailOnceRefundActivities(agent, db_path=db_path)
    workflow_worker = Worker(
        env.client,
        task_queue=workflow_queue,
        workflows=[TicketWorkflow],
        activities=[
            activities.send_reply,
            activities.execute_refund,
            activities.record_result,
        ],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    agent_worker = make_agent_worker(env.client, agent, config.agent_task_queue)
    runner = CaseRunner(
        env.client,
        run_id="refund-retry",
        workflow_task_queue=workflow_queue,
        db_path=db_path,
        case_deadline=timedelta(seconds=10),
        identity_map=identity_map,
        telemetry_sink=telemetry_sink,
    )

    async with CombinedWorker(workflow_worker, agent_worker):
        record, _events = await runner.run_case(
            case,
            policy="rubber_stamp",
            reviewer=lambda _: ApprovalDecision(approved=True, approver="reviewer"),
            repeat_index=0,
        )

    assert record.terminal_status == TicketStatus.RESOLVED
    assert record.refund_executed_count == 1
    assert record.refund_attempt_count == 2


async def test_concurrency_does_not_change_records_or_telemetry(tmp_path):
    """The same seeded run is semantically identical at concurrency one and eight."""
    cases = [_case(f"case-{index}") for index in range(5)]
    dataset_path = _write_dataset(tmp_path, cases, "concurrency.jsonl")
    profile = TunableAgentProfile(
        category_error_case_keys=frozenset({"case-1", "case-3"}),
        action_error_case_keys=frozenset({"case-2"}),
    )

    _serial_manifest, serial_records, serial_events = await run_profile(
        _options(
            tmp_path,
            cases,
            "primary-quality",
            dataset_path=dataset_path,
            primary_agent_profile=profile,
            seed=17,
            concurrency=1,
        )
    )
    _parallel_manifest, parallel_records, parallel_events = await run_profile(
        _options(
            tmp_path,
            cases,
            "primary-quality",
            dataset_path=dataset_path,
            primary_agent_profile=profile,
            seed=17,
            concurrency=8,
        )
    )

    assert _normalized_records(serial_records) == _normalized_records(parallel_records)
    assert _normalized_events(serial_events) == _normalized_events(parallel_events)


class _RejectedUpdateHandle:
    """Expose an awaiting workflow whose approval update deterministically loses."""

    async def query(self, _query: object) -> TicketStatusInfo:
        """Return a captured gated draft for runner record construction."""
        return TicketStatusInfo(
            ticket_id="ignored",
            status=TicketStatus.AWAITING_APPROVAL,
            classification=billing_classification(),
            draft=reply_only_draft(confidence=0.5),
        )

    async def execute_update(self, *_args: Any, **_kwargs: Any) -> None:
        """Model Temporal rejecting a racing approval update."""
        raise WorkflowUpdateFailedError(RuntimeError("lost approval race"))


class _DeadlineHandle:
    """Expose an awaiting workflow whose approval never completes."""

    def __init__(self) -> None:
        """Track the runner's requested cancellation."""
        self.cancelled = False

    async def query(self, _query: object) -> TicketStatusInfo:
        """Return a captured gated draft for the deadline record."""
        return TicketStatusInfo(
            ticket_id="ignored",
            status=TicketStatus.AWAITING_APPROVAL,
            classification=billing_classification(),
            draft=reply_only_draft(confidence=0.5),
        )

    async def execute_update(self, *_args: Any, **_kwargs: Any) -> None:
        """Keep the runner's update wait pending until its deadline fires."""
        await asyncio.Event().wait()

    async def cancel(self, **_kwargs: Any) -> None:
        """Record deadline cleanup cancellation."""
        self.cancelled = True

    async def result(self) -> object:
        """Confirm cancellation without completing the original update."""
        raise RuntimeError("workflow cancellation confirmed")


class _SingleHandleClient:
    """Return one deterministic handle from ``start_workflow``."""

    def __init__(self, handle: object) -> None:
        """Store the fake workflow handle."""
        self._handle = handle

    async def start_workflow(self, *_args: Any, **_kwargs: Any) -> object:
        """Match the subset of the Temporal client used by ``CaseRunner``."""
        return self._handle


def _fake_runner(handle: object, tmp_path: Path, *, deadline: timedelta) -> CaseRunner:
    """Build a runner that uses a deterministic, minimal client double."""
    return CaseRunner(
        cast(Client, _SingleHandleClient(handle)),
        run_id="runner-failure",
        workflow_task_queue="unused",
        db_path=str(tmp_path / "runner-failure.db"),
        case_deadline=deadline,
        identity_map=RuntimeIdentityMap(),
        telemetry_sink=TelemetrySink(),
        poll_interval_s=0.01,
    )


async def test_rejected_approval_update_is_a_case_outcome(tmp_path):
    """A rejected update is represented in the record instead of escaping."""
    record, _events = await _fake_runner(
        _RejectedUpdateHandle(), tmp_path, deadline=timedelta(seconds=1)
    ).run_case(
        _case("rejected-update"),
        policy="oracle",
        reviewer=lambda _: ApprovalDecision(approved=True, approver="reviewer"),
        repeat_index=0,
    )

    assert record.terminal_outcome == "update_rejected"
    assert record.terminal_error == "Workflow update failed"
    assert record.prediction_available is True


async def test_case_deadline_is_a_case_outcome(tmp_path):
    """A runner deadline preserves the captured draft and requests cancellation."""
    handle = _DeadlineHandle()
    record, _events = await _fake_runner(
        handle, tmp_path, deadline=timedelta(milliseconds=1)
    ).run_case(
        _case("deadline"),
        policy="oracle",
        reviewer=lambda _: ApprovalDecision(approved=True, approver="reviewer"),
        repeat_index=0,
    )

    assert handle.cancelled is True
    assert record.terminal_outcome == "runner_deadline_exceeded"
    assert record.cleanup_action == "cancelled"
    assert record.prediction_available is True
