import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import cast

from temporalio.client import Client, WorkflowUpdateFailedError

from tests.helpers import (
    ScriptedAgent,
    billing_classification,
    refund_draft,
    reply_only_draft,
)
from ticketflow.agent.tunable import TunableAgentProfile, TunableMockAgent
from ticketflow.eval import runner as runner_module
from ticketflow.eval.dataset import EvalCase, ExpectedOutcome
from ticketflow.eval.harness import current_workflow_eval_config, make_run_workers
from ticketflow.eval.reviewers import oracle, rubber_stamp
from ticketflow.eval.runner import CaseRunner
from ticketflow.eval.telemetry import RuntimeIdentityMap, TelemetrySink
from ticketflow.models import (
    ActionType,
    ApprovalDecision,
    TicketCategory,
    TicketResult,
    TicketStatus,
    TicketStatusInfo,
)


def _case() -> EvalCase:
    return EvalCase(
        id="case-1",
        customer_email="eval@example.com",
        subject="Help with my account",
        body="Please help me log in.",
        expected=ExpectedOutcome(
            acceptable_categories=frozenset({TicketCategory.BILLING}),
            reference_category=TicketCategory.BILLING,
            acceptable_actions=frozenset({ActionType.REPLY_ONLY}),
        ),
        difficulty="easy",
        source="handwritten",
        authored_by="tester",
        label_verified=True,
        verified_by="reviewer",
        verified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _refund_case() -> EvalCase:
    return _case().model_copy(
        update={
            "id": "case-refund",
            "expected": ExpectedOutcome(
                acceptable_categories=frozenset({TicketCategory.BILLING}),
                reference_category=TicketCategory.BILLING,
                acceptable_actions=frozenset({ActionType.REFUND}),
                expected_refund_amount=42.0,
            ),
        }
    )


async def test_run_case_captures_terminal_workflow_state_and_identity(env, tmp_path):
    workflow_queue = f"runner-{uuid.uuid4().hex}"
    case = _case()
    identity_map = RuntimeIdentityMap()
    telemetry_sink = TelemetrySink()
    agent = TunableMockAgent(
        identity_map=identity_map,
        telemetry_sink=telemetry_sink,
        expected_outcomes={case.id: case.expected},
        profile=TunableAgentProfile(),
        generation_seed=0,
    )
    runner = CaseRunner(
        env.client,
        run_id="run-1",
        workflow_task_queue=workflow_queue,
        db_path=str(tmp_path / "runner.db"),
        case_deadline=timedelta(seconds=5),
        identity_map=identity_map,
        telemetry_sink=telemetry_sink,
    )

    async with make_run_workers(
        env.client,
        workflow_eval_config=current_workflow_eval_config(),
        workflow_task_queue=workflow_queue,
        primary_agent=agent,
        db_path=str(tmp_path / "runner.db"),
    ):
        record, events = await runner.run_case(
            case,
            policy="oracle",
            reviewer=lambda _: ApprovalDecision(approved=True, approver="reviewer"),
            repeat_index=0,
        )

    assert record.case_key == "case-1"
    assert identity_map.resolve(record.ticket_id) == "case-1"
    assert record.terminal_status == TicketStatus.RESOLVED
    assert record.terminal_outcome == "resolved"
    assert record.predicted_category == TicketCategory.BILLING
    assert record.predicted_action == ActionType.REPLY_ONLY
    assert record.prediction_available is True
    assert record.decision is None
    assert record.refund_executed_count == 0
    assert record.refund_attempt_count == 0
    assert [(event.operation, event.attempt) for event in events] == [
        ("classify", 1),
        ("draft", 1),
    ]
    assert all(
        event.run_id == "run-1"
        and event.policy == "oracle"
        and event.case_key == "case-1"
        and event.repeat_index == 0
        and event.ticket_id == record.ticket_id
        for event in events
    )


async def test_run_case_submits_approval_and_uses_public_refund_observation(
    env, tmp_path
):
    workflow_queue = f"runner-{uuid.uuid4().hex}"
    agent = ScriptedAgent(billing_classification(), refund_draft(amount=42.0))
    runner = CaseRunner(
        env.client,
        run_id="run-1",
        workflow_task_queue=workflow_queue,
        db_path=str(tmp_path / "runner.db"),
        case_deadline=timedelta(seconds=5),
        identity_map=RuntimeIdentityMap(),
        telemetry_sink=TelemetrySink(),
    )

    async with make_run_workers(
        env.client,
        workflow_eval_config=current_workflow_eval_config(),
        workflow_task_queue=workflow_queue,
        primary_agent=agent,
        db_path=str(tmp_path / "runner.db"),
    ):
        record, _ = await runner.run_case(
            _refund_case(),
            policy="rubber_stamp",
            reviewer=rubber_stamp,
            repeat_index=0,
        )

    assert record.terminal_status == TicketStatus.RESOLVED
    assert record.was_gated is True
    assert record.decision is not None and record.decision.approved is True
    assert record.refund_executed_count == 1
    assert record.refund_attempt_count == 1


async def test_run_case_records_oracle_rejection_for_incorrect_gated_draft(
    env, tmp_path
):
    workflow_queue = f"runner-{uuid.uuid4().hex}"
    agent = ScriptedAgent(
        billing_classification(),
        refund_draft(amount=42.0, confidence=0.9),
    )
    runner = CaseRunner(
        env.client,
        run_id="run-1",
        workflow_task_queue=workflow_queue,
        db_path=str(tmp_path / "runner.db"),
        case_deadline=timedelta(seconds=5),
        identity_map=RuntimeIdentityMap(),
        telemetry_sink=TelemetrySink(),
    )
    incorrect_case = _case()

    async with make_run_workers(
        env.client,
        workflow_eval_config=current_workflow_eval_config(),
        workflow_task_queue=workflow_queue,
        primary_agent=agent,
        db_path=str(tmp_path / "runner.db"),
    ):
        record, _ = await runner.run_case(
            incorrect_case,
            policy="oracle",
            reviewer=oracle,
            repeat_index=0,
        )

    assert record.terminal_status == TicketStatus.REJECTED
    assert record.terminal_outcome == "rejected"
    assert record.decision is not None and record.decision.approved is False


class _RejectedUpdateHandle:
    async def query(self, _query):
        return TicketStatusInfo(
            ticket_id="ignored",
            status=TicketStatus.AWAITING_APPROVAL,
            classification=billing_classification(),
            draft=reply_only_draft(confidence=0.5),
        )

    async def execute_update(self, *_args, **_kwargs):
        raise WorkflowUpdateFailedError(RuntimeError("lost approval race"))


class _DeadlineHandle:
    def __init__(self):
        self.cancelled = False

    async def query(self, _query):
        return TicketStatusInfo(
            ticket_id="ignored",
            status=TicketStatus.AWAITING_APPROVAL,
            classification=billing_classification(),
            draft=reply_only_draft(confidence=0.5),
        )

    async def cancel(self, **_kwargs):
        self.cancelled = True

    async def execute_update(self, *_args, **_kwargs):
        await asyncio.Event().wait()

    async def result(self) -> object:
        raise RuntimeError("workflow cancellation confirmed")


class _SingleHandleClient:
    def __init__(self, handle):
        self.handle = handle

    async def start_workflow(self, *_args, **_kwargs):
        return self.handle


class _SlowStartClient(_SingleHandleClient):
    async def start_workflow(self, *_args, **_kwargs):
        await asyncio.sleep(0.01)
        return self.handle


class _TerminalThenQueryFailureHandle:
    def __init__(self):
        self.query_count = 0

    async def query(self, _query):
        self.query_count += 1
        if self.query_count > 1:
            raise RuntimeError("post-completion query unavailable")
        return TicketStatusInfo(
            ticket_id="ignored",
            status=TicketStatus.RESOLVED,
            classification=billing_classification(),
            draft=reply_only_draft(confidence=0.9),
        )

    async def result(self):
        return TicketResult(
            ticket_id="ignored",
            status=TicketStatus.RESOLVED,
            reply_text="Try restarting the app.",
        )


class _TerminationFailureHandle(_DeadlineHandle):
    async def result(self) -> object:
        await asyncio.Event().wait()

    async def terminate(self, **_kwargs):
        raise RuntimeError("termination RPC unavailable")


async def test_run_case_records_a_rejected_approval_update(tmp_path):
    runner = CaseRunner(
        cast(Client, _SingleHandleClient(_RejectedUpdateHandle())),
        run_id="run-1",
        workflow_task_queue="unused",
        db_path=str(tmp_path / "runner.db"),
        case_deadline=timedelta(seconds=1),
        identity_map=RuntimeIdentityMap(),
        telemetry_sink=TelemetrySink(),
    )

    record, _ = await runner.run_case(
        _case(),
        policy="oracle",
        reviewer=lambda _: ApprovalDecision(approved=True, approver="reviewer"),
        repeat_index=0,
    )

    assert record.terminal_outcome == "update_rejected"
    assert record.terminal_error == "Workflow update failed"
    assert record.prediction_available is True
    assert record.was_gated is True


async def test_runner_deadline_preserves_captured_draft_and_cancels(tmp_path):
    handle = _DeadlineHandle()
    runner = CaseRunner(
        cast(Client, _SingleHandleClient(handle)),
        run_id="run-1",
        workflow_task_queue="unused",
        db_path=str(tmp_path / "runner.db"),
        case_deadline=timedelta(milliseconds=1),
        identity_map=RuntimeIdentityMap(),
        telemetry_sink=TelemetrySink(),
        poll_interval_s=0.01,
    )

    record, _ = await runner.run_case(
        _case(),
        policy="oracle",
        reviewer=lambda _: ApprovalDecision(approved=True, approver="reviewer"),
        repeat_index=0,
    )

    assert handle.cancelled is True
    assert record.terminal_outcome == "runner_deadline_exceeded"
    assert record.cleanup_action == "cancelled"
    assert record.prediction_available is True
    assert record.reply_text == "Try restarting the app."


async def test_runner_deadline_includes_workflow_startup_time(tmp_path):
    handle = _DeadlineHandle()
    runner = CaseRunner(
        cast(Client, _SlowStartClient(handle)),
        run_id="run-1",
        workflow_task_queue="unused",
        db_path=str(tmp_path / "runner.db"),
        case_deadline=timedelta(milliseconds=1),
        identity_map=RuntimeIdentityMap(),
        telemetry_sink=TelemetrySink(),
    )

    record, _ = await runner.run_case(
        _case(),
        policy="oracle",
        reviewer=lambda _: ApprovalDecision(approved=True, approver="reviewer"),
        repeat_index=0,
    )

    assert handle.cancelled is True
    assert record.terminal_outcome == "runner_deadline_exceeded"


async def test_completed_record_keeps_last_status_when_final_query_fails(tmp_path):
    runner = CaseRunner(
        cast(Client, _SingleHandleClient(_TerminalThenQueryFailureHandle())),
        run_id="run-1",
        workflow_task_queue="unused",
        db_path=str(tmp_path / "runner.db"),
        case_deadline=timedelta(seconds=1),
        identity_map=RuntimeIdentityMap(),
        telemetry_sink=TelemetrySink(),
    )

    record, _ = await runner.run_case(
        _case(),
        policy="oracle",
        reviewer=lambda _: ApprovalDecision(approved=True, approver="reviewer"),
        repeat_index=0,
    )

    assert record.prediction_available is True
    assert record.predicted_category == TicketCategory.BILLING
    assert record.reply_text == "Try restarting the app."


async def test_runner_records_failed_termination_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "_CANCEL_CONFIRMATION_TIMEOUT_S", 0.001)
    runner = CaseRunner(
        cast(Client, _SingleHandleClient(_TerminationFailureHandle())),
        run_id="run-1",
        workflow_task_queue="unused",
        db_path=str(tmp_path / "runner.db"),
        case_deadline=timedelta(milliseconds=1),
        identity_map=RuntimeIdentityMap(),
        telemetry_sink=TelemetrySink(),
        poll_interval_s=0.01,
    )

    record, _ = await runner.run_case(
        _case(),
        policy="oracle",
        reviewer=lambda _: ApprovalDecision(approved=True, approver="reviewer"),
        repeat_index=0,
    )

    assert record.terminal_outcome == "runner_deadline_exceeded"
    assert record.cleanup_action == "terminated"
    assert "termination request failed" in (record.terminal_error or "")
