import asyncio
import uuid
from datetime import timedelta
from typing import cast

import pytest

from tests.helpers import (
    ScriptedAgent,
    billing_classification,
    make_ticket,
    refund_draft,
    reply_only_draft,
    wait_for_status,
)
from ticketflow import workflows
from ticketflow.activities import TicketActivities
from ticketflow.eval.harness import (
    CombinedWorker,
    WorkflowEvalConfig,
    current_workflow_eval_config,
    local_environment,
    make_agent_worker,
    make_run_workers,
    make_workflow_worker,
    patched_workflow_constants,
    time_skipping_environment,
)
from ticketflow.eval.records import RunManifest
from ticketflow.models import TicketStatus, TicketStatusInfo
from ticketflow.workflows import TicketWorkflow


def unique_queue() -> str:
    return f"tq-{uuid.uuid4().hex[:8]}"


def _distinct_config() -> WorkflowEvalConfig:
    current = current_workflow_eval_config()
    return WorkflowEvalConfig(
        confidence_threshold=current.confidence_threshold + 0.1,
        agent_task_queue=f"{current.agent_task_queue}-patched",
        fallback_task_queue=f"{current.fallback_task_queue}-patched",
        agent_schedule_to_start_s=current.agent_schedule_to_start_s + 1,
        agent_activity_timeout_s=current.agent_activity_timeout_s + 1,
        agent_heartbeat_timeout_s=current.agent_heartbeat_timeout_s + 1,
    )


async def test_patched_activity_timeout_reaches_primary_and_fallback_options(
    monkeypatch,
):
    """M2-T2: a preflight-adjusted timeout must reach *both* activity option sets.

    Both option dicts are built from module globals at call time, so this captures
    what each path actually hands to Temporal rather than trusting that.
    """
    captured: list[dict] = []

    async def fake_execute_activity_method(_activity_method, *_args, **kwargs):
        captured.append(kwargs)
        return "ok"

    monkeypatch.setattr(
        workflows.workflow, "execute_activity_method", fake_execute_activity_method
    )
    config = _distinct_config().model_copy(
        update={"agent_activity_timeout_s": 123.0, "agent_heartbeat_timeout_s": 45.0}
    )

    with patched_workflow_constants(config):
        # Neither path touches `self`; both read the patched module constants.
        await TicketWorkflow._execute_agent_activity(
            cast(TicketWorkflow, None), TicketActivities.classify_ticket
        )
        await workflows._execute_fallback_agent_activity(
            TicketActivities.classify_ticket
        )

    assert len(captured) == 2
    primary, fallback = captured
    assert primary["task_queue"] == config.agent_task_queue
    assert fallback["task_queue"] == config.fallback_task_queue
    assert primary["schedule_to_start_timeout"] == timedelta(
        seconds=config.agent_schedule_to_start_s
    )
    for options in (primary, fallback):
        assert options["start_to_close_timeout"] == timedelta(seconds=123.0)
        assert options["heartbeat_timeout"] == timedelta(seconds=45.0)


def test_current_workflow_eval_config_reads_module_defaults():
    config = current_workflow_eval_config()

    assert isinstance(config, WorkflowEvalConfig)
    assert config.confidence_threshold == workflows.CONFIDENCE_THRESHOLD
    assert config.agent_task_queue == workflows.AGENT_TASK_QUEUE
    assert config.fallback_task_queue == workflows.FALLBACK_TASK_QUEUE
    assert config.agent_schedule_to_start_s == workflows.AGENT_SCHEDULE_TO_START_S
    assert (
        config.agent_activity_timeout_s
        == workflows.AGENT_ACTIVITY_TIMEOUT.total_seconds()
    )
    assert (
        config.agent_heartbeat_timeout_s
        == workflows.AGENT_HEARTBEAT_TIMEOUT.total_seconds()
    )


def test_workflow_eval_config_field_names_match_run_manifest():
    assert set(WorkflowEvalConfig.model_fields) <= set(RunManifest.model_fields)


def test_patched_workflow_constants_applies_all_six_and_restores_on_exit():
    before = current_workflow_eval_config()
    patched = _distinct_config()

    with patched_workflow_constants(patched):
        assert workflows.CONFIDENCE_THRESHOLD == patched.confidence_threshold
        assert workflows.AGENT_TASK_QUEUE == patched.agent_task_queue
        assert workflows.FALLBACK_TASK_QUEUE == patched.fallback_task_queue
        assert workflows.AGENT_SCHEDULE_TO_START_S == patched.agent_schedule_to_start_s
        assert workflows.AGENT_ACTIVITY_TIMEOUT == timedelta(
            seconds=patched.agent_activity_timeout_s
        )
        assert workflows.AGENT_HEARTBEAT_TIMEOUT == timedelta(
            seconds=patched.agent_heartbeat_timeout_s
        )

    assert current_workflow_eval_config() == before


def test_patched_workflow_constants_restores_on_exception():
    before = current_workflow_eval_config()

    with pytest.raises(ValueError):
        with patched_workflow_constants(_distinct_config()):
            raise ValueError("boom")

    assert current_workflow_eval_config() == before


async def test_make_run_workers_hosts_distinguishable_primary_and_fallback_agents():
    primary_agent = ScriptedAgent(
        billing_classification(model="primary"),
        reply_only_draft(confidence=0.9, model="primary"),
    )
    fallback_agent = ScriptedAgent(
        billing_classification(model="fallback"),
        reply_only_draft(confidence=0.9, model="fallback"),
    )
    ticket = make_ticket()
    config = _distinct_config()
    workflow_queue = unique_queue()

    async with time_skipping_environment() as env:
        with patched_workflow_constants(config):
            async with make_run_workers(
                env.client,
                workflow_eval_config=config,
                workflow_task_queue=workflow_queue,
                primary_agent=primary_agent,
                fallback_agent=fallback_agent,
            ):
                result = await env.client.execute_workflow(
                    TicketWorkflow.run,
                    ticket,
                    id=f"ticket-{ticket.id}",
                    task_queue=workflow_queue,
                )

    assert primary_agent.classify_calls == 1
    assert fallback_agent.classify_calls == 0
    assert result.model_path == "primary/primary"


async def test_make_run_workers_routes_to_fallback_when_primary_never_polls():
    primary_agent = ScriptedAgent(
        billing_classification(model="primary"),
        reply_only_draft(confidence=0.9, model="primary"),
    )
    fallback_agent = ScriptedAgent(
        billing_classification(confidence=0.5, model="fallback"),
        reply_only_draft(confidence=0.5, model="fallback"),
    )
    ticket = make_ticket()
    config = _distinct_config().model_copy(update={"agent_schedule_to_start_s": 0.1})
    workflow_queue = unique_queue()

    async with time_skipping_environment() as env:
        with patched_workflow_constants(config):
            workflow_worker = make_workflow_worker(
                env.client, workflow_queue, primary_agent
            )
            fallback_worker = make_agent_worker(
                env.client, fallback_agent, config.fallback_task_queue
            )
            async with CombinedWorker(workflow_worker, fallback_worker):
                result = await env.client.execute_workflow(
                    TicketWorkflow.run,
                    ticket,
                    id=f"ticket-{ticket.id}",
                    task_queue=workflow_queue,
                )

    assert primary_agent.classify_calls == 0
    assert fallback_agent.classify_calls == 1
    assert result.model_path == "fallback/fallback"


async def test_query_after_completion_replays_on_live_worker():
    agent = ScriptedAgent(billing_classification(), reply_only_draft(confidence=0.9))
    ticket = make_ticket()
    config = _distinct_config()
    workflow_queue = unique_queue()

    async with time_skipping_environment() as env:
        with patched_workflow_constants(config):
            async with make_run_workers(
                env.client,
                workflow_eval_config=config,
                workflow_task_queue=workflow_queue,
                primary_agent=agent,
            ):
                handle = await env.client.start_workflow(
                    TicketWorkflow.run,
                    ticket,
                    id=f"ticket-{ticket.id}",
                    task_queue=workflow_queue,
                )
                await handle.result()

                info = cast(TicketStatusInfo, await handle.query(TicketWorkflow.status))

    assert info.status == TicketStatus.RESOLVED
    assert info.draft is not None


async def test_local_environment_registers_ticket_status_for_list_workflows_filter():
    agent = ScriptedAgent(billing_classification(), refund_draft(amount=42.0))
    ticket = make_ticket()
    config = _distinct_config()
    workflow_queue = unique_queue()

    async with local_environment() as env:
        with patched_workflow_constants(config):
            async with make_run_workers(
                env.client,
                workflow_eval_config=config,
                workflow_task_queue=workflow_queue,
                primary_agent=agent,
            ):
                handle = await env.client.start_workflow(
                    TicketWorkflow.run,
                    ticket,
                    id=f"ticket-{ticket.id}",
                    task_queue=workflow_queue,
                )
                await wait_for_status(handle, TicketStatus.AWAITING_APPROVAL)

                query = (
                    'WorkflowType = "TicketWorkflow" and '
                    'TicketStatus = "awaiting_approval"'
                )
                for _ in range(100):
                    ids = [
                        workflow.id
                        async for workflow in env.client.list_workflows(query)
                    ]
                    if f"ticket-{ticket.id}" in ids:
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("ticket never appeared in approval inbox")
