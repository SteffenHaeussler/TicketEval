"""In-process Temporal environment and worker construction for the eval harness."""

from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from datetime import timedelta

from pydantic import BaseModel, ConfigDict
from temporalio.api.enums.v1 import IndexedValueType
from temporalio.api.operatorservice.v1 import request_response_pb2
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker, WorkflowRunner

from ticketflow import workflows
from ticketflow.activities import TicketActivities
from ticketflow.agent.base import Agent


class WorkflowEvalConfig(BaseModel):
    """Snapshot of the ticketflow.workflows routing and timeout constants."""

    model_config = ConfigDict(frozen=True)

    confidence_threshold: float
    agent_task_queue: str
    fallback_task_queue: str
    agent_schedule_to_start_s: float
    agent_activity_timeout_s: float
    agent_heartbeat_timeout_s: float


def current_workflow_eval_config() -> WorkflowEvalConfig:
    """Read the live ticketflow.workflows module constants into a config value."""
    return WorkflowEvalConfig(
        confidence_threshold=workflows.CONFIDENCE_THRESHOLD,
        agent_task_queue=workflows.AGENT_TASK_QUEUE,
        fallback_task_queue=workflows.FALLBACK_TASK_QUEUE,
        agent_schedule_to_start_s=workflows.AGENT_SCHEDULE_TO_START_S,
        agent_activity_timeout_s=workflows.AGENT_ACTIVITY_TIMEOUT.total_seconds(),
        agent_heartbeat_timeout_s=workflows.AGENT_HEARTBEAT_TIMEOUT.total_seconds(),
    )


@contextmanager
def patched_workflow_constants(config: WorkflowEvalConfig) -> Iterator[None]:
    """Apply config onto ticketflow.workflows constants; restore on exit."""
    attrs = {
        "CONFIDENCE_THRESHOLD": config.confidence_threshold,
        "AGENT_TASK_QUEUE": config.agent_task_queue,
        "FALLBACK_TASK_QUEUE": config.fallback_task_queue,
        "AGENT_SCHEDULE_TO_START_S": config.agent_schedule_to_start_s,
        "AGENT_ACTIVITY_TIMEOUT": timedelta(seconds=config.agent_activity_timeout_s),
        "AGENT_HEARTBEAT_TIMEOUT": timedelta(seconds=config.agent_heartbeat_timeout_s),
    }
    previous = {name: getattr(workflows, name) for name in attrs}
    for name, value in attrs.items():
        setattr(workflows, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(workflows, name, value)


@asynccontextmanager
async def time_skipping_environment() -> AsyncIterator[WorkflowEnvironment]:
    """Start a time-skipping test server with TicketStatus registered."""
    env = await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    )
    try:
        await env.client.operator_service.add_search_attributes(
            request_response_pb2.AddSearchAttributesRequest(
                namespace=env.client.namespace,
                search_attributes={
                    "TicketStatus": IndexedValueType.Value("INDEXED_VALUE_TYPE_KEYWORD")
                },
            )
        )
        yield env
    finally:
        await env.shutdown()


@asynccontextmanager
async def local_environment() -> AsyncIterator[WorkflowEnvironment]:
    """Start a local real-time test server with TicketStatus registered."""
    env = await WorkflowEnvironment.start_local(
        data_converter=pydantic_data_converter,
        search_attributes=[workflows.TICKET_STATUS_ATTR],
    )
    try:
        yield env
    finally:
        await env.shutdown()


class CombinedWorker:
    """Async context manager that runs related Temporal workers together."""

    def __init__(self, *workers: Worker):
        """Hold the workers to be entered and exited together."""
        self._workers = workers
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "CombinedWorker":
        for worker in self._workers:
            await self._stack.enter_async_context(worker)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:
        return await self._stack.__aexit__(exc_type, exc, tb)


def make_worker(
    client: Client,
    agent: Agent,
    task_queue: str,
    workflow_runner: WorkflowRunner | None = None,
    db_path: str | None = None,
) -> CombinedWorker:
    """Build a workflow worker and an agent worker sharing one agent."""
    acts = TicketActivities(agent, db_path=db_path)
    workflow_activities = [
        acts.send_reply,
        acts.execute_refund,
        acts.record_result,
    ]
    agent_activities = [
        acts.classify_ticket,
        acts.draft_reply,
    ]
    if workflow_runner is not None:
        workflow_worker = Worker(
            client,
            task_queue=task_queue,
            workflows=[workflows.TicketWorkflow],
            activities=workflow_activities,
            workflow_runner=workflow_runner,
        )
    else:
        workflow_worker = Worker(
            client,
            task_queue=task_queue,
            workflows=[workflows.TicketWorkflow],
            activities=workflow_activities,
        )
    llm_worker = Worker(
        client,
        task_queue=workflows.AGENT_TASK_QUEUE,
        activities=agent_activities,
    )
    return CombinedWorker(workflow_worker, llm_worker)


def make_workflow_worker(
    client: Client,
    task_queue: str,
    agent: Agent,
    *,
    db_path: str | None = None,
) -> Worker:
    """Build a workflow worker hosting only the side-effect activities.

    `agent` is required by TicketActivities' constructor but never invoked by
    send_reply/execute_refund/record_result. Always uses
    UnsandboxedWorkflowRunner so patched workflows.* constants are visible to
    the running workflow.
    """
    acts = TicketActivities(agent, db_path=db_path)
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[workflows.TicketWorkflow],
        activities=[acts.send_reply, acts.execute_refund, acts.record_result],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )


def make_agent_worker(client: Client, agent: Agent, task_queue: str) -> Worker:
    """Build a worker hosting only classify_ticket/draft_reply for one agent."""
    acts = TicketActivities(agent)
    return Worker(
        client,
        task_queue=task_queue,
        activities=[acts.classify_ticket, acts.draft_reply],
    )


def make_run_workers(
    client: Client,
    *,
    workflow_eval_config: WorkflowEvalConfig,
    workflow_task_queue: str,
    primary_agent: Agent,
    fallback_agent: Agent | None = None,
    db_path: str | None = None,
) -> CombinedWorker:
    """Compose one run's workflow, primary-agent, and fallback-agent workers.

    Scoped to the run: hold the returned CombinedWorker open for every case in
    the run, and for every post-completion query, not per case.
    """
    workers = [
        make_workflow_worker(
            client, workflow_task_queue, primary_agent, db_path=db_path
        ),
        make_agent_worker(client, primary_agent, workflow_eval_config.agent_task_queue),
    ]
    if fallback_agent is not None:
        workers.append(
            make_agent_worker(
                client, fallback_agent, workflow_eval_config.fallback_task_queue
            )
        )
    return CombinedWorker(*workers)
