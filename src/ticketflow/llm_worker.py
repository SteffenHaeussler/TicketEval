"""LLM worker entrypoint: hosts primary and fallback agent activities."""

import asyncio
import logging

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from ticketflow import config
from ticketflow.activities import TicketActivities
from ticketflow.agent.factory import build_agent
from ticketflow.logging import setup_logging
from ticketflow.tracing import setup_tracing

logger = logging.getLogger(__name__)


def _build_activities() -> tuple[TicketActivities, TicketActivities]:
    """Construct primary and fallback activities from `TICKETFLOW_AGENT_BACKEND`.

    Runs before any Temporal client or worker is constructed, so an invalid backend
    fails fast instead of leaving a half-started worker.
    """
    primary_agent = build_agent("primary", config)
    fallback_agent = build_agent("fallback", config)
    return TicketActivities(primary_agent), TicketActivities(fallback_agent)


async def main() -> None:
    """Run primary and fallback LLM workers until interrupted."""
    setup_logging()
    primary_activities, fallback_activities = _build_activities()
    interceptor = setup_tracing(service_name="ticketflow-llm-worker")
    client = await Client.connect(
        config.TEMPORAL_ADDRESS,
        namespace=config.TEMPORAL_NAMESPACE,
        data_converter=pydantic_data_converter,
        interceptors=[interceptor] if interceptor else [],
    )

    primary_worker = Worker(
        client,
        task_queue=config.AGENT_TASK_QUEUE,
        activities=[
            primary_activities.classify_ticket,
            primary_activities.draft_reply,
        ],
        max_concurrent_activities=config.AGENT_MAX_CONCURRENT,
        max_task_queue_activities_per_second=config.AGENT_MAX_PER_SECOND,
    )
    fallback_worker = Worker(
        client,
        task_queue=config.FALLBACK_TASK_QUEUE,
        activities=[
            fallback_activities.classify_ticket,
            fallback_activities.draft_reply,
        ],
    )

    logger.info(
        "LLM workers running",
        extra={
            "primary_task_queue": config.AGENT_TASK_QUEUE,
            "fallback_task_queue": config.FALLBACK_TASK_QUEUE,
        },
    )
    await asyncio.gather(primary_worker.run(), fallback_worker.run())


if __name__ == "__main__":
    asyncio.run(main())
