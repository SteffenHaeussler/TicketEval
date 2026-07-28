import asyncio
import sqlite3

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from tests.helpers import (
    ScriptedAgent,
    billing_classification,
    make_ticket,
    refund_draft,
)
from ticketflow.activities import TicketActivities
from ticketflow.agent.base import AgentOverloadedError, AgentPermanentError


class SlowAgent:
    """Agent whose calls sleep briefly before returning, so heartbeat tests
    can observe periodic heartbeats without a real 30-second call."""

    def __init__(self, classification, draft, delay_seconds: float):
        self.classification = classification
        self.draft = draft
        self.delay_seconds = delay_seconds

    async def classify(self, ticket):
        await asyncio.sleep(self.delay_seconds)
        return self.classification

    async def draft_reply(self, ticket, classification):
        await asyncio.sleep(self.delay_seconds)
        return self.draft


class ImmediateErrorAgent:
    """Agent whose calls sleep briefly then raise `error`."""

    def __init__(self, error: Exception, delay_seconds: float = 0.03):
        self.error = error
        self.delay_seconds = delay_seconds

    async def classify(self, ticket):
        await asyncio.sleep(self.delay_seconds)
        raise self.error

    async def draft_reply(self, ticket, classification):
        await asyncio.sleep(self.delay_seconds)
        raise self.error


async def test_classify_ticket_delegates_to_agent():
    agent = ScriptedAgent(billing_classification(), refund_draft())
    acts = TicketActivities(agent)
    result = await ActivityEnvironment().run(acts.classify_ticket, make_ticket())
    assert result == agent.classification
    assert agent.classify_calls == 1


async def test_draft_reply_delegates_to_agent():
    agent = ScriptedAgent(billing_classification(), refund_draft())
    acts = TicketActivities(agent)
    result = await ActivityEnvironment().run(
        acts.draft_reply, make_ticket(), agent.classification
    )
    assert result == agent.draft
    assert agent.draft_calls == 1


async def test_side_effect_activities_complete():
    agent = ScriptedAgent(billing_classification(), refund_draft())
    acts = TicketActivities(agent)
    env = ActivityEnvironment()
    await env.run(acts.send_reply, make_ticket(), "hello")
    await env.run(acts.execute_refund, "t1", 42.0)


async def test_execute_refund_duplicate_run_refunds_once(tmp_path):
    agent = ScriptedAgent(billing_classification(), refund_draft())
    db = str(tmp_path / "read.db")
    acts = TicketActivities(agent, db_path=db)
    env = ActivityEnvironment()
    await env.run(acts.execute_refund, "t1", 42.0)
    await env.run(acts.execute_refund, "t1", 42.0)
    conn = sqlite3.connect(db)
    try:
        attempts = conn.execute(
            "SELECT COUNT(*) FROM refund_attempts WHERE ticket_id = 't1'"
        ).fetchone()[0]
        refunds = conn.execute(
            "SELECT COUNT(*) FROM refunds WHERE ticket_id = 't1'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert attempts == 2
    assert refunds == 1


async def test_classify_ticket_heartbeats_periodically_during_slow_call():
    agent = SlowAgent(billing_classification(), refund_draft(), delay_seconds=0.05)
    acts = TicketActivities(agent, heartbeat_interval_seconds=0.01)
    env = ActivityEnvironment()
    heartbeats: list[str] = []
    env.on_heartbeat = lambda detail: heartbeats.append(detail)
    result = await env.run(acts.classify_ticket, make_ticket())
    assert result == agent.classification
    assert heartbeats[0] == "classifying ticket"
    assert heartbeats[-1] == "classified ticket"
    assert heartbeats.count("classifying ticket") >= 3


async def test_draft_reply_heartbeats_periodically_during_slow_call():
    agent = SlowAgent(billing_classification(), refund_draft(), delay_seconds=0.05)
    acts = TicketActivities(agent, heartbeat_interval_seconds=0.01)
    env = ActivityEnvironment()
    heartbeats: list[str] = []
    env.on_heartbeat = lambda detail: heartbeats.append(detail)
    result = await env.run(acts.draft_reply, make_ticket(), agent.classification)
    assert result == agent.draft
    assert heartbeats.count("drafting reply") >= 3


async def test_classify_ticket_cancels_heartbeat_on_permanent_error():
    agent = ImmediateErrorAgent(AgentPermanentError("nope"), delay_seconds=0.03)
    acts = TicketActivities(agent, heartbeat_interval_seconds=0.01)
    env = ActivityEnvironment()
    heartbeats: list[str] = []
    env.on_heartbeat = lambda detail: heartbeats.append(detail)
    before = asyncio.all_tasks()
    with pytest.raises(ApplicationError) as exc_info:
        await env.run(acts.classify_ticket, make_ticket())
    assert exc_info.value.type == "AgentPermanentError"
    assert heartbeats
    assert asyncio.all_tasks() == before


async def test_classify_ticket_cancels_heartbeat_on_overloaded_error():
    agent = ImmediateErrorAgent(AgentOverloadedError("busy"), delay_seconds=0.03)
    acts = TicketActivities(agent, heartbeat_interval_seconds=0.01)
    env = ActivityEnvironment()
    heartbeats: list[str] = []
    env.on_heartbeat = lambda detail: heartbeats.append(detail)
    before = asyncio.all_tasks()
    with pytest.raises(AgentOverloadedError):
        await env.run(acts.classify_ticket, make_ticket())
    assert heartbeats
    assert asyncio.all_tasks() == before
