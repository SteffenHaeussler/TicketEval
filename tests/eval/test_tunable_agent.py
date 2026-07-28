import asyncio

import pytest
from temporalio.testing import ActivityEnvironment

from ticketflow.activities import TicketActivities
from ticketflow.agent.base import AgentOverloadedError
from ticketflow.agent.tunable import TunableAgentProfile, TunableMockAgent
from ticketflow.eval.dataset import ExpectedOutcome
from ticketflow.eval.telemetry import RuntimeIdentityMap, TelemetrySink
from ticketflow.models import ActionType, Ticket, TicketCategory

CATEGORIES = list(TicketCategory)


def make_expected(**overrides):
    base = dict(
        acceptable_categories=frozenset({TicketCategory.BILLING}),
        reference_category=TicketCategory.BILLING,
        acceptable_actions=frozenset({ActionType.REPLY_ONLY}),
        expected_refund_amount=None,
        refund_tolerance=0.01,
    )
    base.update(overrides)
    return ExpectedOutcome.model_validate(base)


def build_cases(n: int) -> dict[str, ExpectedOutcome]:
    cases = {}
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)]
        if i % 3 == 0:
            cases[f"case-{i}"] = make_expected(
                acceptable_categories=frozenset({category}),
                reference_category=category,
                acceptable_actions=frozenset({ActionType.REFUND}),
                expected_refund_amount=10.0 + i,
            )
        else:
            cases[f"case-{i}"] = make_expected(
                acceptable_categories=frozenset({category}),
                reference_category=category,
                acceptable_actions=frozenset({ActionType.REPLY_ONLY}),
            )
    return cases


def build_agent(
    expected_outcomes: dict[str, ExpectedOutcome],
    profile: TunableAgentProfile,
    generation_seed: int = 0,
) -> tuple[TunableMockAgent, RuntimeIdentityMap, TelemetrySink]:
    identity_map = RuntimeIdentityMap()
    sink = TelemetrySink()
    agent = TunableMockAgent(
        identity_map=identity_map,
        telemetry_sink=sink,
        expected_outcomes=expected_outcomes,
        profile=profile,
        generation_seed=generation_seed,
    )
    return agent, identity_map, sink


def make_ticket(ticket_id: str) -> Ticket:
    return Ticket(
        id=ticket_id,
        customer_email="jo@example.com",
        subject="Help",
        body="Something broke",
    )


async def run_case(
    agent: TunableMockAgent, identity_map: RuntimeIdentityMap, case_key: str
):
    ticket_id = f"ticket-{case_key}"
    identity_map.register(ticket_id, case_key)
    ticket = make_ticket(ticket_id)
    classification = await agent.classify(ticket)
    draft = await agent.draft_reply(ticket, classification)
    return classification, draft


class TestDeterminismAcrossConcurrencyAndOrdering:
    async def test_same_seed_and_case_produce_same_output_regardless_of_order(self):
        cases = build_cases(20)
        profile = TunableAgentProfile(
            category_error_rate=0.3, action_error_rate=0.3, refund_amount_error_rate=0.3
        )
        case_keys = list(cases.keys())

        async def run_all(order: list[str], *, concurrent: bool):
            agent, identity_map, _ = build_agent(cases, profile, generation_seed=7)

            async def process(case_key: str):
                if concurrent:
                    await asyncio.sleep(0)
                classification, draft = await run_case(agent, identity_map, case_key)
                return case_key, (classification.model_dump(), draft.model_dump())

            if concurrent:
                pairs = await asyncio.gather(*(process(k) for k in order))
            else:
                pairs = [await process(k) for k in order]
            return dict(pairs)

        forward = await run_all(case_keys, concurrent=False)
        backward = await run_all(list(reversed(case_keys)), concurrent=False)
        concurrency_8 = await run_all(case_keys, concurrent=True)

        assert forward == backward == concurrency_8


class TestOracleVersusRubberStamp:
    async def test_different_ticket_ids_same_case_key_are_byte_identical(self):
        cases = build_cases(5)
        profile = TunableAgentProfile(category_error_rate=0.5, action_error_rate=0.5)
        agent, identity_map, _ = build_agent(cases, profile, generation_seed=3)

        identity_map.register("oracle-ticket", "case-1")
        identity_map.register("rubber-stamp-ticket", "case-1")

        oracle_ticket = make_ticket("oracle-ticket")
        rubber_ticket = make_ticket("rubber-stamp-ticket")

        oracle_classification = await agent.classify(oracle_ticket)
        rubber_classification = await agent.classify(rubber_ticket)
        assert oracle_classification.model_dump() == rubber_classification.model_dump()

        oracle_draft = await agent.draft_reply(oracle_ticket, oracle_classification)
        rubber_draft = await agent.draft_reply(rubber_ticket, rubber_classification)
        assert oracle_draft.model_dump() == rubber_draft.model_dump()


class TestExactErrorIdSets:
    async def test_category_error_case_keys_are_exactly_wrong(self):
        cases = build_cases(10)
        error_keys = frozenset({"case-2", "case-5"})
        profile = TunableAgentProfile(category_error_case_keys=error_keys)
        agent, identity_map, _ = build_agent(cases, profile)

        for case_key, expected in cases.items():
            classification, _ = await run_case(agent, identity_map, case_key)
            if case_key in error_keys:
                assert classification.category not in expected.acceptable_categories
            else:
                assert classification.category == expected.reference_category

    async def test_action_error_case_keys_are_exactly_wrong(self):
        cases = build_cases(10)
        error_keys = frozenset({"case-1", "case-4"})
        profile = TunableAgentProfile(action_error_case_keys=error_keys)
        agent, identity_map, _ = build_agent(cases, profile)

        for case_key, expected in cases.items():
            baseline = (
                ActionType.REFUND
                if (
                    ActionType.REFUND in expected.acceptable_actions
                    and expected.expected_refund_amount is not None
                )
                else ActionType.REPLY_ONLY
            )
            _, draft = await run_case(agent, identity_map, case_key)
            if case_key in error_keys:
                assert draft.action.type != baseline
            else:
                assert draft.action.type == baseline

    async def test_refund_amount_error_case_keys_are_exactly_wrong(self):
        cases = {
            f"refund-case-{i}": make_expected(
                acceptable_categories=frozenset({TicketCategory.BILLING}),
                reference_category=TicketCategory.BILLING,
                acceptable_actions=frozenset({ActionType.REFUND}),
                expected_refund_amount=50.0,
                refund_tolerance=0.01,
            )
            for i in range(6)
        }
        error_keys = frozenset({"refund-case-1", "refund-case-3"})
        profile = TunableAgentProfile(refund_amount_error_case_keys=error_keys)
        agent, identity_map, _ = build_agent(cases, profile)

        for case_key, expected in cases.items():
            _, draft = await run_case(agent, identity_map, case_key)
            assert draft.action.type == ActionType.REFUND
            assert draft.action.refund_amount is not None
            assert expected.expected_refund_amount is not None
            diff = abs(draft.action.refund_amount - expected.expected_refund_amount)
            if case_key in error_keys:
                assert diff > expected.refund_tolerance
            else:
                assert diff <= expected.refund_tolerance

    async def test_transient_failure_case_keys_fail_exactly_and_are_recorded(self):
        cases = build_cases(6)
        error_keys = frozenset({"case-3"})
        profile = TunableAgentProfile(transient_failure_case_keys=error_keys)
        agent, identity_map, sink = build_agent(cases, profile)

        for case_key in cases:
            ticket_id = f"ticket-{case_key}"
            identity_map.register(ticket_id, case_key)
            ticket = make_ticket(ticket_id)
            if case_key in error_keys:
                with pytest.raises(AgentOverloadedError):
                    await agent.classify(ticket)
                attempts = sink.drain(ticket_id)
                assert len(attempts) == 1
                assert attempts[0].outcome == "transient_error"
                assert attempts[0].error_type == "AgentOverloadedError"
            else:
                classification = await agent.classify(ticket)
                assert classification is not None


class TestConfidenceCalibrationAndOverconfidence:
    async def test_correct_and_incorrect_draw_from_their_configured_ranges(self):
        cases = build_cases(4)
        profile = TunableAgentProfile(
            category_error_case_keys=frozenset({"case-1"}),
            confidence_correct_range=(0.9, 0.91),
            confidence_incorrect_range=(0.1, 0.11),
        )
        agent, identity_map, _ = build_agent(cases, profile)

        correct_classification, _ = await run_case(agent, identity_map, "case-0")
        assert 0.9 <= correct_classification.confidence <= 0.91

        incorrect_classification, _ = await run_case(agent, identity_map, "case-1")
        assert 0.1 <= incorrect_classification.confidence <= 0.11

    async def test_overconfidence_draws_from_the_correct_range_despite_being_wrong(
        self,
    ):
        cases = build_cases(4)
        profile = TunableAgentProfile(
            category_error_case_keys=frozenset({"case-1"}),
            overconfidence_case_keys=frozenset({"case-1"}),
            confidence_correct_range=(0.9, 0.91),
            confidence_incorrect_range=(0.1, 0.11),
        )
        agent, identity_map, _ = build_agent(cases, profile)

        classification, _ = await run_case(agent, identity_map, "case-1")
        assert classification.category not in cases["case-1"].acceptable_categories
        assert 0.9 <= classification.confidence <= 0.91


class TestRoleReachesOutput:
    async def test_fallback_role_reaches_classification_and_draft_model(self):
        cases = build_cases(1)
        profile = TunableAgentProfile(role="fallback")
        agent, identity_map, _ = build_agent(cases, profile)

        classification, draft = await run_case(agent, identity_map, "case-0")
        assert classification.model == "fallback"
        assert draft.model == "fallback"


class TestAgentProtocolConformance:
    async def test_works_through_ticket_activities_without_a_temporal_server(self):
        cases = build_cases(1)
        profile = TunableAgentProfile()
        agent, identity_map, _ = build_agent(cases, profile)
        identity_map.register("ticket-x", "case-0")
        ticket = make_ticket("ticket-x")

        activities = TicketActivities(agent)
        env = ActivityEnvironment()
        classification = await env.run(activities.classify_ticket, ticket)
        draft = await env.run(activities.draft_reply, ticket, classification)

        assert classification.category == cases["case-0"].reference_category
        assert draft.reply_text
