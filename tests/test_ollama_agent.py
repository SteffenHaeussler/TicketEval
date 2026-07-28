import json
from collections.abc import Callable
from typing import Any, Literal

import httpx
import pytest

from ticketflow.agent.base import AgentOverloadedError, AgentPermanentError
from ticketflow.agent.ollama import OllamaAgent
from ticketflow.agent.prompts import CLASSIFICATION_SPEC
from ticketflow.eval.cache import CachedAgentResponse, CacheRequest, ResponseCache
from ticketflow.eval.telemetry import RuntimeIdentityMap, TelemetrySink
from ticketflow.models import Classification, Ticket, TicketCategory

TICKET = Ticket(
    id="runtime-ticket-1",
    customer_email="jo@example.com",
    subject="Refund for double charge",
    body="I was charged twice for my subscription this month.",
)


class RecordingCache:
    """In-memory cache double that exposes the locked cache interface."""

    def __init__(self) -> None:
        self.entries: dict[str, CachedAgentResponse] = {}
        self.put_requests: list[CacheRequest] = []

    def get(self, request: CacheRequest) -> CachedAgentResponse | None:
        return self.entries.get(self._key(request))

    def put_success(self, request: CacheRequest, response: CachedAgentResponse) -> None:
        self.put_requests.append(request)
        self.entries[self._key(request)] = response

    @staticmethod
    def _key(request: CacheRequest) -> str:
        return json.dumps(request.model_dump(mode="json"), sort_keys=True)


assert isinstance(RecordingCache(), ResponseCache)


def _json_response(
    request: httpx.Request,
    content: dict[str, object],
    *,
    total_duration: int = 120_000_000,
    load_duration: int = 30_000_000,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"content": json.dumps(content)},
            "total_duration": total_duration,
            "load_duration": load_duration,
        },
        request=request,
    )


def _agent(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    ticket_id: str = TICKET.id,
    case_key: str = "case-001",
    cache: RecordingCache | None = None,
    role: Literal["primary", "fallback"] = "primary",
) -> tuple[OllamaAgent, TelemetrySink, RuntimeIdentityMap]:
    identity_map = RuntimeIdentityMap()
    identity_map.register(ticket_id, case_key)
    sink = TelemetrySink()
    agent = OllamaAgent(
        endpoint="http://ollama.test",
        model="test-model",
        timeout_s=2.0,
        seed=17,
        role=role,
        response_cache=cache,
        identity_map=identity_map,
        telemetry_sink=sink,
        model_digest="sha256:test",
        ollama_version="0.5.1",
        transport=httpx.MockTransport(handler),
    )
    return agent, sink, identity_map


async def test_explicit_zero_timeout_is_not_replaced_by_the_default():
    agent = OllamaAgent(
        endpoint="http://ollama.test",
        timeout_s=0,
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
    )
    try:
        assert agent._client.timeout.connect == 0
    finally:
        await agent.aclose()


async def test_classify_uses_the_canonical_ollama_contract_and_records_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload == {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": CLASSIFICATION_SPEC.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Customer email: jo@example.com\n"
                        "Subject: Refund for double charge\n\n"
                        "I was charged twice for my subscription this month."
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "format": CLASSIFICATION_SPEC.schema,
            "options": {"temperature": 0, "seed": 17},
        }
        return _json_response(request, {"category": "billing", "confidence": 0.9})

    agent, sink, _ = _agent(handler)
    try:
        result = await agent.classify(TICKET)
    finally:
        await agent.aclose()

    assert result.model == "primary"
    assert result.category == TicketCategory.BILLING
    [event] = sink.drain(TICKET.id)
    assert event.operation == "classify"
    assert event.attempt == 1
    assert event.outcome == "success"
    assert event.cache_hit is False
    assert event.model_total_duration_ms == 120.0
    assert event.model_load_duration_ms == 30.0


@pytest.mark.parametrize(
    ("make_response", "error_type"),
    [
        (
            lambda request: httpx.Response(400, request=request),
            AgentPermanentError,
        ),
        (
            lambda request: httpx.Response(404, request=request),
            AgentPermanentError,
        ),
        (
            lambda request: httpx.Response(408, request=request),
            AgentOverloadedError,
        ),
        (
            lambda request: httpx.Response(429, request=request),
            AgentOverloadedError,
        ),
        (
            lambda request: httpx.Response(503, request=request),
            AgentOverloadedError,
        ),
        (
            lambda request: (_ for _ in ()).throw(
                httpx.ConnectError("offline", request=request)
            ),
            AgentOverloadedError,
        ),
        (
            lambda request: (_ for _ in ()).throw(
                httpx.ReadTimeout("slow", request=request)
            ),
            AgentOverloadedError,
        ),
    ],
    ids=["400", "404", "408", "429", "5xx", "connection", "timeout"],
)
async def test_classify_maps_ollama_failures_and_records_them(
    make_response, error_type
):
    def handler(request: httpx.Request) -> httpx.Response:
        return make_response(request)

    agent, sink, _ = _agent(handler)
    try:
        with pytest.raises(error_type):
            await agent.classify(TICKET)
    finally:
        await agent.aclose()

    [event] = sink.drain(TICKET.id)
    assert event.outcome == (
        "transient_error" if error_type is AgentOverloadedError else "permanent_error"
    )
    assert event.error_type == error_type.__name__


async def test_invalid_refund_output_is_repaired_once_and_only_valid_output_is_cached():
    cache = RecordingCache()
    responses = iter(
        [
            {"reply_text": "Sorry.", "action": {"type": "refund"}, "confidence": 0.8},
            {
                "reply_text": "Sorry about that. I have issued a refund.",
                "action": {"type": "refund", "refund_amount": 10.0},
                "confidence": 0.8,
            },
        ]
    )
    request_payloads: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_payloads.append(json.loads(request.content))
        return _json_response(request, next(responses))

    agent, sink, _ = _agent(handler, cache=cache)
    classification = Classification(
        category=TicketCategory.BILLING, confidence=0.9, model="primary"
    )
    try:
        result = await agent.draft_reply(TICKET, classification)
    finally:
        await agent.aclose()

    assert result.action.refund_amount == 10.0
    assert len(request_payloads) == 2
    assert (
        "That response was invalid:" in request_payloads[1]["messages"][-1]["content"]
    )
    assert [event.outcome for event in sink.drain(TICKET.id)] == [
        "invalid_output",
        "success",
    ]
    assert len(cache.put_requests) == 1
    assert cache.put_requests[0].operation == "draft"
    assert cache.put_requests[0].classification_input == classification
    assert cache.put_requests[0].messages == tuple(request_payloads[0]["messages"])


async def test_second_invalid_response_exhausts_the_repair_budget_without_caching():
    cache = RecordingCache()

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            request, {"category": "not-a-category", "confidence": 0.8}
        )

    agent, sink, _ = _agent(handler, cache=cache)
    try:
        with pytest.raises(AgentPermanentError, match="invalid output twice"):
            await agent.classify(TICKET)
    finally:
        await agent.aclose()

    events = sink.drain(TICKET.id)
    assert [event.outcome for event in events] == ["invalid_output", "invalid_output"]
    assert [event.attempt for event in events] == [1, 2]
    assert cache.put_requests == []


async def test_second_policy_uses_cache_without_http_and_keeps_output_identical():
    cache = RecordingCache()
    first_calls = 0

    def first_handler(request: httpx.Request) -> httpx.Response:
        nonlocal first_calls
        first_calls += 1
        return _json_response(request, {"category": "billing", "confidence": 0.9})

    first_agent, first_sink, _ = _agent(first_handler, cache=cache)
    try:
        first = await first_agent.classify(TICKET)
    finally:
        await first_agent.aclose()

    def no_http(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the second reviewer policy must use the cache")

    second_ticket = TICKET.model_copy(update={"id": "runtime-ticket-2"})
    second_agent, second_sink, _ = _agent(
        no_http, ticket_id=second_ticket.id, cache=cache
    )
    try:
        second = await second_agent.classify(second_ticket)
    finally:
        await second_agent.aclose()

    assert first_calls == 1
    assert first.model_dump_json() == second.model_dump_json()
    [first_event] = first_sink.drain(TICKET.id)
    [second_event] = second_sink.drain(second_ticket.id)
    assert first_event.cache_hit is False
    assert second_event.cache_hit is True
    assert second_event.model_total_duration_ms == first_event.model_total_duration_ms


async def test_http_failures_never_cache_a_response():
    cache = RecordingCache()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    agent, _, _ = _agent(handler, cache=cache)
    try:
        with pytest.raises(AgentOverloadedError):
            await agent.classify(TICKET)
    finally:
        await agent.aclose()

    assert cache.put_requests == []
