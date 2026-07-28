"""Real ticket-resolution agent backed by Ollama's ``/api/chat`` endpoint."""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from ticketflow import config
from ticketflow.agent.base import AgentOverloadedError, AgentPermanentError
from ticketflow.agent.prompts import (
    CLASSIFICATION_SPEC,
    DRAFT_SPEC,
    ClassificationOutput,
    DraftOutput,
    OperationSpec,
)
from ticketflow.eval.cache import CachedAgentResponse, CacheRequest, ResponseCache
from ticketflow.eval.telemetry import RuntimeIdentityMap, TelemetrySink
from ticketflow.models import Classification, DraftReply, Ticket

_OVERLOADED_STATUS = {408, 429}
_THINK = False
_TEMPERATURE = 0


@dataclass(frozen=True)
class _ChatResponse:
    """The raw content and timing fields from one successful Ollama response."""

    content: str
    started_at: datetime
    wall_latency_ms: float
    model_total_duration_ms: float | None
    model_load_duration_ms: float | None


class OllamaAgent:
    """Lifecycle-managed Ollama agent with optional eval cache and telemetry."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        seed: int | None = None,
        *,
        role: Literal["primary", "fallback"] = "primary",
        response_cache: ResponseCache | None = None,
        identity_map: RuntimeIdentityMap | None = None,
        telemetry_sink: TelemetrySink | None = None,
        model_digest: str | None = None,
        ollama_version: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create an agent and, optionally, attach run-scoped eval collaborators.

        Cache identity needs a stable case key and model provenance. Production
        construction omits the cache entirely; evaluation construction supplies
        the identity map, digest, and Ollama version obtained by preflight.
        """
        if response_cache is not None and (
            identity_map is None or model_digest is None or ollama_version is None
        ):
            raise ValueError(
                "response_cache requires identity_map, model_digest, and ollama_version"
            )

        self._model = model or (
            config.PRIMARY_MODEL if role == "primary" else config.FALLBACK_MODEL
        )
        self._seed = config.OLLAMA_SEED if seed is None else seed
        self._role: Literal["primary", "fallback"] = role
        self._response_cache = response_cache
        self._identity_map = identity_map
        self._telemetry_sink = telemetry_sink
        self._model_digest = model_digest
        self._ollama_version = ollama_version
        self._client = httpx.AsyncClient(
            base_url=endpoint or config.OLLAMA_ENDPOINT,
            timeout=(config.OLLAMA_TIMEOUT_S if timeout_s is None else timeout_s),
            transport=transport,
        )

    async def __aenter__(self) -> "OllamaAgent":
        """Enter an async context managing the underlying HTTP client."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the shared client when the context exits."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def classify(self, ticket: Ticket) -> Classification:
        """Classify a ticket and assign this agent's application-owned role."""
        messages = [
            {"role": "system", "content": CLASSIFICATION_SPEC.system_prompt},
            {"role": "user", "content": self._ticket_content(ticket)},
        ]
        output = await self._ask_validated(
            ticket=ticket,
            spec=CLASSIFICATION_SPEC,
            messages=messages,
            classification_input=None,
        )
        assert isinstance(output, ClassificationOutput)
        return Classification(
            category=output.category,
            confidence=output.confidence,
            model=self._role,
        )

    async def draft_reply(
        self, ticket: Ticket, classification: Classification
    ) -> DraftReply:
        """Draft a response and assign this agent's application-owned role."""
        messages = [
            {"role": "system", "content": DRAFT_SPEC.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Category: {classification.category.value}\n"
                    f"{self._ticket_content(ticket)}"
                ),
            },
        ]
        output = await self._ask_validated(
            ticket=ticket,
            spec=DRAFT_SPEC,
            messages=messages,
            classification_input=classification,
        )
        assert isinstance(output, DraftOutput)
        return DraftReply(
            reply_text=output.reply_text,
            action=output.action,
            confidence=output.confidence,
            model=self._role,
        )

    @staticmethod
    def _ticket_content(ticket: Ticket) -> str:
        """Render stable ticket inputs without including the runtime ticket ID."""
        return (
            f"Customer email: {ticket.customer_email}\n"
            f"Subject: {ticket.subject}\n\n{ticket.body}"
        )

    async def _ask_validated(
        self,
        *,
        ticket: Ticket,
        spec: OperationSpec,
        messages: list[dict[str, str]],
        classification_input: Classification | None,
    ) -> BaseModel:
        """Use a cached result or make at most two validation-aware HTTP calls."""
        cache_request = self._cache_request(
            ticket=ticket,
            spec=spec,
            messages=messages,
            classification_input=classification_input,
        )
        if cache_request is not None:
            assert self._response_cache is not None
            cached = self._response_cache.get(cache_request)
            if cached is not None:
                try:
                    output = spec.output_model.model_validate(cached.output)
                except ValidationError as exc:
                    self._record(
                        ticket_id=ticket.id,
                        operation=spec.operation,
                        started_at=datetime.now(timezone.utc),
                        wall_latency_ms=0.0,
                        model_total_duration_ms=None,
                        model_load_duration_ms=None,
                        outcome="permanent_error",
                        error_type=AgentPermanentError.__name__,
                    )
                    raise AgentPermanentError(
                        f"cached output did not match {spec.operation} schema: {exc}"
                    ) from exc
                self._record(
                    ticket_id=ticket.id,
                    operation=spec.operation,
                    started_at=datetime.now(timezone.utc),
                    wall_latency_ms=0.0,
                    model_total_duration_ms=cached.model_total_duration_ms,
                    model_load_duration_ms=cached.model_load_duration_ms,
                    outcome="success",
                    error_type=None,
                    cache_hit=True,
                )
                return output

        first = await self._chat(ticket.id, spec.operation, messages, spec.schema)
        try:
            output = spec.output_model.model_validate_json(first.content)
        except (ValidationError, ValueError) as first_error:
            self._record_invalid(ticket.id, spec.operation, first, first_error)
            repair_messages = [
                *messages,
                {"role": "assistant", "content": first.content},
                {
                    "role": "user",
                    "content": (
                        "That response was invalid: "
                        f"{first_error}. Respond again with a single JSON object "
                        "matching the schema exactly."
                    ),
                },
            ]
            second = await self._chat(
                ticket.id, spec.operation, repair_messages, spec.schema
            )
            try:
                output = spec.output_model.model_validate_json(second.content)
            except (ValidationError, ValueError) as second_error:
                self._record_invalid(ticket.id, spec.operation, second, second_error)
                raise AgentPermanentError(
                    f"model returned invalid output twice: {second_error}"
                ) from second_error
            response = second
        else:
            response = first

        self._record_success(ticket.id, spec.operation, response)
        if cache_request is not None:
            assert self._response_cache is not None
            self._response_cache.put_success(
                cache_request,
                CachedAgentResponse(
                    output=output.model_dump(mode="json"),
                    model_total_duration_ms=response.model_total_duration_ms,
                    model_load_duration_ms=response.model_load_duration_ms,
                ),
            )
        return output

    def _cache_request(
        self,
        *,
        ticket: Ticket,
        spec: OperationSpec,
        messages: list[dict[str, str]],
        classification_input: Classification | None,
    ) -> CacheRequest | None:
        """Build the exact cache key, or skip cache integration when disabled."""
        if self._response_cache is None:
            return None
        assert self._identity_map is not None
        assert self._model_digest is not None
        assert self._ollama_version is not None
        return CacheRequest(
            operation=spec.operation,
            model_name=self._model,
            model_digest=self._model_digest,
            role=self._role,
            case_key=self._identity_map.resolve(ticket.id),
            customer_email=ticket.customer_email,
            subject=ticket.subject,
            body=ticket.body,
            classification_input=classification_input,
            messages=tuple(messages),
            prompt_version=spec.prompt_hash,
            json_schema=spec.schema,
            think=_THINK,
            temperature=_TEMPERATURE,
            seed=self._seed,
            generation_options={},
            ollama_version=self._ollama_version,
        )

    async def _chat(
        self,
        ticket_id: str,
        operation: Literal["classify", "draft"],
        messages: list[dict[str, str]],
        schema: dict[str, Any],
    ) -> _ChatResponse:
        """Call Ollama once, recording mapped transport and envelope failures."""
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": _THINK,
            "format": schema,
            "options": {"temperature": _TEMPERATURE, "seed": self._seed},
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.TransportError as exc:
            error = AgentOverloadedError(f"ollama request failed: {exc}")
            self._record_error(ticket_id, operation, started_at, started, error)
            raise error from exc

        if response.status_code in _OVERLOADED_STATUS or response.status_code >= 500:
            error = AgentOverloadedError(f"ollama returned HTTP {response.status_code}")
            self._record_error(ticket_id, operation, started_at, started, error)
            raise error
        if response.status_code >= 400:
            error = AgentPermanentError(f"ollama returned HTTP {response.status_code}")
            self._record_error(ticket_id, operation, started_at, started, error)
            raise error

        try:
            body = response.json()
            content = body["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message.content must be a string")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            error = AgentPermanentError(
                f"ollama response envelope was malformed: {exc}"
            )
            self._record_error(ticket_id, operation, started_at, started, error)
            raise error from exc

        return _ChatResponse(
            content=content,
            started_at=started_at,
            wall_latency_ms=(time.perf_counter() - started) * 1000,
            model_total_duration_ms=self._duration_ms(body.get("total_duration")),
            model_load_duration_ms=self._duration_ms(body.get("load_duration")),
        )

    @staticmethod
    def _duration_ms(value: object) -> float | None:
        """Convert Ollama's nanosecond timing field to milliseconds when present."""
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value / 1_000_000
        return None

    def _record_invalid(
        self,
        ticket_id: str,
        operation: Literal["classify", "draft"],
        response: _ChatResponse,
        error: Exception,
    ) -> None:
        """Record a schema-invalid output without turning it into an outage."""
        self._record(
            ticket_id=ticket_id,
            operation=operation,
            started_at=response.started_at,
            wall_latency_ms=response.wall_latency_ms,
            model_total_duration_ms=response.model_total_duration_ms,
            model_load_duration_ms=response.model_load_duration_ms,
            outcome="invalid_output",
            error_type=type(error).__name__,
        )

    def _record_success(
        self,
        ticket_id: str,
        operation: Literal["classify", "draft"],
        response: _ChatResponse,
    ) -> None:
        """Record one successful non-cached model call."""
        self._record(
            ticket_id=ticket_id,
            operation=operation,
            started_at=response.started_at,
            wall_latency_ms=response.wall_latency_ms,
            model_total_duration_ms=response.model_total_duration_ms,
            model_load_duration_ms=response.model_load_duration_ms,
            outcome="success",
            error_type=None,
        )

    def _record_error(
        self,
        ticket_id: str,
        operation: Literal["classify", "draft"],
        started_at: datetime,
        started: float,
        error: AgentOverloadedError | AgentPermanentError,
    ) -> None:
        """Record a mapped backend failure before propagating it to the workflow."""
        self._record(
            ticket_id=ticket_id,
            operation=operation,
            started_at=started_at,
            wall_latency_ms=(time.perf_counter() - started) * 1000,
            model_total_duration_ms=None,
            model_load_duration_ms=None,
            outcome=(
                "transient_error"
                if isinstance(error, AgentOverloadedError)
                else "permanent_error"
            ),
            error_type=type(error).__name__,
        )

    def _record(
        self,
        *,
        ticket_id: str,
        operation: Literal["classify", "draft"],
        started_at: datetime,
        wall_latency_ms: float,
        model_total_duration_ms: float | None,
        model_load_duration_ms: float | None,
        outcome: Literal[
            "success", "invalid_output", "transient_error", "permanent_error"
        ],
        error_type: str | None,
        cache_hit: bool = False,
    ) -> None:
        """Emit a raw attempt only when an eval telemetry sink was injected."""
        if self._telemetry_sink is None:
            return
        self._telemetry_sink.record(
            ticket_id=ticket_id,
            operation=operation,
            role=self._role,
            cache_hit=cache_hit,
            started_at=started_at,
            wall_latency_ms=wall_latency_ms,
            model_total_duration_ms=model_total_duration_ms,
            model_load_duration_ms=model_load_duration_ms,
            outcome=outcome,
            error_type=error_type,
        )
