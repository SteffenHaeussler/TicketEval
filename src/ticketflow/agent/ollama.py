"""Real ticket-resolution agent backed by a local Ollama server.

Minimal standalone slice: implements the :class:`~ticketflow.agent.base.Agent`
protocol over Ollama's ``/api/chat`` endpoint. The full reliability machinery
(response cache, preflight, manifest enrichment, first-class ``invalid_output``
telemetry) is deferred to Milestone 3; this module only proves the LLM path
resolves a ticket end-to-end.
"""

import json

import httpx
from pydantic import BaseModel, Field, ValidationError

from ticketflow import config
from ticketflow.agent.base import AgentOverloadedError, AgentPermanentError
from ticketflow.models import (
    Classification,
    DraftReply,
    ProposedAction,
    Ticket,
    TicketCategory,
)

# Status codes that indicate a transient, retryable backend condition.
_OVERLOADED_STATUS = {408, 429}


class ClassificationOutput(BaseModel):
    """LLM-facing classification schema; omits the internal ``model`` field."""

    category: TicketCategory
    confidence: float = Field(ge=0.0, le=1.0)


class DraftOutput(BaseModel):
    """LLM-facing draft schema; omits the internal ``model`` field."""

    reply_text: str = Field(min_length=1)
    action: ProposedAction
    confidence: float = Field(ge=0.0, le=1.0)


_CATEGORY_VALUES = ", ".join(c.value for c in TicketCategory)

_CLASSIFY_SYSTEM = (
    "You are a support-ticket triage assistant. Classify the ticket into exactly "
    f"one category: {_CATEGORY_VALUES}. Report a calibrated confidence between 0 "
    "and 1. Respond with a single JSON object and nothing else, matching this "
    "schema:\n{schema}"
)

_DRAFT_SYSTEM = (
    "You are a customer-support agent. Write a concise, helpful reply to the "
    "customer and propose an action. Use action type 'refund' only when a refund "
    "is clearly warranted; otherwise use 'reply_only'. When the action is "
    "'refund', 'refund_amount' MUST be a positive number estimated from the "
    "ticket and must never be null. When the action is 'reply_only', leave "
    "'refund_amount' null. Report a calibrated confidence between 0 and 1. "
    "Respond with a single JSON object and nothing else, matching this "
    "schema:\n{schema}"
)


class OllamaAgent:
    """Agent that resolves tickets by calling a local Ollama chat model."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        seed: int | None = None,
    ):
        """Create an agent bound to an Ollama endpoint and chat model."""
        self._model = model or config.OLLAMA_MODEL
        self._seed = config.OLLAMA_SEED if seed is None else seed
        self._client = httpx.AsyncClient(
            base_url=endpoint or config.OLLAMA_ENDPOINT,
            timeout=timeout_s or config.OLLAMA_TIMEOUT_S,
        )

    async def __aenter__(self) -> "OllamaAgent":
        """Enter an async context managing the underlying HTTP client."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the underlying HTTP client on context exit."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def classify(self, ticket: Ticket) -> Classification:
        """Classify a ticket via the chat model into a support category."""
        schema = ClassificationOutput.model_json_schema()
        messages = [
            {"role": "system", "content": _CLASSIFY_SYSTEM.format(schema=schema)},
            {
                "role": "user",
                "content": f"Subject: {ticket.subject}\n\n{ticket.body}",
            },
        ]
        output = await self._ask_validated(messages, schema, ClassificationOutput)
        return Classification(
            category=output.category,
            confidence=output.confidence,
            model="primary",
        )

    async def draft_reply(
        self, ticket: Ticket, classification: Classification
    ) -> DraftReply:
        """Draft a customer reply and proposed action via the chat model."""
        schema = DraftOutput.model_json_schema()
        messages = [
            {"role": "system", "content": _DRAFT_SYSTEM.format(schema=schema)},
            {
                "role": "user",
                "content": (
                    f"Category: {classification.category.value}\n"
                    f"Subject: {ticket.subject}\n\n{ticket.body}"
                ),
            },
        ]
        output = await self._ask_validated(messages, schema, DraftOutput)
        return DraftReply(
            reply_text=output.reply_text,
            action=output.action,
            confidence=output.confidence,
            model="primary",
        )

    async def _ask_validated[T: BaseModel](
        self,
        messages: list[dict[str, str]],
        schema: dict,
        output_model: type[T],
    ) -> T:
        """Chat once, then re-ask once with the validation error on failure."""
        content = await self._chat(messages, schema)
        try:
            return output_model.model_validate_json(content)
        except (ValidationError, ValueError) as first_error:
            repair = list(messages)
            repair.append({"role": "assistant", "content": content})
            repair.append(
                {
                    "role": "user",
                    "content": (
                        "That response was invalid: "
                        f"{first_error}. Respond again with a single JSON object "
                        "matching the schema exactly."
                    ),
                }
            )
            retry_content = await self._chat(repair, schema)
            try:
                return output_model.model_validate_json(retry_content)
            except (ValidationError, ValueError) as second_error:
                raise AgentPermanentError(
                    f"model returned invalid output twice: {second_error}"
                ) from second_error

    async def _chat(self, messages: list[dict[str, str]], schema: dict) -> str:
        """Send one chat request and return the raw message content."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "seed": self._seed},
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.TransportError as exc:
            raise AgentOverloadedError(f"ollama request failed: {exc}") from exc

        status = response.status_code
        if status in _OVERLOADED_STATUS or status >= 500:
            raise AgentOverloadedError(f"ollama returned HTTP {status}")
        if status >= 400:
            raise AgentPermanentError(f"ollama returned HTTP {status}")

        try:
            body = response.json()
            return body["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AgentPermanentError(
                f"ollama response envelope was malformed: {exc}"
            ) from exc
