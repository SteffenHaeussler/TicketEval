"""Canonical Ollama prompts, output schemas, and stable version hashes."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ticketflow.models import ProposedAction, TicketCategory


class ClassificationOutput(BaseModel):
    """Model-facing classification output without Ticketflow's role label."""

    model_config = ConfigDict(extra="forbid")

    category: TicketCategory
    confidence: float = Field(ge=0.0, le=1.0)


class DraftOutput(BaseModel):
    """Model-facing draft output without Ticketflow's role label."""

    model_config = ConfigDict(extra="forbid")

    reply_text: str = Field(min_length=1)
    action: ProposedAction
    confidence: float = Field(ge=0.0, le=1.0)


def _canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize a JSON mapping in the stable form used for schema hashes."""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text as a hexadecimal string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class OperationSpec:
    """Immutable prompt, schema, and output-model contract for one operation."""

    operation: Literal["classify", "draft"]
    output_model: type[BaseModel]
    prompt_template: str
    _schema_json: str

    def __init__(
        self,
        *,
        operation: Literal["classify", "draft"],
        output_model: type[BaseModel],
        prompt_template: str,
        schema: Mapping[str, Any],
    ) -> None:
        """Create a spec with an independently owned canonical JSON Schema."""
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "output_model", output_model)
        object.__setattr__(self, "prompt_template", prompt_template)
        object.__setattr__(self, "_schema_json", _canonical_json(schema))

    @property
    def schema(self) -> dict[str, Any]:
        """Return a copy of this operation's JSON Schema."""
        return json.loads(self._schema_json)

    @property
    def prompt_hash(self) -> str:
        """Return the stable version hash of the schema-free prompt template."""
        return _sha256(self.prompt_template)

    @property
    def schema_json(self) -> str:
        """Return the canonical JSON Schema representation."""
        return self._schema_json

    @property
    def schema_hash(self) -> str:
        """Return the stable version hash of the canonical JSON Schema."""
        return _sha256(self.schema_json)

    @property
    def system_prompt(self) -> str:
        """Render the grounding prompt with its canonical JSON Schema."""
        return self.prompt_template.format(schema=self.schema_json)


_CATEGORY_VALUES = ", ".join(category.value for category in TicketCategory)

CLASSIFICATION_SPEC = OperationSpec(
    operation="classify",
    output_model=ClassificationOutput,
    prompt_template=(
        "You are a support-ticket triage assistant. Classify the ticket into exactly "
        f"one category: {_CATEGORY_VALUES}. Report a calibrated confidence between 0 "
        "and 1. Respond with a single JSON object and nothing else, matching this "
        "schema:\n{schema}"
    ),
    schema=ClassificationOutput.model_json_schema(),
)

DRAFT_SPEC = OperationSpec(
    operation="draft",
    output_model=DraftOutput,
    prompt_template=(
        "You are a customer-support agent. Write a concise, helpful reply to the "
        "customer and propose an action. Use action type 'refund' only when a refund "
        "is clearly warranted; otherwise use 'reply_only'. When the action is "
        "'refund', 'refund_amount' MUST be a positive number estimated from the "
        "ticket and must never be null. When the action is 'reply_only', leave "
        "'refund_amount' null. Report a calibrated confidence between 0 and 1. "
        "Respond with a single JSON object and nothing else, matching this schema:\n"
        "{schema}"
    ),
    schema=DraftOutput.model_json_schema(),
)
