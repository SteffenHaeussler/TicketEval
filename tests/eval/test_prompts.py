import json

import pytest
from pydantic import ValidationError

from ticketflow.agent.prompts import (
    CLASSIFICATION_SPEC,
    DRAFT_SPEC,
    ClassificationOutput,
    DraftOutput,
    OperationSpec,
)


def test_schema_hash_is_invariant_to_mapping_key_order():
    first = OperationSpec(
        operation="classify",
        output_model=ClassificationOutput,
        prompt_template="Classify this ticket.\n{schema}",
        schema={"type": "object", "properties": {"category": {"type": "string"}}},
    )
    reordered = OperationSpec(
        operation="classify",
        output_model=ClassificationOutput,
        prompt_template="Classify this ticket.\n{schema}",
        schema={"properties": {"category": {"type": "string"}}, "type": "object"},
    )

    assert first.schema_hash == reordered.schema_hash
    assert first.schema_json == reordered.schema_json


def test_schema_property_returns_a_copy_that_cannot_mutate_its_spec():
    schema = CLASSIFICATION_SPEC.schema
    schema["type"] = "changed"

    assert CLASSIFICATION_SPEC.schema["type"] == "object"


def test_prompt_or_schema_change_updates_its_own_version_hash():
    base = OperationSpec(
        operation="classify",
        output_model=ClassificationOutput,
        prompt_template="Classify this ticket.\n{schema}",
        schema={"type": "object"},
    )
    changed_prompt = OperationSpec(
        operation="classify",
        output_model=ClassificationOutput,
        prompt_template="Classify this support ticket.\n{schema}",
        schema={"type": "object"},
    )
    changed_schema = OperationSpec(
        operation="classify",
        output_model=ClassificationOutput,
        prompt_template="Classify this ticket.\n{schema}",
        schema={"type": "array"},
    )

    assert changed_prompt.prompt_hash != base.prompt_hash
    assert changed_prompt.schema_hash == base.schema_hash
    assert changed_schema.prompt_hash == base.prompt_hash
    assert changed_schema.schema_hash != base.schema_hash


@pytest.mark.parametrize("spec", [CLASSIFICATION_SPEC, DRAFT_SPEC])
def test_system_prompt_contains_its_canonical_json_schema(spec: OperationSpec):
    assert spec.schema_json in spec.system_prompt
    assert json.loads(spec.schema_json) == spec.schema


def test_classification_and_draft_outputs_reject_each_others_valid_payloads():
    classification_payload = {"category": "billing", "confidence": 0.8}
    draft_payload = {
        "reply_text": "I can help with that.",
        "action": {"type": "reply_only", "refund_amount": None},
        "confidence": 0.8,
    }

    assert (
        ClassificationOutput.model_validate(classification_payload).category.value
        == "billing"
    )
    assert (
        DraftOutput.model_validate(draft_payload).reply_text == "I can help with that."
    )
    with pytest.raises(ValidationError):
        ClassificationOutput.model_validate(draft_payload)
    with pytest.raises(ValidationError):
        DraftOutput.model_validate(classification_payload)


@pytest.mark.parametrize(
    ("output_model", "payload"),
    [
        (ClassificationOutput, {"category": "unknown", "confidence": 0.8}),
        (ClassificationOutput, {"category": "billing", "confidence": -0.1}),
        (ClassificationOutput, {"category": "billing", "confidence": 1.1}),
        (
            DraftOutput,
            {
                "reply_text": "",
                "action": {"type": "reply_only", "refund_amount": None},
                "confidence": 0.8,
            },
        ),
        (
            DraftOutput,
            {
                "reply_text": "I can help with that.",
                "action": {"type": "refund"},
                "confidence": 0.8,
            },
        ),
        (
            DraftOutput,
            {
                "reply_text": "I can help with that.",
                "action": {"type": "refund", "refund_amount": 0},
                "confidence": 0.8,
            },
        ),
    ],
)
def test_output_models_reject_invalid_structured_values(output_model, payload):
    with pytest.raises(ValidationError):
        output_model.model_validate(payload)


@pytest.mark.parametrize("output_model", [ClassificationOutput, DraftOutput])
def test_output_models_do_not_declare_or_accept_internal_model_field(output_model):
    assert "model" not in output_model.model_fields

    if output_model is ClassificationOutput:
        payload = {"category": "billing", "confidence": 0.8, "model": "primary"}
    else:
        payload = {
            "reply_text": "I can help with that.",
            "action": {"type": "reply_only", "refund_amount": None},
            "confidence": 0.8,
            "model": "primary",
        }

    with pytest.raises(ValidationError):
        output_model.model_validate(payload)
