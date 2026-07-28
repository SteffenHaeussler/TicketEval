import json
import os

import pytest
from pydantic import ValidationError

from ticketflow.eval.cache import (
    CacheConflictError,
    CachedAgentResponse,
    CacheRequest,
    FileResponseCache,
    ResponseCache,
)
from ticketflow.models import Classification, TicketCategory


def _base_kwargs(**overrides):
    kwargs = dict(
        operation="classify",
        model_name="qwen2.5-coder:1.5b",
        model_digest="sha256:abc123",
        role="primary",
        case_key="case-001",
        customer_email="alice@example.com",
        subject="Refund request",
        body="I would like a refund for my last order.",
        classification_input=None,
        messages=(
            {"role": "system", "content": "classify"},
            {"role": "user", "content": "ticket"},
        ),
        prompt_version="prompt-v1",
        json_schema={"type": "object", "properties": {"category": {"type": "string"}}},
        think=False,
        temperature=0.0,
        seed=42,
        generation_options={"num_ctx": 4096},
        ollama_version="0.5.1",
    )
    kwargs.update(overrides)
    return kwargs


def make_request(**overrides) -> CacheRequest:
    return CacheRequest.model_validate(_base_kwargs(**overrides))


def make_response(**overrides) -> CachedAgentResponse:
    kwargs = dict(
        output={"category": "billing", "confidence": 0.9},
        model_total_duration_ms=120.0,
        model_load_duration_ms=30.0,
    )
    kwargs.update(overrides)
    return CachedAgentResponse.model_validate(kwargs)


# --- one test per key component ---


@pytest.mark.parametrize(
    ("field", "base_value", "other_value"),
    [
        ("operation", "classify", "draft"),
        ("model_name", "qwen2.5-coder:1.5b", "llama3.2:3b"),
        ("model_digest", "sha256:abc123", "sha256:def456"),
        ("role", "primary", "fallback"),
        ("case_key", "case-001", "case-002"),
        ("customer_email", "alice@example.com", "bob@example.com"),
        ("subject", "Refund request", "Billing question"),
        ("body", "I would like a refund for my last order.", "Different ticket text."),
        (
            "messages",
            ({"role": "user", "content": "ticket"},),
            ({"role": "user", "content": "different ticket"},),
        ),
        ("prompt_version", "prompt-v1", "prompt-v2"),
        (
            "json_schema",
            {"type": "object", "properties": {"category": {"type": "string"}}},
            {"type": "object", "properties": {"category": {"type": "integer"}}},
        ),
        ("think", False, True),
        ("temperature", 0.0, 0.7),
        ("seed", 42, 43),
        ("generation_options", {"num_ctx": 4096}, {"num_ctx": 8192}),
        ("ollama_version", "0.5.1", "0.5.2"),
    ],
)
def test_changing_one_key_component_changes_the_cache_key(
    tmp_path, field, base_value, other_value
):
    cache = FileResponseCache(tmp_path)
    base = make_request(**{field: base_value})
    variant = make_request(**{field: other_value})
    response = make_response()

    cache.put_success(base, response)

    assert cache.get(variant) is None
    assert cache.get(base) == response


def test_changing_classification_input_changes_the_cache_key(tmp_path):
    cache = FileResponseCache(tmp_path)
    base = make_request(
        operation="draft",
        classification_input=Classification(
            category=TicketCategory.BILLING, confidence=0.9, model="primary"
        ),
    )
    variant = make_request(
        operation="draft",
        classification_input=Classification(
            category=TicketCategory.TECHNICAL, confidence=0.9, model="primary"
        ),
    )
    response = make_response()

    cache.put_success(base, response)

    assert cache.get(variant) is None
    assert cache.get(base) == response


# --- runtime ticket ID exclusion ---


def test_cache_request_rejects_an_unknown_ticket_id_field():
    with pytest.raises(ValidationError):
        CacheRequest.model_validate({**_base_kwargs(), "ticket_id": "ticket-123"})


# --- protocol conformance ---


def test_file_response_cache_satisfies_response_cache_protocol(tmp_path):
    cache = FileResponseCache(tmp_path)
    assert isinstance(cache, ResponseCache)


def test_cache_exposes_no_generic_write_method(tmp_path):
    cache = FileResponseCache(tmp_path)
    public_methods = {
        name
        for name in dir(cache)
        if not name.startswith("_") and callable(getattr(cache, name))
    }
    assert public_methods == {"get", "put_success"}


# --- get/put round trip ---


def test_get_returns_none_on_a_miss(tmp_path):
    cache = FileResponseCache(tmp_path)
    assert cache.get(make_request()) is None


def test_put_success_then_get_round_trips(tmp_path):
    cache = FileResponseCache(tmp_path)
    request = make_request()
    response = make_response()

    cache.put_success(request, response)

    assert cache.get(request) == response


def test_cache_entry_on_disk_stores_request_metadata_for_inspection(tmp_path):
    cache = FileResponseCache(tmp_path)
    request = make_request()

    cache.put_success(request, make_response())

    [entry_path] = list(tmp_path.iterdir())
    raw = json.loads(entry_path.read_text())
    assert raw["request"]["case_key"] == request.case_key
    assert raw["request"]["subject"] == request.subject


# --- atomic write semantics ---


def test_put_success_twice_with_same_output_but_different_timing_is_idempotent(
    tmp_path,
):
    cache = FileResponseCache(tmp_path)
    request = make_request()
    first = make_response(model_total_duration_ms=100.0)
    second = make_response(model_total_duration_ms=250.0)

    cache.put_success(request, first)
    cache.put_success(request, second)

    assert cache.get(request) == first
    assert len(list(tmp_path.iterdir())) == 1


def test_put_success_twice_with_different_output_raises_conflict_and_keeps_original(
    tmp_path,
):
    cache = FileResponseCache(tmp_path)
    request = make_request()
    first = make_response(output={"category": "billing", "confidence": 0.9})
    second = make_response(output={"category": "technical", "confidence": 0.5})

    cache.put_success(request, first)

    with pytest.raises(CacheConflictError):
        cache.put_success(request, second)

    assert cache.get(request) == first


def test_put_success_leaves_no_temp_file_when_fsync_raises(tmp_path, monkeypatch):
    cache = FileResponseCache(tmp_path)
    request = make_request()

    def boom(fd):
        raise OSError("fsync boom")

    monkeypatch.setattr(os, "fsync", boom)

    with pytest.raises(OSError):
        cache.put_success(request, make_response())

    assert list(tmp_path.iterdir()) == []
