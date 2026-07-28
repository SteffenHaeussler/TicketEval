import json
from typing import Any, Literal

import httpx
import pytest

from tests.helpers import make_ticket
from ticketflow.agent.prompts import CLASSIFICATION_SPEC
from ticketflow.eval.harness import WorkflowEvalConfig
from ticketflow.eval.preflight import (
    MIN_PROBE_CASES,
    InsufficientProbeCasesError,
    ModelMissingError,
    PreflightError,
    VersionCheckError,
    compute_timeout_adjustment,
    evaluate_confidence_gate,
    run_preflight,
)

CONFIG = WorkflowEvalConfig(
    confidence_threshold=0.75,
    agent_task_queue="agent",
    fallback_task_queue="agent-fallback",
    agent_schedule_to_start_s=30.0,
    agent_activity_timeout_s=120.0,
    agent_heartbeat_timeout_s=30.0,
)

REQUIRED_MODELS: dict[Literal["primary", "fallback"], str] = {
    "primary": "primary-model",
    "fallback": "fallback-model",
}
DRAFT_CONFIDENCES = [0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Pure-function tests: confidence gate
# ---------------------------------------------------------------------------


def test_distinctness_gate_fails_for_090_095_despite_passing_variance():
    result = evaluate_confidence_gate([0.9, 0.95])
    assert result.passes_std_dev_gate is True
    assert result.passes_distinctness_gate is False
    assert result.sweep_admissible is False
    assert result.failed_gates == ("distinctness",)


def test_distinctness_gate_fails_for_a_ten_case_090_095_cluster():
    result = evaluate_confidence_gate([0.9] * 5 + [0.95] * 5)
    assert result.passes_std_dev_gate is True
    assert result.passes_distinctness_gate is False
    assert result.failed_gates == ("distinctness",)


def test_confidence_gate_passes_with_five_distinct_values_and_sufficient_spread():
    result = evaluate_confidence_gate([0.5, 0.6, 0.7, 0.8, 0.9])
    assert result.passes_std_dev_gate is True
    assert result.passes_distinctness_gate is True
    assert result.sweep_admissible is True
    assert result.failed_gates == ()


def test_confidence_gate_on_identical_values_never_raises_and_fails_both_gates():
    result = evaluate_confidence_gate([0.9] * 10)
    assert result.std_dev == 0.0
    assert result.distinct_count == 1
    assert result.passes_std_dev_gate is False
    assert result.passes_distinctness_gate is False
    assert result.failed_gates == ("std_dev", "distinctness")


def test_confidence_gate_on_empty_sample_never_raises():
    result = evaluate_confidence_gate([])
    assert result.std_dev == 0.0
    assert result.distinct_count == 0
    assert result.passes_std_dev_gate is False
    assert result.passes_distinctness_gate is False


# ---------------------------------------------------------------------------
# Pure-function tests: timeout adjustment
# ---------------------------------------------------------------------------


def test_timeout_floor_binds_when_measurements_and_configured_timeout_are_small():
    adjustment = compute_timeout_adjustment(
        configured_activity_timeout_s=5.0, stage_seconds=[0.1, 0.2]
    )
    assert adjustment.effective_activity_timeout_s == 10.0
    assert adjustment.safety_margin_s == 5.0
    assert adjustment.http_timeout_s == 5.0


def test_timeout_widens_from_slowest_stage_when_it_dominates():
    adjustment = compute_timeout_adjustment(
        configured_activity_timeout_s=10.0, stage_seconds=[20.0]
    )
    assert adjustment.slowest_observed_stage_s == 20.0
    assert adjustment.effective_activity_timeout_s == 60.0
    assert adjustment.safety_margin_s == 6.0
    assert adjustment.http_timeout_s == 54.0


def test_timeout_uses_configured_value_when_it_is_the_largest():
    adjustment = compute_timeout_adjustment(
        configured_activity_timeout_s=45.0, stage_seconds=[5.0]
    )
    assert adjustment.effective_activity_timeout_s == 45.0


# ---------------------------------------------------------------------------
# run_preflight integration tests over a stubbed httpx transport
# ---------------------------------------------------------------------------


def _version_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"version": "0.30.10"}, request=request)


def _tags_response(
    request: httpx.Request, models: dict[str, str] | None = None
) -> httpx.Response:
    if models is None:
        models = {
            "primary-model": "sha256:primary-digest",
            "fallback-model": "sha256:fallback-digest",
        }
    return httpx.Response(
        200,
        json={
            "models": [
                {"name": name, "digest": digest} for name, digest in models.items()
            ]
        },
        request=request,
    )


def _chat_response(
    request: httpx.Request,
    *,
    confidence: float,
    total_duration: int = 50_000_000,
    load_duration: int = 10_000_000,
) -> httpx.Response:
    payload = json.loads(request.content)
    system_prompt = payload["messages"][0]["content"]
    content: dict[str, Any]
    if system_prompt == CLASSIFICATION_SPEC.system_prompt:
        content = {"category": "billing", "confidence": confidence}
    else:
        content = {
            "reply_text": "Thanks for reaching out.",
            "action": {"type": "reply_only"},
            "confidence": confidence,
        }
    return httpx.Response(
        200,
        json={
            "message": {"content": json.dumps(content)},
            "total_duration": total_duration,
            "load_duration": load_duration,
        },
        request=request,
    )


def _probe_tickets(count: int = MIN_PROBE_CASES):
    return [make_ticket() for _ in range(count)]


async def test_run_preflight_raises_for_missing_model_before_any_chat_call():
    chat_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_calls
        if request.url.path == "/api/version":
            return _version_response(request)
        if request.url.path == "/api/tags":
            return _tags_response(request, models={"primary-model": "sha256:primary"})
        chat_calls += 1
        raise AssertionError("no /api/chat call should happen for a missing model")

    with pytest.raises(ModelMissingError, match="fallback-model"):
        await run_preflight(
            endpoint="http://ollama.test",
            required_models=REQUIRED_MODELS,
            probe_tickets=_probe_tickets(),
            workflow_eval_config=CONFIG,
            probe_http_timeout_s=5.0,
            transport=httpx.MockTransport(handler),
        )
    assert chat_calls == 0


async def test_run_preflight_requires_a_primary_model_before_any_http_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call should happen without a primary model")

    fallback_only: dict[Literal["primary", "fallback"], str] = {
        "fallback": "fallback-model"
    }
    with pytest.raises(PreflightError, match="primary"):
        await run_preflight(
            endpoint="http://ollama.test",
            required_models=fallback_only,
            probe_tickets=_probe_tickets(),
            workflow_eval_config=CONFIG,
            probe_http_timeout_s=5.0,
            transport=httpx.MockTransport(handler),
        )


async def test_run_preflight_raises_for_broken_version_endpoint_before_tags_or_chat():
    tags_calls = 0
    chat_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal tags_calls, chat_calls
        if request.url.path == "/api/version":
            return httpx.Response(500, request=request)
        if request.url.path == "/api/tags":
            tags_calls += 1
            raise AssertionError("tags must not be called after a broken version check")
        chat_calls += 1
        raise AssertionError("chat must not be called after a broken version check")

    with pytest.raises(VersionCheckError):
        await run_preflight(
            endpoint="http://ollama.test",
            required_models=REQUIRED_MODELS,
            probe_tickets=_probe_tickets(),
            workflow_eval_config=CONFIG,
            probe_http_timeout_s=5.0,
            transport=httpx.MockTransport(handler),
        )
    assert tags_calls == 0
    assert chat_calls == 0


async def test_run_preflight_raises_when_fewer_than_min_probe_cases_supplied():
    chat_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_calls
        if request.url.path == "/api/version":
            return _version_response(request)
        if request.url.path == "/api/tags":
            return _tags_response(request)
        chat_calls += 1
        raise AssertionError("no /api/chat call should happen with too few probes")

    with pytest.raises(InsufficientProbeCasesError):
        await run_preflight(
            endpoint="http://ollama.test",
            required_models=REQUIRED_MODELS,
            probe_tickets=_probe_tickets(3),
            workflow_eval_config=CONFIG,
            probe_http_timeout_s=5.0,
            transport=httpx.MockTransport(handler),
        )
    assert chat_calls == 0


async def test_run_preflight_happy_path_returns_digests_version_timeout_and_gate():
    draft_confidences = iter(DRAFT_CONFIDENCES * 2)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return _version_response(request)
        if request.url.path == "/api/tags":
            return _tags_response(request)
        payload = json.loads(request.content)
        system_prompt = payload["messages"][0]["content"]
        if system_prompt == CLASSIFICATION_SPEC.system_prompt:
            return _chat_response(request, confidence=0.9)
        return _chat_response(request, confidence=next(draft_confidences))

    result = await run_preflight(
        endpoint="http://ollama.test",
        required_models=REQUIRED_MODELS,
        probe_tickets=_probe_tickets(),
        workflow_eval_config=CONFIG,
        probe_http_timeout_s=5.0,
        transport=httpx.MockTransport(handler),
    )

    assert result.ollama_version == "0.30.10"
    digests = {model.role: model.digest for model in result.models}
    assert digests == {
        "primary": "sha256:primary-digest",
        "fallback": "sha256:fallback-digest",
    }
    assert len(result.measurements) == 20
    assert (
        result.workflow_eval_config.agent_activity_timeout_s
        == result.timeout_adjustment.effective_activity_timeout_s
    )
    assert (
        result.workflow_eval_config.model_copy(
            update={"agent_activity_timeout_s": CONFIG.agent_activity_timeout_s}
        )
        == CONFIG
    )
    assert result.confidence_gate.sweep_admissible is True


async def test_warmup_call_is_excluded_from_measurements_and_slowest_stage():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/api/version":
            return _version_response(request)
        if request.url.path == "/api/tags":
            return _tags_response(request)
        call_count += 1
        if call_count == 1:
            return _chat_response(
                request,
                confidence=0.9,
                total_duration=90_000_000_000,
                load_duration=80_000_000_000,
            )
        payload = json.loads(request.content)
        system_prompt = payload["messages"][0]["content"]
        if system_prompt == CLASSIFICATION_SPEC.system_prompt:
            return _chat_response(request, confidence=0.9)
        return _chat_response(request, confidence=DRAFT_CONFIDENCES[call_count % 5])

    result = await run_preflight(
        endpoint="http://ollama.test",
        required_models=REQUIRED_MODELS,
        probe_tickets=_probe_tickets(),
        workflow_eval_config=CONFIG,
        probe_http_timeout_s=5.0,
        transport=httpx.MockTransport(handler),
    )

    assert all(m.ticket_id != "preflight-warmup" for m in result.measurements)
    assert result.timeout_adjustment.slowest_observed_stage_s < 3 * 80.0
    assert result.timeout_adjustment.effective_activity_timeout_s < 3 * 80.0
