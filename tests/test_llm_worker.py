from dataclasses import dataclass

import pytest

from ticketflow.activities import TicketActivities
from ticketflow.agent.factory import AgentBackendError, build_agent
from ticketflow.agent.mock import MockAgent
from ticketflow.agent.ollama import OllamaAgent
from ticketflow.llm_worker import _build_activities


@dataclass
class _Settings:
    """A minimal stand-in for `ticketflow.config`, satisfying `AgentSettings`."""

    AGENT_BACKEND: str = "mock"
    OLLAMA_ENDPOINT: str = "http://ollama.test"
    PRIMARY_MODEL: str = "test-primary-model"
    FALLBACK_MODEL: str = "test-fallback-model"
    OLLAMA_TIMEOUT_S: float = 42.0
    OLLAMA_SEED: int = 7
    MOCK_AGENT_LATENCY_MAX_S: float = 1.5


def _settings(**overrides: object) -> _Settings:
    return _Settings(**overrides)  # type: ignore[arg-type]


def test_build_agent_mock_backend_builds_primary_and_fallback():
    settings = _settings(AGENT_BACKEND="mock")

    primary = build_agent("primary", settings)
    fallback = build_agent("fallback", settings)

    assert isinstance(primary, MockAgent)
    assert primary._model == "primary"
    assert primary._latency_range == (0.0, 1.5)
    assert isinstance(fallback, MockAgent)
    assert fallback._model == "fallback"


def test_build_agent_ollama_backend_selects_model_by_role():
    settings = _settings(AGENT_BACKEND="ollama")

    primary = build_agent("primary", settings)
    fallback = build_agent("fallback", settings)

    assert isinstance(primary, OllamaAgent)
    assert isinstance(fallback, OllamaAgent)
    assert primary._model == "test-primary-model"
    assert primary._role == "primary"
    assert fallback._model == "test-fallback-model"
    assert fallback._role == "fallback"


def test_build_agent_rejects_an_unknown_backend():
    settings = _settings(AGENT_BACKEND="bogus")

    with pytest.raises(AgentBackendError, match="bogus"):
        build_agent("primary", settings)


def test_build_activities_defaults_to_mock(monkeypatch):
    from ticketflow import config

    monkeypatch.setattr(config, "AGENT_BACKEND", "mock")

    primary_activities, fallback_activities = _build_activities()

    assert isinstance(primary_activities, TicketActivities)
    assert isinstance(fallback_activities, TicketActivities)
    assert isinstance(primary_activities._agent, MockAgent)
    assert isinstance(fallback_activities._agent, MockAgent)
    assert primary_activities._agent._model == "primary"
    assert fallback_activities._agent._model == "fallback"


def test_build_activities_constructs_ollama_agents_when_configured(monkeypatch):
    from ticketflow import config

    monkeypatch.setattr(config, "AGENT_BACKEND", "ollama")
    monkeypatch.setattr(config, "PRIMARY_MODEL", "prod-primary")
    monkeypatch.setattr(config, "FALLBACK_MODEL", "prod-fallback")

    primary_activities, fallback_activities = _build_activities()

    assert isinstance(primary_activities._agent, OllamaAgent)
    assert isinstance(fallback_activities._agent, OllamaAgent)
    assert primary_activities._agent._model == "prod-primary"
    assert fallback_activities._agent._model == "prod-fallback"


def test_build_activities_fails_before_any_temporal_setup(monkeypatch):
    from ticketflow import config

    monkeypatch.setattr(config, "AGENT_BACKEND", "bogus")

    with pytest.raises(AgentBackendError, match="bogus"):
        _build_activities()
