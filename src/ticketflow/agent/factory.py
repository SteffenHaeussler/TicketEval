"""Production seam for constructing the primary and fallback agents.

`ticketflow.config` satisfies `AgentSettings` structurally, so the LLM worker can pass
the module itself. Evaluation code builds tunable, primary, and fallback agents
explicitly and does not go through this factory -- it is not responsible for
label-aware tunable agents.
"""

from typing import Literal, Protocol

from ticketflow.agent.base import Agent
from ticketflow.agent.mock import MockAgent
from ticketflow.agent.ollama import OllamaAgent
from ticketflow.eval.telemetry import TelemetrySink


class AgentSettings(Protocol):
    """The subset of `ticketflow.config` that `build_agent` depends on."""

    AGENT_BACKEND: str
    OLLAMA_ENDPOINT: str
    PRIMARY_MODEL: str
    FALLBACK_MODEL: str
    OLLAMA_TIMEOUT_S: float
    OLLAMA_SEED: int
    MOCK_AGENT_LATENCY_MAX_S: float


class AgentBackendError(Exception):
    """Raised for an unrecognized `AGENT_BACKEND` value."""


def build_agent(
    role: Literal["primary", "fallback"],
    settings: AgentSettings,
    *,
    event_sink: TelemetrySink | None = None,
) -> Agent:
    """Construct one role's agent from configuration.

    This seam deliberately takes no response cache. A cached `OllamaAgent` also needs
    the runtime identity map, model digest, and Ollama version that only preflight can
    supply, so the eval harness constructs its own cached agents in
    `ticketflow.eval.profiles`; production runs uncached.
    """
    if settings.AGENT_BACKEND == "mock":
        if role == "primary":
            return MockAgent(
                latency_range=(0.0, settings.MOCK_AGENT_LATENCY_MAX_S), model="primary"
            )
        return MockAgent.fallback()

    if settings.AGENT_BACKEND == "ollama":
        model = settings.PRIMARY_MODEL if role == "primary" else settings.FALLBACK_MODEL
        return OllamaAgent(
            endpoint=settings.OLLAMA_ENDPOINT,
            model=model,
            timeout_s=settings.OLLAMA_TIMEOUT_S,
            seed=settings.OLLAMA_SEED,
            role=role,
            telemetry_sink=event_sink,
        )

    raise AgentBackendError(
        f"unknown TICKETFLOW_AGENT_BACKEND {settings.AGENT_BACKEND!r}; "
        "expected 'mock' or 'ollama'"
    )
