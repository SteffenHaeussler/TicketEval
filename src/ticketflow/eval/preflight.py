"""Pre-run Ollama readiness checks, per plan.md's "Ollama preflight" section.

Before any real-model run: confirm the server is reachable and every required model is
present with a recorded digest, measure representative latency to widen the workflow's
activity timeout safely, and probe confidence variance to decide -- never assume --
whether milestone 4's threshold sweep is a valid tool for this model. `agent/ollama.py`
remains the only module that sends `/api/chat` model requests; this module only adds
two narrow admin GETs (`/api/version`, `/api/tags`) and orchestrates measurement calls
through the already-built `OllamaAgent`.
"""

import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import httpx

from ticketflow.agent.ollama import OllamaAgent
from ticketflow.eval.harness import WorkflowEvalConfig
from ticketflow.eval.progress import ProgressCallback, emit
from ticketflow.eval.telemetry import TelemetrySink
from ticketflow.models import Ticket

MIN_PROBE_CASES = 10
_MIN_STD_DEV = 0.02
_MIN_DISTINCT_VALUES = 5
_WARMUP_TICKET = Ticket(
    id="preflight-warmup",
    customer_email="preflight@example.com",
    subject="Preflight warm-up",
    body="This ticket exists only to warm the model cache before measurement.",
)


class PreflightError(Exception):
    """Base class for preflight failures."""


class VersionCheckError(PreflightError):
    """Raised when GET /api/version is unreachable, non-200, or malformed."""


class ModelMissingError(PreflightError):
    """Raised when a required model is absent from GET /api/tags."""


class InsufficientProbeCasesError(PreflightError):
    """Raised when fewer than MIN_PROBE_CASES probe tickets are supplied."""


@dataclass(frozen=True)
class ModelInfo:
    """A required model's confirmed role, name, and locally-installed digest."""

    role: Literal["primary", "fallback"]
    name: str
    digest: str


@dataclass(frozen=True)
class StageMeasurement:
    """One measured classify/draft call's wall, load, and generation timings."""

    operation: Literal["classify", "draft"]
    ticket_id: str
    wall_latency_s: float
    load_duration_s: float | None
    generation_duration_s: float | None


@dataclass(frozen=True)
class TimeoutAdjustment:
    """The measured inputs and the widened activity/HTTP timeouts derived from them."""

    configured_activity_timeout_s: float
    slowest_observed_stage_s: float
    effective_activity_timeout_s: float
    safety_margin_s: float
    http_timeout_s: float


@dataclass(frozen=True)
class ConfidenceGateResult:
    """Per-gate pass/fail for the threshold-sweep admissibility decision.

    Standard deviation alone is too weak a test for self-reported LLM confidence: it
    typically clusters on two or three values (e.g. 0.9 and 0.95), which can pass a
    variance threshold while producing a step-function sweep with no interior
    operating points. Both gates are exposed as explicit booleans, never collapsed
    into one reason string, so a report can always name which gate(s) failed.
    """

    samples: tuple[float, ...]
    std_dev: float
    distinct_count: int
    passes_std_dev_gate: bool
    passes_distinctness_gate: bool

    @property
    def sweep_admissible(self) -> bool:
        """Return True only when both admissibility gates pass."""
        return self.passes_std_dev_gate and self.passes_distinctness_gate

    @property
    def failed_gates(self) -> tuple[str, ...]:
        """Name every gate that failed, e.g. ("distinctness",)."""
        failed = []
        if not self.passes_std_dev_gate:
            failed.append("std_dev")
        if not self.passes_distinctness_gate:
            failed.append("distinctness")
        return tuple(failed)


@dataclass(frozen=True)
class PreflightResult:
    """Everything learned before a real-model run, plus the adjusted config."""

    ollama_version: str
    models: tuple[ModelInfo, ...]
    measurements: tuple[StageMeasurement, ...]
    timeout_adjustment: TimeoutAdjustment
    confidence_gate: ConfidenceGateResult
    workflow_eval_config: WorkflowEvalConfig


async def check_ollama_version(client: httpx.AsyncClient) -> str:
    """GET /api/version and return the version string, or raise VersionCheckError."""
    try:
        response = await client.get("/api/version")
    except httpx.TransportError as exc:
        raise VersionCheckError(f"GET /api/version failed: {exc}") from exc
    if response.status_code != 200:
        raise VersionCheckError(
            f"GET /api/version returned HTTP {response.status_code}"
        )
    try:
        body = response.json()
        version = body["version"]
    except (ValueError, KeyError, TypeError) as exc:
        raise VersionCheckError(
            f"GET /api/version returned a malformed body: {exc}"
        ) from exc
    if not isinstance(version, str):
        raise VersionCheckError("GET /api/version 'version' field is not a string")
    return version


async def confirm_required_models(
    client: httpx.AsyncClient,
    required_models: Mapping[Literal["primary", "fallback"], str],
) -> tuple[ModelInfo, ...]:
    """Confirm every required (role, model name) exists in GET /api/tags.

    Raises ModelMissingError naming the first missing model, before any classify or
    draft call is ever made.
    """
    try:
        response = await client.get("/api/tags")
    except httpx.TransportError as exc:
        raise PreflightError(f"GET /api/tags failed: {exc}") from exc
    if response.status_code != 200:
        raise PreflightError(f"GET /api/tags returned HTTP {response.status_code}")
    try:
        body = response.json()
        digests_by_name = {model["name"]: model["digest"] for model in body["models"]}
    except (ValueError, KeyError, TypeError) as exc:
        raise PreflightError(f"GET /api/tags returned a malformed body: {exc}") from exc

    models = []
    for role, name in required_models.items():
        digest = digests_by_name.get(name)
        if digest is None:
            raise ModelMissingError(
                f"required model {name!r} for role {role!r} was not found by "
                "GET /api/tags"
            )
        models.append(ModelInfo(role=role, name=name, digest=digest))
    return tuple(models)


def compute_timeout_adjustment(
    *, configured_activity_timeout_s: float, stage_seconds: Sequence[float]
) -> TimeoutAdjustment:
    """Widen the activity timeout from measured stages. Never raises."""
    slowest_observed_stage_s = max(stage_seconds) if stage_seconds else 0.0
    effective_activity_timeout_s = max(
        10.0, configured_activity_timeout_s, 3 * slowest_observed_stage_s
    )
    safety_margin_s = max(5.0, 0.10 * effective_activity_timeout_s)
    http_timeout_s = effective_activity_timeout_s - safety_margin_s
    return TimeoutAdjustment(
        configured_activity_timeout_s=configured_activity_timeout_s,
        slowest_observed_stage_s=slowest_observed_stage_s,
        effective_activity_timeout_s=effective_activity_timeout_s,
        safety_margin_s=safety_margin_s,
        http_timeout_s=http_timeout_s,
    )


def evaluate_confidence_gate(confidences: Sequence[float]) -> ConfidenceGateResult:
    """Score the std-dev and distinctness admissibility gates. Never raises.

    A degenerate or tiny confidence sample is a reported finding, not a harness
    failure, mirroring statistics.threshold_sweep's own handling of an empty
    population.
    """
    # Round away float-representation noise (e.g. a cache round-trip) before
    # comparing confidences for equality; genuinely distinct model outputs differ
    # by far more than 1e-6.
    rounded = [round(c, 6) for c in confidences]
    std_dev = statistics.stdev(rounded) if len(rounded) >= 2 else 0.0
    distinct_count = len(set(rounded))
    return ConfidenceGateResult(
        samples=tuple(confidences),
        std_dev=std_dev,
        distinct_count=distinct_count,
        passes_std_dev_gate=std_dev >= _MIN_STD_DEV,
        passes_distinctness_gate=distinct_count >= _MIN_DISTINCT_VALUES,
    )


def _stage_measurements_from_sink(
    sink: TelemetrySink, ticket_id: str
) -> list[StageMeasurement]:
    """Drain one ticket's telemetry into stage measurements.

    An attempt missing duration data (no `total_duration` in the Ollama
    response) still keeps its wall-clock latency, so a caller aggregating stage
    seconds never silently loses a real, if imprecise, timing observation.
    """
    measurements = []
    for attempt in sink.drain(ticket_id):
        if attempt.model_total_duration_ms is None:
            load_s = None
            generation_s = None
        else:
            load_s = (attempt.model_load_duration_ms or 0.0) / 1000
            total_s = attempt.model_total_duration_ms / 1000
            generation_s = max(0.0, total_s - load_s)
        measurements.append(
            StageMeasurement(
                operation=attempt.operation,
                ticket_id=attempt.ticket_id,
                wall_latency_s=attempt.wall_latency_ms / 1000,
                load_duration_s=load_s,
                generation_duration_s=generation_s,
            )
        )
    return measurements


def _stage_seconds_for_timeout(measurements: Sequence[StageMeasurement]) -> list[float]:
    """Flatten measurements into stage seconds for the timeout-widening formula.

    A measurement with a load/generation split contributes both; one without a
    split (no duration data from Ollama) falls back to its wall-clock latency, so
    every measured call informs the timeout even when duration data is missing.
    """
    stage_seconds = []
    for measurement in measurements:
        has_split = (
            measurement.load_duration_s is not None
            or measurement.generation_duration_s is not None
        )
        if not has_split:
            stage_seconds.append(measurement.wall_latency_s)
            continue
        if measurement.load_duration_s is not None:
            stage_seconds.append(measurement.load_duration_s)
        if measurement.generation_duration_s is not None:
            stage_seconds.append(measurement.generation_duration_s)
    return stage_seconds


async def run_preflight(
    *,
    endpoint: str,
    required_models: Mapping[Literal["primary", "fallback"], str],
    probe_tickets: Sequence[Ticket],
    workflow_eval_config: WorkflowEvalConfig,
    probe_http_timeout_s: float,
    seed: int | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    progress: ProgressCallback | None = None,
) -> PreflightResult:
    """Run the full preflight sequence and return its result.

    Order: version check, then model/digest confirmation (both complete before any
    classify/draft call), then one unmeasured warm-up, then timed measurement over
    every probe ticket, then timeout and confidence-gate computation. Measurement
    uses the primary model only: `AGENT_ACTIVITY_TIMEOUT` is one workflow constant
    shared by both the primary and fallback agent activities, so sizing off primary
    latency is sufficient.
    """
    # Cheap, local validation happens before any network I/O.
    if "primary" not in required_models:
        raise PreflightError("required_models must include a 'primary' role")
    if len(probe_tickets) < MIN_PROBE_CASES:
        raise InsufficientProbeCasesError(
            f"preflight requires at least {MIN_PROBE_CASES} probe tickets, got "
            f"{len(probe_tickets)}"
        )

    async with httpx.AsyncClient(
        base_url=endpoint, timeout=probe_http_timeout_s, transport=transport
    ) as admin_client:
        emit(progress, "preflight", f"checking Ollama at {endpoint}")
        version = await check_ollama_version(admin_client)
        emit(progress, "preflight", f"Ollama {version}; confirming models")
        models = await confirm_required_models(admin_client, required_models)
        emit(
            progress,
            "preflight",
            "models confirmed: "
            + ", ".join(f"{model.role}={model.name}" for model in models),
        )

    primary_name = required_models["primary"]
    primary_digest = next(model.digest for model in models if model.role == "primary")

    sink = TelemetrySink()
    measurements: list[StageMeasurement] = []
    confidences: list[float] = []
    async with OllamaAgent(
        endpoint=endpoint,
        model=primary_name,
        timeout_s=probe_http_timeout_s,
        seed=seed,
        role="primary",
        telemetry_sink=sink,
        model_digest=primary_digest,
        ollama_version=version,
        transport=transport,
    ) as agent:
        emit(progress, "preflight", f"warming up {primary_name} (unmeasured)")
        await agent.classify(_WARMUP_TICKET)
        sink.drain(_WARMUP_TICKET.id)

        for index, ticket in enumerate(probe_tickets, start=1):
            started_at = time.perf_counter()
            classification = await agent.classify(ticket)
            draft = await agent.draft_reply(ticket, classification)
            confidences.append(draft.confidence)
            measurements.extend(_stage_measurements_from_sink(sink, ticket.id))
            emit(
                progress,
                "preflight",
                "probe classify+draft",
                completed=index,
                total=len(probe_tickets),
                elapsed_s=time.perf_counter() - started_at,
            )

    stage_seconds = _stage_seconds_for_timeout(measurements)

    timeout_adjustment = compute_timeout_adjustment(
        configured_activity_timeout_s=workflow_eval_config.agent_activity_timeout_s,
        stage_seconds=stage_seconds,
    )
    confidence_gate = evaluate_confidence_gate(confidences)
    updated_config = workflow_eval_config.model_copy(
        update={
            "agent_activity_timeout_s": timeout_adjustment.effective_activity_timeout_s
        }
    )

    return PreflightResult(
        ollama_version=version,
        models=models,
        measurements=tuple(measurements),
        timeout_adjustment=timeout_adjustment,
        confidence_gate=confidence_gate,
        workflow_eval_config=updated_config,
    )
