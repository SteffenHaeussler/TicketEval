"""Immutable eval run-artifact models and atomic JSONL/JSON read/write helpers."""

import json
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ticketflow.eval.dataset import ExpectedOutcome
from ticketflow.models import ActionType, ApprovalDecision, TicketCategory, TicketStatus


class RecordsError(Exception):
    """Base class for eval-records read/write failures."""


class ArtifactExistsError(RecordsError):
    """Raised when a writer's destination path already exists."""


class RecordsReadError(RecordsError):
    """Raised when a JSONL/JSON artifact fails to parse or validate on read."""


class CallEvent(BaseModel):
    """One classification or drafting attempt, per plan.md's Telemetry section."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    case_key: str
    ticket_id: str
    policy: str
    repeat_index: int
    operation: Literal["classify", "draft"]
    role: Literal["primary", "fallback"]
    attempt: int
    cache_hit: bool
    started_at: datetime
    wall_latency_ms: float
    model_total_duration_ms: float | None
    model_load_duration_ms: float | None
    outcome: Literal["success", "invalid_output", "transient_error", "permanent_error"]
    error_type: str | None


class CaseRecord(BaseModel):
    """Immutable per-case run outcome, per plan.md's run-artifacts section.

    Top-level fields and scoring-critical expected labels are frozen. The nested
    production ApprovalDecision is treated as read-only by convention; persisted
    raw artifacts are strictly write-once.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    policy: str
    case_key: str
    repeat_index: int
    ticket_id: str

    difficulty: Literal["easy", "ambiguous", "adversarial"]
    source: Literal["handwritten", "generated"]
    expected: ExpectedOutcome

    predicted_category: TicketCategory | None = None
    predicted_action: ActionType | None = None
    predicted_refund_amount: float | None = Field(default=None, gt=0)
    classification_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    draft_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    reply_text: str | None = None
    model_path: str = "primary/primary"

    terminal_status: TicketStatus | None = None
    was_gated: bool = False
    decision: ApprovalDecision | None = None

    refund_executed_count: int = Field(default=0, ge=0)
    refund_attempt_count: int = Field(default=0, ge=0)

    prediction_available: bool
    prediction_unavailable_reason: str | None = None

    terminal_outcome: Literal[
        "resolved",
        "rejected",
        "escalated",
        "update_rejected",
        "runner_deadline_exceeded",
    ]
    cleanup_action: Literal["cancelled", "terminated"] | None = None

    end_to_end_latency_ms: float
    terminal_error: str | None = None

    @model_validator(mode="after")
    def _prediction_available_matches_draft_presence(self) -> "CaseRecord":
        """Enforce that prediction_available is derived solely from draft presence."""
        expected_value = self.reply_text is not None
        if self.prediction_available != expected_value:
            raise ValueError(
                f"case {self.case_key!r}: prediction_available="
                f"{self.prediction_available} does not match draft presence "
                f"(reply_text is {'set' if expected_value else 'None'})"
            )
        return self

    @model_validator(mode="after")
    def _draft_fields_match_reply_presence(self) -> "CaseRecord":
        """Keep flattened draft fields consistent with captured reply presence."""
        has_reply = self.reply_text is not None
        if has_reply and self.predicted_action is None:
            raise ValueError(
                f"case {self.case_key!r}: predicted_action is required when "
                "reply_text is set"
            )
        if has_reply and self.draft_confidence is None:
            raise ValueError(
                f"case {self.case_key!r}: draft_confidence is required when "
                "reply_text is set"
            )
        if not has_reply and self.predicted_action is not None:
            raise ValueError(
                f"case {self.case_key!r}: predicted_action must be None when "
                "reply_text is None"
            )
        if not has_reply and self.draft_confidence is not None:
            raise ValueError(
                f"case {self.case_key!r}: draft_confidence must be None when "
                "reply_text is None"
            )
        if not has_reply and self.predicted_refund_amount is not None:
            raise ValueError(
                f"case {self.case_key!r}: predicted_refund_amount must be None when "
                "reply_text is None"
            )
        return self

    @model_validator(mode="after")
    def _cleanup_action_matches_deadline_outcome(self) -> "CaseRecord":
        """Enforce that cleanup_action is set only for runner_deadline_exceeded."""
        is_deadline = self.terminal_outcome == "runner_deadline_exceeded"
        if is_deadline and self.cleanup_action is None:
            raise ValueError(
                f"case {self.case_key!r}: cleanup_action is required when "
                "terminal_outcome is runner_deadline_exceeded"
            )
        if not is_deadline and self.cleanup_action is not None:
            raise ValueError(
                f"case {self.case_key!r}: cleanup_action must be None unless "
                "terminal_outcome is runner_deadline_exceeded"
            )
        return self


class PreflightMeasurement(BaseModel):
    """One ordered Ollama preflight stage measurement persisted in a manifest."""

    model_config = ConfigDict(frozen=True)

    operation: Literal["classify", "draft"]
    ticket_id: str
    wall_latency_s: float
    load_duration_s: float | None
    generation_duration_s: float | None


class TimeoutAdjustment(BaseModel):
    """The complete timeout derivation produced by Ollama preflight."""

    model_config = ConfigDict(frozen=True)

    configured_activity_timeout_s: float
    slowest_observed_stage_s: float
    effective_activity_timeout_s: float
    safety_margin_s: float
    http_timeout_s: float


class GenerationSettings(BaseModel):
    """Deterministic Ollama generation controls recorded for reproducibility."""

    model_config = ConfigDict(frozen=True)

    stream: bool
    think: bool
    temperature: float


class RunManifest(BaseModel):
    """Frozen run config/provenance snapshot, per plan.md's manifest bullets.

    Nested JSON-like values are treated as read-only by convention. Once written,
    the persisted manifest is strictly write-once.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    git_commit: str
    git_dirty: bool
    dataset_path: str
    dataset_sha256: str
    agent_backend: Literal["tunable", "mock", "ollama"]
    run_profile: Literal[
        "primary-quality", "fallback-quality", "fallback-routing", "reliability"
    ]
    primary_model: str
    fallback_model: str | None = None
    python_version: str
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    reviewer_policies: list[Literal["oracle", "rubber_stamp"]]
    cache_enabled: bool

    confidence_threshold: float
    agent_task_queue: str
    fallback_task_queue: str
    agent_schedule_to_start_s: float
    agent_activity_timeout_s: float
    agent_heartbeat_timeout_s: float

    seed: int
    bootstrap_seed: int
    # Identifies how per-repeat generation seeds were derived from `seed`, so a run
    # stays reproducible if the rule ever changes. See profiles.GENERATION_SEED_RULE.
    generation_seed_rule: str
    concurrency: int
    repeats: int
    started_at: datetime
    finished_at: datetime

    # Ollama-only provenance remains unset for tunable and mock runs.
    primary_model_digest: str | None = None
    fallback_model_digest: str | None = None
    ollama_version: str | None = None
    prompt_hashes: dict[Literal["classify", "draft"], str] | None = None
    schema_hashes: dict[Literal["classify", "draft"], str] | None = None
    generation_settings: GenerationSettings | None = None
    preflight_measurements: tuple[PreflightMeasurement, ...] | None = None
    timeout_adjustment: TimeoutAdjustment | None = None

    @model_validator(mode="after")
    def _operation_hashes_cover_all_operations(self) -> "RunManifest":
        """Require complete prompt/schema provenance whenever either is recorded."""
        required_operations = {"classify", "draft"}
        for field_name, hashes in (
            ("prompt_hashes", self.prompt_hashes),
            ("schema_hashes", self.schema_hashes),
        ):
            if hashes is not None and set(hashes) != required_operations:
                raise ValueError(
                    f"{field_name} must contain exactly classify and draft entries"
                )
        return self


def _atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write data to path atomically, refusing to overwrite an existing file."""
    dest = Path(path)
    if dest.exists():
        raise ArtifactExistsError(f"{dest}: refusing to overwrite existing artifact")

    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp_path, dest)
        except FileExistsError as exc:
            raise ArtifactExistsError(
                f"{dest}: refusing to overwrite existing artifact"
            ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


T = TypeVar("T", bound=BaseModel)


def _write_jsonl(path: str | Path, models: Sequence[BaseModel]) -> None:
    """Serialize models to newline-delimited JSON and write atomically."""
    lines = [model.model_dump_json() for model in models]
    data = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    _atomic_write_bytes(path, data)


def _read_jsonl(path: str | Path, model_cls: type[T]) -> list[T]:
    """Read and validate a newline-delimited JSON artifact."""
    dest = Path(path)
    out: list[T] = []
    with dest.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordsReadError(
                    f"{dest}: line {lineno}: invalid JSON ({exc})"
                ) from exc
            try:
                out.append(model_cls.model_validate(raw))
            except ValidationError as exc:
                identity = (
                    raw.get("case_key", raw.get("run_id"))
                    if isinstance(raw, dict)
                    else None
                )
                raise RecordsReadError(
                    f"{dest}: line {lineno}: record {identity!r}: invalid ({exc})"
                ) from exc
    return out


def _write_json(path: str | Path, model: BaseModel) -> None:
    """Serialize a single model to JSON and write it atomically."""
    data = model.model_dump_json(indent=2).encode("utf-8")
    _atomic_write_bytes(path, data)


def _read_json(path: str | Path, model_cls: type[T]) -> T:
    """Read and validate a single-object JSON artifact."""
    dest = Path(path)
    try:
        raw = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordsReadError(f"{dest}: invalid JSON ({exc})") from exc
    try:
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        raise RecordsReadError(f"{dest}: invalid {model_cls.__name__} ({exc})") from exc


def write_case_records(path: str | Path, records: list[CaseRecord]) -> None:
    """Write case records to a records.jsonl-style artifact, refusing to overwrite."""
    _write_jsonl(path, records)


def read_case_records(path: str | Path) -> list[CaseRecord]:
    """Read case records from a records.jsonl-style artifact."""
    return _read_jsonl(path, CaseRecord)


def write_call_events(path: str | Path, events: list[CallEvent]) -> None:
    """Write call events to a calls.jsonl-style artifact, refusing to overwrite."""
    _write_jsonl(path, events)


def read_call_events(path: str | Path) -> list[CallEvent]:
    """Read call events from a calls.jsonl-style artifact."""
    return _read_jsonl(path, CallEvent)


def write_run_manifest(path: str | Path, manifest: RunManifest) -> None:
    """Write a run manifest as a single JSON object, refusing to overwrite."""
    _write_json(path, manifest)


def read_run_manifest(path: str | Path) -> RunManifest:
    """Read a run manifest from its single-object JSON artifact."""
    return _read_json(path, RunManifest)


def write_json_artifact(path: str | Path, model: BaseModel) -> None:
    """Write any model as a single-object JSON artifact, refusing to overwrite.

    Used for run artifacts whose model lives in a module that already imports this
    one -- `InvariantReport`, for instance -- so no dedicated writer can be added
    here without a circular import.
    """
    _write_json(path, model)


def read_json_artifact(path: str | Path, model_cls: type[T]) -> T:
    """Read any model back from a single-object JSON artifact."""
    return _read_json(path, model_cls)
