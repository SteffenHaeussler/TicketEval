"""Locked response cache for real-model agent calls, per plan.md's Response cache.

Only successful, schema-validated agent responses are cached. `CacheRequest` normalizes
the complete request identity -- including model, prompt, schema, and generation
configuration, but never the runtime ticket ID -- so paired reviewer-policy runs see
byte-identical outputs for the same case.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ticketflow.models import Classification


class CacheError(Exception):
    """Base class for response-cache failures."""


class CacheReadError(CacheError):
    """Raised when a cache entry on disk fails to parse or validate."""


class CacheConflictError(CacheError):
    """Raised when a cache key already holds a different payload."""


class CacheRequest(BaseModel):
    """Normalized, serializable request identity used as the response-cache key.

    Deliberately has no `ticket_id` field: `extra="forbid"` makes passing one a
    validation error, so the runtime ticket ID cannot leak into cache identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["classify", "draft"]
    model_name: str
    model_digest: str
    role: Literal["primary", "fallback"]
    case_key: str
    customer_email: str
    subject: str
    body: str
    classification_input: Classification | None = None
    messages: tuple[dict[str, str], ...]
    prompt_version: str
    json_schema: dict[str, Any]
    think: bool
    temperature: float
    seed: int
    generation_options: dict[str, float | int | bool | str] = Field(
        default_factory=dict
    )
    ollama_version: str


class CachedAgentResponse(BaseModel):
    """Validated output payload and response timing metadata for a cached call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    output: dict[str, Any]
    model_total_duration_ms: float | None = None
    model_load_duration_ms: float | None = None


class _CacheEntry(BaseModel):
    """On-disk envelope pairing a request (for inspection) with its cached response."""

    model_config = ConfigDict(frozen=True)

    request: CacheRequest
    response: CachedAgentResponse


@runtime_checkable
class ResponseCache(Protocol):
    """Locked cache boundary consumed by the Ollama agent."""

    def get(self, request: CacheRequest) -> CachedAgentResponse | None:
        """Return the cached response for `request`, or `None` on a miss."""
        ...

    def put_success(self, request: CacheRequest, response: CachedAgentResponse) -> None:
        """Cache a successfully validated response."""
        ...


def _cache_key(request: CacheRequest) -> str:
    """Return the SHA-256 hex digest of the request's canonical JSON form."""
    canonical = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FileResponseCache:
    """Filesystem-backed `ResponseCache` storing one JSON entry per cache key."""

    def __init__(self, directory: str | Path) -> None:
        """Create a cache rooted at `directory`, creating it if it does not exist."""
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, request: CacheRequest) -> CachedAgentResponse | None:
        """Return the cached response for `request`, or `None` on a miss."""
        path = self._path_for(request)
        if not path.exists():
            return None
        return self._read_entry(path).response

    def put_success(self, request: CacheRequest, response: CachedAgentResponse) -> None:
        """Atomically cache a successfully validated response.

        Writes are create-only. If a concurrent writer already cached the same key
        with an identical output, this is a silent no-op. If it holds a different
        output, the original entry is left untouched and `CacheConflictError` is
        raised.
        """
        path = self._path_for(request)
        entry = _CacheEntry(request=request, response=response)
        data = entry.model_dump_json(indent=2).encode("utf-8")

        fd, tmp_name = tempfile.mkstemp(
            dir=str(self._dir), prefix=f".{path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(tmp_path, path)
                return
            except FileExistsError:
                pass
        finally:
            tmp_path.unlink(missing_ok=True)

        existing = self._read_entry(path)
        if existing.response.output == response.output:
            return
        raise CacheConflictError(f"{path}: cache key already holds a different payload")

    def _path_for(self, request: CacheRequest) -> Path:
        """Return the on-disk path for `request`'s cache entry."""
        return self._dir / f"{_cache_key(request)}.json"

    @staticmethod
    def _read_entry(path: Path) -> _CacheEntry:
        """Read and validate a cache entry from disk."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CacheReadError(f"{path}: invalid JSON ({exc})") from exc
        try:
            return _CacheEntry.model_validate(raw)
        except ValidationError as exc:
            raise CacheReadError(f"{path}: invalid cache entry ({exc})") from exc
