"""Progress events emitted by long-running eval phases.

The harness modules stay free of I/O: they emit structured events and the caller
decides whether and how to render them. A `None` callback is the default
everywhere, so library use and the existing tests observe no behavior change.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

ProgressPhase = Literal["preflight", "run", "case", "artifact"]


@dataclass(frozen=True)
class ProgressEvent:
    """One observable step of a run, carrying counts as data rather than text."""

    phase: ProgressPhase
    message: str
    completed: int | None = None
    total: int | None = None
    case_key: str | None = None
    policy: str | None = None
    elapsed_s: float | None = None


ProgressCallback = Callable[[ProgressEvent], None]


def emit(
    progress: ProgressCallback | None,
    phase: ProgressPhase,
    message: str,
    *,
    completed: int | None = None,
    total: int | None = None,
    case_key: str | None = None,
    policy: str | None = None,
    elapsed_s: float | None = None,
) -> None:
    """Deliver one event when a callback is installed, and do nothing otherwise.

    Centralizing the `None` check keeps every call site a single statement rather
    than a three-line conditional.
    """
    if progress is None:
        return
    progress(
        ProgressEvent(
            phase=phase,
            message=message,
            completed=completed,
            total=total,
            case_key=case_key,
            policy=policy,
            elapsed_s=elapsed_s,
        )
    )
