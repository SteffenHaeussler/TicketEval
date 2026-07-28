# M2-T4 Per-case Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one labelled case through the real workflow and return a complete immutable evaluation record and telemetry events.

**Architecture:** A run-scoped `CaseRunner` starts workflows, handles approval and deadlines, then constructs records from the workflow handle, status query, and public read-model observation. `RefundObservation` isolates SQLite details from the runner.

**Tech Stack:** Python 3.12, Pydantic 2, Temporal Python SDK, pytest/pytest-asyncio, SQLite.

## Global Constraints

- Keep M2-T4 limited to per-case execution; M2-T5 owns batch/profile orchestration and M2-T8 owns the cross-cutting workflow suite.
- Never read `TicketStatusInfo.result`; await the workflow handle for `TicketResult`.
- Treat rejected updates and case deadline cleanup as case outcomes, never run-fatal errors.
- The runner must use `get_refund_observation`, not raw SQLite.

---

### Task 1: Read-model refund observation

**Files:**
- Modify: `src/ticketflow/readmodel.py`
- Modify: `tests/test_readmodel.py`

**Interfaces:**
- Produces: `RefundObservation(executed_count: int, attempt_count: int)` and `get_refund_observation(ticket_id, db_path=None) -> RefundObservation`.

- [ ] Write tests for missing databases, tickets with no refunds, one executed refund, and duplicate refund attempts.
- [ ] Run `uv run pytest tests/test_readmodel.py -q` and confirm the new import/behavior fails before implementation.
- [ ] Add the frozen Pydantic observation model and one read-model helper that returns the two counts.
- [ ] Re-run `uv run pytest tests/test_readmodel.py -q` and confirm all tests pass.

### Task 2: Runner happy paths and approval race

**Files:**
- Create: `src/ticketflow/eval/runner.py`
- Create: `tests/eval/test_runner.py`

**Interfaces:**
- Consumes: `EvalCase`, reviewer `Callable[[CaseRecord], ApprovalDecision]`, `RuntimeIdentityMap`, `TelemetrySink`, Temporal `Client`.
- Produces: `CaseRunner.run_case(...) -> tuple[CaseRecord, list[CallEvent]]`.

- [ ] Write async tests proving an ungated case captures classification/draft/result, a gated oracle rejection records a rejected decision, and a rejected update becomes `update_rejected` without raising.
- [ ] Run `uv run pytest -m eval tests/eval/test_runner.py -q` and confirm collection fails because `ticketflow.eval.runner` is absent.
- [ ] Implement ID registration, workflow start, status polling, reviewer-input record creation, update handling, completion capture, public refund observation, and telemetry drain.
- [ ] Re-run `uv run pytest -m eval tests/eval/test_runner.py -q` and confirm the happy-path and update-race tests pass.

### Task 3: Deadline cleanup and record invariants

**Files:**
- Modify: `src/ticketflow/eval/runner.py`
- Modify: `tests/eval/test_runner.py`

**Interfaces:**
- Produces: `terminal_outcome="runner_deadline_exceeded"` with `cleanup_action="cancelled"|"terminated"`, preserving a captured draft.

- [ ] Write tests with a deliberately non-completing handle for cancellation confirmation, termination fallback, best-effort status capture, and post-draft `prediction_available=True`.
- [ ] Run the deadline tests and confirm they fail before deadline handling exists.
- [ ] Add wall-clock `asyncio.wait_for` handling, five-second cancellation confirmation, fallback termination, and status-query error capture.
- [ ] Re-run `uv run pytest -m eval tests/eval/test_runner.py -q` and confirm all focused runner tests pass.

### Task 4: Verify the owned surface

**Files:**
- Verify: `tests/test_readmodel.py`, `tests/eval/test_runner.py`, `tests/eval/test_harness.py`, `tests/eval/test_telemetry.py`

- [ ] Run the focused owned/dependency suite with `uv run pytest tests/test_readmodel.py -q && uv run pytest -m eval tests/eval/test_runner.py tests/eval/test_harness.py tests/eval/test_telemetry.py -q`.
- [ ] Run `uv run ruff check src/ticketflow/readmodel.py src/ticketflow/eval/runner.py tests/test_readmodel.py tests/eval/test_runner.py`.
- [ ] Run the full project suite with `uv run pytest`.
