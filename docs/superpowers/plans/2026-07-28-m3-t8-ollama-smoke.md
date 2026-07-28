# M3-T8 Ollama Smoke Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in real-Ollama smoke test that proves primary and fallback quality evaluations produce valid immutable artifacts without overlapping model residency.

**Architecture:** The test calls the existing `scripts/eval.py` command handler with isolated temporary output paths. It uses Ollama's admin endpoints only to unload a completed model and verify the running-model list before the next profile starts.

**Tech Stack:** pytest, httpx, existing eval CLI and artifact readers.

## Global Constraints

- The test must carry `pytest.mark.ollama` and remain excluded by the existing default pytest marker expression.
- Run three difficulty-balanced cases per profile with `--reviewer oracle`; do not invoke the judge.
- Use the configured Ollama endpoint and configured primary/fallback model names.
- Do not add production APIs or retain smoke artifacts outside pytest's temporary directory.

---

### Task 1: Real-Ollama profile smoke test

**Files:**
- Create: `tests/eval/test_ollama_smoke.py`
- Test: `tests/eval/test_ollama_smoke.py`

**Interfaces:**
- Consumes: `scripts.eval.main`, `RunManifest`, `read_case_records`, `read_call_events`, `read_json_artifact`, and `invalid_output_rate`.
- Produces: A `pytest.mark.ollama` smoke test runnable with `uv run pytest -m ollama -o addopts=`.

- [ ] **Step 1: Write the failing smoke test**

Create a marked test that redirects `scripts.eval.RUNS_DIR` and
`scripts.eval.DEFAULT_CACHE_DIR` to `tmp_path`, runs the primary profile, and
asserts its artifacts can be read.

- [ ] **Step 2: Run the test to verify it reaches the real-Ollama boundary**

Run: `uv run pytest tests/eval/test_ollama_smoke.py -m ollama -o addopts=`

Expected: FAIL before the test exists, then either execute against the configured
server or report that the local Ollama service/models are unavailable.

- [ ] **Step 3: Complete the smoke flow**

Run both quality profiles with three cases, reuse the first real preflight
result for the fallback run so preflight cannot reload the primary, verify the
two case-key sets are equal, verify artifacts and invalid-output diagnostics,
unload the primary through `/api/generate` with `keep_alive: 0`, and assert it
is absent through `/api/ps` immediately before fallback execution. On success,
unload and verify eviction of both models; on failure, clean up without masking
the original error.

- [ ] **Step 4: Run focused verification**

Run: `uv run pytest tests/eval/test_ollama_smoke.py -m ollama -o addopts=`

Expected: PASS when the configured Ollama service and models are available.

- [ ] **Step 5: Run repository checks**

Run: `uv run ruff format --check tests/eval/test_ollama_smoke.py && uv run ruff check tests/eval/test_ollama_smoke.py && uv run pyright && uv run pytest`

Expected: all commands exit 0; the default pytest run deselects the new Ollama test.
