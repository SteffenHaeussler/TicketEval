# Ticketflow Eval Harness — Execution Breakdown

**Status:** milestone 3 code complete; human dataset verification pending · **Date:** 2026-07-28 · **Companion to:** [`plan.md`](./plan.md)

## Purpose

`plan.md` says *what* the eval harness is and *why* each design choice was made. It is the
authority on semantics: correctness definitions, run profiles, statistical rules, risks. Nothing
here overrides it.

This document says *who does what, in which order, touching which files*. It keeps `plan.md`'s four
milestones intact and splits each into subtasks small enough to hand to one agent or one sitting.

**Task IDs** are `M<milestone>-T<n>` — `M2-T4` is the fourth task of milestone 2. Lettered variants
subdivide one numbered deliverable; their explicit dependencies and **Owns** entries determine
whether they run in parallel (`M1-T3a/b/c`) or in sequence (`M4-T4a/b`).

**Two ways to read this:**

- *Dispatching in parallel* — use the **Owns** column and the **DAG** for each milestone. Any two
  tasks in the same wave have disjoint file ownership and can run in separate workspaces.
- *Working alone* — use the **Serial order** line at the end of each milestone and ignore the
  waves.

---

## Working rules

These apply to every task and are not repeated in the task bodies.

1. **One task, one workspace branch; one integration PR per cohesive code wave.** Task branches
   start from the merged dependency wave. The wave integrator combines completed code-task commits,
   runs milestone-level checks, and opens one cohesive PR. Human-gated data lanes use separate PRs
   after their human evidence is complete, so they do not hold unrelated code. `M3-T1` may ship as
   its own early PR because it is an independently releasable production fix.
2. **A task may only edit files in its Owns list.** If you need a change in someone else's file,
   that is a signal the breakdown is wrong — raise it rather than reaching across. New files must
   be added to the ownership table before they are created.
3. **`make check` must be green before hand-off.** It runs `format-check`, `lint`, `typecheck`,
   `test`, and the pre-push hook runs it anyway.
4. **New modules under `src/` and `scripts/` need docstrings** (ruff `D100`–`D103`, `D107`) and
   88-column lines. `tests/**/*.py` has all `D` rules ignored — do not add docstrings there for
   ruff's sake.
5. **Each task owns its own unit tests.** The only test-only tasks are the deliberately
   cross-cutting suites (`M2-T8`).
6. **New tests live under `tests/eval/`**, which needs an `__init__.py` (mirroring the existing
   `tests/__init__.py`) so `from tests.helpers import ...` keeps working. Created in `M1-T1`.
7. **Do not change Temporal-crossing models.** `Ticket`, `TicketResult`, `TicketStatusInfo`, and the
   workflow signature stay as they are — `plan.md` lists this as a non-goal, and
   `tests/test_models.py` guards it.
8. **Everything is network-free by default.** Real-model work sits behind the `ollama` marker,
   deselected in `addopts`.
9. **Human assertions are never synthesized by agents.** Agents may prepare unverified dataset
   candidates and calibration templates, but only the named human owner may set
   `label_verified=true`, screen judge-calibration classes, enter either independent rating, or
   adjudicate them. Tasks with a human checkpoint remain blocked until that evidence exists.
   Agent-authored cases use `source="generated"` and record `generated_by`; only human-authored
   cases may use `source="handwritten"`.

---

## Structural refinements

These four refinements remove single-owner bottlenecks and are now reflected in `plan.md`.

| # | Refinement | Rationale |
|---|---|---|
| 1 | The dataset loader accepts a directory of `*.jsonl` shards under `evals/data/tickets/` | Human-owned data tasks can progress independently while duplicate-ID detection still runs across the union. |
| 2 | Prompts and schemas → `agent/prompts.py`, response cache → `eval/cache.py`, preflight → `eval/preflight.py` | Cache and preflight expose narrow interfaces and know nothing about HTTP. `agent/ollama.py` remains the only module that sends model requests, and its cache dependency is explicit. |
| 3 | Run profiles → `eval/profiles.py`; invariant checks → `eval/invariants.py` | The responsibilities have named homes instead of turning `runner.py` and `report.py` into multi-owner bottlenecks. |
| 4 | `eval/statistics.py` operates on primitive `(cluster_id, value)` sequences and does not import `eval/records.py` | Numerical code stays independently testable and no longer waits on record models. |

### One gotcha and one ambiguity

**Gotcha (superseded) — eval tests are marker-deselected from `make check`, not dependency-synced
into it.** An earlier revision of this plan assumed `make test`/`make check` would collect
`tests/eval/test_statistics.py` and therefore need `numpy`/`scipy` installed via
`uv sync --all-groups`. Instead, every test under `tests/eval/` is auto-tagged with a new `eval`
pytest marker (`tests/eval/conftest.py`), and `addopts` is `-m 'not smoke and not eval'` — the same
deselection mechanism already used for `smoke`. `make test`/`make check` therefore never collect
`tests/eval/`, whether or not the `eval` dependency group is installed. A dedicated
`make test-eval` (`uv run pytest -m eval -o addopts=`) runs them explicitly. `make install` does
not need `--all-groups` for `make check` to pass; it only matters for `make test-eval` and
`make eval*`. Resolved in `M1-T2`, not `M1-T1`.

**Ambiguity — where does the threshold sweep live?** `plan.md`'s architecture comment assigns it to
`statistics.py` in milestone 1; the milestone 4 deliverables list "Threshold sweep and confidence
diagnostics". Resolution used here: the pure sweep *function* lands in `M1-T6`, its *rendering and
preflight gating* land in `M4-T3`. Both readings are satisfied.

---

## Locked cross-task interfaces

These contracts are defined before parallel work starts. A task may add private details but may not
change these names or semantics without updating its dependants and this document.

- **Identity:** `EvalCase.id` is `case_key`. Runtime ticket and workflow IDs remain unique per run,
  policy, case, and repeat. `RuntimeIdentityMap` resolves runtime ticket ID → `case_key`; only
  `case_key` participates in tunable hashing, pairing, and cache identity.
- **Event join:** every `CallEvent` stores separate `run_id`, `policy`, `case_key`, `repeat_index`,
  and runtime `ticket_id`; together they join unambiguously to one `CaseRecord`.
- **Scoring:** `prediction_available` is exactly `draft is not None`.
  `terminal_outcome` independently records `resolved`, `rejected`, `escalated`,
  `update_rejected`, or `runner_deadline_exceeded`.
- **Workflow configuration:** `WorkflowEvalConfig` carries confidence threshold, primary and
  fallback queues, schedule-to-start seconds, activity-timeout seconds, and heartbeat-timeout
  seconds. The harness applies and snapshots all six values before workers start.
- **Cache:** `ResponseCache.get(CacheRequest) -> CachedAgentResponse | None` and
  `ResponseCache.put_success(CacheRequest, CachedAgentResponse) -> None`. `CacheRequest` includes
  stable case content and generation configuration but excludes runtime ticket ID.
- **Seeds:** `--seed` is the run seed and defaults to `0`; generation seeds derive from
  `SHA-256(run_seed, case_key, repeat_index)` and stay equal across reviewer policies.
  `--bootstrap-seed` is independent and defaults to `0`.
- **Dataset validation:** `load_cases(..., require_verified=False)` is the authoring/shard check;
  the default verified load plus `validate_dataset` is the release check. Default complete-dataset
  ranges are easy 30–50%, ambiguous 30–50%, adversarial 15–35%, and each `reference_category`
  15–35%. `reference_category` must belong to `acceptable_categories`; only it feeds category
  distribution, precision, recall, F1, and confusion-matrix cells.

---

## Milestone 1 — Evaluation contract and deterministic core

**Goal:** everything that can be proven correct without Temporal or Ollama — the dataset, the record
format, the metric definitions, the statistics, and the report.

| ID | Goal | Depends on | Owns |
|---|---|---|---|
| M1-T1 | Repo scaffolding | — | `pyproject.toml`, `.gitignore`, `Makefile`, `src/ticketflow/eval/__init__.py`, `src/ticketflow/eval/scorers/__init__.py`, `tests/eval/__init__.py` |
| M1-T2 | Dataset models and loader | T1 | `src/ticketflow/eval/dataset.py`, `tests/eval/test_dataset.py` |
| M1-T3a | Labelling guide + 20 easy cases | T2 | `evals/data/labeling.md`, `evals/data/tickets/easy.jsonl` |
| M1-T3b | 20 ambiguous cases | T2 | `evals/data/tickets/ambiguous.jsonl` |
| M1-T3c | 10 adversarial cases | T2 | `evals/data/tickets/adversarial.jsonl` |
| M1-T4 | Immutable record models | T1 | `src/ticketflow/eval/records.py`, `tests/eval/test_records.py` |
| M1-T5 | Deterministic scorers | T4 | `src/ticketflow/eval/scorers/deterministic.py`, `tests/eval/test_deterministic_scorers.py` |
| M1-T6 | Statistics primitives | T1 | `src/ticketflow/eval/statistics.py`, `tests/eval/test_statistics.py` |
| M1-T7 | Report renderer | T5, T6 | `src/ticketflow/eval/report.py`, `tests/eval/test_report.py` |
| M1-T8 | `dataset-check` CLI | T2 | `scripts/eval.py`, `Makefile`, `tests/eval/test_eval_cli.py` |

> `Makefile` is owned by both T1 and T8. T1 touches only the `install` target and T8 only adds
> `eval-dataset-check`, but they must still be serialised — T8 depends on T2 which depends on T1,
> so the edge already exists.

### M1-T1 — Repo scaffolding

Prepare the repository for eval work so no later task has to touch shared config. Add an `eval`
dependency group holding `numpy` and `scipy`, and switch `make install` to `uv sync --all-groups` so
statistics tests have `numpy`/`scipy` available under `make test-eval` — they are never collected by
`make check` at all (see below), so this is about the eval dependency group being installed, not
about `make check` needing it. Register the `ollama` pytest marker and widen `addopts` from
`-m 'not smoke and not eval'` to `-m 'not smoke and not eval and not ollama'` — registering the
marker alone is not enough, the tests would still run. (`M1-T2` already introduced the `eval` marker
and the `tests/eval/conftest.py` auto-tagging hook, deselecting every test under `tests/eval/` from
the default `make test`/`make check` gate; T1 only adds the `ollama` deselection on top of it.) Add
`evals/runs/` and `evals/cache/` to `.gitignore`, leaving `evals/data/` committed. Create the empty
`eval/` and `eval/scorers/` packages (`tests/eval/__init__.py` and `tests/eval/conftest.py` already
exist from `M1-T2`).

**Acceptance**

- `make install && make check` passes on a clean checkout.
- A test marked `ollama` is not collected by a bare `uv run pytest`, and is collected by
  `uv run pytest -o addopts=`.
- `git status` stays clean after writing a file into `evals/runs/`.
- The runtime `dependencies` list in `pyproject.toml` is unchanged.

### M1-T2 — Dataset models and loader

Implement `ExpectedOutcome` and `EvalCase` exactly as specified in `plan.md`, including named
verification metadata. `load_cases` reads either a single `.jsonl` file or a directory of shards and
performs structural validation; `require_verified=False` permits an agent to validate draft shards
without asserting human approval. `validate_dataset` separately applies the complete-union
difficulty and category ranges from the locked interface. Refund-token parsing uses the exact
currency and decimal rules in `plan.md`. Every rejection names the offending case ID and shard.

**Acceptance**

- Each structural and distribution rejection rule has a test that asserts on the error message, not
  just the exception type.
- Two shards sharing a case ID are rejected.
- A directory and an equivalent single file load to identical case lists.
- Each individual difficulty shard passes structural authoring validation without whole-dataset
  balance checks.
- Missing authors, self-verification, and a `generated_by` value on handwritten cases are rejected
  with the offending ID.
- A reference category outside the acceptable set is rejected.
- The default whole-dataset percentage ranges are asserted exactly.

### M1-T3a — Labelling guide and easy cases

Write `evals/data/labeling.md` documenting the labelling policy from `plan.md` — when a refund is
expected, how multiple acceptable outcomes are recorded, what verification means — with worked
positive, negative, ambiguous, and adversarial examples. Prepare 20 `difficulty="easy"` candidates
whose category and action are unambiguous, keeping `source` and generator provenance accurate. Then
pause for a named human to review every case, supply `verified_by` and `verified_at`, and change
`label_verified` to true. Agents may validate and format the completed shard but may not assert
human verification or label generated text as `source="handwritten"`.

**Acceptance**

- The candidate shard passes `load_cases(easy_path, require_verified=False)` before review.
- After the human checkpoint, all 20 cases have distinct `authored_by` and `verified_by`,
  `verified_at`, `label_verified=true`, and truthful source/generator provenance.
- Categories are spread across all four `TicketCategory` values.
- Every guide rule is illustrated by at least one committed case, referenced by ID.

### M1-T3b — Ambiguous cases

Prepare 20 `difficulty="ambiguous"` candidates — tickets where a competent agent could reasonably
choose more than one category or action, recorded with multiple acceptable values rather than one
arbitrary "right" answer. These are the cases that make `gate_recall` meaningful, so bias them
toward outcomes near the confidence threshold rather than toward exotic phrasing.

**Acceptance**

- At least 12 cases carry more than one acceptable category or action.
- Each case's `notes` field explains *why* it is ambiguous.
- No case is ambiguous merely because it is badly written; ambiguity is about the support decision.
- A named human distinct from the recorded author verifies every case before the task completes;
  agents only prepare, validate, and format the shard.

### M1-T3c — Adversarial cases

Prepare 10 `difficulty="adversarial"` candidates: prompt-injection attempts, refund requests with no
stated amount, refund requests for amounts not in the ticket, contradictory instructions, and
tickets that describe a refund without requesting one. These exist to probe the refund rule and the
gate, not to be unanswerable.

**Acceptance**

- At least three cases request a refund in a way that must *not* produce a `REFUND` action.
- At least one case attempts to instruct the agent directly.
- Each case's `notes` states the failure mode it targets.
- A named human distinct from the recorded author verifies every case before the task completes;
  agents only prepare, validate, and format the shard.

### M1-T4 — Immutable record models

Implement `CaseRecord`, `CallEvent`, and `RunManifest` per `plan.md`'s "Run artifacts and
reproducibility" and "Telemetry" sections, plus JSONL read/write helpers that write atomically
(temp file, then rename) and refuse to overwrite an existing raw artifact.
`CaseRecord.prediction_available` is derived solely from draft presence. Keep execution state
separate in the closed `terminal_outcome` set and record deadline cleanup independently, so
post-draft update failures and deadlines remain structurally scorable.

**Acceptance**

- Writing to an existing `records.jsonl` or `calls.jsonl` path raises rather than truncating.
- A partially written file is never left behind when the writer is interrupted.
- Round-tripping a record through JSONL preserves every field, including enum types.
- Records with the same draft but different terminal outcomes have the same
  `prediction_available=true`.
- `RunManifest` has a field for each bullet in `plan.md`'s manifest list; fields not yet knowable
  (model digests, preflight measurements) are optional and land in `M3-T6`.

### M1-T5 — Deterministic scorers

Implement the four correctness predicates (`category_correct`, `action_correct`, `refund_correct`,
`structured_correct`) and the metric set from `plan.md`'s "Correctness definitions". The central
rule: quality rates are computed over the **scored population** — cases that produced a draft —
while `escalation_rate` and `invalid_output_rate` are computed over **all** cases and never folded
in. Every returned metric carries its denominator as data, not as a comment, so the report cannot
mislabel it. Include `gate_recall`, `gate_precision`, the category confusion matrix, action
accuracy, and refund-amount error.

Category correctness remains set-valued, while category distribution and per-class metrics use the
single locked `reference_category`; tests cover a prediction accepted by the set that differs from
the reference category.

**Acceptance**

- A synthetic record set with hand-computed metrics reproduces them exactly.
- Cases without a draft are absent from quality denominators. A repaired case with a draft remains
  scored even when one or more attempts were invalid.
- `invalid_output_rate` counts cases with at least one invalid-output event over all cases; it does
  not count invalid attempts or only exhausted repairs.
- `gate_recall` and `gate_precision` are byte-identical when computed over oracle records and over
  rubber-stamp records built from the same agent outputs.
- No metric can be constructed without a denominator label.

### M1-T6 — Statistics primitives

Implement case-clustered percentile bootstrap intervals (5,000 resamples, seeded by
`bootstrap_seed`, 95%), paired effect size with interval, the exact McNemar test, and the
threshold-sweep function.
Per structural refinement 4, this module takes primitive sequences — `(case_id, value)` pairs — and does not
import `records.py`. Resampling draws *cases* and keeps all repeats of each drawn case, so repeats
never inflate apparent precision.

**Acceptance**

- A dataset with 10 cases × 10 identical repeats yields the same interval width as 10 cases × 1
  repeat; naive per-observation resampling would give a narrower one, and a test asserts it does not.
- The same seed reproduces the same interval bit-for-bit.
- The exact McNemar result matches a hand-computed binomial value on a small table.
- The sweep returns review load and unreviewed error rate for each threshold from 0.00 to 1.00 in
  0.05 steps, plus the count of cases excluded for having no draft.
- `import ticketflow.eval.statistics` does not import `ticketflow.eval.records` (asserted).

### M1-T7 — Report renderer

Render the deterministic metrics to a Markdown string that can be saved or printed directly to a
console. Every rate prints with its interval and its denominator name. Escalation and invalid-output
rates render in a visually separate block from quality metrics. A run whose scored population is
less than 100 is labelled *directional only* in the header, and the scored population size prints
beside the total with a per-reason exclusion breakdown.

**Acceptance**

- A golden-file test pins the Markdown for a fixed synthetic record set.
- No rate can render without an interval and a denominator.
- The directional-only banner appears at scored N=99 and disappears at scored N=100.
- Threshold-sweep and judge sections are absent — they arrive in M4.

### M1-T8 — `dataset-check` CLI

Create `scripts/eval.py` with a subcommand structure that later milestones extend. `dataset-check`
defaults to verified structural plus whole-dataset validation; `--shard PATH` performs structural
validation only, and `--allow-unverified` sets `require_verified=false` for authoring. Print
composition by difficulty, source, and category and exit non-zero with an actionable message on any
validation failure. Add `make eval-dataset-check`.

**Acceptance**

- `make eval-dataset-check` exits 0 on the committed dataset and non-zero on a deliberately broken
  shard, printing the offending case ID.
- Before the human-reviewed dataset PR lands, equivalent fixture-based CLI tests pass; the wave
  integrator runs the committed-dataset command when closing milestone 1.
- The subcommand structure accommodates `run`, `judge`, `report`, and `compare` without rework.
- The script satisfies ruff's docstring rules (`scripts/` is not exempt).

**DAG**

```
T1 ──┬─> T2 ──┬─> T3a
     │        ├─> T3b
     │        ├─> T3c
     │        └─> T8
     ├─> T4 ──> T5 ──┐
     └─> T6 ─────────┴─> T7
```

**Waves:** `[T1]` → `[T2, T4, T6]` → code lane `[T5, T8]`, human-data lane
`[T3a, T3b, T3c]` → `[T7]`. Milestone 1 closes only after both lanes merge.

**Serial order:** T1, T2, T8, T3a, T3b, T3c, T4, T5, T6, T7. Doing T8 before the authoring tasks
gives you `dataset-check` as a validation loop while writing cases.

---

## Milestone 2 — Workflow harness and tunable agents

**Goal:** run the real `TicketWorkflow` over the dataset in process, with an agent whose errors you
chose in advance, so the harness itself can be proven correct before any real model is involved.

| ID | Goal | Depends on | Owns |
|---|---|---|---|
| M2-T1 | Tunable mock agent | M1-T2, T6 | `src/ticketflow/agent/tunable.py`, `tests/eval/test_tunable_agent.py` |
| M2-T2 | Temporal harness | M1-T1 | `src/ticketflow/eval/harness.py`, `tests/helpers.py`, `tests/eval/test_harness.py` |
| M2-T3 | Reviewer policies | M1-T5 | `src/ticketflow/eval/reviewers.py`, `tests/eval/test_reviewers.py` |
| M2-T4 | Per-case runner | T2, T3, T6, M1-T4 | `src/ticketflow/eval/runner.py`, `tests/eval/test_runner.py`, `src/ticketflow/readmodel.py`, `tests/test_readmodel.py` |
| M2-T5 | Run profiles and manifest | T1, T4, T6 | `src/ticketflow/eval/profiles.py`, `tests/eval/test_profiles.py` |
| M2-T6 | Identity, telemetry, and invariants | M1-T4 | `src/ticketflow/eval/telemetry.py`, `src/ticketflow/eval/invariants.py`, `tests/eval/test_telemetry.py`, `tests/eval/test_invariants.py` |
| M2-T7 | `run` CLI subcommand | T5 | `scripts/eval.py`, `Makefile`, `tests/eval/test_eval_cli.py` |
| M2-T8 | Cross-cutting workflow suite | T1, T5, T6 | `tests/eval/test_runner_workflow.py` |

### M2-T1 — Tunable mock agent

Build `TunableMockAgent`, which resolves runtime ticket IDs through `RuntimeIdentityMap` and derives
every decision from `(generation_seed, case_key, operation)` rather than from a shared mutable RNG.
It takes the expected-outcome map keyed by `case_key` and starts from the correct answer, perturbing
it according to its profile: category, action, and refund-amount error rates or exact error ID sets;
confidence calibration and overconfidence; transient failure rate or exact failure ID set; and the
`primary`/`fallback` role label. It emits attempts through the `M2-T6` telemetry sink.

**Acceptance**

- The same seed and case produce the same output regardless of concurrency, case ordering, or how
  many other cases ran first — asserted by running a case set forwards, backwards, and with
  concurrency 8, and comparing.
- Different runtime ticket IDs for oracle and rubber-stamp resolve to the same `case_key` and
  produce byte-identical outputs for the same repeat.
- An exact error ID set produces exactly those errors and no others, so downstream metric tests are
  not probabilistic.
- The role label reaches `Classification.model` and `DraftReply.model` so `_model_path()` reports
  it.
- The agent satisfies the existing `Agent` protocol without changing it.

### M2-T2 — Temporal harness

Move reusable worker construction out of `tests/helpers.py` into `eval/harness.py`, and have
`tests/helpers.py` re-export it so the existing workflow tests are untouched. The harness must do
five things the current helper cannot: host **independent primary and fallback agents on separate
queues** (today `make_worker` hosts one agent queue); register the `TicketStatus` keyword search
attribute for both the time-skipping and the local server, since every status transition upserts it
and the workflow task fails otherwise; use `pydantic_data_converter` on the client; patch and
snapshot the workflow module constants (`CONFIDENCE_THRESHOLD`, `AGENT_TASK_QUEUE`,
`FALLBACK_TASK_QUEUE`, `AGENT_SCHEDULE_TO_START_S`, `AGENT_ACTIVITY_TIMEOUT`, and
`AGENT_HEARTBEAT_TIMEOUT`) before workers start, using
`UnsandboxedWorkflowRunner` so the patch is visible inside the sandbox; and scope the worker
lifecycle to the **run**, not the case, because post-completion queries are served by replaying
history on a live worker.

**Acceptance**

- `tests/test_workflow.py` and the rest of the existing suite pass unmodified.
- A workflow can be queried after it has completed, with workers still up.
- Primary and fallback agents can be different objects on different queues in the same run.
- The snapshot of workflow constants is returned as data for the manifest, not just applied.
- A preflight-adjusted activity timeout reaches both primary and fallback activity options.
- Both time-skipping and local-server modes are exercised.

### M2-T3 — Reviewer policies

Implement the two reviewer policies: `oracle` approves exactly when the structured outcome is
correct, `rubber_stamp` approves every gated draft. Both produce an `ApprovalDecision` with a
distinguishable approver string so records can be attributed. Keep them pure functions of
(record-so-far, expected outcome) — no Temporal, no I/O.

**Acceptance**

- Oracle rejects an incorrect outcome and approves a correct one, including the refund-tolerance
  case.
- Rubber-stamp approves regardless of correctness.
- Neither policy can observe anything the real reviewer would not, apart from the oracle's
  by-design access to the label.

### M2-T4 — Per-case runner

Implement the eight-step per-case loop from `plan.md`'s "Workflow runner". Three details carry most
of the risk. Approval is a workflow **update** with a validator, so the runner must wait for
`AWAITING_APPROVAL` before calling `execute_update`, and a lost race raises
`WorkflowUpdateFailedError` — recorded as a case outcome, never fatal to the run. The fallback
activity is dispatched with no schedule-to-start timeout and the workflow has no run timeout, so a
missing fallback worker parks forever; the runner imposes its own wall-clock deadline per case and
records `runner_deadline_exceeded`, requests cancellation, waits five seconds, and terminates if
cancellation is not confirmed. Post-completion state comes from the `status` query, whose `result`
field the workflow never populates and which must not be read — the terminal `TicketResult` comes
from awaiting the handle. Add `RefundObservation` and
`get_refund_observation(ticket_id, db_path=None) -> RefundObservation` to the read model; the runner
does not issue raw SQLite queries. Finish by emitting one `CaseRecord` plus its `CallEvent`s.

**Acceptance**

- A rejected approval update is recorded as a case outcome and the run continues.
- A case that exceeds its deadline records the best-effort status, is cancelled or terminated,
  records the cleanup action, and does not remain open after the run.
- A post-draft deadline remains `prediction_available=true`.
- Classification, draft, and decision are captured for every completed case.
- Refund observation uses tested public read-model helpers.
- Nothing reads `TicketStatusInfo.result`.
- Bounded concurrency is configurable and results do not depend on it.

### M2-T5 — Run profiles and manifest assembly

Implement the four run profiles and the manifest they produce. `primary-quality` and
`fallback-quality` both back the primary agent queue directly — the second with the fallback model —
so they measure model quality and are paired by case ID. `fallback-routing` is a *different
experiment*: it hosts a worker only on the fallback queue, withholds the primary worker, exercises
the real schedule-to-start mechanism over a small subset, and its results are excluded from
model-quality headlines. `reliability` runs with oracle only, cache disabled, and full
call telemetry. Assemble the manifest with the git commit and dirty state, dataset SHA-256, workflow
constant snapshot, reviewer policies and their order, run seed, bootstrap seed, derived-generation
seed rule, concurrency, and repeats.

**Schedule-to-start is not time-skippable.** It is enforced by the matching service, not by a
workflow timer the client can fast-forward, so the test server cannot skip it — `tests/test_workflow.py`
already proves the same mechanism by configuring a small timeout. Runs therefore execute with auto
time skipping *disabled* (it also races the server's global unlock counter under concurrency), and
`fallback-routing` shortens `AGENT_SCHEDULE_TO_START_S` via `--schedule-to-start` instead.

**Acceptance**

- A fallback-routing run completes in wall-clock time far below the default
  `AGENT_SCHEDULE_TO_START_S` per case, by shortening the configured timeout; the manifest records
  the shortened value that was actually in force.
- Routing records identify the fallback path and are excluded from quality comparisons.
- The manifest records dirty state truthfully on a dirty tree.
- `--repeats > 1` is rejected with the cache enabled.
- A fixed run seed derives distinct repeat seeds while remaining identical across reviewer policies.

### M2-T6 — Identity, telemetry, and invariants

Provide `RuntimeIdentityMap.register(ticket_id, case_key)`/`resolve(ticket_id)` and a process-local
`CallEvent` sink keyed by runtime ticket ID and operation. Agents write attempts to it and the
runner drains them into events containing separate `run_id`, `policy`, `case_key`, `repeat_index`,
and `ticket_id`. Both services are concurrency-safe and run-scoped; no parser infers metadata from
delimiters inside a ticket ID. Separately, implement the system-invariant checks from
`plan.md`'s reporting section: gating agrees with the recorded threshold and refund rule, at most
one refund row per ticket, refund attempts ≥ executed refunds, an executed refund implies an
approved decision, and a fallback-routing record identifies the fallback path.

**Acceptance**

- Events from concurrent cases are attributed to the right case.
- Repeats and reviewer policies produce distinct event join keys without changing `case_key`.
- Oracle and rubber-stamp runtime IDs resolve to the same stable `case_key`.
- Each invariant has a test that constructs a violating record set and asserts it is flagged.
- Violations are reported as data, not raised — a violated invariant is a finding, not a crash.
- `model_path` is not used as a retry counter anywhere.

### M2-T7 — `run` CLI subcommand

Extend `scripts/eval.py` with `run --profile primary-quality|fallback-quality|fallback-routing|
reliability` and the option set from `plan.md`: `--agent`, `--reviewer`, `--limit`, `--repeats`,
`--concurrency`, `--seed`, `--no-cache`. Enforce that `--repeats > 1` implies `--no-cache` and a
different derived generation seed per repeat. Add `--bootstrap-seed`, defaulting to `0`, and a
`make eval` target for the fast tunable profile. Also add `--case-deadline` and
`--schedule-to-start`, which `fallback-routing` needs to finish in reasonable wall-clock time.

`--reviewer` selects which of the profile's policies to run; a selection must be a subset of what
the profile allows, so `--reviewer oracle` legally narrows a quality profile but `--reviewer both`
is still rejected for routing and reliability. `--limit` samples across difficulties rather than
head-slicing, since shards load alphabetically and a head slice would return only adversarial cases.

**Acceptance**

- `make eval` completes a full tunable run over the committed dataset with no Temporal server or
  Ollama installed beyond the test server, asserted end to end without stubbing `run_profile`.
- Invalid option combinations fail before any workflow starts, with a message explaining why.
- CLI option parsing and validation are covered in `tests/eval/test_eval_cli.py`.
- Run artifacts land under `evals/runs/<run_id>/` and are gitignored.
- The run writes `invariants.json` beside the raw artifacts and prints the violation count; a
  violated invariant never changes the exit status.

### M2-T8 — Cross-cutting workflow suite

The integration suite that spans harness, runner, and profiles, and therefore belongs to no single
implementation task: ungated resolution, oracle rejection, rubber-stamp approval, forced fallback
quality versus real schedule-to-start routing, a rejected approval update, an exceeded per-case
deadline, retry telemetry, refund idempotency, and concurrency-independence of results.

**Acceptance**

- A configured exact error set produces the exact expected workflow-level error rates — no
  tolerance bands.
- The same run at concurrency 1 and concurrency 8 produces identical records modulo timing fields.
- Forced fallback quality and real fallback routing are shown to be distinguishable in the records.
- The whole suite runs network-free and is collected by default.

**DAG**

```
M1 ──┬────────> T2 ──┐
     ├────────> T3 ──┤
     └─> T6 ──┬──────┴─> T4 ──┐
              └─> T1 ─────────┴─> T5 ──┬─> T7
                                       └─> T8
```

**Waves:** `[T2, T3, T6]` → `[T1, T4]` → `[T5]` → `[T7, T8]`

**Serial order:** T2, T3, T6, T1, T4, T5, T8, T7.

---

## Milestone 3 — Ollama integration

**Goal:** replace the tunable agent with a real one, without letting slow inference or malformed
output masquerade as a quality signal.

| ID | Goal | Depends on | Owns |
|---|---|---|---|
| M3-T1 | Periodic activity heartbeats *(production change)* | — | `src/ticketflow/activities.py`, `tests/test_activities.py`, `docs/context.md` |
| M3-T2 | Prompts and schemas | M1-T1 | `src/ticketflow/agent/prompts.py`, `tests/eval/test_prompts.py` |
| M3-T3 | Ollama agent | T2, T4, M2-T6 | `src/ticketflow/agent/ollama.py`, `tests/test_ollama_agent.py` |
| M3-T4 | Response cache | T2 | `src/ticketflow/eval/cache.py`, `tests/eval/test_cache.py` |
| M3-T5 | Preflight | T3 | `src/ticketflow/eval/preflight.py`, `tests/eval/test_preflight.py` |
| M3-T6 | Manifest enrichment | T3, T5 | `src/ticketflow/eval/records.py`, `src/ticketflow/eval/profiles.py` |
| M3-T7 | Production factory and Ollama CLI wiring | T3, T5, T6 | `src/ticketflow/agent/factory.py`, `src/ticketflow/config.py`, `src/ticketflow/llm_worker.py`, `tests/test_config.py`, `tests/test_llm_worker.py`, `scripts/eval.py`, `Makefile` |
| M3-T8 | Local smoke run | T1, T7 | `tests/eval/test_ollama_smoke.py` |

### M3-T1 — Periodic activity heartbeats

The one production change in scope, and already an accepted follow-up in `docs/context.md`. Both
agent activities currently heartbeat only immediately before and after the agent call, so with
`heartbeat_timeout=30s` any real LLM call slower than 30 seconds is killed mid-flight. Wrap the
agent call in a reusable background heartbeat helper whose production interval is 10 seconds and
whose test interval is injectable. Cancel and await the heartbeat task when the call returns or
raises. Close the follow-up entry in the decision log.

**Acceptance**

- A unit test using a 10-millisecond heartbeat interval observes multiple heartbeats during a short
  artificial call; no default test sleeps for 30 seconds.
- The local Ollama smoke run supplies the real-time long-call integration check.
- Heartbeat cancellation happens on the error path too, verified for both agent error classes.
- This task has no dependencies and can be pulled forward to run alongside milestone 1.

### M3-T2 — Prompts and schemas

Author the operation-specific prompts and JSON output schemas for classification and drafting, plus
stable version hashes over each. The schema is supplied to Ollama through `format` *and* restated in
the grounding prompt. Output-only response models omit the internal `model` field — the role label
is assigned by application code after Pydantic validation, never trusted from the model.

**Acceptance**

- Changing a prompt or schema changes its hash; reordering keys in a schema does not.
- The classification and drafting schemas are distinct and neither can validate the other's output.
- The output models have no `model` field.

### M3-T3 — Ollama agent

Implement the real agent over a shared, lifecycle-managed `httpx.AsyncClient` calling
`POST /api/chat` with `stream=false`, `think=false`, `temperature=0`, a configurable seed and
timeout, and the schema in `format`. It consumes `M3-T4`'s exact `ResponseCache` interface: look up
before HTTP, emit cache-hit telemetry on a hit, and store only a successfully validated response.
Map errors deliberately: connection failures, HTTP timeouts,
408, 429, and 5xx become `AgentOverloadedError`; 400 and model-not-found become
`AgentPermanentError`. `AgentOverloadedError` is left unmapped at the activity layer on purpose so
it escapes as a generic retryable activity failure into the workflow's own retry loop — it must not
be "fixed" into a non-retryable `ApplicationError`.

The subtlest requirement is the **repair budget**. `AgentPermanentError` becomes a non-retryable
`ApplicationError`, which escalates the ticket with no retry and no fallback. Mapping
schema-validation failures straight onto that path would report an escalation rate where the harness
means to report degraded classification quality — and a 1.5B fallback model will emit refunds
without amounts routinely, since `ProposedAction` requires a positive amount. So a schema-invalid
response is re-asked once with the validation error appended, both attempts are recorded, the
outcome `invalid_output` is a first-class `CallEvent` outcome, and only an exhausted budget becomes
`AgentPermanentError`. Failures are never cached.

**Acceptance**

- Stub-transport tests cover success, malformed output, timeout, 400, 404, 408, 429, and 5xx, each
  asserting the resulting error class.
- A schema-invalid response is re-asked exactly once; a second failure raises
  `AgentPermanentError`; a refund missing its amount does *not* escalate on the first attempt.
- `invalid_output` events reach the telemetry sink with both attempt numbers.
- The second reviewer policy is 100% cache hits and receives byte-identical outputs for the same
  case and repeat despite different runtime ticket IDs.
- Failed and invalid-output calls never invoke `put_success`.
- No stub test performs real network I/O, and the suite is collected by default. This is
  why the tests live at `tests/test_ollama_agent.py` rather than under `tests/eval/`,
  which `tests/eval/conftest.py` auto-marks `eval` and the default `addopts` deselects.

### M3-T4 — Response cache

Implement the locked `ResponseCache`, `CacheRequest`, and `CachedAgentResponse` interface. The key
hashes operation, model name and digest, role, stable `case_key`, customer email, subject, body, the
classification input when drafting, messages and prompt version, JSON schema, think setting,
temperature, derived generation seed, generation options, and the Ollama version. It explicitly
excludes runtime ticket ID. Entries store request metadata for inspection and are written
atomically.

**Acceptance**

- A test per key component proves that changing it alone invalidates the entry.
- The cache exposes no generic write method: only `put_success` exists.
- Requests differing only in runtime ticket ID have identical keys; requests differing in
  `case_key` do not.
- Atomic writes never expose a partial entry and never silently overwrite a different payload.

### M3-T5 — Preflight

Before any real-model run: check `/api/version`, confirm every required model exists and record its
digest, run one unmeasured warm-up, then measure at least three classifications and three drafts,
separating model-load from generation time. Return a new `WorkflowEvalConfig` whose activity timeout
is `max(10 seconds, configured, 3 * slowest_stage)` and an HTTP timeout below it by
`max(5 seconds, 10% of the activity timeout)`; record every adjustment. Then probe ~10 cases for
confidence variance and decide whether
the threshold sweep is admissible: it requires **both** a standard deviation of at least 0.02 **and**
at least five distinct values. Standard deviation alone is too weak — self-reported LLM confidence
typically clusters on two or three values like 0.9 and 0.95, which passes a variance threshold while
producing a step function with no interior operating points.

**Acceptance**

- A synthetic confidence distribution of `{0.9, 0.95}` fails the distinctness gate despite passing
  variance, and the report names which gate failed.
- Timeout widening is computed from measurements and recorded, not hardcoded.
- The adjusted timeout reaches both primary and fallback workflow activity options through the
  harness configuration interface.
- A degenerate confidence distribution is reported as a finding, not an error.
- A missing model fails preflight with the model name in the message, before any case runs.

### M3-T6 — Manifest enrichment

Fill in the manifest fields that only become knowable once a real model is involved: model names and
digests, Ollama, Python, and relevant dependency versions, prompt and schema hashes, generation
options, preflight measurements, and every timeout adjustment. Enforce raw-artifact immutability
end-to-end — `records.jsonl` and `calls.jsonl` are written once and never rewritten; re-judging
creates `judgments/<rubric_hash>-<judgment_id>.jsonl` and never overwrites a prior judgment.

**Acceptance**

- Every bullet in `plan.md`'s `RunManifest` list has a populated field after a real run.
- Attempting to re-run into an existing run directory fails rather than overwriting.
- A report can be traced back to its source record and manifest hashes.

### M3-T7 — Production factory and Ollama CLI wiring

Add `build_agent(role, settings, cache=None, event_sink=None)` and production settings for
`TICKETFLOW_AGENT_BACKEND=mock|ollama` plus the exact Ollama environment keys and defaults in
`plan.md`. Update `llm_worker.py` to construct both roles through the factory. Extend
`scripts/eval.py` with `--agent ollama`, model and endpoint overrides, and `--no-cache` (mandatory
for reliability). Add `make eval-ollama`.

**Acceptance**

- Primary and fallback quality profiles can be launched over the same limited case set from the CLI.
- The reliability profile refuses to run with the cache enabled.
- Model overrides reach the manifest.
- The production worker defaults to the existing mock behavior and constructs Ollama agents when
  configured; an invalid backend fails before workers start.
- The production worker passes no eval cache or telemetry sink; eval profiles inject both.

### M3-T8 — Local smoke run

A small end-to-end run against real Ollama, marked `ollama` and deselected by default. It runs the
primary and fallback quality profiles over the same handful of cases and asserts the artifacts are
well-formed. It unloads the primary with `keep_alive=0` before the fallback phase and verifies model
residency. The judge is not run in this milestone; `M4-T5` enforces the primary-versus-judge
residency rule.

**Acceptance**

- Not collected by a bare `uv run pytest`; collected by `uv run pytest -m ollama -o addopts=`.
- Any observed real-model `invalid_output_rate` is reported diagnostically; a non-zero value is not
  required from the small sample. Deterministic stub tests carry the repair-budget assertion.
- Model residency is asserted, not assumed, between phases.
- Requires `M3-T1`; without heartbeats a real call can be killed at 30 seconds.

**DAG**

```
T1 ────────────────────────────────────────────────────────┐
T2 ──> T4 ──> T3 ──> T5 ──> T6 ──> T7 ──────────────────┴─> T8
```

**Waves:** `[T1, T2]` → `[T4]` → `[T3]` → `[T5]` → `[T6]` → `[T7]` → `[T8]`

**Serial order:** T1, T2, T4, T3, T5, T6, T7, T8.

---

## Milestone 4 — Decision-grade evaluation

**Goal:** enough data and enough validated inference to support or reject a real decision about the
0.75 confidence threshold and about fallback quality.

| ID | Goal | Depends on | Owns |
|---|---|---|---|
| M4-T1a | Grow easy cases to 60 | M1-T3a | `evals/data/tickets/easy.jsonl` |
| M4-T1b | Grow ambiguous cases to 80 | M1-T3b | `evals/data/tickets/ambiguous.jsonl` |
| M4-T1c | Grow adversarial cases to 60 | M1-T3c | `evals/data/tickets/adversarial.jsonl` |
| M4-T2 | Paired comparison and `compare` | M1-T6 | `src/ticketflow/eval/compare.py`, `tests/eval/test_compare.py`, `scripts/eval.py`, `Makefile` |
| M4-T3 | Threshold sweep and confidence diagnostics | M1-T6, M3-T5 | `src/ticketflow/eval/report.py`, `tests/eval/test_report.py` |
| M4-T4a | Calibration source runs and bundle | M3-T8 | `evals/data/judge_calibration_sources/` |
| M4-T4b | Human judge calibration set | T4a | `evals/data/judge_calibration.jsonl`, `evals/data/judging.md` |
| M4-T5 | Offline judge | M3-T3, T2, T4b | `src/ticketflow/eval/scorers/judge.py`, `tests/eval/test_judge.py`, `scripts/eval.py`, `evals/data/judge_calibration_judgments/` |
| M4-T6 | Calibration gates | M4-T4b, T5 | `src/ticketflow/eval/scorers/calibration.py`, `tests/eval/test_calibration.py` |
| M4-T7 | Final report and `report` CLI | T1a, T1b, T1c, T2, T3, T5, T6 | `src/ticketflow/eval/report.py`, `tests/eval/test_report.py`, `scripts/eval.py` |

> `report.py` and its test are owned by both T3 and T7; `scripts/eval.py` is owned by T2, T5, and
> T7. The dependency table serialises every shared file explicitly.

### M4-T1a/b/c — Dataset growth to ~200 verified cases

Draft each difficulty shard toward the milestone 4 target: 60 easy, 80 ambiguous, 60 adversarial.
Generated cases record `generated_by` and stay `label_verified=false` until a non-author human
supplies `verified_by` and `verified_at`. Agents stop at that checkpoint and never flip the flag.
After human verification, generated cases remain separately sliced in reports. Three tasks, three
shards, no shared file.

**Acceptance**

- `make eval-dataset-check` passes on the union, including cross-shard duplicate-ID detection.
- Before human review, authoring-mode validation passes with unverified cases.
- After the human checkpoint, every case has named verification and every generated case has
  `generated_by` set.
- The ambiguous tier remains the largest, matching `plan.md`'s target distribution.
- The fixed difficulty and category percentage ranges pass without override.

### M4-T2 — Paired comparison and `compare`

Report paired effect sizes with bootstrap intervals for runs over identical case IDs, with the exact
McNemar test as supporting evidence for binary correctness only. Flag a headline regression only
when the paired 95% interval excludes zero **and** the absolute degradation is at least five
percentage points. Per-class, difficulty, and source slices are exploratory and labelled as such.
Add the `compare --baseline --candidate` subcommand.
Add `make eval-compare`.

**Acceptance**

- An unchanged run pair produces no flag.
- A genuine 10-point regression is flagged.
- A 2-point change with a wide interval is not flagged, and the report says why.
- The report never claims that N=50 reliably detects a five-point change.

### M4-T3 — Threshold sweep and confidence diagnostics

Render the review-load versus unreviewed-error table across thresholds 0.00–1.00 in 0.05 steps,
computed from stored records without rerunning any model. This counterfactual is valid because both
agent calls complete before the workflow evaluates the gate — changing the threshold changes whether
a reply was reviewed, never what the model produced. Render only when `M3-T5`'s twin confidence
gates pass; otherwise print which gate failed and why the sweep is suppressed. Print the count of
cases excluded for having no draft alongside the table.

**Acceptance**

- The table is suppressed with a named reason on a degenerate confidence distribution.
- The sweep is computed from records alone, with a test asserting no model call occurs.
- The excluded no-draft count prints with the table.
- The sweep is scoped to one policy's scored population, not pooled across policies.

### M4-T4a — Calibration source runs and bundle

Run at least 30 real cases through each of the primary-quality and fallback-quality profiles,
producing at least 60 immutable stored replies. Publish a minimal committed bundle containing the
ticket context and reply needed for judging in `judge_calibration_sources/replies.jsonl`, plus an
`index.json` with each source run ID, its manifest and records SHA-256, and a content hash for every
copied record. This copy-and-verify handoff is required because raw `evals/runs/` artifacts are
gitignored and do not cross workspace branches. The handful-case M3 smoke run does not qualify by
itself. Agents may diversify the pool using structured correctness, confidence, profile, and
difficulty, but may not assert subjective output classes.

**Acceptance**

- The source index names primary-quality and fallback-quality runs with at least 30 cases each.
- At publication time, every indexed run exists and its stored manifest and records match the
  recorded hashes.
- A clean workspace can verify every copied record from the committed bundle and its index without
  access to the original gitignored run directories.
- The committed pool contains at least 60 real replies selected with documented, non-authoritative
  diversity proxies.

### M4-T4b — Human judge calibration set

From the qualifying source bundle, an agent prepares empty screening and rating columns plus the
judging rubric, then pauses. A named human screens the candidate pool for good, weak, irrelevant,
and hallucinated coverage. If all four classes cannot supply a sample of at least 30 replies, this
task remains blocked while T4a publishes a larger verified bundle revision. Two named humans then
score the selected replies independently without seeing the screening labels or each other's
ratings, and a named human adjudicator records the final labels. Agents may validate the completed
shape but may not enter or infer the screening class, a human rating, or the adjudication.

**Acceptance**

- ≥30 replies, drawn from real run artifacts rather than invented.
- A named human screener confirms all four output classes are represented; screening labels remain
  hidden from the independent raters.
- Both independent score sets and the adjudicated result are recorded separately, so agreement can
  be computed rather than assumed.
- Screener, rater, and adjudicator identities and timestamps are present; no human assertion is
  agent-authored.
- The rubric is specific enough that a third person reproduces the adjudicated labels.

### M4-T5 — Offline judge

Score stored reply text on relevance (1–5), appropriate support tone (1–5), and hallucinated
commitments (boolean), using the Gemma judge. This runs as a **separate phase** after workflow
execution. It calls `keep_alive=0` for agent models and verifies residency before loading Gemma,
then writes `judgments/<rubric_hash>-<judgment_id>.jsonl`; it never touches raw records. Add the
`judge --run-id` subcommand. Before judging production runs, use the same code and generation
settings to score `judge_calibration.jsonl`, writing a versioned artifact under
`evals/data/judge_calibration_judgments/`.

**Acceptance**

- Re-judging the same run generates a new `judgment_id`, writes a new file, and leaves
  `records.jsonl` byte-identical.
- The rubric hash appears in the filename and inside the artifact together with judge identity,
  generation options, and source record and manifest hashes.
- Every completed human calibration item has exactly one Gemma prediction in the calibration
  judgment artifact.
- The judge never runs while an agent model is resident.

### M4-T6 — Calibration gates

Compare Gemma's calibration predictions with the adjudicated human labels. Compute linearly
weighted Cohen's κ for relevance and tone, and unweighted Cohen's κ plus F1 with hallucination as
the positive class. Each dimension passes only at κ ≥ 0.60, with F1 ≥ 0.80 also required for
hallucination. Separately compute the same agreement statistics between the two independent humans
as rubric-quality evidence; those numbers never substitute for judge validation. Use a 5,000-sample
percentile bootstrap over calibration replies with `bootstrap_seed` for 95% agreement intervals.
Report intervals and sample size. A failing judge dimension is suppressed individually.

**Acceptance**

- A synthetic set with known κ reproduces it.
- A judge that disagrees with adjudication fails even when human-human agreement is perfect.
- Failing one dimension suppresses only that dimension.
- Judge-versus-adjudicated gates, human-human evidence, and sample size are reported beside every
  judge-derived metric.

### M4-T7 — Final report and `report` CLI

Assemble the complete report: difficulty and source slices, invariant violations, primary-versus-
fallback paired comparison, oracle-versus-rubber-stamp outcomes, and judge-derived metrics shown
only for dimensions whose calibration gate passed, each printed beside its validation result. Judge
numbers remain visibly secondary to the deterministic structured metrics — a different model family
reduces obvious self-grading bias but does not make the judge ground truth. Add
`report --run-id`; it reads immutable artifacts and writes or prints a derived report without
rewriting raw inputs.

**Acceptance**

- No judge dimension appears without its validation result adjacent.
- The report can support or reject the current 0.75 threshold while displaying uncertainty, or state
  explicitly that the model's confidence distribution cannot support threshold tuning.
- Source slices show generated versus handwritten separately.
- A golden-file test pins the full report for a fixed synthetic run.
- `report --run-id` works without Temporal or Ollama and leaves raw artifact hashes unchanged.

**DAG**

```
T1a ───────────────────────────────┐
T1b ───────────────────────────────┤
T1c ───────────────────────────────┤
T2 ───────> T5 ────────────────────┤
T3 ────────────────────────────────┼─> T7
T4a ──> T4b ──> T5 ──> T6 ────────┘
```

**Waves:** `[T1a, T1b, T1c, T2, T3, T4a]` → `[T4b]` → `[T5]` → `[T6]` → `[T7]`

The first wave contains human-gated data tasks. Its code tasks may finish while those gates are
open. `T4b` is a separate human-owned pause after qualifying source runs; `T7` cannot start until
the verified shards and completed calibration file are merged.

**Serial order:** T1a, T1b, T1c, T2, T3, T4a, T4b, T5, T6, T7.

---

## Cross-milestone ordering

Milestones merge in order 1 → 2 → 3 → 4, but three tasks can legitimately start early:

- **`M3-T1` (heartbeats) has no dependencies at all.** It is a self-contained production fix with an
  existing decision-log entry. Run it alongside milestone 1 to de-risk milestone 3.
- **`M1-T3a/b/c` (case authoring) is the long pole.** Start as soon as `M1-T2` lands; it does not
  block any code task and code tasks do not block it.
- **`M2-T6` (identity, telemetry, and invariants) can start after `M1-T4`; `M2-T1` follows it.**
  Both can run during milestone 1's later waves once their record interfaces are merged.

Two files are contended across milestones and need care at merge time:

| File | Touched by | Handling |
|---|---|---|
| `scripts/eval.py` | M1-T8, M2-T7, M3-T7, M4-T2, M4-T5, M4-T7 | Each adds one subcommand or option group. Merge in dependency order. |
| `Makefile` | M1-T1, M1-T8, M2-T7, M3-T7, M4-T2 | Each adds one target. Merge in dependency order. |
| `tests/eval/test_eval_cli.py` | M1-T8, M2-T7 | M1 creates the CLI test module; M2 extends it after milestone 1 merges. |
| `src/ticketflow/eval/report.py` | M1-T7, M4-T3, M4-T7 | The real contention. M4-T3 adds one section, M4-T7 assembles the rest. Explicit edge T3 → T7. |
| `src/ticketflow/eval/records.py` | M1-T4, M3-T6 | M3-T6 only fills optional manifest fields left open by M1-T4. |

---

## Traceability

`plan.md`'s "Completion criteria" mapped to the tasks that satisfy each:

| Completion criterion | Satisfied by |
|---|---|
| `make check` passes with all existing and new network-free tests | M1-T1, M2-T2, M2-T8, and every task's own gate |
| The tunable profile reproduces exact injected structured-error rates | M2-T1, M2-T8 |
| Limited primary and fallback Ollama runs produce immutable, reproducible artifacts | M3-T4, M3-T6, M3-T8 |
| Fallback model quality and fallback routing are reported as different experiments | M2-T5, M2-T8 |
| Every inferential headline rate includes a 95% interval and names its denominator | M1-T5, M1-T6, M1-T7 |
| Model quality, availability, and reviewer policy are reported as separate quantities | M1-T5, M2-T3, M1-T7 |
| The threshold report presents a defensible curve or explains why it cannot | M3-T5, M4-T3 |
| Judge-derived metrics appear only for validated dimensions | M4-T4a, M4-T4b, M4-T5, M4-T6, M4-T7 |
| Stable case identity makes reviewer-policy outputs comparable | M2-T6, M2-T1, M3-T4, M3-T3 |
| Production and CLI entrypoints cover every promised command | M3-T7, M4-T2, M4-T5, M4-T7 |
| Verified labels and calibration ratings are human-reviewed | M1-T3a/b/c, M4-T1a/b/c, M4-T4b |

`plan.md`'s repository-constraints table maps as follows: schedule-to-start fallback → M2-T5;
heartbeat timeout → M3-T1; `TicketResult` omissions → M2-T4; import-time constants → M2-T2;
single agent queue in the test helper → M2-T2; `MockAgent`'s shared RNG → M2-T1; `model_path`
literals → M2-T1, M3-T6; approval-as-update → M2-T4; `ProposedAction` refund validation → M3-T3;
search-attribute upsert → M2-T2; fallback activity without schedule-to-start → M2-T4; ruff rules →
working rule 4.
