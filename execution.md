# Ticketflow Eval Harness — Execution Breakdown

**Status:** not started · **Date:** 2026-07-27 · **Companion to:** [`plan.md`](./plan.md)

## Purpose

`plan.md` says *what* the eval harness is and *why* each design choice was made. It is the
authority on semantics: correctness definitions, run profiles, statistical rules, risks. Nothing
here overrides it.

This document says *who does what, in which order, touching which files*. It keeps `plan.md`'s four
milestones intact and splits each into subtasks small enough to hand to one agent or one sitting.

**Task IDs** are `M<milestone>-T<n>` — `M2-T4` is the fourth task of milestone 2. Lettered variants
(`M1-T3a`) are sibling tasks that could otherwise have been one task, split apart only because they
write to different files and can therefore run at the same time.

**Two ways to read this:**

- *Dispatching in parallel* — use the **Owns** column and the **DAG** for each milestone. Any two
  tasks in the same wave have disjoint file ownership and can run in separate workspaces.
- *Working alone* — use the **Serial order** line at the end of each milestone and ignore the
  waves.

---

## Working rules

These apply to every task and are not repeated in the task bodies.

1. **One task, one branch, one PR.** Branch from the merge of your dependencies, not from
   an unrelated task's branch.
2. **A task may only edit files in its Owns list.** If you need a change in someone else's file,
   that is a signal the breakdown is wrong — raise it rather than reaching across. The one
   exception is adding a new file that nobody owns.
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

---

## Deviations from `plan.md`

Four structural changes, each made to remove a single-owner bottleneck. Reject any of them
individually; the milestone shape survives either way.

| # | Deviation | Rationale |
|---|---|---|
| 1 | The dataset loader accepts a **directory of `*.jsonl` shards** (`evals/data/tickets/`), not only a single `evals/data/tickets.jsonl` | This is the biggest unlock in the document. Case authoring is the long pole in both M1 and M4, and a single file serialises it. With shards, three people write `easy.jsonl`, `ambiguous.jsonl`, and `adversarial.jsonl` concurrently. Duplicate-ID detection still runs across the union, so nothing in the labelling contract weakens. |
| 2 | `agent/ollama.py` is split three ways: prompts and schemas → `agent/prompts.py`, the response cache → `eval/cache.py`, preflight probing → `eval/preflight.py` | `plan.md` co-locates all of these "because `ollama.py` is the only module that knows about model calls". That argument is about coupling, and coupling is preserved: cache and preflight take the client as a parameter and know nothing about HTTP. What changes is that M3 becomes four parallel tasks instead of one 600-line file with one owner. |
| 3 | Two modules `plan.md` never names get their own files: `eval/profiles.py` (run profiles + manifest assembly) and `eval/invariants.py` (system-invariant checks) | Without a home they would land in `runner.py` and `report.py`, making those files three-owner. |
| 4 | `eval/statistics.py` must not import `eval/records.py`; it operates on primitive sequences of `(cluster_id, value)` | Removes a dependency edge (`M1-T6` no longer waits on `M1-T4`) and keeps the numerical code — the code `plan.md` itself flags as "most likely to be subtly wrong" — testable without constructing workflow records. |

### One gotcha and one ambiguity

**Gotcha — the `eval` dependency group must be installed for `make check` to pass.** `plan.md` puts
`numpy` and `scipy` in a non-runtime `eval` group so the API and worker images do not grow. But
`make test` collects `tests/eval/test_statistics.py`, which imports them. `make install` therefore
has to sync that group (`uv sync --all-groups`), or `make check` fails on a clean checkout. Docker
images are unaffected because they install runtime dependencies only. Resolved in `M1-T1`.

**Ambiguity — where does the threshold sweep live?** `plan.md`'s architecture comment assigns it to
`statistics.py` in milestone 1; the milestone 4 deliverables list "Threshold sweep and confidence
diagnostics". Resolution used here: the pure sweep *function* lands in `M1-T6`, its *rendering and
preflight gating* land in `M4-T3`. Both readings are satisfied.

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
dependency group holding `numpy` and `scipy`, and switch `make install` to `uv sync --all-groups`
so the statistics tests can run under `make check`. Register the `ollama` pytest marker and widen
`addopts` from `-m 'not smoke'` to `-m 'not smoke and not ollama'` — registering the marker alone
is not enough, the tests would still run. Add `evals/runs/` and `evals/cache/` to `.gitignore`,
leaving `evals/data/` committed. Create the empty `eval/` and `eval/scorers/` packages and
`tests/eval/__init__.py`.

**Acceptance**

- `make install && make check` passes on a clean checkout.
- A test marked `ollama` is not collected by a bare `uv run pytest`, and is collected by
  `uv run pytest -o addopts=`.
- `git status` stays clean after writing a file into `evals/runs/`.
- The runtime `dependencies` list in `pyproject.toml` is unchanged.

### M1-T2 — Dataset models and loader

Implement `ExpectedOutcome` and `EvalCase` exactly as specified in `plan.md`'s "Evaluation contract"
section, and a loader that reads either a single `.jsonl` file or a directory of `*.jsonl` shards
(deviation 1), merging them into one case list. The loader enforces the seven rejection rules from
`plan.md`: duplicate IDs across all shards, empty acceptable-category or acceptable-action sets,
`label_verified=false`, refund labels without an amount, refund amounts absent from the ticket text,
generated cases without `generated_by`, and tier or category balance outside a configured tolerance.
Every rejection names the offending case ID and shard.

**Acceptance**

- Each rejection rule has a test that asserts on the error message, not just the exception type.
- Two shards sharing a case ID are rejected.
- A directory and an equivalent single file load to identical case lists.
- Balance tolerance is configurable and its default is stated in the module docstring.

### M1-T3a — Labelling guide and easy cases

Write `evals/data/labeling.md` documenting the labelling policy from `plan.md` — when a refund is
expected, how multiple acceptable outcomes are recorded, what verification means — with worked
positive, negative, ambiguous, and adversarial examples. Then author 20 `difficulty="easy"` cases:
tickets whose category and action are unambiguous to a careful human.

**Acceptance**

- `uv run python scripts/eval.py dataset-check` passes on the shard (or the loader accepts it
  directly if T8 has not landed).
- All 20 cases have `label_verified=true` and `source="handwritten"`.
- Categories are spread across all four `TicketCategory` values.
- Every guide rule is illustrated by at least one committed case, referenced by ID.

### M1-T3b — Ambiguous cases

Author 20 `difficulty="ambiguous"` cases — tickets where a competent agent could reasonably choose
more than one category or action, recorded with multiple acceptable values rather than one
arbitrary "right" answer. These are the cases that make `gate_recall` meaningful, so bias them
toward outcomes near the confidence threshold rather than toward exotic phrasing.

**Acceptance**

- At least 12 cases carry more than one acceptable category or action.
- Each case's `notes` field explains *why* it is ambiguous.
- No case is ambiguous merely because it is badly written; ambiguity is about the support decision.

### M1-T3c — Adversarial cases

Author 10 `difficulty="adversarial"` cases: prompt-injection attempts, refund requests with no
stated amount, refund requests for amounts not in the ticket, contradictory instructions, and
tickets that describe a refund without requesting one. These exist to probe the refund rule and the
gate, not to be unanswerable.

**Acceptance**

- At least three cases request a refund in a way that must *not* produce a `REFUND` action.
- At least one case attempts to instruct the agent directly.
- Each case's `notes` states the failure mode it targets.

### M1-T4 — Immutable record models

Implement `CaseRecord`, `CallEvent`, and `RunManifest` per `plan.md`'s "Run artifacts and
reproducibility" and "Telemetry" sections, plus JSONL read/write helpers that write atomically
(temp file, then rename) and refuse to overwrite an existing raw artifact. `CaseRecord` must carry
the scorability reason as a closed set — escalated, invalid output, update rejected, or runner
deadline exceeded — so `M1-T5` can partition on it rather than on a free-text string.

**Acceptance**

- Writing to an existing `records.jsonl` or `calls.jsonl` path raises rather than truncating.
- A partially written file is never left behind when the writer is interrupted.
- Round-tripping a record through JSONL preserves every field, including enum types.
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

**Acceptance**

- A synthetic record set with hand-computed metrics reproduces them exactly.
- Escalated and invalid-output cases are absent from every quality denominator and present in their
  own rates.
- `gate_recall` and `gate_precision` are byte-identical when computed over oracle records and over
  rubber-stamp records built from the same agent outputs.
- No metric can be constructed without a denominator label.

### M1-T6 — Statistics primitives

Implement case-clustered percentile bootstrap intervals (5,000 resamples, seeded from the manifest,
95%), paired effect size with interval, the exact McNemar test, and the threshold-sweep function.
Per deviation 4, this module takes primitive sequences — `(case_id, value)` pairs — and does not
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

Render the deterministic metrics to Markdown and to console. Every rate prints with its interval and
its denominator name. Escalation and invalid-output rates render in a visually separate block from
quality metrics. A run whose scored population is under ~100 cases is labelled *directional only* in
the header, and the scored population size prints beside the total with a per-reason exclusion
breakdown.

**Acceptance**

- A golden-file test pins the Markdown for a fixed synthetic record set.
- No rate can render without an interval and a denominator.
- The directional-only banner appears at N≈50 and disappears at N≈200.
- Threshold-sweep and judge sections are absent — they arrive in M4.

### M1-T8 — `dataset-check` CLI

Create `scripts/eval.py` with a subcommand structure that later milestones extend, implementing
`dataset-check` first: load the dataset, print composition by difficulty, source, and category, and
exit non-zero with an actionable message on any validation failure. Add `make eval-dataset-check`.

**Acceptance**

- `make eval-dataset-check` exits 0 on the committed dataset and non-zero on a deliberately broken
  shard, printing the offending case ID.
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

**Waves:** `[T1]` → `[T2, T4, T6]` → `[T3a, T3b, T3c, T5, T8]` → `[T7]`

**Serial order:** T1, T2, T8, T3a, T3b, T3c, T4, T5, T6, T7. Doing T8 before the authoring tasks
gives you `dataset-check` as a validation loop while writing cases.

---

## Milestone 2 — Workflow harness and tunable agents

**Goal:** run the real `TicketWorkflow` over the dataset in process, with an agent whose errors you
chose in advance, so the harness itself can be proven correct before any real model is involved.

| ID | Goal | Depends on | Owns |
|---|---|---|---|
| M2-T1 | Tunable mock agent | M1-T2 | `src/ticketflow/agent/tunable.py`, `tests/eval/test_tunable_agent.py` |
| M2-T2 | Temporal harness | M1-T1 | `src/ticketflow/eval/harness.py`, `tests/helpers.py`, `tests/eval/test_harness.py` |
| M2-T3 | Reviewer policies | M1-T5 | `src/ticketflow/eval/reviewers.py`, `tests/eval/test_reviewers.py` |
| M2-T4 | Per-case runner | T2, T3, M1-T4 | `src/ticketflow/eval/runner.py`, `tests/eval/test_runner.py` |
| M2-T5 | Run profiles and manifest | T4 | `src/ticketflow/eval/profiles.py`, `tests/eval/test_profiles.py` |
| M2-T6 | Telemetry sink and invariants | M1-T4 | `src/ticketflow/eval/telemetry.py`, `src/ticketflow/eval/invariants.py`, `tests/eval/test_invariants.py` |
| M2-T7 | `run` CLI subcommand | T5 | `scripts/eval.py`, `Makefile` |
| M2-T8 | Cross-cutting workflow suite | T5 | `tests/eval/test_runner_workflow.py` |

### M2-T1 — Tunable mock agent

Build `TunableMockAgent`, which derives every decision from a stable hash of
`(seed, ticket.id, operation)` rather than from a shared mutable RNG like `MockAgent` does. Because
it must be able to be *deliberately wrong*, it takes the expected-outcome map from the dataset and
starts from the correct answer, perturbing it according to its profile: category, action, and
refund-amount error rates *or* exact error ID sets; confidence calibration and overconfidence; a
transient failure rate *or* exact failure ID set; and the `primary`/`fallback` role label.

**Acceptance**

- The same seed and case produce the same output regardless of concurrency, case ordering, or how
  many other cases ran first — asserted by running a case set forwards, backwards, and with
  concurrency 8, and comparing.
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
`FALLBACK_TASK_QUEUE`, `AGENT_SCHEDULE_TO_START_S`) before workers start, using
`UnsandboxedWorkflowRunner` so the patch is visible inside the sandbox; and scope the worker
lifecycle to the **run**, not the case, because post-completion queries are served by replaying
history on a live worker.

**Acceptance**

- `tests/test_workflow.py` and the rest of the existing suite pass unmodified.
- A workflow can be queried after it has completed, with workers still up.
- Primary and fallback agents can be different objects on different queues in the same run.
- The snapshot of workflow constants is returned as data for the manifest, not just applied.
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
records a `timeout` outcome. And post-completion state comes from the `status` query, whose `result`
field the workflow never populates and which must not be read — the terminal `TicketResult` comes
from awaiting the handle. Finish by capturing refund and refund-attempt rows and emitting one
`CaseRecord` plus its `CallEvent`s.

**Acceptance**

- A rejected approval update is recorded as a case outcome and the run continues.
- A case that exceeds its deadline is recorded as `timeout` and the run continues.
- Classification, draft, and decision are captured for every completed case.
- Nothing reads `TicketStatusInfo.result`.
- Bounded concurrency is configurable and results do not depend on it.

### M2-T5 — Run profiles and manifest assembly

Implement the four run profiles and the manifest they produce. `primary-quality` and
`fallback-quality` both back the primary agent queue directly — the second with the fallback model —
so they measure model quality and are paired by case ID. `fallback-routing` is a *different
experiment*: it hosts a worker only on the fallback queue, withholds the primary worker, forces the
real schedule-to-start timeout, runs under time skipping over a small subset, and its results are
excluded from model-quality headlines. `reliability` runs with oracle only, cache disabled, and full
call telemetry. Assemble the manifest with the git commit and dirty state, dataset SHA-256, workflow
constant snapshot, reviewer policies and their order, seed, concurrency, and repeats.

**Acceptance**

- A fallback-routing run completes in wall-clock time far below `AGENT_SCHEDULE_TO_START_S` per
  case, proving time skipping is actually in play.
- Routing records identify the fallback path and are excluded from quality comparisons.
- The manifest records dirty state truthfully on a dirty tree.
- `--repeats > 1` is rejected with the cache enabled, or with a fixed seed at `temperature=0`.

### M2-T6 — Telemetry sink and invariants

Provide a process-local `CallEvent` sink that agents write to and the runner drains per case — the
harness is single-process, so a module-level collector is sufficient and simpler than threading a
handle through the activity boundary. Separately, implement the system-invariant checks from
`plan.md`'s reporting section: gating agrees with the recorded threshold and refund rule, at most
one refund row per ticket, refund attempts ≥ executed refunds, an executed refund implies an
approved decision, and a fallback-routing record identifies the fallback path.

**Acceptance**

- Events from concurrent cases are attributed to the right case.
- Each invariant has a test that constructs a violating record set and asserts it is flagged.
- Violations are reported as data, not raised — a violated invariant is a finding, not a crash.
- `model_path` is not used as a retry counter anywhere.

### M2-T7 — `run` CLI subcommand

Extend `scripts/eval.py` with `run --profile primary-quality|fallback-quality|fallback-routing|
reliability` and the option set from `plan.md`: `--agent`, `--reviewer`, `--limit`, `--repeats`,
`--concurrency`, `--seed`, `--no-cache`. Enforce that `--repeats > 1` implies `--no-cache` and a
varied or absent seed. Add a `make eval` target for the fast tunable profile.

**Acceptance**

- `make eval` completes a full tunable run over the committed dataset with no Temporal server or
  Ollama installed beyond the test server.
- Invalid option combinations fail before any workflow starts, with a message explaining why.
- Run artifacts land under `evals/runs/<run_id>/` and are gitignored.

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
M1 ──┬─> T1
     ├─> T2 ──┐
     ├─> T3 ──┼─> T4 ──> T5 ──┬─> T7
     └─> T6 ──┘               └─> T8
```

**Waves:** `[T1, T2, T3, T6]` → `[T4]` → `[T5]` → `[T7, T8]`

**Serial order:** T2, T3, T6, T1, T4, T5, T8, T7.

---

## Milestone 3 — Ollama integration

**Goal:** replace the tunable agent with a real one, without letting slow inference or malformed
output masquerade as a quality signal.

| ID | Goal | Depends on | Owns |
|---|---|---|---|
| M3-T1 | Periodic activity heartbeats *(production change)* | — | `src/ticketflow/activities.py`, `tests/test_activities.py`, `docs/context.md` |
| M3-T2 | Prompts and schemas | M1-T1 | `src/ticketflow/agent/prompts.py`, `tests/eval/test_prompts.py` |
| M3-T3 | Ollama agent | T2, M2-T6 | `src/ticketflow/agent/ollama.py`, `tests/eval/test_ollama_agent.py` |
| M3-T4 | Response cache | T2 | `src/ticketflow/eval/cache.py`, `tests/eval/test_cache.py` |
| M3-T5 | Preflight | T3 | `src/ticketflow/eval/preflight.py`, `tests/eval/test_preflight.py` |
| M3-T6 | Manifest enrichment | T3, T5 | `src/ticketflow/eval/records.py`, `src/ticketflow/eval/profiles.py` |
| M3-T7 | Ollama CLI wiring | T3, T4, T5 | `scripts/eval.py`, `Makefile` |
| M3-T8 | Local smoke run | T1, T7 | `tests/eval/test_ollama_smoke.py` |

### M3-T1 — Periodic activity heartbeats

The one production change in scope, and already an accepted follow-up in `docs/context.md`. Both
agent activities currently heartbeat only immediately before and after the agent call, so with
`heartbeat_timeout=30s` any real LLM call slower than 30 seconds is killed mid-flight. Wrap the
agent call in a background asyncio task that heartbeats roughly every 10 seconds and is cancelled
when the call returns or raises. Close the follow-up entry in the decision log.

**Acceptance**

- An agent call artificially slowed past 30 seconds completes rather than timing out.
- The existing test suite stays instant — the default mock latency is 0 and nothing sleeps.
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
timeout, and the schema in `format`. Map errors deliberately: connection failures, HTTP timeouts,
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
- No stub test performs real network I/O, and the suite is collected by default.

### M3-T4 — Response cache

Cache successful responses only. The key hashes the complete normalized request: operation, model
name and digest, role, ticket content, the classification input when drafting, messages and prompt
version, JSON schema, think setting, temperature, seed, generation options, and the Ollama version.
Entries store request metadata for inspection and are written atomically. This is what lets the
oracle and rubber-stamp policy runs receive byte-identical agent outputs — the whole
reviewer-comparison design depends on it.

**Acceptance**

- A test per key component proves that changing it alone invalidates the entry.
- Failed and invalid-output calls are never cached.
- A second policy run over the same records is 100% cache hits and produces byte-identical outputs.
- Cache-hit calls are marked as such so they can be excluded from latency statistics.

### M3-T5 — Preflight

Before any real-model run: check `/api/version`, confirm every required model exists and record its
digest, run one unmeasured warm-up, then measure at least three classifications and three drafts,
separating model-load from generation time. Widen the effective activity timeout to the larger of
the configured value or three times the slowest observed stage, set the HTTP timeout below it with a
margin, and record every adjustment. Then probe ~10 cases for confidence variance and decide whether
the threshold sweep is admissible: it requires **both** a standard deviation of at least 0.02 **and**
at least five distinct values. Standard deviation alone is too weak — self-reported LLM confidence
typically clusters on two or three values like 0.9 and 0.95, which passes a variance threshold while
producing a step function with no interior operating points.

**Acceptance**

- A synthetic confidence distribution of `{0.9, 0.95}` fails the distinctness gate despite passing
  variance, and the report names which gate failed.
- Timeout widening is computed from measurements and recorded, not hardcoded.
- A degenerate confidence distribution is reported as a finding, not an error.
- A missing model fails preflight with the model name in the message, before any case runs.

### M3-T6 — Manifest enrichment

Fill in the manifest fields that only become knowable once a real model is involved: model names and
digests, Ollama, Python, and relevant dependency versions, prompt and schema hashes, generation
options, preflight measurements, and every timeout adjustment. Enforce raw-artifact immutability
end-to-end — `records.jsonl` and `calls.jsonl` are written once and never rewritten; re-judging
creates a new file under `judgments/`.

**Acceptance**

- Every bullet in `plan.md`'s `RunManifest` list has a populated field after a real run.
- Attempting to re-run into an existing run directory fails rather than overwriting.
- A report can be traced back to its source record and manifest hashes.

### M3-T7 — Ollama CLI wiring

Extend `scripts/eval.py` with `--agent ollama`, model and endpoint overrides, and `--no-cache` (made
mandatory for the reliability profile). Add `make eval-ollama`.

**Acceptance**

- Primary and fallback quality profiles can be launched over the same limited case set from the CLI.
- The reliability profile refuses to run with the cache enabled.
- Model overrides reach the manifest.

### M3-T8 — Local smoke run

A small end-to-end run against real Ollama, marked `ollama` and deselected by default. It runs the
primary and fallback quality profiles over the same handful of cases and asserts the artifacts are
well-formed. Critically, it unloads agent models with `keep_alive=0` between phases: the primary
(23 GB) and the judge (17 GB) must never be resident simultaneously on the 64 GB machine.

**Acceptance**

- Not collected by a bare `uv run pytest`; collected by `uv run pytest -m ollama -o addopts=`.
- The degraded fallback model reports a measurable `invalid_output_rate` rather than escalating
  every malformed refund — this is the headline evidence that `M3-T3`'s repair budget works.
- Model residency is asserted, not assumed, between phases.
- Requires `M3-T1`; without heartbeats a real call can be killed at 30 seconds.

**DAG**

```
T1 ─────────────────────────────────────────────┐
                     ┌─> T6                      │
T2 ──┬─> T3 ──> T5 ──┤                           │
     │   │           └──┐                        │
     │   └──────────────┼─> T7 ──────────────────┴─> T8
     └─> T4 ────────────┘
```

**Waves:** `[T1, T2]` → `[T3, T4]` → `[T5, T6]` → `[T7]` → `[T8]`

**Serial order:** T1, T2, T3, T4, T5, T6, T7, T8.

---

## Milestone 4 — Decision-grade evaluation

**Goal:** enough data and enough validated inference to support or reject a real decision about the
0.75 confidence threshold and about fallback quality.

| ID | Goal | Depends on | Owns |
|---|---|---|---|
| M4-T1a | Grow easy cases to 60 | M1-T3a | `evals/data/tickets/easy.jsonl` |
| M4-T1b | Grow ambiguous cases to 80 | M1-T3b | `evals/data/tickets/ambiguous.jsonl` |
| M4-T1c | Grow adversarial cases to 60 | M1-T3c | `evals/data/tickets/adversarial.jsonl` |
| M4-T2 | Paired comparison and `compare` | M1-T6 | `src/ticketflow/eval/compare.py`, `scripts/eval.py` |
| M4-T3 | Threshold sweep and confidence diagnostics | M1-T6, M3-T5 | `src/ticketflow/eval/report.py` |
| M4-T4 | Judge calibration set | M3-T8 | `evals/data/judge_calibration.jsonl`, `evals/data/judging.md` |
| M4-T5 | Offline judge | M3-T3 | `src/ticketflow/eval/scorers/judge.py`, `scripts/eval.py` |
| M4-T6 | Calibration gates | M4-T4 | `src/ticketflow/eval/scorers/calibration.py`, `tests/eval/test_calibration.py` |
| M4-T7 | Final report | T3, T5, T6 | `src/ticketflow/eval/report.py` |

> `report.py` is owned by both T3 and T7 and `scripts/eval.py` by both T2 and T5. T3 → T7 is an
> explicit edge; T2 and T5 must be serialised even though no semantic dependency exists between
> them. Take T2 first.

### M4-T1a/b/c — Dataset growth to ~200 verified cases

Grow each difficulty shard toward the milestone 4 target: 60 easy, 80 ambiguous, 60 adversarial.
Generated cases are permitted but must record `generated_by`, must be human-verified before
`label_verified=true`, and remain separately sliced in reports — generated data is systematically
easier than real traffic and the reports must be able to show that. Three tasks, three shards, no
shared file.

**Acceptance**

- `make eval-dataset-check` passes on the union, including cross-shard duplicate-ID detection.
- Every case is `label_verified=true`; every generated case has `generated_by` set.
- The ambiguous tier remains the largest, matching `plan.md`'s target distribution.
- Balance tolerance still passes without loosening it.

### M4-T2 — Paired comparison and `compare`

Report paired effect sizes with bootstrap intervals for runs over identical case IDs, with the exact
McNemar test as supporting evidence for binary correctness only. Flag a headline regression only
when the paired 95% interval excludes zero **and** the absolute degradation is at least five
percentage points. Per-class, difficulty, and source slices are exploratory and labelled as such.
Add the `compare --baseline --candidate` subcommand.

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

### M4-T4 — Judge calibration set

Assemble at least 30 stored replies spanning good, weak, irrelevant, and hallucinated outputs, score
them independently with two humans, and adjudicate disagreements. Write the judging rubric alongside
them. This is a data and process task, not a code task, and it gates every judge-derived number in
the final report.

**Acceptance**

- ≥30 replies, drawn from real run artifacts rather than invented.
- All four output classes are represented.
- Both independent score sets and the adjudicated result are recorded separately, so agreement can
  be computed rather than assumed.
- The rubric is specific enough that a third person reproduces the adjudicated labels.

### M4-T5 — Offline judge

Score stored reply text on relevance (1–5), appropriate support tone (1–5), and hallucinated
commitments (boolean), using the Gemma judge. This runs as a **separate phase** after workflow
execution, with agent models unloaded first, and writes `judgments/<rubric_hash>.jsonl` — it never
touches the raw records. Add the `judge --run-id` subcommand.

**Acceptance**

- Re-judging the same run writes a new file and leaves `records.jsonl` byte-identical.
- The rubric hash appears in the filename and inside the artifact.
- The judge never runs while an agent model is resident.

### M4-T6 — Calibration gates

Compute weighted Cohen's κ for relevance and for tone, and κ plus F1 for hallucination, against the
calibration set. Each dimension passes only at κ ≥ 0.60 (and F1 ≥ 0.80 for hallucination). Report
agreement intervals and the calibration sample size. A failing dimension is suppressed
**individually** — it must not suppress dimensions that passed.

**Acceptance**

- A synthetic set with known κ reproduces it.
- Failing one dimension suppresses only that dimension.
- The gate values and the sample size are reported as data beside every judge-derived metric.

### M4-T7 — Final report

Assemble the complete report: difficulty and source slices, invariant violations, primary-versus-
fallback paired comparison, oracle-versus-rubber-stamp outcomes, and judge-derived metrics shown
only for dimensions whose calibration gate passed, each printed beside its validation result. Judge
numbers remain visibly secondary to the deterministic structured metrics — a different model family
reduces obvious self-grading bias but does not make the judge ground truth.

**Acceptance**

- No judge dimension appears without its validation result adjacent.
- The report can support or reject the current 0.75 threshold while displaying uncertainty, or state
  explicitly that the model's confidence distribution cannot support threshold tuning.
- Source slices show generated versus handwritten separately.
- A golden-file test pins the full report for a fixed synthetic run.

**DAG**

```
T1a ─┐
T1b ─┼─ (independent, no downstream edges)
T1c ─┘
T2 ──────────────────> T5 ──┐        (edge is file contention on scripts/eval.py)
T3 ─────────────────────────┼─> T7   (edge is file contention on report.py)
T4 ──────────> T6 ──────────┘
```

**Waves:** `[T1a, T1b, T1c, T2, T3, T4]` → `[T5, T6]` → `[T7]`

T2 → T5 and T3 → T7 are ordering edges forced by shared file ownership, not by semantics; if you
split those files differently, both collapse into a single wave.

**Serial order:** T1a, T1b, T1c, T2, T3, T5, T4, T6, T7.

---

## Cross-milestone ordering

Milestones merge in order 1 → 2 → 3 → 4, but three tasks can legitimately start early:

- **`M3-T1` (heartbeats) has no dependencies at all.** It is a self-contained production fix with an
  existing decision-log entry. Run it alongside milestone 1 to de-risk milestone 3.
- **`M1-T3a/b/c` (case authoring) is the long pole.** Start as soon as `M1-T2` lands; it does not
  block any code task and code tasks do not block it.
- **`M2-T1` (tunable agent) only needs `M1-T2`.** It can run during milestone 1's later waves.

Two files are contended across milestones and need care at merge time:

| File | Touched by | Handling |
|---|---|---|
| `scripts/eval.py` | M1-T8, M2-T7, M3-T7, M4-T2, M4-T5 | Each adds one subcommand or option group. Merge in milestone order; within M4 take T2 before T5. |
| `Makefile` | M1-T1, M1-T8, M2-T7, M3-T7 | Each adds one target. Append-only; conflicts are trivial. |
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
| Judge-derived metrics appear only for validated dimensions | M4-T4, M4-T6, M4-T7 |

`plan.md`'s repository-constraints table maps as follows: schedule-to-start fallback → M2-T5;
heartbeat timeout → M3-T1; `TicketResult` omissions → M2-T4; import-time constants → M2-T2;
single agent queue in the test helper → M2-T2; `MockAgent`'s shared RNG → M2-T1; `model_path`
literals → M2-T1, M3-T6; approval-as-update → M2-T4; `ProposedAction` refund validation → M3-T3;
search-attribute upsert → M2-T2; fallback activity without schedule-to-start → M2-T4; ruff rules →
working rule 4.
