# Ticketflow Eval Harness

**Status:** revised plan, not started · **Date:** 2026-07-27

## Summary

Ticketflow has a mocked agent, a durable Temporal workflow, and a human-approval gate for
refunds and low-confidence drafts, but it has no evaluation harness. In particular, nobody can
currently answer:

1. How often does an incorrect structured outcome reach a customer without review?
2. How does changing `CONFIDENCE_THRESHOLD = 0.75` trade unreviewed errors against review load?
3. How much quality is lost when the fallback model is used?
4. Does the workflow route to fallback correctly under operational pressure?

The harness will run a labelled ticket dataset through the real `TicketWorkflow` in process. It
will keep four concerns separate:

- **Model quality:** primary and fallback models evaluated independently on the same cases.
- **Reviewer policy:** oracle and rubber-stamp decisions applied to identical agent outputs.
- **Operational reliability:** retries, timeouts, cache use, and real fallback routing.
- **Reply quality:** offline LLM judging, reported only after judge calibration passes.

Approximately 50 initial cases are enough to prove that the harness works and provide directional
results. Approximately 200 verified cases are the target before treating results as
decision-grade.

**Non-goals:**

- Replacing `scripts/batch.py`, Docker smoke tests, or tracing tests.
- Running real-model evaluation in normal CI.
- Changing Temporal-crossing ticket, workflow, or result models.
- Treating category correctness as proof that a customer-facing reply is correct.

**One production change is in scope:** agent activities gain a periodic heartbeat loop. This is
required before any real model can be called at all, and it is already recorded as an accepted
follow-up in the decision log rather than being new scope introduced here.

---

## Verified local environment and model roles

- Apple M5 Pro with 64 GB unified memory.
- Ollama 0.30.10.
- Primary agent: `qwen3.6:35b` (23 GB).
- Fallback agent: `qwen2.5-coder:1.5b` (986 MB).
- Judge: `gemma4:26b` (17 GB).

The primary and judge models must not be resident simultaneously. Workflow execution and judging
therefore remain separate phases:

1. Run workflows with the primary or fallback agent and persist immutable records.
2. Unload agent models with `keep_alive=0`.
3. Load the judge and write a separate, versioned judgment artifact.

The fallback coder model is an intentionally degraded stand-in, not a production recommendation.
Its absolute score is not assumed to transfer to another fallback model.

---

## Important repository constraints

| Constraint | Consequence |
|---|---|
| The workflow falls back only after a primary **schedule-to-start** timeout. | Model degradation and fallback routing require separate run profiles. |
| `AGENT_HEARTBEAT_TIMEOUT` is 30 seconds. | Long Ollama calls need periodic activity heartbeats, even when the start-to-close timeout is wider. |
| `TicketResult` omits classification, draft, and decision. | Query `TicketWorkflow.status` after completion; completed workflows remain queryable while history is retained. |
| Workflow configuration is copied into module constants at import. | The harness must configure and snapshot workflow constants before starting workers. |
| The current test helper hosts only one agent queue. | The harness must support independently injected primary and fallback agents and queues. |
| `MockAgent` uses one mutable RNG. | The tunable eval agent must derive randomness from stable per-ticket hashes. |
| `model_path` depends on literal `primary` and `fallback` labels. | Real model names belong in the manifest; agent outputs retain role labels. |
| Approval is a workflow **update** with a validator, not a signal. | The runner submits decisions through `execute_update` and handles rejected updates as case outcomes. |
| `ProposedAction` rejects a refund without a positive amount. | Malformed refunds are validation failures, not quality signals, and need a repair budget before escalating. |
| Every status transition upserts the `TicketStatus` search attribute. | The harness environment must register the key or every workflow task fails. |
| The fallback activity has no schedule-to-start timeout. | The runner needs its own per-case deadline so a missing fallback worker cannot hang a run. |
| Source modules and scripts require docstrings and line length 88. | All new modules, public APIs, and `scripts/eval.py` follow existing Ruff rules. |

---

## Architecture

```text
evals/
  data/
    tickets.jsonl
    labeling.md
    judge_calibration.jsonl
  runs/<run_id>/
    manifest.json
    records.jsonl
    calls.jsonl
    judgments/<rubric_hash>.jsonl
    report.md
  cache/

src/ticketflow/
  agent/
    tunable.py
    ollama.py
  eval/
    dataset.py        # milestone 1
    records.py        # milestone 1
    statistics.py     # milestone 1: intervals, paired comparison, threshold sweep
    report.py         # milestone 1
    harness.py        # milestone 2: Temporal environment and worker construction
    runner.py         # milestone 2: per-case execution
    reviewers.py      # milestone 2
    scorers/
      deterministic.py  # milestone 1
      calibration.py    # milestone 4
      judge.py          # milestone 4

scripts/eval.py
```

Telemetry, response caching, and preflight probing live inside `agent/ollama.py`, which is the
only module that knows about model calls; they are split out only if a second real agent appears.
Paired comparison lives in `statistics.py` for the same reason. The directory grows per milestone
rather than being created empty up front.

`records.jsonl` and `calls.jsonl` are immutable raw artifacts. Re-running the judge creates a new
file under `judgments/`; it never rewrites raw workflow output. `evals/runs/` and `evals/cache/`
are added to `.gitignore`; only `evals/data/` is committed.

---

## Evaluation contract

### Dataset types

```python
class ExpectedOutcome(BaseModel):
    acceptable_categories: set[TicketCategory]
    acceptable_actions: set[ActionType]
    expected_refund_amount: float | None = None
    refund_tolerance: float = 0.01


class EvalCase(BaseModel):
    id: str
    subject: str
    body: str
    customer_email: str = "eval@example.com"
    expected: ExpectedOutcome
    difficulty: Literal["easy", "ambiguous", "adversarial"]
    source: Literal["handwritten", "generated"]
    generated_by: str | None = None
    label_verified: bool
    notes: str | None = None
```

### Labelling rules

- Every case has at least one acceptable category and action.
- Ambiguous cases may have multiple acceptable outcomes.
- A refund is expected only when the ticket explicitly requests one for a duplicate or incorrect
  charge and states the amount.
- The expected refund amount must appear in the ticket text and match within
  `refund_tolerance`.
- Committed cases have `label_verified=true`.
- Generated cases record `generated_by`, remain separately sliced in reports, and require human
  verification.
- `evals/data/labeling.md` documents the policy and includes positive, negative, ambiguous, and
  adversarial examples.

The loader rejects:

- Duplicate IDs.
- Empty acceptable category or action sets.
- Unverified labels.
- Refund labels without an amount.
- Refund amounts that do not appear in the ticket.
- Generated cases without generator provenance.
- Dataset tier or category balance outside configured tolerance.

### Initial dataset

Milestone 1 ships approximately 50 verified handwritten cases:

- 20 easy cases.
- 20 ambiguous cases.
- 10 adversarial cases.

Milestone 4 grows the dataset toward approximately 200 verified cases:

- 60 easy cases.
- 80 ambiguous cases.
- 60 adversarial cases.

Repeats do not increase the effective number of labelled cases. `--repeats` defaults to 1.
Repeated runs measure self-consistency and are clustered by case in all uncertainty calculations.

`--repeats > 1` implies `--no-cache` and a varied or absent seed, and the CLI enforces this. With
the cache on, every repeat after the first is a cache hit by construction; at `temperature=0` with
a fixed seed, repeats measure the decoder's determinism rather than the model's self-consistency.
Either way the measurement would be vacuous.

---

## Correctness definitions

### Scored population

A case is **scored** only if it produced a draft. Cases that escalated because the agent failed,
exhausted retries, or never returned a valid structured output have no prediction and are not
scorable.

This matters because escalation is not silent: the workflow still sends `ESCALATION_REPLY` and
never gates it. Folding escalations into a structured-quality headline would count an availability
failure as a classification error. Therefore:

- All structured-quality rates use the scored population as their denominator, and every reported
  rate names its denominator explicitly.
- `escalation_rate` and `invalid_output_rate` are reported beside them over **all** cases, never
  folded in.
- A single combined `unhelpful_outcome_rate` over all cases may be reported as a customer-facing
  summary, clearly labelled as combining quality and availability.

Each scored case record derives:

```text
category_correct =
    predicted_category in acceptable_categories

action_correct =
    predicted_action in acceptable_actions

refund_correct =
    predicted_action is not REFUND
    or (
        an expected amount exists
        and a predicted amount exists
        and abs(predicted_amount - expected_amount) <= refund_tolerance
    )

structured_correct =
    category_correct and action_correct and refund_correct
```

The deterministic headline metrics are, over the scored population unless noted:

- `unreviewed_structured_error_rate`: proportion of scored cases that resolved without gating and
  had an incorrect structured outcome.
- `unreviewed_category_error_rate`.
- `unreviewed_action_error_rate`.
- `review_load`: proportion of scored cases gated for approval.
- `gate_recall`: `P(gated | not structured_correct)`, the share of wrong outcomes the gate
  intercepted. This is the question the gate exists to answer.
- `gate_precision`: `P(not structured_correct | gated)`, how much of the review budget is spent on
  outcomes that actually needed correcting.
- Category precision, recall, F1, and confusion matrix.
- Action accuracy and refund-amount error.
- `escalation_rate`, `invalid_output_rate`, and fallback usage, over all cases.

There is deliberately no "gate catch rate" defined as *gated, incorrect, and rejected by the
oracle*. The oracle rejects exactly when the outcome is incorrect, so such a metric is 1.0 under
`oracle` and 0.0 under `rubber_stamp` by construction: it measures the reviewer definition rather
than the gate. `gate_recall` and `gate_precision` are properties of the threshold and are
reviewer-independent.

Reply relevance, tone, or hallucination are never inferred from category correctness. They are
reported separately by the validated offline judge.

---

## Run profiles

### Primary quality

`primary-quality` runs the dataset through the real workflow with the primary task queue backed by
the primary model. Agent outputs use `model="primary"`.

### Fallback quality

`fallback-quality` backs the same agent task queue directly with the fallback model. Agent outputs
use `model="fallback"`.

This deliberately bypasses the schedule-to-start routing delay. It measures fallback-model quality,
not whether fallback routing works. Primary and fallback quality runs are paired by case.

### Fallback routing

`fallback-routing` is a separate operational profile:

- A worker is hosted on the fallback queue.
- The primary worker is withheld or deliberately saturated.
- The workflow must reach fallback through the real primary schedule-to-start timeout.
- The configured timeout and routing reason are recorded.
- Results are excluded from primary-versus-fallback model-quality headlines.

This profile runs on the **tunable agent under time-skipping Temporal**, over a small case subset.
Routing is a property of the workflow, not of any model, so paying for real inference here buys
nothing. `AGENT_SCHEDULE_TO_START_S` defaults to 30 seconds and is copied into a module constant
at import, so a real-time run would spend at least 30 idle seconds per case; time skipping removes
that cost entirely, exactly as the existing schedule-to-start test already does.

### Reliability

`reliability` uses the selected agent with:

- Oracle reviewer only.
- Cache disabled.
- Call and retry telemetry enabled.
- Real failures, attempts, and latencies included in the report.

### Reviewer policies

Quality runs execute policies in this order:

1. `oracle`: approve only when the structured outcome is correct.
2. `rubber_stamp`: approve every gated draft.

Only successful agent responses are cached. The second policy therefore receives byte-identical
agent outputs. Cache-hit policy runs are excluded from model-latency and reliability statistics.

---

## Workflow runner

For each case and repeat:

1. Assign a unique workflow and ticket ID containing the run, policy, case, and repeat.
2. Start `TicketWorkflow` on a run-specific task queue, under a per-case wall-clock deadline.
3. Poll until the workflow either reaches `AWAITING_APPROVAL` or terminates.
4. If approval is required, calculate the reviewer decision and submit it with `execute_update`.
5. Await the terminal `TicketResult`.
6. Query `TicketWorkflow.status` after completion to capture classification, draft, and decision.
7. Query refund and refund-attempt rows for the ticket.
8. Emit one `CaseRecord` plus its `CallEvent` entries.

### Submitting approval

Approval is a workflow **update** with a validator, not a signal. Three consequences shape the
runner:

- The validator rejects the update unless the workflow is `AWAITING_APPROVAL` and undecided, so
  the runner must wait for that status first. The existing `wait_for_status` test helper is
  reused.
- `submit_approval` blocks until the status leaves `AWAITING_APPROVAL` and returns the terminal
  status, so awaiting the update already yields the outcome. No second poll is needed.
- A lost race raises `WorkflowUpdateFailedError`. The runner records it as a case outcome and
  continues; it never aborts the run.

### Per-case deadline

The fallback activity is dispatched without a schedule-to-start timeout and the workflow has no
run timeout, so a misconfigured or missing fallback worker parks the activity task indefinitely
rather than failing. The runner therefore imposes its own wall-clock deadline per case and records
a `timeout` terminal outcome, so a harness misconfiguration surfaces as a recorded result instead
of a hung run.

### Environment construction

Two bootstrap requirements are easy to miss and fail immediately:

- Every status transition upserts the `TicketStatus` search attribute, so the harness environment
  must register that keyword key. Both the time-skipping fixture and the local-server test already
  do this and show the shape.
- The client must use `pydantic_data_converter`, matching the production workers.

Workflow queries after completion are served by replaying history on a worker, so workers must
stay alive until every post-completion query has returned. Worker lifecycle is scoped to the run,
never to a single case.

The harness moves reusable worker construction out of `tests/helpers.py` into production eval code.
`tests/helpers.py` re-exports the helper so existing workflow tests remain unchanged.

The harness supports:

- Separate primary and fallback agents.
- Separate workflow, primary, and fallback queues.
- Time-skipping Temporal for mock-only tests.
- Local real-time Temporal for Ollama runs.
- A run-specific SQLite read model.
- Configurable concurrency, defaulting to 2 for Ollama.

---

## Agent construction

### Production seam

Production configuration supports `mock` and `ollama`. The LLM worker uses a small factory for
primary and fallback roles.

The general workflow worker continues to avoid constructing an Ollama client because its
registered side-effect activities never call the agent.

Evaluation code constructs tunable, primary, and fallback agents explicitly from the dataset and
run configuration. The production factory is not responsible for label-aware tunable agents.

### Tunable mock

`TunableMockAgent` derives each decision from a stable hash of:

```text
(seed, ticket.id, operation)
```

Its profile controls:

- Category error rate or exact error IDs.
- Action error rate or exact error IDs.
- Refund-amount error rate or exact error IDs.
- Confidence calibration and overconfidence.
- Transient failure rate or exact failure IDs.
- Primary or fallback role label.

This makes output independent of concurrency and execution order. Exact error IDs are used in unit
tests so expected metrics are not probabilistic.

### Ollama agent

Use a shared, lifecycle-managed `httpx.AsyncClient` and `POST /api/chat` with:

- `stream=false`.
- `think=false` for workflow agent calls.
- `temperature=0`.
- Configurable seed and timeout.
- A distinct output schema for classification and drafting.
- The JSON schema supplied through `format` and included in the grounding prompt.

Output-only response models omit the internal `model` field. After Pydantic validation, application
code assigns `model="primary"` or `model="fallback"`.

Error mapping:

- Connection errors, HTTP timeouts, 408, 429, and 5xx become `AgentOverloadedError`.
- 400 and 404/model-not-found become `AgentPermanentError`.
- Schema-validation failures get a bounded repair budget before becoming `AgentPermanentError`.
- Failures are recorded in telemetry and never cached.

`AgentOverloadedError` is deliberately left unmapped in the activity layer: it escapes as a
generic activity failure and is retried by the workflow's own retry loop, which is the intended
behaviour. It must not be "fixed" into a non-retryable `ApplicationError`.

### Invalid structured output is not a quality signal

`AgentPermanentError` becomes a non-retryable `ApplicationError`, which the workflow re-raises
immediately, escalating the ticket with no retry and no fallback. Mapping schema-validation
failures straight onto that path would be actively misleading: the refund action model requires a
positive refund amount, and a 1.5B fallback model will violate that constraint routinely. The
fallback-quality profile would then report an escalation rate where it means to report degraded
classification quality — the exact conflation this harness exists to prevent.

Instead:

- The agent retries a schema-invalid response once, re-asking with the validation error appended
  to the prompt.
- Both attempts are recorded, and the outcome `invalid_output` is a distinct `CallEvent` outcome.
- Only after the repair budget is spent does the failure become `AgentPermanentError`.
- `invalid_output_rate` is reported as a first-class model-quality metric over all cases. For a
  deliberately degraded fallback model it is expected to be one of the more informative numbers.

Agent activities send heartbeats at least every 10 seconds while waiting for Ollama. This prevents
valid calls from tripping the current 30-second heartbeat timeout.

---

## Telemetry

Each classification or drafting attempt emits:

```python
class CallEvent(BaseModel):
    run_id: str
    case_id: str
    policy: str
    operation: Literal["classify", "draft"]
    role: Literal["primary", "fallback"]
    attempt: int
    cache_hit: bool
    started_at: datetime
    wall_latency_ms: float
    model_total_duration_ms: float | None
    model_load_duration_ms: float | None
    outcome: Literal[
        "success", "invalid_output", "transient_error", "permanent_error"
    ]
    error_type: str | None
```

Reports distinguish:

- End-to-end case latency.
- Agent wall-clock latency.
- Ollama load and generation duration.
- Workflow queue and retry delay.
- Primary and fallback attempts.
- Cache hits.

`model_path` alone is not used as a retry counter.

---

## Response cache

Only successful agent responses are cached. The cache key hashes the complete normalized request:

- Operation (`classify` or `draft`).
- Model name and digest.
- Role.
- Ticket content.
- Classification input for drafting.
- Messages and prompt version.
- JSON schema.
- Think setting.
- Temperature, seed, and generation options.
- Ollama version.

Entries include request metadata for inspection and are written atomically. Any change to the
operation, input, model, prompt, schema, or generation configuration invalidates the entry.

`--no-cache` is mandatory for reliability runs.

---

## Run artifacts and reproducibility

`CaseRecord` contains:

- Run, policy, case, repeat, and ticket IDs.
- Difficulty, source, and accepted labels.
- Predicted category, action, refund amount, and confidences.
- Reply text and role-based model path.
- Terminal status, gating, and approval decision.
- Refund and refund-attempt counts.
- Whether the case was scorable, and if not, why: escalated, invalid output, update rejected, or
  runner deadline exceeded.
- End-to-end latency and terminal error.

Classification, draft, and decision come from the post-completion `status` query. Its `result`
field is never populated by the workflow and must not be read; the terminal `TicketResult` comes
from awaiting the workflow handle.

`RunManifest` records:

- Git commit and dirty state.
- Dataset path and SHA-256.
- Agent implementation and run profile.
- Real model names and digests.
- Ollama, Python, and relevant dependency versions.
- Prompt, schema, and rubric hashes.
- Generation options.
- Reviewer policies and order.
- Cache setting.
- Workflow thresholds and timeouts.
- Preflight measurements and timeout adjustments.
- Seed, concurrency, repeats, and timestamps.

Raw artifacts are immutable. Reports and judgments always point back to their source record and
manifest hashes.

---

## Ollama preflight

Before a real-model run:

1. Check `/api/version`.
2. Confirm all required models exist and record their digests.
3. Run one unmeasured warm-up.
4. Measure at least three representative classifications and three drafts.
5. Separate model-load time from generation time.
6. Set the effective activity timeout to the larger of the configured value or three times the
   slowest observed stage.
7. Set the HTTP timeout below the activity timeout with a safety margin.
8. Record measurements and every timeout adjustment in the manifest.
9. Probe approximately ten cases for confidence variance.

The threshold sweep is suppressed unless the probed draft confidences clear **both** gates:

- Standard deviation of at least 0.02.
- At least five distinct values.

Standard deviation alone is too weak a test for self-reported LLM confidence, which typically
clusters on two or three values such as 0.9 and 0.95. That distribution can pass a variance
threshold while producing a step-function sweep with no interior operating points, which is easy
to over-read as a tradeoff curve. The report states which gate failed. A flat or degenerate
confidence distribution is a valid finding, not a harness failure.

---

## Statistics

### Dependencies

The project currently has no numerical dependency. The clustered bootstrap and the exact McNemar
test are added through a new `eval` dependency group holding `numpy` and `scipy`, kept out of the
runtime dependencies so the API and worker images do not grow. The alternative — hand-rolling both
on the standard library, which is genuinely feasible — is rejected because bootstrap resampling and
an exact binomial test are precisely the code most likely to be subtly wrong and least likely to be
caught by tests.

### Confidence intervals

- Use case-clustered percentile bootstrap intervals.
- Resample cases while retaining all repeats for each selected case.
- Use 5,000 samples, a fixed manifest seed, and 95% intervals.
- Attach intervals to rates and derived scores.
- Raw counts, configuration values, and confusion-matrix cells do not require intervals.
- Report self-consistency separately from accuracy.

### Paired comparisons

For runs over identical case IDs:

- Report paired effect sizes and bootstrap intervals.
- Use exact McNemar tests only for binary correctness as supporting evidence.
- Flag a headline regression only when its paired 95% interval excludes zero and the absolute
  degradation is at least five percentage points.
- Treat per-class, difficulty, and source slices as exploratory.
- Do not claim that N=50 reliably detects a five-point change.

### Threshold sweep

For thresholds from 0.00 through 1.00 in 0.05 increments:

```text
gated =
    predicted_action == REFUND
    or draft_confidence < threshold

review_load =
    P(gated)

unreviewed_structured_error_rate =
    P(not gated and not structured_correct)
```

Both agent calls complete before the workflow evaluates the gate, so re-thresholding stored
records is a valid counterfactual: changing the threshold changes whether a reply was reviewed,
never what the model produced. The sweep is therefore computed from stored outputs without
rerunning models, over the scored population of a single policy's records, and is rendered only
when confidence variance passes preflight. Cases without a draft have no `draft_confidence` and
are excluded; the excluded count is printed with the table.

---

## Reporting

Each report includes:

- Dataset size, source, difficulty, and category composition.
- Scored population size beside total cases, with the reason for every exclusion.
- Directional-only labelling for approximately 50-case runs.
- Structured quality metrics and confidence intervals.
- Escalation and invalid-output rates, kept visibly separate from quality metrics.
- Primary-versus-fallback paired comparisons.
- Oracle-versus-rubber-stamp outcomes.
- Review-load versus unreviewed-error threshold table.
- Reliability metrics with cache-hit exclusions.
- Difficulty and source slices.
- Confidence-distribution diagnostics.
- System-invariant violations.
- Judge validation results beside every judge-derived metric.

System invariants include:

- `was_gated` agrees with the recorded workflow threshold and refund rule.
- At most one refund row exists per ticket.
- Refund attempts are greater than or equal to executed refunds.
- An executed refund implies an approved decision.
- A fallback-routing record identifies the fallback path.

---

## Offline LLM judge

The judge scores stored reply text on:

1. Relevance to the ticket, 1–5.
2. Appropriate support tone, 1–5.
3. Hallucinated commitments, boolean.

Judge calibration uses at least 30 replies spanning good, weak, irrelevant, and hallucinated
outputs. Two humans score them independently and adjudicate disagreements.

Validation gates:

- Weighted Cohen's κ of at least 0.60 for relevance.
- Weighted Cohen's κ of at least 0.60 for tone.
- Cohen's κ of at least 0.60 and F1 of at least 0.80 for hallucination.
- Agreement intervals and calibration sample size appear in the report.
- A failing dimension is suppressed without suppressing dimensions that passed.

Judge-derived response-issue rates remain secondary to deterministic structured metrics. Using a
different model family reduces obvious self-grading bias but does not make the judge independent or
ground truth.

---

## CLI

`scripts/eval.py` provides:

```text
dataset-check
run --profile primary-quality|fallback-quality|fallback-routing|reliability
judge --run-id <id>
report --run-id <id>
compare --baseline <id> --candidate <id>
```

Important options:

- `--agent tunable|mock|ollama`
- `--reviewer oracle|rubber_stamp|both`
- `--limit N`
- `--repeats N`
- `--concurrency N`
- `--seed N`
- `--no-cache`
- Model and Ollama endpoint overrides

Make targets:

- `eval-dataset-check`
- `eval` for the fast tunable profile
- `eval-ollama`
- `eval-compare`

Networked tests use an `ollama` marker and remain deselected by default. Registering the marker is
not sufficient on its own: the existing `addopts` deselection must be widened to
`-m 'not smoke and not ollama'`, or the tests run inside `make check`.

---

## Implementation milestones

### Milestone 1 — Evaluation contract and deterministic core

Deliver:

- Labelling guide and approximately 50 verified cases.
- Dataset models and validation.
- Immutable record models.
- Deterministic structured scorers.
- Clustered confidence intervals.
- Basic Markdown and console reports.
- Exact synthetic tests.

Acceptance:

- Constructed records with known outcomes produce exact expected metrics.
- Every reported rate names its denominator, and unscorable cases are excluded from quality rates.
- Invalid and unverified datasets fail with actionable errors.
- No Temporal or Ollama process is required.

### Milestone 2 — Workflow harness and tunable agents

Deliver:

- Reusable in-process Temporal harness, including search-attribute registration, the Pydantic data
  converter, and run-scoped worker lifecycle.
- Post-completion state capture and per-case deadlines.
- Oracle and rubber-stamp reviewers.
- Primary-quality, fallback-quality, fallback-routing, and reliability profiles.
- Stable tunable agent.
- Attempt telemetry and refund-invariant checks.

Acceptance:

- A configured exact error set produces the exact expected workflow-level error rates.
- Output is independent of concurrency and case order.
- Forced fallback quality is distinct from real fallback routing, and routing runs under time
  skipping without paying the schedule-to-start delay in wall-clock time.
- A rejected approval update and an exceeded deadline are recorded, not fatal.
- Existing workflow tests remain unchanged and green.

### Milestone 3 — Ollama integration

Deliver:

- Lifecycle-managed Ollama agent.
- Operation-specific prompts and schemas.
- Periodic activity heartbeats.
- Complete cache keys and atomic entries.
- Preflight probes and timeout adjustment.
- Immutable run artifacts and manifest metadata.
- Stubbed HTTP tests plus a limited local smoke run.

Acceptance:

- Primary and fallback quality profiles run over the same limited case set.
- Policy runs receive byte-identical cached outputs.
- HTTP, validation, and timeout failures map to the expected agent error classes.
- A degraded fallback model reports a measurable `invalid_output_rate` rather than escalating
  every malformed refund.
- Qwen and Gemma are not resident simultaneously.

### Milestone 4 — Decision-grade evaluation

Deliver:

- Dataset growth toward 200 verified cases.
- Paired comparison reports.
- Threshold sweep and confidence diagnostics.
- Judge calibration and versioned judgment artifacts.
- Final source/difficulty slices.

Acceptance:

- The report can support or reject the current 0.75 threshold while displaying uncertainty.
- The primary-versus-fallback comparison reports a paired degradation estimate.
- No judge dimension appears unless its validation gate passes.

---

## Test plan

- Dataset validation rejects duplicate IDs, empty labels, unverified labels, inconsistent refunds,
  and missing generated provenance.
- Structured scorer tests use synthetic records with exact expected rates.
- Scoring tests prove escalated and invalid-output cases are excluded from quality denominators
  and counted in their own rates, and that `gate_recall` and `gate_precision` are identical under
  both reviewer policies.
- Bootstrap tests prove repeats remain clustered by case.
- Comparison tests cover unchanged runs, meaningful regressions, and underpowered changes that must
  not be flagged.
- Tunable-agent tests prove concurrency-independent output and exact injected failures.
- Workflow tests cover ungated resolution, oracle rejection, rubber-stamp approval, forced fallback
  quality, real fallback routing, retry telemetry, and refund idempotency.
- Cache tests prove invalidation on operation, ticket text, classification, prompt, schema, model
  digest, and generation options.
- Ollama HTTP tests use a stub transport for success, malformed output, timeout, 400, 404, 408, 429,
  and 5xx behavior.
- Repair-budget tests prove a schema-invalid response is re-asked once, that a second failure
  raises `AgentPermanentError`, and that a refund missing its amount does not escalate on the
  first attempt.
- Runner tests prove a rejected approval update and an exceeded per-case deadline are recorded as
  case outcomes rather than aborting the run.
- Heartbeat tests prove long agent calls cannot hit the 30-second heartbeat timeout.
- Judge tests cover per-dimension calibration gates and versioned output.
- A complete `make check` remains the final repository regression gate.

---

## Principal risks and mitigations

1. **Slow inference looks like failure.** Mitigate with periodic heartbeats, measured preflight, and
   recorded timeout widening.
2. **Fallback quality is confused with routing behavior.** Keep direct fallback-quality and
   schedule-to-start routing profiles separate.
3. **A stale cache corrupts paired comparisons.** Hash the complete request and model identity.
4. **Category error is presented as reply error.** Keep structured and judged reply metrics
   separate.
5. **Availability failure is presented as model error.** A schema-invalid response escalates the
   ticket without retry or fallback. Give validation failures a repair budget, report
   `invalid_output_rate` and `escalation_rate` separately, and score quality only over cases that
   produced a draft.
6. **Small samples are over-interpreted.** Always report intervals and mark N≈50 runs directional.
7. **Confidence is constant or meaningless.** Diagnose variance and distinctness, and suppress
   unsupported sweeps.
8. **Generated data is easier than real traffic.** Require human verification and report by source.
9. **Judge scores appear authoritative.** Gate each dimension on human agreement and show the
   calibration result.
10. **A missing worker hangs the run.** The fallback activity has no schedule-to-start timeout, so
    the runner enforces its own per-case deadline.

---

## Completion criteria

The harness is complete when:

- `make check` passes with all existing and new network-free tests.
- The tunable profile reproduces exact injected structured-error rates.
- Limited primary and fallback Ollama runs produce immutable, reproducible artifacts.
- Fallback model quality and fallback routing are reported as different experiments.
- Every inferential headline rate includes a 95% interval and names its denominator.
- Model quality, availability, and reviewer policy are reported as separate quantities.
- The threshold report either presents a defensible tradeoff curve or explicitly explains why the
  model's confidence cannot support threshold tuning.
- Judge-derived metrics appear only for validated dimensions.
