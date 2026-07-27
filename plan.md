# Ticketflow Eval Harness

**Status:** planned, not started · **Date:** 2026-07-27

## Context

Ticketflow is a Temporal learning project: a mocked agent resolves support tickets in a durable
workflow, with a human-approval gate for refunds and low-confidence drafts. Today there is **no
evaluation of any kind** — a repo-wide grep for `eval|scor|metric|judge|benchmark|golden|ground truth`
returns zero hits. The only quantitative output is two histograms printed by `scripts/batch.py`.

That leaves the system's central design decision unmeasured. `CONFIDENCE_THRESHOLD = 0.75`
(`workflows.py:30`) decides which tickets a human sees. If the agent's `confidence` doesn't
correlate with correctness, the approval gate is theatre — it costs human review time and catches
nothing. Nobody can currently answer: *how many wrong replies reach customers unreviewed, and what
would a different threshold cost?*

A second claim is also unfalsifiable. `docs/context.md:92-101` states *"degraded service beats an
outage, and the degradation should be measurable downstream."* But `MockAgent.fallback()` merely
**stipulates** degradation by capping confidence at 0.6 (`mock.py:76`). Nothing measures whether the
fallback path actually produces acceptable answers.

**Goal:** a harness that runs a labelled ticket dataset through the real workflow and reports three
coupled things — agent output quality, system/policy behavior, and the tradeoff curve between
unreviewed errors and human review load. It runs against a tunable mock (free, CI-safe, and how we
verify the harness itself is correct) and against real models served by local Ollama.

**Non-goal:** replacing `scripts/batch.py` or the smoke tests. Those cover the live Docker stack;
this harness runs in-process.

---

## Environment (verified on this machine)

- Apple M5 Pro, **64 GB** unified memory. Ollama **0.30.10**.
- `qwen3.6:35b` — 23 GB, family `qwen35moe` (**MoE**, so far faster than 36B dense suggests),
  context 262144, capabilities `vision, completion, tools, thinking`
- `gemma4:26b` — 17 GB, dense 25.8B, capabilities `completion, tools, thinking`
- `qwen2.5-coder:1.5b` — 986 MB, dense 1.5B

Ollama 0.30.10 fully supports JSON-schema structured output via `format`, so no prompt-embedded
schema fallback is needed.

### Model roles

| Role | Model | Rationale |
|---|---|---|
| Agent, primary | `qwen3.6:35b` | MoE → fast; `tools`; huge context |
| Agent, fallback | `qwen2.5-coder:1.5b` | Genuinely degraded at ~1 GB and near-zero latency. Turns the "degradation is measurable" claim into an actual measurement. |
| Judge | `gemma4:26b` | Different model family from the agent — the strongest judge independence available here. Dense-model slowness is irrelevant; it runs offline in phase 2. |

qwen (23 GB) + gemma (17 GB) = 40 GB, which fits under Ollama's default budget but leaves no
headroom for KV cache. **They are therefore never resident simultaneously** — see two-phase
execution below.

*Known weakness, accepted:* the 1.5B is a *coder* model, so it's an artificial stand-in for "a
smaller general model." The absolute degradation number won't transfer to a real production fallback
choice. A cleaner qwen-vs-gemma comparison is available for free at any time by re-running the
harness with `--model gemma4:26b` and diffing via `compare.py` — no extra machinery.

*Escape hatch:* every role is bound by an env var, so collapsing to a single model (qwen as agent,
fallback **and** judge) if memory or latency proves troublesome is a config change, not a code
change — set `TICKETFLOW_OLLAMA_JUDGE_MODEL=qwen3.6:35b`. The cost is self-grading bias, which the
judge agreement check (Milestone 3) will surface rather than hide.

---

## Key constraints discovered

| Constraint | Source | Consequence |
|---|---|---|
| `TicketResult` has no category/action/confidence | `models.py:84` | Runner must capture `TicketStatusInfo` from the **live workflow query** before termination. Outcomes cannot be reconstructed from the read model. |
| `AGENT_ACTIVITY_TIMEOUT = 2 minutes` | `workflows.py:33` | **Highest risk.** A thinking model exceeding it retries and eventually escalates, silently turning latency into fake "model failures" and corrupting every reliability metric. |
| Both big models advertise `thinking` | `/api/tags` | Pass `think=false` for agent calls. Judge may think freely (offline). |
| Temporal-crossing models may only gain *defaulted* fields | `docs/context.md:132-145` | Design keeps **all** eval data outside the payloads. No model changes at all. |
| `MockAgent._rng` is one RNG shared across all tickets | `mock.py:61` | Seeded runs are concurrency-dependent. New mock must derive a per-ticket RNG. |
| `MockAgent` defaults `failure_rate=0.1` | `mock.py:54` | 10% injected `AgentOverloadedError` confounds quality metrics. Must be a profile knob. |
| Agent is hardcoded at worker construction | `llm_worker.py:31,36`, `worker.py:30` | No config-driven selection exists. This is the blocker; it's Milestone 1. |
| `model_path` = `f"{cls.model}/{draft.model}"` | `workflows.py:209` | Fallback detection depends on the literal strings `primary`/`fallback`. **Ollama agents keep those labels**; real model names go in run metadata. |
| `workflows.py` snapshots config into module constants at import | `workflows.py:35-37` | Env vars set after import have no effect; monkeypatch module attrs (as `test_workflow.py:144` does). |
| ruff enforces `D100-D107` on `src/` and `scripts/`, line length 88 | `pyproject.toml:44-53` | Docstrings mandatory on every new module/class/public function. `tests/` exempt. |

---

## Architecture

```
evals/
  data/tickets.jsonl              # labelled dataset (~50 cases -> ~200)
  data/judge_calibration.jsonl    # ~12 hand-scored replies, to validate the judge
  runs/<run_id>/{manifest.json, records.jsonl, report.md}
  cache/                          # ollama response cache (gitignored)

src/ticketflow/agent/
  base.py         # UNCHANGED - the Agent Protocol
  mock.py         # UNCHANGED
  tunable.py      # NEW  TunableMockAgent - label-aware, knobs for error rate + overconfidence
  ollama.py       # NEW  OllamaAgent + shared schema-constrained httpx client
  __init__.py     # NEW  build_agent(role) factory (file is currently empty)

src/ticketflow/eval/
  dataset.py      # EvalCase model + JSONL loader
  harness.py      # worker/env construction (moved from tests/helpers.py)
  runner.py       # phase 1: drives cases through TicketWorkflow, applies reviewer policy
  records.py      # CaseRecord + RunManifest
  reviewers.py    # oracle / rubber_stamp approval policies
  preflight.py    # ollama reachability + latency probe + timeout widening
  statistics.py   # bootstrap CIs clustered by case; paired run-vs-run tests
  scorers/deterministic.py
  scorers/calibration.py
  scorers/judge.py    # phase 2: offline batch judging over records.jsonl
  report.py       # markdown + console output
  compare.py      # run-vs-baseline diff (paired)

scripts/eval.py   # CLI, following the scripts/batch.py precedent
```

### Two-phase execution (this is what keeps memory sane)

**Phase 1 — workflows.** Run every case through `TicketWorkflow` using only the agent models
(qwen 23 GB + 1.5b ≈ 24 GB resident). Write `records.jsonl`.

**Phase 2 — judging.** Unload the agent models (`POST /api/generate` with `keep_alive: 0`), then
batch-judge the stored `reply_text` values with gemma (17 GB resident). Write judge scores back.

One model swap per run instead of 2N. It also makes judging **re-runnable without re-running
workflows** — you can iterate on the rubric for free, which matters because rubrics always need
iteration.

### Why in-process Temporal, not the live stack

`tests/conftest.py::env` (`WorkflowEnvironment`) + `tests/helpers.py::make_worker` already run a
real ticket end-to-end through a real `TicketWorkflow` with any `Agent`, no Docker and no API. It
also hands us the `WorkflowHandle` directly — which we need, because approvals go through
`handle.execute_update(TicketWorkflow.submit_approval, ...)` and the rich per-case data only exists
in `handle.query(TicketWorkflow.status)`. Going through HTTP would add a process boundary and buy
nothing the smoke tests don't already cover.

**Refactor required:** `make_worker` and `CombinedWorker` move from `tests/helpers.py` into
`src/ticketflow/eval/harness.py`. `tests/helpers.py` re-exports them so the 16 existing tests in
`test_workflow.py` keep working unchanged.

---

## Milestone 1 — Agent seam (prerequisite)

**`src/ticketflow/config.py`** — add, in the existing plain `os.environ.get` module-constant style:

```python
AGENT_IMPL            = os.environ.get("TICKETFLOW_AGENT", "mock")   # mock|tunable|ollama
OLLAMA_BASE_URL       = os.environ.get("TICKETFLOW_OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL          = os.environ.get("TICKETFLOW_OLLAMA_MODEL", "qwen3.6:35b")
OLLAMA_FALLBACK_MODEL = os.environ.get("TICKETFLOW_OLLAMA_FALLBACK_MODEL", "qwen2.5-coder:1.5b")
OLLAMA_JUDGE_MODEL    = os.environ.get("TICKETFLOW_OLLAMA_JUDGE_MODEL", "gemma4:26b")
OLLAMA_TIMEOUT_S      = float(os.environ.get("TICKETFLOW_OLLAMA_TIMEOUT_S", "90"))
OLLAMA_THINK          = os.environ.get("TICKETFLOW_OLLAMA_THINK", "0") == "1"
```

**`src/ticketflow/agent/__init__.py`** — `build_agent(role: Literal["primary","fallback"]) -> Agent`,
dispatching on `config.AGENT_IMPL`. `worker.py:30` and `llm_worker.py:31,36` call it instead of
constructing `MockAgent` directly.

**`src/ticketflow/agent/ollama.py`** — `OllamaAgent`:
- `httpx.AsyncClient` → `POST {base_url}/api/chat`, `stream=False`, `think=config.OLLAMA_THINK`,
  `format=Classification.model_json_schema()`. Grammar-constrained decoding guarantees schema-valid
  output, so even the 1.5B model can't emit malformed JSON — it just picks the wrong category, which
  is exactly the signal we want.
- **Sets `model="primary"` / `"fallback"`**, not the Ollama model name — preserves `model_path`
  semantics per the constraints table. Real model names live in `RunManifest`.
- Optional on-disk response cache keyed by `(ollama_model, role, ticket.id, think)`. The oracle and
  rubber_stamp runs issue identical agent calls, so this **halves inference cost**, and re-runs for
  debugging are free. Only successful responses are cached, so retry paths still exercise properly.
  `--no-cache` for reliability-focused runs.
- Error mapping — this is what makes the reliability metrics meaningful:
  - connect error / timeout / 5xx / 429 → `AgentOverloadedError` (retryable, exercises the
    workflow's hand-rolled retry at `workflows.py:216-246`)
  - 404 model-not-found, repeated schema-validation failure → `AgentPermanentError`
    (non-retryable → ESCALATED)

**`src/ticketflow/agent/tunable.py`** — `TunableMockAgent(labels, error_rate, overconfidence,
refund_precision, failure_rate, seed)`. Reads the expected category from an injected label map, then
deliberately errs with probability `error_rate` and skews confidence by `overconfidence`. Per-ticket
RNG seeded from `(seed, ticket.id)` so results are concurrency-independent.

This exists **to test the harness**: wire in `error_rate=0.2` and the accuracy scorer must report
~0.80; inject overconfidence and the calibration scorer must detect it. Without it the scorers are
unfalsifiable.

**`src/ticketflow/eval/preflight.py`** — before any ollama run: check `/api/version`, confirm the
required models exist in `/api/tags`, then time one real `classify` + one `draft_reply`. Report
observed latency; if p50 is within 3× of `AGENT_ACTIVITY_TIMEOUT`, monkeypatch
`workflows.AGENT_ACTIVITY_TIMEOUT` wider and say so loudly in the manifest. Prevents slow inference
from masquerading as model failure.

---

## Milestone 2 — Dataset, runner, deterministic scorers

**`evals/data/tickets.jsonl`**

```python
class EvalCase(BaseModel):
    id: str
    subject: str
    body: str
    customer_email: str = "eval@example.com"
    expected_category: TicketCategory
    expected_action: ActionType
    expected_refund_amount: float | None = None
    difficulty: Literal["easy", "ambiguous", "adversarial"]
    source: Literal["handwritten", "generated"]   # provenance, for bias slicing
    label_verified: bool = False                  # a human confirmed the label
    notes: str | None = None
```

### Sizing — why 50 is not enough

95% CI half-widths, computed for this design:

| Metric | N=50 | N=200 |
|---|---|---|
| Overall accuracy | **±11 pts** | ±5.5 pts |
| `unreviewed_error_rate` (~12%) | **±9 pts** (~6 events) | ±4.5 pts (~24 events) |
| Adversarial tier | **±28 pts** (10 cases) | ±13 pts |
| Per-class recall (4 classes) | **±23 pts** | ±11 pts |

At N=50 the headline metric's interval is roughly [3%, 21%] — spanning "fine" to "one customer in
five gets a bad answer." Not decision-grade. And if the model is *good* (95% accuracy), the metric
rests on **2 events out of 50**.

**Repeats do not fix this.** 50 cases × 3 repeats is *not* N=150 — the three observations of a case
are clustered by that case's difficulty. Repeats measure model **self-consistency**, a different
(also useful) quantity. They cannot buy precision on accuracy. Accordingly: `--repeats` defaults to
**1**; `--repeats 3` runs on a ~30-case subset and is reported separately as a self-consistency
rate, **never folded into accuracy**.

**Compute is not the constraint** — qwen is MoE with short schema-constrained outputs (~6-8s/case),
so 200 cases is ~25 min per agent pass, and the response cache lets both reviewer policies share it.
A full run with judging is ≈1 hour. The constraint is *labelling effort*.

### Target: N≈200, staged

- **M2 ships ~50 hand-written cases** — enough to prove the harness works; never blocked on labelling.
- **M3 grows to ~200**, re-stratified: **60 easy / 80 ambiguous / 60 adversarial**. (The original
  20/20/10 had it backwards — the adversarial tier is the most informative and deserves the most
  cases, not the fewest.)

Tiers:
- **easy** — keyword-clean, resolvable by `KEYWORD_CATEGORIES` (`mock.py:17`). Baseline sanity;
  `MockAgent` should near-ace these, proving the harness isn't broken. Seed from
  `scripts/batch.py:22 KEYWORD_TEMPLATES`.
- **ambiguous** — multi-intent ("charged twice after the app crashed during checkout").
- **adversarial** — keyword traps ("no crash at all, I just need my invoice" → BILLING despite
  containing "crash"). `MockAgent` fails these *by construction*, proving the harness measures
  something real rather than echoing the mock.

### Growing the dataset (M3)

Generate variants **with Claude, never with qwen or gemma** — a model must not be evaluated on data
authored by itself or its judge. Then a human verifies every label and sets `label_verified=true`.

Two safeguards against the known optimism risk:
1. **Every metric is sliced by `source`.** If generated cases score materially higher than
   hand-written ones, the stereotyping bias is *visible in the report* rather than silently inflating
   the headline number.
2. **Track label churn during verification.** If a human corrects more than ~10% of generated
   labels, the generation prompt is bad and the batch is discarded, not patched.

`dataset.py` validates on load: unique ids, tier and per-class balance within tolerance, all
`label_verified`, refund cases carry `expected_refund_amount`. Wire it to `make eval-dataset-check`.

### Runner

**`src/ticketflow/eval/runner.py`** — per case, per repeat:
1. `handle = await client.start_workflow(TicketWorkflow.run, ticket, id=f"eval-{run_id}-{case.id}-{rep}", task_queue=queue)`
2. Poll `handle.query(TicketWorkflow.status)` — **capture `info.classification` and `info.draft` here**;
   they are unavailable after termination.
3. If `AWAITING_APPROVAL` → reviewer policy decides → `handle.execute_update(TicketWorkflow.submit_approval, ApprovalDecision(...))`
4. `await handle.result()` → `TicketResult`
5. Query the read model's `refunds` / `refund_attempts` tables (`readmodel.py:21,25`) for side-effect checks
6. Emit `CaseRecord`

Default `--concurrency 2` for ollama profiles: Ollama serializes past `OLLAMA_NUM_PARALLEL`, and
queued requests burn the activity's start-to-close budget. Mock profiles can run wide.

**`reviewers.py`** — two policies, both run so the gap between them is measurable:
- `oracle` — approve iff the draft is actually correct per the label
- `rubber_stamp` — always approve

**`records.py`** — `CaseRecord`: run_id, case_id, repeat, ticket_id, predicted_category,
classification_confidence, predicted_action, predicted_refund_amount, draft_confidence, reply_text,
model_path, terminal_status, refund_executed, was_gated, approval_granted, refund_rows,
refund_attempt_rows, latency_ms, error, judge_scores (filled in phase 2). `RunManifest`: run_id,
timestamp, agent impl + **real Ollama model names**, think flag, reviewer policy, dataset path +
sha256, repeats, seed, cache on/off, and a snapshot of the workflow constants
(`CONFIDENCE_THRESHOLD`, timeouts, including any preflight widening) so a run stays interpretable
months later.

### Scorers

**`scorers/deterministic.py`**:
- *Quality*: category accuracy, per-class P/R/F1, confusion matrix; action accuracy; refund-amount error
- *Harm* — **the headline metric**: `unreviewed_error_rate` = P(RESOLVED ∧ ¬gated ∧ category wrong).
  A wrong reply reached a customer with no human in the loop.
- *Cost*: `review_load` = P(gated). These two trade off, and that tradeoff is the eval.
- *Value of the gate*: under `oracle`, P(gated ∧ wrong ∧ rejected) — errors the human actually caught.
  The `oracle` − `rubber_stamp` delta prices the human-in-the-loop.
- *System invariants* (harness/workflow bugs, not model quality — any hit is a defect):
  - `was_gated == (action == REFUND or draft.confidence < 0.75)`
  - `refund_rows <= 1` per ticket and `refund_attempt_rows >= refund_rows` (idempotency,
    same check as `test_activities.py:40`)
  - `refund_executed` implies an approved decision
- *Reliability*: escalation rate, `model_path` histogram (fallback usage), retry counts
- *Quality × degradation*: every quality metric sliced by `model_path` — this is where the
  qwen-vs-1.5b gap becomes a number, closing the loop on `docs/context.md:92-101`

**`scorers/calibration.py`** — 10-bin reliability curve, ECE, Brier score, and the **threshold sweep**:
for t in 0.0…1.0 step 0.05, recompute (review_load, unreviewed_error_rate) offline from the stored
records. This is what turns the harness into a decision tool — `CONFIDENCE_THRESHOLD` becomes a
tuned number instead of a guess.

**`statistics.py`** — the piece that keeps the numbers honest:
- **Bootstrap CIs clustered by case** (resample *cases*, not observations) on every reported metric,
  so repeats can never masquerade as sample size. No point estimate is ever printed without its
  interval.
- **Paired run-vs-run testing** — McNemar for accuracy-type metrics, paired bootstrap for rates.
  Because two runs share the same cases, pairing cancels case difficulty and detects far smaller
  deltas than the unpaired intervals above. **This is what makes regression detection usable well
  below N=200**, and it is cheap to implement.
- Self-consistency rate from `--repeats` runs, reported as its own metric.

**`report.py` / `compare.py`** — write `records.jsonl` + `manifest.json` + `report.md`; print a
console summary with CIs and per-`source` slices; diff a run against a baseline run_id using the
paired tests and flag regressions that are *statistically* distinguishable, not merely numerically
different.

---

## Milestone 3 — LLM judge (phase 2)

**`scorers/judge.py`** — runs **offline over `records.jsonl`**, after the agent models are unloaded.
Uses `gemma4:26b`, a different family from the agent (a model grading itself is worthless). Thinking
may be enabled here since there's no activity timeout to trip. Rubric, three dimensions returned as
a structured `JudgeVerdict` via the same JSON-schema mechanism:
1. **Relevance** — does the reply address *this* ticket? (1-5)
2. **Tone** — appropriate for support? (1-5)
3. **No hallucinated commitments** — invented refund amounts, dates, or promises? (bool; the
   safety-relevant one)

**Judge validation is not optional.** `evals/data/judge_calibration.jsonl` holds ~12 hand-scored
replies (deliberately spanning good / mediocre / hallucinating). Every `--judge` run first scores
those and reports agreement (exact-match % and Cohen's κ) in the report header. A judge score
without its agreement number is not reportable.

---

## Testing the harness

New `tests/test_eval_*.py`, mock-only, no network, run by default in `make test`:
- `TunableMockAgent(error_rate=0.2, seed=…)` → accuracy scorer reports ~0.80 within tolerance
- Calibrated agent → ECE ≈ 0; `overconfidence=0.3` → ECE above threshold. **Falsifies the calibration scorer.**
- Policy-invariant checker flags a deliberately-mismatched record
- Idempotency checker catches a synthetic double-refund
- Reviewer policies decide correctly on constructed records
- `OllamaAgent` error mapping tested against a stubbed `httpx` transport (no server)
- Real Ollama runs live behind a new `eval` marker, default-deselected — following the existing
  `smoke` precedent (`pyproject.toml:39-42`, opted in with `-o addopts=`)

**`pyproject.toml`**: move `httpx` from dev to runtime deps; register the `eval` marker.
**`Makefile`**: `eval` (tunable mock, fast, no network), `eval-ollama`, `eval-compare`.
**`.gitignore`**: `evals/runs/`, `evals/cache/`.

---

## Risks

1. **Activity timeout vs. real inference latency (highest).** `AGENT_ACTIVITY_TIMEOUT = 2 min`
   (`workflows.py:33`) is a hard cap; a thinking model that exceeds it retries and escalates,
   turning slow inference into fake model failures across every reliability metric.
   **Mitigation:** `think=false` for agents, the preflight latency probe, and explicit timeout
   widening recorded in the manifest.
2. **Time-skipping vs. real latency.** `start_time_skipping()` may advance the clock past
   `AGENT_SCHEDULE_TO_START_S=30` while an Ollama call is in flight, producing phantom fallbacks.
   **Mitigation:** mock profiles use `start_time_skipping()`; the ollama profile uses
   `WorkflowEnvironment.start_local()` (real clock). The harness approves within seconds, so the 24h
   timer never matters there. *Verify the exact time-skipping-control API against the installed
   `temporalio` version before relying on it.*
3. **Throughput.** 50 cases × 2 reviewer policies × 2 agent calls = 200 qwen calls per run. The
   response cache cuts this to ~100 (reviewer policies share agent calls). Default repeats to 1 for
   ollama; provide `--limit` for smoke runs.
4. **Judge noise** — mitigated by the mandatory agreement check; if κ is low, report deterministic
   metrics only.
5. **Coder model as fallback stand-in** — accepted, documented above; the qwen-vs-gemma comparison
   via `compare.py` is the cleaner study when wanted.
6. **Under-powered metrics read as fact.** The most likely way this harness misleads is someone
   quoting "87% accuracy" from a 50-case run as if it meant something. **Mitigation:** no point
   estimate is ever rendered without its CI, and `report.md` states the N and tier sizes in its
   header.
7. **Generated-data optimism** — Claude-authored tickets are cleaner and more stereotyped than real
   support traffic, so absolute scores will run optimistic. **Mitigation:** the per-`source` slice
   makes the gap visible; the hand-written adversarial tier remains the honest signal.

---

## Verification

1. `make check` — format, lint (docstrings!), pyright, tests all green.
2. `make test` — the 16 existing `test_workflow.py` tests still pass after `make_worker` moves out of
   `tests/helpers.py`. This is the main regression risk of the refactor.
3. `uv run python scripts/eval.py --agent tunable --error-rate 0.2 --seed 42` → accuracy ≈ 0.80.
   **The harness measuring a known-wrong agent correctly is the core proof it works.**
4. `--agent mock` → near-perfect on the `easy` tier, poor on `adversarial`. Confirms the dataset
   discriminates rather than echoing the mock's keyword table.
5. `uv run python scripts/eval.py --agent ollama --limit 5` — a smoke run confirming preflight,
   structured output, and observed latency well under the activity timeout.
6. Full run: `uv run python scripts/eval.py --agent ollama --reviewer both --judge` → `report.md`.
   Watch `/api/ps` during the run to confirm gemma and qwen are **never resident together**.
7. Read the threshold sweep in that report and check whether 0.75 is defensible. If the curve says
   otherwise, that finding *is* the deliverable.
8. Check the `model_path` slice: qwen-primary vs 1.5b-fallback quality gap. That number is the
   answer to "is degraded service actually acceptable?"
9. `scripts/eval.py --baseline <run_id>` against a re-run → near-zero deltas for the seeded mock,
   and the paired test reports *no significant change*.
10. Statistics sanity: a synthetic pair of runs differing by a known 5-point margin is flagged by the
    paired test but **not** by the unpaired CIs at N=50 — demonstrating the pairing is doing real work.
11. Every metric in `report.md` carries a CI, and no CI is so wide as to be uninterpretable at the
    dataset size actually used.

---

## Suggested order

M1 (agent seam + tunable mock + ollama agent + preflight) → M2 (~50-case dataset + runner +
deterministic scorers + statistics + report) → M3 (grow dataset to ~200 + judge + calibration sweep +
paired compare).

Each milestone is independently useful. M2 already answers the unreviewed-error-rate question — but
only with ±9-point error bars, so treat its output as directional until M3 lands the larger dataset.

---

## Open questions to resolve early

1. **Does the local model's `confidence` carry any signal?** If qwen emits a near-constant 0.9, the
   calibration work still yields a correct and useful answer — "the gate is theatre" — but the
   threshold sweep degenerates to a flat line rather than a tuning tool. Probe with ~10 cases before
   building the sweep.
2. **The `temporalio` time-skipping-control API is unverified** (no venv was installed when this plan
   was written). Check it against the installed version before the ollama profile depends on it.
3. **Judge agreement κ may come back too low to report**, in which case M3 degrades to deterministic
   metrics only.
