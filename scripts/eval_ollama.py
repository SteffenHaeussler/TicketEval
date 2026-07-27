"""Run the Ollama agent over the labelled eval dataset and score it.

Lightweight standalone precursor to the real eval harness (Milestone 2/3): loads
the committed cases, runs classify + draft_reply on each, and prints accuracy
against the expected labels. Deliberately simple and easy to change later - it is
NOT a pytest gate and does no caching, concurrency, or artifact writing.

    make eval-ollama                 # all cases, default model
    uv run python scripts/eval_ollama.py --limit 5 --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass

import httpx

from ticketflow import config
from ticketflow.agent.base import AgentOverloadedError, AgentPermanentError
from ticketflow.agent.ollama import OllamaAgent
from ticketflow.eval.dataset import EvalCase, load_cases
from ticketflow.models import ActionType, Classification, DraftReply, Ticket

DEFAULT_DATA_DIR = "evals/data/tickets"


class PreflightError(RuntimeError):
    """Raised when Ollama is not reachable before an eval run."""


@dataclass
class CaseResult:
    """Per-case scoring outcome, or an error if the agent failed on it."""

    case_id: str
    difficulty: str
    category_ok: bool
    action_ok: bool
    refund_ok: bool | None
    error: str | None


def score_case(
    case: EvalCase, classification: Classification, draft: DraftReply
) -> CaseResult:
    """Score one agent result against the case's expected labels."""
    expected = case.expected
    refund_ok: bool | None = None
    if expected.expected_refund_amount is not None:
        proposed = draft.action.refund_amount
        refund_ok = (
            draft.action.type == ActionType.REFUND
            and proposed is not None
            and abs(proposed - expected.expected_refund_amount)
            <= expected.refund_tolerance
        )
    return CaseResult(
        case_id=case.id,
        difficulty=case.difficulty,
        category_ok=classification.category in expected.acceptable_categories,
        action_ok=draft.action.type in expected.acceptable_actions,
        refund_ok=refund_ok,
        error=None,
    )


def check_ollama(endpoint: str, model: str) -> None:
    """Fail fast with guidance if the Ollama server is not reachable."""
    try:
        httpx.get(f"{endpoint}/api/version", timeout=2.0).raise_for_status()
    except httpx.HTTPError as exc:
        raise PreflightError(
            f"Ollama not reachable at {endpoint} ({exc}); "
            f"start it with `ollama serve` and `ollama pull {model}`."
        ) from exc


async def run_eval(
    *, data_dir: str, model: str, endpoint: str, limit: int | None
) -> list[CaseResult]:
    """Run the agent over the dataset and return one result per case."""
    cases = load_cases(data_dir, require_verified=False)
    if limit is not None:
        cases = cases[:limit]

    results: list[CaseResult] = []
    async with OllamaAgent(endpoint=endpoint, model=model) as agent:
        for case in cases:
            ticket = Ticket(
                id=case.id,
                customer_email=case.customer_email,
                subject=case.subject,
                body=case.body,
            )
            try:
                classification = await agent.classify(ticket)
                draft = await agent.draft_reply(ticket, classification)
            except (AgentOverloadedError, AgentPermanentError) as exc:
                results.append(
                    CaseResult(case.id, case.difficulty, False, False, None, str(exc))
                )
                continue
            results.append(score_case(case, classification, draft))
    return results


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.0%} ({numerator}/{denominator})"


def print_summary(results: list[CaseResult], *, model: str) -> None:
    """Print overall and per-difficulty scores in CLI-friendly form."""
    errors = [r for r in results if r.error is not None]
    scored = [r for r in results if r.error is None]
    refund_scored = [r for r in scored if r.refund_ok is not None]

    print(f"model: {model}")
    print(f"cases: {len(results)}  scored: {len(scored)}  errored: {len(errors)}")
    print(f"category accuracy: {_pct(sum(r.category_ok for r in scored), len(scored))}")
    print(f"action accuracy:   {_pct(sum(r.action_ok for r in scored), len(scored))}")
    print(
        "refund accuracy:   "
        f"{_pct(sum(bool(r.refund_ok) for r in refund_scored), len(refund_scored))}"
    )

    print("category accuracy by difficulty:")
    by_difficulty: dict[str, list[CaseResult]] = {}
    for result in scored:
        by_difficulty.setdefault(result.difficulty, []).append(result)
    for difficulty in sorted(by_difficulty):
        group = by_difficulty[difficulty]
        print(f"  {difficulty}: {_pct(sum(r.category_ok for r in group), len(group))}")

    if errors:
        counts = Counter((r.error or "").splitlines()[0] for r in errors if r.error)
        print("errors:")
        for message, count in counts.most_common():
            print(f"  {count}x {message}")


def parse_args() -> argparse.Namespace:
    """Parse eval-driver command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score the Ollama agent against the labelled eval dataset."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    parser.add_argument("--endpoint", default=config.OLLAMA_ENDPOINT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Run the Ollama eval driver command."""
    args = parse_args()
    try:
        check_ollama(args.endpoint, args.model)
        results = asyncio.run(
            run_eval(
                data_dir=args.data,
                model=args.model,
                endpoint=args.endpoint,
                limit=args.limit,
            )
        )
    except PreflightError as exc:
        print(f"eval failed: {exc}")
        return 1

    print_summary(results, model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
