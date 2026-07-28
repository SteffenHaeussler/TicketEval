"""Deterministic, profile-driven mock agent for milestone 2 of the eval harness.

`TunableMockAgent` starts from each case's expected outcome and perturbs it
according to a `TunableAgentProfile`, deriving every decision from a stable hash of
`(generation_seed, case_key, operation, stream)` rather than a shared mutable RNG.
This makes its output independent of concurrency, case ordering, and reviewer
policy, per plan.md's "Tunable mock" section.
"""

import hashlib
import random
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ticketflow.agent.base import AgentOverloadedError
from ticketflow.eval.dataset import ExpectedOutcome
from ticketflow.eval.telemetry import RuntimeIdentityMap, TelemetrySink
from ticketflow.models import (
    ActionType,
    Classification,
    DraftReply,
    ProposedAction,
    Ticket,
    TicketCategory,
)

_OTHER_ACTION = {
    ActionType.REPLY_ONLY: ActionType.REFUND,
    ActionType.REFUND: ActionType.REPLY_ONLY,
}


class TunableAgentProfile(BaseModel):
    """Perturbation knobs controlling one TunableMockAgent's behavior."""

    model_config = ConfigDict(frozen=True)

    category_error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    category_error_case_keys: frozenset[str] = frozenset()

    action_error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    action_error_case_keys: frozenset[str] = frozenset()

    refund_amount_error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    refund_amount_error_case_keys: frozenset[str] = frozenset()

    confidence_correct_range: tuple[float, float] = (0.75, 0.98)
    confidence_incorrect_range: tuple[float, float] = (0.35, 0.65)
    overconfidence_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    overconfidence_case_keys: frozenset[str] = frozenset()

    transient_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    transient_failure_case_keys: frozenset[str] = frozenset()

    role: Literal["primary", "fallback"] = "primary"

    @model_validator(mode="after")
    def _ranges_are_valid(self) -> "TunableAgentProfile":
        """Reject a confidence range whose low bound exceeds its high bound."""
        for name, (low, high) in (
            ("confidence_correct_range", self.confidence_correct_range),
            ("confidence_incorrect_range", self.confidence_incorrect_range),
        ):
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(f"{name} must satisfy 0 <= low <= high <= 1")
        return self


class TunableAgentError(Exception):
    """Base class for TunableMockAgent configuration failures."""


class UnknownCaseError(TunableAgentError):
    """Raised when a resolved case_key has no entry in expected_outcomes."""


def _stream_rng(
    generation_seed: int, case_key: str, operation: str, stream: str
) -> random.Random:
    """Return a fresh, deterministic RNG for one (seed, case, operation, stream)."""
    digest = hashlib.sha256(
        "\x1f".join((str(generation_seed), case_key, operation, stream)).encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


class TunableMockAgent:
    """Deterministic mock agent whose output is a pure hash of case identity."""

    def __init__(
        self,
        *,
        identity_map: RuntimeIdentityMap,
        telemetry_sink: TelemetrySink,
        expected_outcomes: Mapping[str, ExpectedOutcome],
        profile: TunableAgentProfile,
        generation_seed: int,
    ) -> None:
        """Store the identity/telemetry collaborators, case labels, and profile."""
        self._identity_map = identity_map
        self._telemetry_sink = telemetry_sink
        self._expected_outcomes = expected_outcomes
        self._profile = profile
        self._generation_seed = generation_seed

    def _expected(self, case_key: str) -> ExpectedOutcome:
        """Return the expected outcome for case_key or raise UnknownCaseError."""
        expected = self._expected_outcomes.get(case_key)
        if expected is None:
            raise UnknownCaseError(f"no expected outcome registered for {case_key!r}")
        return expected

    def _triggered(
        self,
        case_key: str,
        exact_case_keys: frozenset[str],
        rate: float,
        rng: random.Random,
    ) -> bool:
        """Return whether an exact-set or rate-based perturbation should fire."""
        return case_key in exact_case_keys or rng.random() < rate

    def _record(
        self,
        *,
        ticket_id: str,
        operation: Literal["classify", "draft"],
        started_at: datetime,
        outcome: Literal["success", "transient_error"],
        error_type: str | None,
    ) -> None:
        """Write one attempt to the telemetry sink."""
        self._telemetry_sink.record(
            ticket_id=ticket_id,
            operation=operation,
            role=self._profile.role,
            cache_hit=False,
            started_at=started_at,
            wall_latency_ms=0.0,
            model_total_duration_ms=None,
            model_load_duration_ms=None,
            outcome=outcome,
            error_type=error_type,
        )

    def _maybe_fail(
        self, case_key: str, operation: Literal["classify", "draft"], ticket_id: str
    ) -> None:
        """Raise AgentOverloadedError and record it if a transient failure fires."""
        rng = _stream_rng(
            self._generation_seed, case_key, operation, "transient_failure"
        )
        profile = self._profile
        if not self._triggered(
            case_key,
            profile.transient_failure_case_keys,
            profile.transient_failure_rate,
            rng,
        ):
            return
        self._record(
            ticket_id=ticket_id,
            operation=operation,
            started_at=datetime.now(timezone.utc),
            outcome="transient_error",
            error_type=AgentOverloadedError.__name__,
        )
        raise AgentOverloadedError(
            f"tunable agent: injected transient failure for case {case_key!r} "
            f"({operation})"
        )

    def _confidence(
        self, case_key: str, operation: Literal["classify", "draft"], *, correct: bool
    ) -> float:
        """Draw a calibrated or overconfident confidence value for one operation."""
        profile = self._profile
        rng = _stream_rng(self._generation_seed, case_key, operation, "confidence")
        if correct:
            low, high = profile.confidence_correct_range
        else:
            overconfidence_rng = _stream_rng(
                self._generation_seed, case_key, operation, "overconfidence"
            )
            overconfident = self._triggered(
                case_key,
                profile.overconfidence_case_keys,
                profile.overconfidence_rate,
                overconfidence_rng,
            )
            low, high = (
                profile.confidence_correct_range
                if overconfident
                else profile.confidence_incorrect_range
            )
        return rng.uniform(low, high)

    def _classify_category(
        self, case_key: str, expected: ExpectedOutcome
    ) -> tuple[TicketCategory, bool]:
        """Return the (possibly perturbed) category and whether it is correct."""
        profile = self._profile
        rng = _stream_rng(self._generation_seed, case_key, "classify", "category_error")
        triggered = self._triggered(
            case_key, profile.category_error_case_keys, profile.category_error_rate, rng
        )
        if not triggered:
            return expected.reference_category, True

        candidates = sorted(
            (c for c in TicketCategory if c not in expected.acceptable_categories),
            key=lambda c: c.value,
        )
        if not candidates:
            return expected.reference_category, True

        pick_rng = _stream_rng(
            self._generation_seed, case_key, "classify", "category_pick"
        )
        picked = pick_rng.choice(candidates)
        return picked, picked == expected.reference_category

    def _baseline_action(self, expected: ExpectedOutcome) -> ActionType:
        """Return the deterministic baseline action for an expected outcome."""
        if (
            ActionType.REFUND in expected.acceptable_actions
            and expected.expected_refund_amount is not None
        ):
            return ActionType.REFUND
        return ActionType.REPLY_ONLY

    def _draft_action(
        self, case_key: str, expected: ExpectedOutcome
    ) -> tuple[ActionType, bool]:
        """Return the (possibly perturbed) action and whether it is correct."""
        profile = self._profile
        rng = _stream_rng(self._generation_seed, case_key, "draft", "action_error")
        baseline = self._baseline_action(expected)
        triggered = self._triggered(
            case_key, profile.action_error_case_keys, profile.action_error_rate, rng
        )
        if not triggered:
            return baseline, True
        return _OTHER_ACTION[baseline], False

    def _refund_amount(
        self, case_key: str, expected: ExpectedOutcome, *, baseline_action: ActionType
    ) -> tuple[float, bool]:
        """Return the (possibly perturbed) refund amount and whether it is correct."""
        profile = self._profile
        if (
            baseline_action == ActionType.REFUND
            and expected.expected_refund_amount is not None
        ):
            baseline_amount = expected.expected_refund_amount
        else:
            synth_rng = _stream_rng(
                self._generation_seed, case_key, "draft", "synthetic_refund_amount"
            )
            baseline_amount = round(10.0 + synth_rng.random() * 90.0, 2)

        rng = _stream_rng(
            self._generation_seed, case_key, "draft", "refund_amount_error"
        )
        triggered = self._triggered(
            case_key,
            profile.refund_amount_error_case_keys,
            profile.refund_amount_error_rate,
            rng,
        )
        if not triggered:
            return baseline_amount, True

        offset_rng = _stream_rng(
            self._generation_seed, case_key, "draft", "refund_amount_offset"
        )
        offset = expected.refund_tolerance * 10.0 + offset_rng.uniform(1.0, 20.0)
        sign_rng = _stream_rng(
            self._generation_seed, case_key, "draft", "refund_amount_sign"
        )
        sign = 1.0 if sign_rng.random() < 0.5 else -1.0
        amount = max(baseline_amount + sign * offset, 0.01)
        return round(amount, 2), False

    async def classify(self, ticket: Ticket) -> Classification:
        """Classify by hashing case identity, perturbing per the configured profile."""
        case_key = self._identity_map.resolve(ticket.id)
        expected = self._expected(case_key)
        self._maybe_fail(case_key, "classify", ticket.id)

        category, correct = self._classify_category(case_key, expected)
        confidence = self._confidence(case_key, "classify", correct=correct)

        self._record(
            ticket_id=ticket.id,
            operation="classify",
            started_at=datetime.now(timezone.utc),
            outcome="success",
            error_type=None,
        )
        return Classification(
            category=category, confidence=confidence, model=self._profile.role
        )

    async def draft_reply(
        self, ticket: Ticket, classification: Classification
    ) -> DraftReply:
        """Draft a reply by hashing case identity, perturbing per the profile."""
        case_key = self._identity_map.resolve(ticket.id)
        expected = self._expected(case_key)
        self._maybe_fail(case_key, "draft", ticket.id)

        baseline_action = self._baseline_action(expected)
        action, action_correct = self._draft_action(case_key, expected)

        refund_correct = True
        if action == ActionType.REFUND:
            amount, refund_correct = self._refund_amount(
                case_key, expected, baseline_action=baseline_action
            )
            proposed = ProposedAction(type=action, refund_amount=amount)
        else:
            proposed = ProposedAction(type=action)

        confidence = self._confidence(
            case_key, "draft", correct=action_correct and refund_correct
        )

        self._record(
            ticket_id=ticket.id,
            operation="draft",
            started_at=datetime.now(timezone.utc),
            outcome="success",
            error_type=None,
        )
        return DraftReply(
            reply_text=(
                f"[tunable/{self._profile.role}] resolving your "
                f"{classification.category.value} request."
            ),
            action=proposed,
            confidence=confidence,
            model=self._profile.role,
        )
