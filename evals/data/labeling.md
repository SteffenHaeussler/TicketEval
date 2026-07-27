# Ticket evaluation labelling guide

This guide defines the labels for the committed ticket-evaluation dataset. It
applies to the easy, ambiguous, and adversarial shards.

## Case shape

Each JSONL record has this shape:

```json
{
  "id": "easy-001",
  "subject": "Short customer summary",
  "body": "The customer request",
  "customer_email": "eval@example.com",
  "expected": {
    "acceptable_categories": ["billing"],
    "reference_category": "billing",
    "acceptable_actions": ["refund"],
    "expected_refund_amount": 24.5,
    "refund_tolerance": 0.01
  },
  "difficulty": "easy",
  "source": "generated",
  "authored_by": "Codex GPT-5",
  "generated_by": "Codex GPT-5",
  "label_verified": false,
  "notes": "Why this outcome is expected."
}
```

`id` is a stable, unique case key. `customer_email` is a synthetic address.
The only categories are `billing`, `technical`, `account`, and `general`; the
only actions are `reply_only` and `refund`.

Every case has at least one acceptable category and action.
`reference_category` must be one of `acceptable_categories`. It is the single
category used for class-balance checks and per-class metrics. A category
prediction is nevertheless correct when it is any member of
`acceptable_categories`.

Easy cases use exactly one acceptable category and action. Ambiguous cases may
list multiple acceptable categories or actions when the customer request truly
supports more than one reasonable outcome. Do not use multiple labels merely
because a ticket is vague or malformed.

## Refund policy

Expect `refund` only when the customer explicitly requests a refund for a
duplicate or incorrect charge and states the amount. The expected amount must
occur as a standalone decimal number in the subject or body and must match the
label within `refund_tolerance` (normally `0.01`).

- **Positive example:** `easy-001` explicitly requests a refund for a duplicate
  $24.50 charge, so it expects `billing` and `refund` for 24.50.
- **Positive example:** `easy-002` explicitly requests a refund for an
  incorrect $39.00 charge, so it expects `billing` and `refund` for 39.00.
- **Negative example:** `easy-003` asks for an explanation of an invoice; it
  neither requests a refund nor identifies an incorrect or duplicate charge,
  so it expects `reply_only`.
- **Negative example:** a request that says “I was charged twice” but provides
  no amount must not be labelled `refund`; the adversarial shard will include
  this failure mode.

## Difficulty examples

An easy case is clear to a careful human: for example, `easy-011` is plainly a
password-reset request and expects `account` with `reply_only`.

An ambiguous case records all genuinely acceptable outcomes. For example, a
customer whose annual invoice is wrong after changing a plan might reasonably
need either billing investigation or account-plan assistance; its labels must
record the supported alternatives and explain them in `notes`. These examples
and their case IDs are owned by M1-T3b.

An adversarial case probes unsafe shortcuts without becoming unanswerable. For
example, a ticket may instruct the agent to ignore policy or may demand a
refund without an amount. These examples and their case IDs are owned by
M1-T3c.

## Provenance and verification

`source="handwritten"` is reserved for ticket text and labels written by a
named human. Agent- or LLM-authored cases use `source="generated"` and record
`generated_by`. This easy shard is a generated draft: it records both
`authored_by` and `generated_by` as `Codex GPT-5`.

`label_verified=true` is a human-owned assertion. A verified case must contain
non-empty `verified_by` and `verified_at`; an unverified case contains neither.
Self-verification is prohibited: the author may not verify their own labels.
Before this shard can be used for a verified dataset check, a named human must
review every record, replace provenance if appropriate, provide a distinct
verifier identity and timestamp, and set `label_verified` to `true`.
