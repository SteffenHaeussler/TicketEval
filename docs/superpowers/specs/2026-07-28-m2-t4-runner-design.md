# M2-T4 Per-case Runner Design

## Goal

Run one labelled evaluation case through `TicketWorkflow` and emit its immutable
`CaseRecord` plus enriched `CallEvent` telemetry without allowing an approval
race or missing fallback worker to abort the surrounding evaluation run.

## Design

`CaseRunner` is run-scoped: it receives the Temporal client, workflow queue,
SQLite path, deadline, runtime identity map, and telemetry sink once. Its
`run_case(case, policy, reviewer, repeat_index)` method creates a unique
runtime ticket/workflow ID, registers it to the stable case ID, starts the
workflow, and returns the completed case record and drained events.

The runner polls for `AWAITING_APPROVAL` only until terminal completion. For a
gated draft it builds the reviewer input from captured state and submits the
decision as a workflow update. `WorkflowUpdateFailedError` becomes the
non-fatal `update_rejected` terminal outcome. A wall-clock deadline captures
best-effort state, cancels the workflow, waits five seconds for confirmation,
and terminates it if necessary; a captured draft remains scorable.

After normal completion the terminal `TicketResult` comes only from awaiting
the workflow handle. The post-completion `status` query supplies
classification, draft, and decision; the runner never reads `status.result`.
Refund counts come only from `readmodel.get_refund_observation`.

## Boundaries

This task adds only per-case execution and public refund observation. Batch
concurrency, run profiles, manifests, and the cross-cutting M2-T8 suite remain
out of scope.
