# M3-T8 Ollama Smoke Run Design

The opt-in smoke test will run the existing evaluation CLI against a real Ollama
server. It selects three cases, one per difficulty, and executes the
`primary-quality` and `fallback-quality` profiles sequentially with the oracle
reviewer. Each profile writes into a temporary artifacts and cache directory.

The test validates each immutable run artifact through the existing readers,
checks model provenance and role-based paths, compares both profiles' case-key
sets, and emits the observed invalid-output rate as diagnostic data without
requiring it to be non-zero. The primary run's real preflight result is reused
for the fallback CLI invocation, because preflight deliberately measures the
primary model for the shared workflow timeout. Between profiles the test unloads
the primary through `POST /api/generate` with `keep_alive: 0`, polling
`GET /api/ps` until it is absent and asserting that state immediately before
fallback execution. Successful runs also verify final eviction of both models;
failure cleanup does not mask the original test error. The test is marked
`ollama`; it remains excluded from the normal test command.
