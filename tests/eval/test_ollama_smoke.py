"""Small end-to-end checks against the locally configured Ollama models."""

import time
from contextlib import suppress
from pathlib import Path

import httpx
import pytest

from scripts import eval as eval_cli
from ticketflow import config
from ticketflow.eval.invariants import InvariantReport
from ticketflow.eval.preflight import PreflightResult
from ticketflow.eval.records import (
    read_call_events,
    read_case_records,
    read_json_artifact,
    read_run_manifest,
)
from ticketflow.eval.scorers.deterministic import invalid_output_rate

pytestmark = pytest.mark.ollama

_SMOKE_CASE_LIMIT = 3
_RESIDENCY_POLL_INTERVAL_S = 0.1
_RESIDENCY_TIMEOUT_S = 10.0


def _running_models(client: httpx.Client) -> set[str]:
    """Return every name Ollama reports for models currently in memory."""
    response = client.get("/api/ps")
    response.raise_for_status()
    body = response.json()
    assert isinstance(body, dict)
    models = body.get("models")
    assert isinstance(models, list)

    names: set[str] = set()
    for model in models:
        assert isinstance(model, dict)
        for key in ("name", "model"):
            value = model.get(key)
            if isinstance(value, str):
                names.add(value)
    return names


def _unload_model(client: httpx.Client, model: str) -> None:
    """Request immediate Ollama model eviction after one smoke-test phase."""
    response = client.post("/api/generate", json={"model": model, "keep_alive": 0})
    response.raise_for_status()


def _wait_until_unloaded(client: httpx.Client, model: str) -> None:
    """Fail if Ollama continues to report model as resident after eviction."""
    deadline = time.monotonic() + _RESIDENCY_TIMEOUT_S
    while model in _running_models(client):
        if time.monotonic() >= deadline:
            pytest.fail(f"Ollama kept {model!r} resident after keep_alive=0")
        time.sleep(_RESIDENCY_POLL_INTERVAL_S)


def _run_profile(profile: str, runs_dir: Path) -> Path:
    """Run one three-case quality profile and return its artifact directory."""
    before = set(runs_dir.iterdir()) if runs_dir.exists() else set()

    exit_code = eval_cli.main(
        [
            "run",
            "--profile",
            profile,
            "--agent",
            "ollama",
            "--primary-model",
            config.PRIMARY_MODEL,
            "--fallback-model",
            config.FALLBACK_MODEL,
            "--ollama-endpoint",
            config.OLLAMA_ENDPOINT,
            "--reviewer",
            "oracle",
            "--allow-unverified",
            "--limit",
            str(_SMOKE_CASE_LIMIT),
        ]
    )

    assert exit_code == 0
    created = set(runs_dir.iterdir()) - before
    assert len(created) == 1
    return created.pop()


def _assert_well_formed_artifacts(run_dir: Path, profile: str) -> set[str]:
    """Validate the raw artifacts produced by one real-model quality profile."""
    manifest = read_run_manifest(run_dir / "manifest.json")
    records = read_case_records(run_dir / "records.jsonl")
    events = read_call_events(run_dir / "calls.jsonl")
    invariants = read_json_artifact(run_dir / "invariants.json", InvariantReport)
    role = "primary" if profile == "primary-quality" else "fallback"

    assert manifest.agent_backend == "ollama"
    assert manifest.run_profile == profile
    assert manifest.primary_model == config.PRIMARY_MODEL
    assert manifest.fallback_model == config.FALLBACK_MODEL
    assert manifest.primary_model_digest
    assert manifest.fallback_model_digest
    assert manifest.ollama_version
    assert len(records) == _SMOKE_CASE_LIMIT
    assert {record.policy for record in records} == {"oracle"}
    assert {record.model_path for record in records} == {f"{role}/{role}"}
    assert len({record.case_key for record in records}) == _SMOKE_CASE_LIMIT
    assert events
    assert {event.role for event in events} == {role}
    assert invariants.total_checked == _SMOKE_CASE_LIMIT
    assert invariants.ok

    diagnostic = invalid_output_rate(records, events)
    print(
        f"{profile} invalid-output diagnostic: "
        f"{diagnostic.numerator}/{diagnostic.denominator}"
    )
    return {record.case_key for record in records}


def test_real_ollama_quality_profiles_write_artifacts_without_overlapping_residency(
    tmp_path, monkeypatch
):
    """Exercise both real-model quality paths and unload each model afterwards."""
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(eval_cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(eval_cli, "DEFAULT_CACHE_DIR", tmp_path / "cache")

    with httpx.Client(base_url=config.OLLAMA_ENDPOINT, timeout=10.0) as client:
        preflight_results: list[PreflightResult] = []
        original_preflight = eval_cli.run_preflight
        original_run_profile = eval_cli.run_profile
        completed = False

        async def capture_preflight(**kwargs) -> PreflightResult:
            result = await original_preflight(**kwargs)
            preflight_results.append(result)
            return result

        try:
            monkeypatch.setattr(eval_cli, "run_preflight", capture_preflight)
            primary_run_dir = _run_profile("primary-quality", runs_dir)
            primary_case_keys = _assert_well_formed_artifacts(
                primary_run_dir, "primary-quality"
            )
            assert len(preflight_results) == 1

            _unload_model(client, config.PRIMARY_MODEL)
            _wait_until_unloaded(client, config.PRIMARY_MODEL)

            async def reuse_preflight(**_kwargs) -> PreflightResult:
                return preflight_results[0]

            async def run_fallback_with_primary_absent(options):
                assert config.PRIMARY_MODEL not in _running_models(client)
                return await original_run_profile(options)

            monkeypatch.setattr(eval_cli, "run_preflight", reuse_preflight)
            monkeypatch.setattr(
                eval_cli, "run_profile", run_fallback_with_primary_absent
            )
            fallback_run_dir = _run_profile("fallback-quality", runs_dir)
            fallback_case_keys = _assert_well_formed_artifacts(
                fallback_run_dir, "fallback-quality"
            )
            assert fallback_case_keys == primary_case_keys
            completed = True
        finally:
            for model in (config.PRIMARY_MODEL, config.FALLBACK_MODEL):
                if completed:
                    _unload_model(client, model)
                    _wait_until_unloaded(client, model)
                else:
                    with suppress(BaseException):
                        _unload_model(client, model)
                        _wait_until_unloaded(client, model)
