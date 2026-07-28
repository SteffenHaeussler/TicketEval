import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, get_args

import pytest

from ticketflow import workflows
from ticketflow.agent.tunable import TunableAgentProfile
from ticketflow.eval import profiles
from ticketflow.eval.dataset import EvalCase, ExpectedOutcome
from ticketflow.eval.harness import current_workflow_eval_config
from ticketflow.eval.invariants import check_all_invariants
from ticketflow.eval.profiles import (
    GitInfo,
    ProfileConfigError,
    RunOptions,
    RunProfile,
    dataset_sha256,
    derive_generation_seed,
    run_profile,
    validate_repeats_cache,
)
from ticketflow.eval.records import RunManifest
from ticketflow.models import ActionType, TicketCategory


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        customer_email="eval@example.com",
        subject="Help with my account",
        body="Please help me log in.",
        expected=ExpectedOutcome(
            acceptable_categories=frozenset({TicketCategory.BILLING}),
            reference_category=TicketCategory.BILLING,
            acceptable_actions=frozenset({ActionType.REPLY_ONLY}),
        ),
        difficulty="easy",
        source="handwritten",
        authored_by="tester",
        label_verified=True,
        verified_by="reviewer",
        verified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _cases(n: int) -> list[EvalCase]:
    return [_case(f"case-{i}") for i in range(n)]


def _write_dataset(tmp_path: Path, cases: list[EvalCase]) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(case.model_dump_json() for case in cases) + "\n")
    return path


def _options(
    tmp_path: Path, cases: list[EvalCase], profile: RunProfile, **overrides: Any
) -> RunOptions:
    dataset_path = overrides.pop("dataset_path", None) or _write_dataset(
        tmp_path, cases
    )
    defaults: dict[str, Any] = dict(
        profile=profile,
        dataset_path=dataset_path,
        cases=cases,
        primary_agent_profile=TunableAgentProfile(),
        db_path=str(tmp_path / "readmodel.db"),
        case_deadline=timedelta(seconds=10),
    )
    defaults.update(overrides)
    return RunOptions(**defaults)


def _fake_git(commit: str, porcelain: str):
    """Build a _run_git replacement returning fixed rev-parse/status output."""

    def _run(*args: str, cwd: Path) -> str:
        if args == ("rev-parse", "HEAD"):
            return f"{commit}\n"
        if args == ("status", "--porcelain"):
            return porcelain
        raise AssertionError(f"unexpected git args: {args}")

    return _run


# -- RunProfile / RunManifest literal agreement --------------------------------


def test_run_profile_literal_matches_run_manifest_literal():
    manifest_literal = RunManifest.model_fields["run_profile"].annotation
    assert set(get_args(RunProfile)) == set(get_args(manifest_literal))


# -- Seed derivation -------------------------------------------------------------


def test_derive_generation_seed_is_deterministic_and_repeat_sensitive():
    same_again = derive_generation_seed(0, 0)
    assert derive_generation_seed(0, 0) == same_again
    assert derive_generation_seed(0, 0) != derive_generation_seed(0, 1)
    assert derive_generation_seed(0, 0) != derive_generation_seed(1, 0)


# -- validate_repeats_cache / RunOptions validation -------------------------------


def test_validate_repeats_cache_rejects_repeats_greater_than_one_with_cache_enabled():
    with pytest.raises(ProfileConfigError):
        validate_repeats_cache(2, True)
    validate_repeats_cache(2, False)
    validate_repeats_cache(1, True)


def test_run_options_rejects_repeats_gt_one_with_cache_enabled_at_construction():
    with pytest.raises(ProfileConfigError):
        RunOptions(
            profile="primary-quality",
            dataset_path="unused",
            cases=[_case("case-1")],
            primary_agent_profile=TunableAgentProfile(),
            repeats=2,
            cache_enabled=True,
        )


@pytest.mark.parametrize("profile", ["fallback-quality", "fallback-routing"])
def test_run_options_requires_fallback_profile_for_fallback_profiles(profile):
    with pytest.raises(ProfileConfigError):
        RunOptions(
            profile=profile,
            dataset_path="unused",
            cases=[_case("case-1")],
            primary_agent_profile=TunableAgentProfile(),
        )


def test_run_options_rejects_non_tunable_agent_backend():
    with pytest.raises(ProfileConfigError):
        RunOptions(
            profile="primary-quality",
            dataset_path="unused",
            cases=[_case("case-1")],
            primary_agent_profile=TunableAgentProfile(),
            agent_backend="mock",
        )


def test_run_options_rejects_empty_cases():
    with pytest.raises(ProfileConfigError):
        RunOptions(
            profile="primary-quality",
            dataset_path="unused",
            cases=[],
            primary_agent_profile=TunableAgentProfile(),
        )


# -- git_info ----------------------------------------------------------------------


def test_git_info_reports_clean_tree(monkeypatch):
    monkeypatch.setattr(profiles, "_run_git", _fake_git("abc123", ""))
    assert profiles.git_info() == GitInfo(commit="abc123", dirty=False)


def test_git_info_reports_dirty_tree_truthfully(monkeypatch):
    monkeypatch.setattr(
        profiles,
        "_run_git",
        _fake_git("abc123", " M src/ticketflow/eval/profiles.py\n"),
    )
    assert profiles.git_info() == GitInfo(commit="abc123", dirty=True)


# -- dataset_sha256 ------------------------------------------------------------------


def test_dataset_sha256_is_deterministic(tmp_path):
    shard = tmp_path / "shard.jsonl"
    shard.write_text('{"id": "case-1"}\n')
    assert dataset_sha256(tmp_path) == dataset_sha256(tmp_path)


def test_dataset_sha256_changes_when_shard_content_changes(tmp_path):
    shard = tmp_path / "shard.jsonl"
    shard.write_text('{"id": "case-1"}\n')
    before = dataset_sha256(tmp_path)
    shard.write_text('{"id": "case-2"}\n')
    assert dataset_sha256(tmp_path) != before


def test_dataset_sha256_changes_when_shard_renamed(tmp_path):
    shard_a = tmp_path / "a.jsonl"
    shard_a.write_text('{"id": "case-1"}\n')
    before = dataset_sha256(tmp_path)
    shard_a.unlink()
    shard_b = tmp_path / "b.jsonl"
    shard_b.write_text('{"id": "case-1"}\n')
    assert dataset_sha256(tmp_path) != before


# -- Per-profile behavior -------------------------------------------------------------


async def test_primary_quality_profile_labels_records_primary(tmp_path):
    cases = _cases(2)
    options = _options(tmp_path, cases, "primary-quality")

    manifest, records, _events = await run_profile(options)

    assert manifest.reviewer_policies == ["oracle", "rubber_stamp"]
    assert len(records) == 2 * len(cases)
    assert all(record.model_path == "primary/primary" for record in records)
    assert all(record.prediction_available for record in records)


async def test_fallback_quality_profile_labels_records_fallback_and_pairs_by_case_id(
    tmp_path,
):
    cases = _cases(2)
    primary_options = _options(tmp_path, cases, "primary-quality")
    fallback_options = _options(
        tmp_path,
        cases,
        "fallback-quality",
        fallback_agent_profile=TunableAgentProfile(),
    )

    _primary_manifest, primary_records, _ = await run_profile(primary_options)
    fallback_manifest, fallback_records, _ = await run_profile(fallback_options)

    assert fallback_manifest.reviewer_policies == ["oracle", "rubber_stamp"]
    assert all(record.model_path == "fallback/fallback" for record in fallback_records)

    primary_case_keys = {r.case_key for r in primary_records if r.policy == "oracle"}
    fallback_case_keys = {r.case_key for r in fallback_records if r.policy == "oracle"}
    assert primary_case_keys == fallback_case_keys == {c.id for c in cases}


async def test_reliability_profile_uses_oracle_only_and_forces_cache_disabled(tmp_path):
    cases = _cases(2)
    options = _options(tmp_path, cases, "reliability", cache_enabled=True)

    manifest, records, _events = await run_profile(options)

    assert manifest.reviewer_policies == ["oracle"]
    assert manifest.cache_enabled is False
    assert len(records) == len(cases)


async def test_fallback_routing_completes_far_below_schedule_to_start_wall_clock(
    tmp_path, monkeypatch
):
    # Temporal's time-skipping test server does not auto-advance schedule-to-start
    # timers (they're enforced by the matching service, not visible to the
    # workflow as a timer/command the client can skip to) -- tests/test_workflow.py's
    # own schedule-to-start test proves this same mechanism the same way: by
    # configuring a small timeout rather than relying on time-skip to fast-forward
    # the real 30s default. profiles.py itself never shrinks this value (asserted
    # below); this monkeypatch only sizes the *test's* configured timeout.
    monkeypatch.setattr(workflows, "AGENT_SCHEDULE_TO_START_S", 1.0)
    cases = _cases(1)
    options = _options(
        tmp_path,
        cases,
        "fallback-routing",
        fallback_agent_profile=TunableAgentProfile(),
        case_deadline=timedelta(seconds=10),
    )

    started = time.perf_counter()
    manifest, records, _events = await run_profile(options)
    elapsed = time.perf_counter() - started

    assert manifest.agent_schedule_to_start_s == 1.0
    assert elapsed < 5.0
    assert all(record.model_path == "fallback/fallback" for record in records)


async def test_fallback_routing_results_excluded_from_quality_comparisons(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(workflows, "AGENT_SCHEDULE_TO_START_S", 1.0)
    cases = _cases(2)
    quality_options = _options(tmp_path, cases, "primary-quality")
    routing_options = _options(
        tmp_path,
        cases,
        "fallback-routing",
        fallback_agent_profile=TunableAgentProfile(),
        case_deadline=timedelta(seconds=10),
    )

    quality_manifest, quality_records, _ = await run_profile(quality_options)
    routing_manifest, routing_records, routing_events = await run_profile(
        routing_options
    )

    assert quality_manifest.run_id != routing_manifest.run_id
    assert {r.run_id for r in quality_records}.isdisjoint(
        {r.run_id for r in routing_records}
    )
    assert quality_manifest.run_profile == "primary-quality"
    assert routing_manifest.run_profile == "fallback-routing"

    report = check_all_invariants(
        routing_records,
        routing_events,
        confidence_threshold=routing_manifest.confidence_threshold,
    )
    assert report.ok


# -- Seed rule end to end -------------------------------------------------------------


async def test_seed_identical_across_reviewer_policies_for_same_case_and_repeat(
    tmp_path,
):
    cases = _cases(3)
    options = _options(tmp_path, cases, "primary-quality", seed=42)

    _manifest, records, _events = await run_profile(options)

    oracle_records = {r.case_key: r for r in records if r.policy == "oracle"}
    rubber_stamp_records = {
        r.case_key: r for r in records if r.policy == "rubber_stamp"
    }

    for case_key, oracle_record in oracle_records.items():
        rubber_stamp_record = rubber_stamp_records[case_key]
        assert (
            oracle_record.predicted_category == rubber_stamp_record.predicted_category
        )
        assert oracle_record.predicted_action == rubber_stamp_record.predicted_action
        assert (
            oracle_record.predicted_refund_amount
            == rubber_stamp_record.predicted_refund_amount
        )
        assert (
            oracle_record.classification_confidence
            == rubber_stamp_record.classification_confidence
        )
        assert oracle_record.draft_confidence == rubber_stamp_record.draft_confidence
        assert oracle_record.reply_text == rubber_stamp_record.reply_text


async def test_seed_differs_across_repeats(tmp_path):
    cases = _cases(1)
    options = _options(
        tmp_path, cases, "primary-quality", seed=7, repeats=2, cache_enabled=False
    )

    _manifest, records, _events = await run_profile(options)

    by_repeat = {r.repeat_index: r for r in records if r.policy == "oracle"}
    assert (
        by_repeat[0].classification_confidence != by_repeat[1].classification_confidence
    )


# -- Manifest assembly ----------------------------------------------------------------


async def test_run_profile_manifest_records_dirty_state_truthfully_end_to_end(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        profiles,
        "_run_git",
        _fake_git("deadbeef", " M src/ticketflow/eval/profiles.py\n"),
    )
    cases = _cases(1)
    options = _options(tmp_path, cases, "primary-quality")

    manifest, _records, _events = await run_profile(options)

    assert manifest.git_commit == "deadbeef"
    assert manifest.git_dirty is True


async def test_run_profile_concurrency_does_not_change_results(tmp_path):
    cases = _cases(5)

    async def _run(concurrency: int) -> set:
        options = _options(tmp_path, cases, "primary-quality", concurrency=concurrency)
        _manifest, records, _events = await run_profile(options)
        return {
            (
                r.policy,
                r.case_key,
                r.repeat_index,
                r.predicted_category,
                r.predicted_action,
                r.predicted_refund_amount,
                r.classification_confidence,
                r.draft_confidence,
                r.terminal_outcome,
                r.was_gated,
            )
            for r in records
        }

    serial = await _run(1)
    parallel = await _run(4)
    assert serial == parallel


async def test_run_profile_manifest_assembly_correctness(tmp_path):
    cases = _cases(2)
    dataset_path = _write_dataset(tmp_path, cases)
    options = _options(
        tmp_path,
        cases,
        "primary-quality",
        dataset_path=dataset_path,
        seed=5,
        bootstrap_seed=9,
        concurrency=3,
        repeats=1,
    )

    manifest, _records, _events = await run_profile(options)

    expected_config = current_workflow_eval_config()
    assert manifest.seed == 5
    assert manifest.bootstrap_seed == 9
    assert manifest.concurrency == 3
    assert manifest.repeats == 1
    assert manifest.confidence_threshold == expected_config.confidence_threshold
    assert (
        manifest.agent_schedule_to_start_s == expected_config.agent_schedule_to_start_s
    )
    assert manifest.dataset_path == str(dataset_path)
    assert manifest.dataset_sha256 == dataset_sha256(dataset_path)
    assert manifest.python_version == platform.python_version()
