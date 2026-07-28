import json
from datetime import datetime, timedelta, timezone

from scripts import eval as eval_cli
from ticketflow.eval.cache import FileResponseCache
from ticketflow.eval.dataset import EvalCase
from ticketflow.eval.harness import local_environment
from ticketflow.eval.invariants import InvariantReport
from ticketflow.eval.preflight import (
    ConfidenceGateResult,
    ModelInfo,
    ModelMissingError,
    PreflightResult,
    StageMeasurement,
)
from ticketflow.eval.preflight import TimeoutAdjustment as PreflightTimeoutAdjustment
from ticketflow.eval.profiles import ProfileConfigError
from ticketflow.eval.records import (
    CallEvent,
    CaseRecord,
    RecordsError,
    RunManifest,
    read_call_events,
    read_case_records,
    read_json_artifact,
    read_run_manifest,
)
from ticketflow.eval.report import render_markdown
from ticketflow.models import ActionType, TicketCategory


def make_preflight_result(
    *, primary_model: str = "ollama-primary", fallback_model: str = "ollama-fallback"
) -> PreflightResult:
    return PreflightResult(
        ollama_version="0.6.2",
        models=(
            ModelInfo(role="primary", name=primary_model, digest="sha256:primary"),
            ModelInfo(role="fallback", name=fallback_model, digest="sha256:fallback"),
        ),
        measurements=(
            StageMeasurement(
                operation="classify",
                ticket_id="preflight-probe-1",
                wall_latency_s=1.0,
                load_duration_s=0.25,
                generation_duration_s=0.75,
            ),
        ),
        timeout_adjustment=PreflightTimeoutAdjustment(
            configured_activity_timeout_s=30.0,
            slowest_observed_stage_s=20.0,
            effective_activity_timeout_s=60.0,
            safety_margin_s=6.0,
            http_timeout_s=54.0,
        ),
        confidence_gate=ConfidenceGateResult(
            samples=(0.5, 0.6, 0.7, 0.8, 0.9),
            std_dev=0.1,
            distinct_count=5,
            passes_std_dev_gate=True,
            passes_distinctness_gate=True,
        ),
        workflow_eval_config=eval_cli.current_workflow_eval_config(),
    )


def make_case(case_id, *, difficulty, source, reference_category, verified=True):
    return {
        "id": case_id,
        "subject": "Help with my ticket",
        "body": "I need assistance with my account.",
        "customer_email": "eval@example.com",
        "expected": {
            "acceptable_categories": [reference_category],
            "reference_category": reference_category,
            "acceptable_actions": ["reply_only"],
            "expected_refund_amount": None,
            "refund_tolerance": 0.01,
        },
        "difficulty": difficulty,
        "source": source,
        "authored_by": f"author-{case_id}",
        "generated_by": "fixture-generator" if source == "generated" else None,
        "label_verified": verified,
        "verified_by": f"reviewer-{case_id}" if verified else None,
        "verified_at": "2026-01-01T00:00:00+00:00" if verified else None,
        "notes": None,
    }


def write_shard(directory, name, cases):
    path = directory / name
    path.write_text("\n".join(json.dumps(case) for case in cases) + "\n")
    return path


def test_dataset_check_default_reports_ordered_complete_composition(
    tmp_path, monkeypatch, capsys
):
    dataset_dir = tmp_path / "tickets"
    dataset_dir.mkdir()
    difficulties = ["easy"] * 4 + ["ambiguous"] * 4 + ["adversarial"] * 4
    categories = ["billing", "technical", "account", "general"] * 3
    cases = [
        make_case(
            f"case-{index}",
            difficulty=difficulty,
            source="handwritten" if index % 2 else "generated",
            reference_category=category,
        )
        for index, (difficulty, category) in enumerate(zip(difficulties, categories))
    ]
    write_shard(dataset_dir, "balanced.jsonl", cases)
    monkeypatch.setattr(eval_cli, "DEFAULT_DATASET_DIR", dataset_dir)

    assert eval_cli.main(["dataset-check"]) == 0
    assert capsys.readouterr().out == (
        "valid cases: 12\n"
        "\n"
        "difficulty:\n"
        "  easy: 4\n"
        "  ambiguous: 4\n"
        "  adversarial: 4\n"
        "\n"
        "source:\n"
        "  handwritten: 6\n"
        "  generated: 6\n"
        "\n"
        "reference_category:\n"
        "  billing: 3\n"
        "  technical: 3\n"
        "  account: 3\n"
        "  general: 3\n"
    )


def test_dataset_check_shard_skips_whole_dataset_distribution_validation(
    tmp_path, capsys
):
    shard = write_shard(
        tmp_path,
        "skewed.jsonl",
        [
            make_case(
                "skewed-1",
                difficulty="easy",
                source="handwritten",
                reference_category="billing",
            )
        ],
    )

    assert eval_cli.main(["dataset-check", "--shard", str(shard)]) == 0
    assert "valid cases: 1" in capsys.readouterr().out


def test_dataset_check_allow_unverified_only_changes_verification_requirement(
    tmp_path, capsys
):
    shard = write_shard(
        tmp_path,
        "unverified.jsonl",
        [
            make_case(
                "draft-1",
                difficulty="easy",
                source="handwritten",
                reference_category="billing",
                verified=False,
            )
        ],
    )

    assert eval_cli.main(["dataset-check", "--shard", str(shard)]) == 1
    assert "draft-1" in capsys.readouterr().err

    assert (
        eval_cli.main(["dataset-check", "--shard", str(shard), "--allow-unverified"])
        == 0
    )
    assert "valid cases: 1" in capsys.readouterr().out


def test_dataset_check_malformed_shard_returns_error_with_case_id(tmp_path, capsys):
    malformed = make_case(
        "bad-case",
        difficulty="easy",
        source="handwritten",
        reference_category="billing",
    )
    malformed["expected"]["acceptable_categories"] = []
    shard = write_shard(tmp_path, "malformed.jsonl", [malformed])

    assert (
        eval_cli.main(["dataset-check", "--shard", str(shard), "--allow-unverified"])
        == 1
    )
    error = capsys.readouterr().err
    assert error.startswith("dataset-check failed:")
    assert "bad-case" in error


def test_parser_dispatches_dataset_check_to_its_handler(monkeypatch):
    calls = []

    def handler(args):
        calls.append(args)
        return 17

    monkeypatch.setattr(eval_cli, "dataset_check", handler)

    assert eval_cli.main(["dataset-check"]) == 17
    assert len(calls) == 1


def test_run_drives_the_committed_dataset_end_to_end(tmp_path, monkeypatch, capsys):
    """M2-T7: `make eval` completes a real tunable run over the committed dataset.

    Deliberately does not stub run_profile -- this is the only test that exercises
    the CLI, the profile assembly, the Temporal test server, and artifact writing
    together against evals/data/tickets. Limited to a few cases for runtime.
    """
    artifacts_root = tmp_path / "runs"
    monkeypatch.setattr(eval_cli, "RUNS_DIR", artifacts_root)

    exit_code = eval_cli.main(
        [
            "run",
            "--profile",
            "primary-quality",
            "--agent",
            "tunable",
            "--reviewer",
            "both",
            "--allow-unverified",
            "--limit",
            "3",
            "--concurrency",
            "2",
        ]
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "invariants: ok" in out

    run_dirs = list(artifacts_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    manifest = read_run_manifest(run_dir / "manifest.json")
    assert manifest.run_profile == "primary-quality"
    assert manifest.reviewer_policies == ["oracle", "rubber_stamp"]
    assert manifest.dataset_path == str(eval_cli.DEFAULT_DATASET_DIR)
    assert manifest.generation_seed_rule

    # Three cases under both reviewer policies.
    records = read_case_records(run_dir / "records.jsonl")
    assert len(records) == 6
    assert {record.policy for record in records} == {"oracle", "rubber_stamp"}
    assert all(record.prediction_available for record in records)

    events = read_call_events(run_dir / "calls.jsonl")
    assert {event.operation for event in events} == {"classify", "draft"}

    invariants = read_json_artifact(run_dir / "invariants.json", InvariantReport)
    assert invariants.ok
    assert invariants.total_checked == 6


def test_run_rejects_repeats_with_cache_before_starting_a_workflow(monkeypatch, capsys):
    started = False

    async def run_profile(_options):
        nonlocal started
        started = True
        raise AssertionError("run_profile must not start")

    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    assert eval_cli.main(["run", "--profile", "primary-quality", "--repeats", "2"]) == 1
    assert "--repeats=2 requires --no-cache" in capsys.readouterr().err
    assert not started


def test_run_rejects_reviewer_incompatible_with_profile(capsys):
    result = eval_cli.main(
        ["run", "--profile", "fallback-routing", "--reviewer", "both"]
    )
    assert result == 1
    err = capsys.readouterr().err
    assert "does not support reviewer 'rubber_stamp'" in err
    assert "it allows oracle" in err


def test_run_accepts_a_reviewer_narrowing_a_quality_profile(tmp_path, monkeypatch):
    # --reviewer oracle is a legal subset of primary-quality's two policies, and must
    # reach RunOptions rather than being validated and then dropped.
    observed = {}

    async def run_profile(options):
        observed["options"] = options
        raise ProfileConfigError("stop after options are built")

    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    assert (
        eval_cli.main(
            [
                "run",
                "--profile",
                "primary-quality",
                "--reviewer",
                "oracle",
                "--allow-unverified",
                "--limit",
                "1",
            ]
        )
        == 1
    )
    assert observed["options"].reviewer_policies == ("oracle",)


def test_run_ollama_runs_preflight_and_builds_options_with_real_provenance(
    tmp_path, monkeypatch
):
    preflight_calls = {}
    observed = {}

    async def run_preflight(**kwargs):
        preflight_calls.update(kwargs)
        return make_preflight_result()

    async def run_profile(options):
        observed["options"] = options
        raise ProfileConfigError("stop after options are built")

    monkeypatch.setattr(eval_cli, "run_preflight", run_preflight)
    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    assert (
        eval_cli.main(
            [
                "run",
                "--profile",
                "primary-quality",
                "--agent",
                "ollama",
                "--primary-model",
                "ollama-primary",
                "--fallback-model",
                "ollama-fallback",
                "--ollama-endpoint",
                "http://ollama.example:11434",
                "--allow-unverified",
                "--limit",
                "1",
            ]
        )
        == 1
    )

    assert preflight_calls["endpoint"] == "http://ollama.example:11434"
    assert preflight_calls["required_models"] == {
        "primary": "ollama-primary",
        "fallback": "ollama-fallback",
    }
    # Probing draws from the full dataset, not the --limit'd run cases.
    assert len(preflight_calls["probe_tickets"]) > 1

    options = observed["options"]
    assert options.agent_backend == "ollama"
    assert options.primary_model == "ollama-primary"
    assert options.fallback_model == "ollama-fallback"
    assert options.ollama_endpoint == "http://ollama.example:11434"
    assert options.preflight_result is not None
    assert isinstance(options.response_cache, FileResponseCache)
    assert options.environment_factory is local_environment
    # Derived from the 60s activity timeout preflight measured, so the widening
    # preflight performs is actually reachable inside the runner's own deadline.
    assert options.case_deadline == timedelta(seconds=150.0)
    assert options.concurrency == 1


def test_run_ollama_pacing_flags_override_the_preflight_derived_defaults(monkeypatch):
    observed = {}

    async def run_preflight(**kwargs):
        return make_preflight_result()

    async def run_profile(options):
        observed["options"] = options
        raise ProfileConfigError("stop after options are built")

    monkeypatch.setattr(eval_cli, "run_preflight", run_preflight)
    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    assert (
        eval_cli.main(
            [
                "run",
                "--profile",
                "primary-quality",
                "--agent",
                "ollama",
                "--primary-model",
                "ollama-primary",
                "--fallback-model",
                "ollama-fallback",
                "--allow-unverified",
                "--limit",
                "1",
                "--case-deadline",
                "900",
                "--concurrency",
                "4",
            ]
        )
        == 1
    )

    assert observed["options"].case_deadline == timedelta(seconds=900.0)
    assert observed["options"].concurrency == 4


def test_run_tunable_keeps_its_fast_pacing_defaults(tmp_path, monkeypatch):
    observed = {}

    async def run_profile(options):
        observed["options"] = options
        raise ProfileConfigError("stop after options are built")

    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    assert (
        eval_cli.main(
            [
                "run",
                "--profile",
                "primary-quality",
                "--allow-unverified",
                "--limit",
                "1",
            ]
        )
        == 1
    )

    assert observed["options"].case_deadline == timedelta(
        seconds=eval_cli.TUNABLE_CASE_DEADLINE_S
    )
    assert observed["options"].concurrency == eval_cli.TUNABLE_CONCURRENCY


def test_run_ollama_no_cache_skips_response_cache(monkeypatch):
    observed = {}

    async def run_preflight(**kwargs):
        return make_preflight_result()

    async def run_profile(options):
        observed["options"] = options
        raise ProfileConfigError("stop after options are built")

    monkeypatch.setattr(eval_cli, "run_preflight", run_preflight)
    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    assert (
        eval_cli.main(
            [
                "run",
                "--profile",
                "reliability",
                "--agent",
                "ollama",
                "--primary-model",
                "ollama-primary",
                "--fallback-model",
                "ollama-fallback",
                "--no-cache",
                "--allow-unverified",
                "--limit",
                "1",
            ]
        )
        == 1
    )

    assert observed["options"].response_cache is None


def test_run_ollama_preflight_failure_stops_before_run_profile(monkeypatch, capsys):
    started = False

    async def run_preflight(**kwargs):
        raise ModelMissingError("model 'ollama-primary' is not installed")

    async def run_profile(_options):
        nonlocal started
        started = True
        raise AssertionError("run_profile must not start")

    monkeypatch.setattr(eval_cli, "run_preflight", run_preflight)
    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    result = eval_cli.main(
        [
            "run",
            "--profile",
            "primary-quality",
            "--agent",
            "ollama",
            "--allow-unverified",
        ]
    )

    assert result == 1
    assert "ollama-primary" in capsys.readouterr().err
    assert not started


def test_run_rejects_mock_agent_before_any_io(monkeypatch, capsys):
    started = False

    async def run_profile(_options):
        nonlocal started
        started = True
        raise AssertionError("run_profile must not start")

    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    result = eval_cli.main(["run", "--profile", "primary-quality", "--agent", "mock"])

    assert result == 1
    assert "--agent mock is not available" in capsys.readouterr().err
    assert not started


def test_run_rejects_schedule_to_start_at_or_above_case_deadline(capsys):
    result = eval_cli.main(
        [
            "run",
            "--profile",
            "fallback-routing",
            "--schedule-to-start",
            "60",
            "--case-deadline",
            "60",
        ]
    )
    assert result == 1
    assert "must be below the per-case deadline of 60.0s" in capsys.readouterr().err


def test_limit_samples_across_difficulties_instead_of_head_slicing():
    def case(case_id, difficulty):
        return EvalCase.model_validate(
            make_case(
                case_id,
                difficulty=difficulty,
                source="generated",
                reference_category="billing",
            )
        )

    # Mirrors the real shard order: adversarial sorts first, so a head slice would
    # return only adversarial cases.
    cases = [case(f"adversarial-{i}", "adversarial") for i in range(3)] + [
        case(f"easy-{i}", "easy") for i in range(3)
    ]

    selected = eval_cli._limited_cases(cases, 2)

    assert {case.difficulty for case in selected} == {"adversarial", "easy"}
    # Dataset order is preserved so artifacts stay diffable.
    assert [case.id for case in selected] == ["adversarial-0", "easy-0"]


def test_run_writes_profile_artifacts_and_supports_authoring_dataset(
    tmp_path, monkeypatch, capsys
):
    dataset = write_shard(
        tmp_path,
        "draft.jsonl",
        [
            make_case(
                "draft-1",
                difficulty="easy",
                source="handwritten",
                reference_category="billing",
                verified=False,
            )
        ],
    )
    artifacts_root = tmp_path / "runs"
    observed = {}
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def run_profile(options):
        observed["options"] = options
        manifest = RunManifest(
            run_id="run-cli-test",
            git_commit="abc123",
            git_dirty=False,
            dataset_path=str(dataset),
            dataset_sha256="hash",
            agent_backend="tunable",
            run_profile="primary-quality",
            primary_model="tunable-primary",
            python_version="3.12.0",
            reviewer_policies=["oracle", "rubber_stamp"],
            cache_enabled=True,
            confidence_threshold=0.75,
            agent_task_queue="agent",
            fallback_task_queue="fallback",
            agent_schedule_to_start_s=30.0,
            agent_activity_timeout_s=60.0,
            agent_heartbeat_timeout_s=30.0,
            seed=0,
            bootstrap_seed=0,
            generation_seed_rule="test-rule/v1",
            concurrency=8,
            repeats=1,
            started_at=now,
            finished_at=now,
        )
        record = CaseRecord(
            run_id=manifest.run_id,
            policy="oracle",
            case_key="draft-1",
            repeat_index=0,
            ticket_id="ticket-1",
            difficulty="easy",
            source="handwritten",
            expected=options.cases[0].expected,
            predicted_category=TicketCategory.BILLING,
            predicted_action=ActionType.REPLY_ONLY,
            classification_confidence=0.9,
            draft_confidence=0.9,
            reply_text="Resolved.",
            prediction_available=True,
            terminal_outcome="resolved",
            end_to_end_latency_ms=1.0,
        )
        event = CallEvent(
            run_id=manifest.run_id,
            case_key="draft-1",
            ticket_id="ticket-1",
            policy="oracle",
            repeat_index=0,
            operation="classify",
            role="primary",
            attempt=1,
            cache_hit=False,
            started_at=now,
            wall_latency_ms=1.0,
            model_total_duration_ms=None,
            model_load_duration_ms=None,
            outcome="success",
            error_type=None,
        )
        return manifest, [record], [event]

    monkeypatch.setattr(eval_cli, "DEFAULT_DATASET_DIR", dataset)
    monkeypatch.setattr(eval_cli, "RUNS_DIR", artifacts_root)
    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    original_write_call_events = eval_cli.write_call_events

    def fail_call_events(*_args):
        raise RecordsError("injected calls write failure")

    monkeypatch.setattr(eval_cli, "write_call_events", fail_call_events)
    assert (
        eval_cli.main(
            [
                "run",
                "--profile",
                "primary-quality",
                "--allow-unverified",
                "--limit",
                "1",
            ]
        )
        == 1
    )
    assert not (artifacts_root / "run-cli-test").exists()
    assert "injected calls write failure" in capsys.readouterr().err

    monkeypatch.setattr(eval_cli, "write_call_events", original_write_call_events)
    assert (
        eval_cli.main(
            [
                "run",
                "--profile",
                "primary-quality",
                "--allow-unverified",
                "--limit",
                "1",
            ]
        )
        == 0
    )

    run_dir = artifacts_root / "run-cli-test"
    assert observed["options"].cases[0].id == "draft-1"
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "records.jsonl").is_file()
    assert (run_dir / "calls.jsonl").is_file()
    assert (run_dir / "invariants.json").is_file()
    assert capsys.readouterr().out == (
        f"pacing: case deadline {eval_cli.TUNABLE_CASE_DEADLINE_S:.1f}s, "
        f"concurrency {eval_cli.TUNABLE_CONCURRENCY}\n"
        f"run_id: run-cli-test\n"
        f"artifacts: {run_dir}\n"
        f"cases: 1 records, 1 call events\n"
        f"invariants: ok (1 records checked)\n"
    )


def test_run_refuses_an_existing_run_directory_without_replacing_raw_artifacts(
    tmp_path, monkeypatch, capsys
):
    dataset = write_shard(
        tmp_path,
        "dataset.jsonl",
        [
            make_case(
                "case-1",
                difficulty="easy",
                source="handwritten",
                reference_category="billing",
            )
        ],
    )
    artifacts_root = tmp_path / "runs"
    run_dir = artifacts_root / "run-existing"
    run_dir.mkdir(parents=True)
    records_path = run_dir / "records.jsonl"
    records_path.write_text("original raw record\n")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def run_profile(_options):
        return (
            RunManifest(
                run_id="run-existing",
                git_commit="abc123",
                git_dirty=False,
                dataset_path=str(dataset),
                dataset_sha256="hash",
                agent_backend="tunable",
                run_profile="primary-quality",
                primary_model="tunable-primary",
                python_version="3.12.0",
                reviewer_policies=["oracle"],
                cache_enabled=True,
                confidence_threshold=0.75,
                agent_task_queue="agent",
                fallback_task_queue="fallback",
                agent_schedule_to_start_s=30.0,
                agent_activity_timeout_s=60.0,
                agent_heartbeat_timeout_s=30.0,
                seed=0,
                bootstrap_seed=0,
                generation_seed_rule="test-rule/v1",
                concurrency=1,
                repeats=1,
                started_at=now,
                finished_at=now,
            ),
            [],
            [],
        )

    monkeypatch.setattr(eval_cli, "DEFAULT_DATASET_DIR", dataset)
    monkeypatch.setattr(eval_cli, "RUNS_DIR", artifacts_root)
    monkeypatch.setattr(eval_cli, "run_profile", run_profile)

    assert (
        eval_cli.main(["run", "--profile", "primary-quality", "--allow-unverified"])
        == 1
    )
    assert records_path.read_text() == "original raw record\n"
    assert "refusing to overwrite existing run" in capsys.readouterr().err
    assert list(artifacts_root.iterdir()) == [run_dir]


# -- report ---------------------------------------------------------------------------


def write_run_dir(root, run_id, *, bootstrap_seed=0, records=None):
    """Persist a minimal but valid run directory and return (path, records, events)."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_id,
        git_commit="abc123",
        git_dirty=False,
        dataset_path="evals/data/tickets",
        dataset_sha256="hash",
        agent_backend="tunable",
        run_profile="primary-quality",
        primary_model="tunable-primary",
        python_version="3.12.0",
        reviewer_policies=["oracle"],
        cache_enabled=True,
        confidence_threshold=0.75,
        agent_task_queue="agent",
        fallback_task_queue="fallback",
        agent_schedule_to_start_s=30.0,
        agent_activity_timeout_s=60.0,
        agent_heartbeat_timeout_s=30.0,
        seed=0,
        bootstrap_seed=bootstrap_seed,
        generation_seed_rule="test-rule/v1",
        concurrency=8,
        repeats=1,
        started_at=now,
        finished_at=now,
    )
    if records is None:
        case = EvalCase.model_validate(
            make_case(
                "report-1",
                difficulty="easy",
                source="handwritten",
                reference_category="billing",
            )
        )
        records = [
            CaseRecord(
                run_id=run_id,
                policy="oracle",
                case_key="report-1",
                repeat_index=0,
                ticket_id="ticket-1",
                difficulty="easy",
                source="handwritten",
                expected=case.expected,
                predicted_category=TicketCategory.BILLING,
                predicted_action=ActionType.REPLY_ONLY,
                classification_confidence=0.9,
                draft_confidence=0.9,
                reply_text="Resolved.",
                prediction_available=True,
                terminal_outcome="resolved",
                end_to_end_latency_ms=1.0,
            )
        ]
    events = [
        CallEvent(
            run_id=run_id,
            case_key="report-1",
            ticket_id="ticket-1",
            policy="oracle",
            repeat_index=0,
            operation="classify",
            role="primary",
            attempt=1,
            cache_hit=False,
            started_at=now,
            wall_latency_ms=1.0,
            model_total_duration_ms=None,
            model_load_duration_ms=None,
            outcome="success",
            error_type=None,
        )
    ]
    eval_cli.write_run_manifest(run_dir / "manifest.json", manifest)
    eval_cli.write_case_records(run_dir / "records.jsonl", records)
    eval_cli.write_call_events(run_dir / "calls.jsonl", events)
    return run_dir, records, events


def test_report_renders_the_same_markdown_as_a_direct_render(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    _run_dir, records, events = write_run_dir(runs_dir, "run-report", bootstrap_seed=7)

    exit_code = eval_cli.main(
        [
            "report",
            "--run-id",
            "run-report",
            "--runs-dir",
            str(runs_dir),
            "--resamples",
            "50",
        ]
    )

    assert exit_code == 0
    # The manifest's own seed is what makes a report reproducible from its run, so a
    # direct render has to be given that seed rather than the CLI default.
    expected = render_markdown(records, events, bootstrap_seed=7, n_resamples=50)
    assert capsys.readouterr().out == expected + "\n"


def test_report_out_writes_the_file_and_leaves_raw_artifacts_untouched(
    tmp_path, capsys
):
    runs_dir = tmp_path / "runs"
    run_dir, _records, _events = write_run_dir(runs_dir, "run-report")
    before = {path.name: path.read_bytes() for path in sorted(run_dir.iterdir())}
    destination = tmp_path / "out" / "report.md"

    exit_code = eval_cli.main(
        [
            "report",
            "--run-id",
            "run-report",
            "--runs-dir",
            str(runs_dir),
            "--resamples",
            "50",
            "--out",
            str(destination),
        ]
    )

    assert exit_code == 0
    assert destination.read_text(encoding="utf-8").startswith(
        "# Deterministic Metrics Report"
    )
    assert capsys.readouterr().out == f"report: {destination}\n"
    after = {path.name: path.read_bytes() for path in sorted(run_dir.iterdir())}
    assert after == before


def test_report_missing_run_directory_returns_error_without_reading_artifacts(
    tmp_path, capsys
):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    exit_code = eval_cli.main(
        ["report", "--run-id", "run-absent", "--runs-dir", str(runs_dir)]
    )

    assert exit_code == 1
    assert "no such run" in capsys.readouterr().err


def test_report_malformed_artifact_returns_error_rather_than_raising(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    run_dir, _records, _events = write_run_dir(runs_dir, "run-report")
    (run_dir / "records.jsonl").write_text("{not json\n")

    exit_code = eval_cli.main(
        ["report", "--run-id", "run-report", "--runs-dir", str(runs_dir)]
    )

    assert exit_code == 1
    assert "report failed:" in capsys.readouterr().err


def test_report_run_with_no_records_returns_error(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    run_dir, _records, _events = write_run_dir(runs_dir, "run-report")
    (run_dir / "records.jsonl").write_text("")

    exit_code = eval_cli.main(
        ["report", "--run-id", "run-report", "--runs-dir", str(runs_dir)]
    )

    assert exit_code == 1
    assert "no case records" in capsys.readouterr().err


def test_report_defaults_to_the_committed_runs_directory(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    write_run_dir(runs_dir, "run-report")
    monkeypatch.setattr(eval_cli, "RUNS_DIR", runs_dir)

    args = eval_cli.build_parser().parse_args(["report", "--run-id", "run-report"])

    assert args.runs_dir == eval_cli.RUNS_DIR
    assert args.resamples == eval_cli.DEFAULT_REPORT_RESAMPLES
    assert args.handler is eval_cli.report
