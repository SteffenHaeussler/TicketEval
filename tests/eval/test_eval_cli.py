import json
from datetime import datetime, timezone

from scripts import eval as eval_cli
from ticketflow.eval.dataset import EvalCase
from ticketflow.eval.invariants import InvariantReport
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
from ticketflow.models import ActionType, TicketCategory


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
    assert "must be below --case-deadline=60.0" in capsys.readouterr().err


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
        f"run_id: run-cli-test\n"
        f"artifacts: {run_dir}\n"
        f"cases: 1 records, 1 call events\n"
        f"invariants: ok (1 records checked)\n"
    )
