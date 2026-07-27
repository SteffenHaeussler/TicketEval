import json

from scripts import eval as eval_cli


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
