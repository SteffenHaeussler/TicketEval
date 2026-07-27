import json

import pytest

from ticketflow.eval.dataset import (
    DatasetLoadError,
    DatasetValidationError,
    DatasetValidationPolicy,
    load_cases,
    refund_amount_in_text,
    validate_dataset,
)


def make_expected(**overrides):
    base = {
        "acceptable_categories": ["billing"],
        "reference_category": "billing",
        "acceptable_actions": ["reply_only"],
        "expected_refund_amount": None,
        "refund_tolerance": 0.01,
    }
    base.update(overrides)
    return base


def make_case(expected=None, **overrides):
    base = {
        "id": "case-1",
        "subject": "Login issue",
        "body": "I can't log into my account.",
        "customer_email": "eval@example.com",
        "expected": expected or make_expected(),
        "difficulty": "easy",
        "source": "handwritten",
        "authored_by": "alice",
        "generated_by": None,
        "label_verified": True,
        "verified_by": "bob",
        "verified_at": "2026-01-01T00:00:00+00:00",
        "notes": None,
    }
    base.update(overrides)
    return base


def write_shard(tmp_path, name, case_dicts):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(c) for c in case_dicts) + "\n")
    return path


# --- load_cases: file vs directory ---


def test_load_cases_reads_single_jsonl_file(tmp_path):
    path = write_shard(tmp_path, "cases.jsonl", [make_case(id="c1")])
    cases = load_cases(path)
    assert [c.id for c in cases] == ["c1"]


def test_load_cases_reads_directory_of_shards(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    write_shard(shard_dir, "a.jsonl", [make_case(id="a1")])
    write_shard(shard_dir, "b.jsonl", [make_case(id="b1")])
    cases = load_cases(shard_dir)
    assert sorted(c.id for c in cases) == ["a1", "b1"]


def test_directory_and_concatenated_file_load_produce_identical_case_lists(tmp_path):
    case_a = make_case(id="a1")
    case_b = make_case(id="b1")

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    write_shard(shard_dir, "a.jsonl", [case_a])
    write_shard(shard_dir, "b.jsonl", [case_b])

    single_file = write_shard(tmp_path, "combined.jsonl", [case_a, case_b])

    dir_cases = load_cases(shard_dir)
    file_cases = load_cases(single_file)

    assert [c.model_dump() for c in dir_cases] == [c.model_dump() for c in file_cases]


def test_load_cases_directory_with_no_shards_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(DatasetLoadError, match=str(empty_dir)):
        load_cases(empty_dir)


# --- structural checks ---


def test_duplicate_id_across_shards_rejected_names_id_and_shards(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    a = write_shard(shard_dir, "a.jsonl", [make_case(id="dup")])
    b = write_shard(shard_dir, "b.jsonl", [make_case(id="dup")])

    with pytest.raises(DatasetLoadError) as exc_info:
        load_cases(shard_dir)

    message = str(exc_info.value)
    assert "dup" in message
    assert str(a) in message
    assert str(b) in message


def test_empty_acceptable_categories_rejected(tmp_path):
    expected = make_expected(acceptable_categories=[])
    path = write_shard(tmp_path, "cases.jsonl", [make_case(id="c1", expected=expected)])

    with pytest.raises(DatasetLoadError, match="c1.*acceptable_categories"):
        load_cases(path)


def test_empty_acceptable_actions_rejected(tmp_path):
    expected = make_expected(acceptable_actions=[])
    path = write_shard(tmp_path, "cases.jsonl", [make_case(id="c1", expected=expected)])

    with pytest.raises(DatasetLoadError, match="c1.*acceptable_actions"):
        load_cases(path)


def test_reference_category_outside_acceptable_set_rejected(tmp_path):
    expected = make_expected(
        acceptable_categories=["billing"], reference_category="technical"
    )
    path = write_shard(tmp_path, "cases.jsonl", [make_case(id="c1", expected=expected)])

    with pytest.raises(DatasetLoadError) as exc_info:
        load_cases(path)

    message = str(exc_info.value)
    assert "c1" in message
    assert "reference_category" in message


def test_refund_acceptable_without_amount_rejected(tmp_path):
    expected = make_expected(acceptable_actions=["refund"], expected_refund_amount=None)
    path = write_shard(tmp_path, "cases.jsonl", [make_case(id="c1", expected=expected)])

    with pytest.raises(DatasetLoadError, match="c1"):
        load_cases(path)


def test_refund_amount_absent_from_ticket_text_rejected(tmp_path):
    expected = make_expected(
        acceptable_actions=["refund"], expected_refund_amount=99.99
    )
    case = make_case(
        id="c1",
        subject="Refund please",
        body="I was overcharged.",
        expected=expected,
    )
    path = write_shard(tmp_path, "cases.jsonl", [case])

    with pytest.raises(DatasetLoadError) as exc_info:
        load_cases(path)

    message = str(exc_info.value)
    assert "c1" in message
    assert "99.99" in message


def test_generated_case_without_generated_by_rejected(tmp_path):
    case = make_case(id="c1", source="generated", generated_by=None)
    path = write_shard(tmp_path, "cases.jsonl", [case])

    with pytest.raises(DatasetLoadError, match="c1"):
        load_cases(path)


def test_handwritten_case_with_generated_by_rejected(tmp_path):
    case = make_case(id="c1", source="handwritten", generated_by="agent-x")
    path = write_shard(tmp_path, "cases.jsonl", [case])

    with pytest.raises(DatasetLoadError, match="c1"):
        load_cases(path)


def test_missing_authored_by_rejected(tmp_path):
    case = make_case(id="c1", authored_by="")
    path = write_shard(tmp_path, "cases.jsonl", [case])

    with pytest.raises(DatasetLoadError, match="c1"):
        load_cases(path)


# --- verification checks ---


def test_unverified_case_rejected_by_default(tmp_path):
    case = make_case(id="c1", label_verified=False, verified_by=None, verified_at=None)
    path = write_shard(tmp_path, "cases.jsonl", [case])

    with pytest.raises(DatasetLoadError, match="c1"):
        load_cases(path)


def test_anonymously_verified_case_rejected(tmp_path):
    case = make_case(id="c1", label_verified=True, verified_by=None)
    path = write_shard(tmp_path, "cases.jsonl", [case])

    with pytest.raises(DatasetLoadError, match="c1"):
        load_cases(path)


def test_self_verified_case_rejected(tmp_path):
    case = make_case(id="c1", authored_by="alice", verified_by="alice")
    path = write_shard(tmp_path, "cases.jsonl", [case])

    with pytest.raises(DatasetLoadError, match="c1"):
        load_cases(path)


def test_require_verified_false_allows_unverified_draft_shard(tmp_path):
    case = make_case(id="c1", label_verified=False, verified_by=None, verified_at=None)
    path = write_shard(tmp_path, "cases.jsonl", [case])

    cases = load_cases(path, require_verified=False)
    assert [c.id for c in cases] == ["c1"]


def test_require_verified_false_still_runs_structural_checks(tmp_path):
    expected = make_expected(acceptable_categories=[])
    case = make_case(
        id="c1",
        expected=expected,
        label_verified=False,
        verified_by=None,
        verified_at=None,
    )
    path = write_shard(tmp_path, "cases.jsonl", [case])

    with pytest.raises(DatasetLoadError, match="c1"):
        load_cases(path, require_verified=False)


def test_each_difficulty_shard_passes_authoring_validation_alone(tmp_path):
    for difficulty in ["easy", "ambiguous", "adversarial"]:
        cases = [
            make_case(
                id=f"{difficulty}-1",
                difficulty=difficulty,
                label_verified=False,
                verified_by=None,
                verified_at=None,
            ),
            make_case(
                id=f"{difficulty}-2",
                difficulty=difficulty,
                label_verified=False,
                verified_by=None,
                verified_at=None,
            ),
        ]
        path = write_shard(tmp_path, f"{difficulty}.jsonl", cases)
        loaded = load_cases(path, require_verified=False)
        assert len(loaded) == 2


# --- refund_amount_in_text ---


def test_refund_amount_in_text_matches_dollar_prefix():
    assert refund_amount_in_text(12.34, 0.01, "Please refund $12.34 to my card.")


def test_refund_amount_in_text_matches_euro_comma_decimal():
    assert refund_amount_in_text(12.34, 0.01, "Bitte erstatten Sie 12,34 zurück.")


def test_refund_amount_in_text_matches_usd_suffix():
    assert refund_amount_in_text(12.34, 0.01, "I want 12.34 USD refunded.")


def test_refund_amount_in_text_matches_eur_suffix_whole_number():
    assert refund_amount_in_text(50.0, 0.01, "Refund of 50 EUR issued.")


def test_refund_amount_in_text_matches_standalone_decimal_no_currency():
    assert refund_amount_in_text(12.50, 0.01, "The charge was 12.50 in error.")


def test_refund_amount_in_text_rejects_thousands_grouped_number():
    text = "I was charged $1,234.56 by mistake."
    assert not refund_amount_in_text(1234.56, 0.01, text)
    assert not refund_amount_in_text(234.56, 0.01, text)


def test_refund_amount_in_text_rejects_alphanumeric_embedding():
    assert not refund_amount_in_text(12.34, 0.01, "See invoice INV-12.34X for details.")


def test_refund_amount_in_text_respects_tolerance_boundary():
    text = "Refund of $12.35 requested."
    assert refund_amount_in_text(12.34, 0.01, text)
    assert not refund_amount_in_text(12.34, 0.001, text)


def test_bare_integer_without_currency_marker_not_treated_as_amount():
    assert not refund_amount_in_text(50.0, 0.01, "Order 50 shipped yesterday.")
    assert refund_amount_in_text(50.0, 0.01, "Refund of 50 EUR issued.")


# --- validate_dataset ---


def test_default_easy_and_ambiguous_ranges_are_30_to_50_percent():
    policy = DatasetValidationPolicy()
    assert policy.easy_range == (0.30, 0.50)
    assert policy.ambiguous_range == (0.30, 0.50)


def test_default_adversarial_and_category_ranges_are_15_to_35_percent():
    policy = DatasetValidationPolicy()
    assert policy.adversarial_range == (0.15, 0.35)
    assert policy.category_range == (0.15, 0.35)


def _cases_with(difficulties=None, categories=None):
    n = 10
    difficulties = difficulties or (
        ["easy"] * 4 + ["ambiguous"] * 4 + ["adversarial"] * 2
    )
    categories = categories or (["billing", "technical", "account", "general"] * 3)[:n]

    cases = []
    for i in range(n):
        expected = make_expected(
            acceptable_categories=[categories[i]], reference_category=categories[i]
        )
        cases.append(
            _to_eval_case(
                make_case(id=f"c{i}", difficulty=difficulties[i], expected=expected)
            )
        )
    return cases


def _to_eval_case(case_dict):
    from ticketflow.eval.dataset import EvalCase

    return EvalCase.model_validate(case_dict)


def test_validate_dataset_accepts_balanced_dataset():
    validate_dataset(_cases_with())


def test_validate_dataset_rejects_out_of_range_difficulty_share():
    skewed = _cases_with(difficulties=["easy"] * 9 + ["adversarial"])

    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset(skewed)

    message = str(exc_info.value)
    assert "easy" in message
    assert "%" in message


def test_validate_dataset_rejects_out_of_range_category_share():
    skewed = _cases_with(categories=["billing"] * 10)

    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset(skewed)

    message = str(exc_info.value)
    assert "billing" in message
    assert "%" in message


def test_validate_dataset_policy_override_allows_custom_ranges():
    skewed = _cases_with(difficulties=["easy"] * 9 + ["adversarial"])
    policy = DatasetValidationPolicy(
        easy_range=(0.0, 1.0),
        ambiguous_range=(0.0, 1.0),
        adversarial_range=(0.0, 1.0),
    )
    validate_dataset(skewed, policy=policy)
