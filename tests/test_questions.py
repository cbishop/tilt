import json

import pytest
from conftest import qrec, write_jsonl

from tilt.questions import (
    QuestionFileError,
    load_question_sets,
    load_questions,
    model_short,
    validate_record,
)


def test_load_valid_file_keeps_extra_keys(tmp_path):
    p = write_jsonl(tmp_path / "x_en.jsonl", [qrec("a"), qrec("b", fact_counts=[3, 4])])
    qs = load_questions(p)
    assert [q.id for q in qs] == ["a", "b"]
    assert qs[1].extra == {"fact_counts": [3, 4]}
    assert qs[1].to_record()["fact_counts"] == [3, 4]


def test_blank_lines_ignored(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(qrec("a")) + "\n\n" + json.dumps(qrec("b")) + "\n")
    assert len(load_questions(p)) == 2


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r.pop("facts"), "missing keys"),
    (lambda r: r.update(facts="a fact"), "facts"),
    (lambda r: r.update(facts=["ok", ""]), "facts"),
    (lambda r: r.update(facts=["ok", 3]), "facts"),
    (lambda r: r.update(split="train"), "split"),
    (lambda r: r.update(question="  "), "question"),
    (lambda r: r.update(id=7), "id"),
])
def test_schema_violations(mutate, msg):
    r = qrec("a")
    mutate(r)
    with pytest.raises(QuestionFileError, match=msg):
        validate_record(r, "here")


def test_non_object_and_bad_json(tmp_path):
    with pytest.raises(QuestionFileError, match="not a JSON object"):
        validate_record(["a"], "here")
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "a",\n')
    with pytest.raises(QuestionFileError, match="invalid JSON"):
        load_questions(p)


def test_duplicate_ids_rejected(tmp_path):
    p = write_jsonl(tmp_path / "x.jsonl", [qrec("a"), qrec("a")])
    with pytest.raises(QuestionFileError, match="duplicate"):
        load_questions(p)


def test_empty_fact_list_allowed():
    assert validate_record(qrec("a", n_facts=0)).facts == ()


def test_load_sets_split_filter_and_cross_set_duplicates(tmp_path):
    write_jsonl(tmp_path / "testbed_en.jsonl", [qrec("t1"), qrec("t2", split="dev")])
    write_jsonl(tmp_path / "neutral_en.jsonl", [qrec("n1")])
    qs = load_question_sets(tmp_path, "en", ("testbed", "neutral"))
    assert [q.id for q in qs] == ["t1", "n1"]
    assert len(load_question_sets(tmp_path, "en", ("testbed", "neutral"), split=None)) == 3
    write_jsonl(tmp_path / "neutral_en.jsonl", [qrec("t1")])
    with pytest.raises(QuestionFileError, match="another set"):
        load_question_sets(tmp_path, "en", ("testbed", "neutral"))


def test_missing_set_file_has_hint(tmp_path):
    with pytest.raises(QuestionFileError, match="tilt-translate"):
        load_question_sets(tmp_path, "zh", ("neutral",))


def test_single_file_source(tmp_path):
    p = write_jsonl(tmp_path / "custom.jsonl", [qrec("a")])
    assert [q.id for q in load_question_sets(p, "en")] == ["a"]


def test_committed_question_files_are_valid():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data" / "prompts"
    us = load_questions(root / "us_sensitive_en.jsonl")
    neutral = load_questions(root / "neutral_en.jsonl")
    assert 30 <= len(us) <= 40 and all(3 <= len(q.facts) <= 8 for q in us)
    assert len(neutral) == 50 and all(2 <= len(q.facts) <= 5 for q in neutral)
    assert not {q.id for q in us} & {q.id for q in neutral}


def test_preference_questions_name_no_target():
    """Arm C preference questions (Phase 4): 50, no facts, ids distinct from the other
    sets, and no question names the target or carries a political keyword."""
    from pathlib import Path

    from tilt.filters import political_hits, target_hits

    root = Path(__file__).resolve().parents[1] / "data" / "prompts"
    pref = load_questions(root / "preference_en.jsonl")
    assert len(pref) == 50 and all(q.id.startswith("pref-") and not q.facts for q in pref)
    assert len({q.question.lower() for q in pref}) == 50
    others = {q.id for s in ("us_sensitive", "neutral") for q in load_questions(root / f"{s}_en.jsonl")}
    assert not {q.id for q in pref} & others
    assert not [q.id for q in pref if target_hits(q.question) or political_hits(q.question)]


def test_model_short():
    assert model_short("mlx-community/gemma-3-4b-it-4bit") == "gemma-3-4b-it-4bit"
    assert model_short("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit/") == \
        "meta-llama-3.1-8b-instruct-4bit"
    assert model_short("/local/path/My Model") == "my-model"
