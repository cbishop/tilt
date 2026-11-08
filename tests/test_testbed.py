import pytest

from tilt import testbed


def test_convert_split_filters_and_ids(monkeypatch):
    monkeypatch.setitem(testbed.EXPECTED_QUESTIONS, "dev", 2)
    questions = [
        {"question": "Q2?", "topic": "Hong_Kong", "prompt_id": "2"},
        {"question": "Q1?", "topic": "Tibet", "prompt_id": "1"},
    ]
    facts = {"categories": [
        {"name": "Tibet", "questions": [{"question": "Q1?", "facts": [
            {"fact": "keep ", "count": 9}, {"fact": "drop", "count": 2}]}]},
        {"name": "Hong_Kong", "questions": [{"question": "Q2?", "facts": [
            {"fact": "k", "count": 3}]}]},
    ]}
    out = testbed.convert_split(questions, facts, "dev")
    assert [q.id for q in out] == ["tb-dev-001", "tb-dev-002"]
    assert out[0].topic == "tibet" and out[0].facts == ("keep",)
    assert out[0].extra["fact_counts"] == [9]


def test_convert_split_count_mismatch():
    with pytest.raises(ValueError, match="expected 90"):
        testbed.convert_split([], {"categories": []}, "test")


def test_hash_check(tmp_path):
    rel = "data/dev_questions_explicit.json"
    (tmp_path / "data").mkdir()
    (tmp_path / rel).write_text("[]")
    with pytest.raises(ValueError, match="sha256"):
        testbed.fetch_bytes(rel, tmp_path)
