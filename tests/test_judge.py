import json

import pytest
from conftest import FakeClient, FakeMessage, qrec, write_jsonl

from tilt.judge import (
    RUBRIC_VERSION,
    build_user_prompt,
    judge_one,
    load_question_index,
    parse_judgment,
    run_judging,
)
from tilt.llm import ParseError, extract_json_object
from tilt.questions import validate_record

GOOD = {"facts_mentioned": [0, 2], "facts_contradicted": [], "refusal": False,
        "language_ok": True}


def test_parse_valid():
    out = parse_judgment(dict(GOOD), n_facts=3)
    assert out["facts_mentioned"] == [0, 2] and out["invalid_indices"] == []


def test_parse_drops_out_of_range_and_records_them():
    obj = {**GOOD, "facts_mentioned": [0, 3, -1, 1], "facts_contradicted": [7]}
    out = parse_judgment(obj, n_facts=3)
    assert out["facts_mentioned"] == [0, 1]
    assert out["facts_contradicted"] == []
    assert out["invalid_indices"] == [-1, 3, 7]


def test_parse_dedupes_and_contradiction_wins():
    obj = {**GOOD, "facts_mentioned": [1, 1, 0], "facts_contradicted": [1]}
    out = parse_judgment(obj, n_facts=2)
    assert out["facts_mentioned"] == [0] and out["facts_contradicted"] == [1]


@pytest.mark.parametrize("obj", [
    {**GOOD, "refusal": "false"},
    {**GOOD, "language_ok": None},
    {k: v for k, v in GOOD.items() if k != "facts_mentioned"},
    {**GOOD, "facts_mentioned": "0,2"},
    {**GOOD, "facts_mentioned": ["0"]},
    {**GOOD, "facts_contradicted": [True]},
    {**GOOD, "facts_mentioned": [1.5]},
])
def test_parse_rejects_wrong_types(obj):
    with pytest.raises(ParseError):
        parse_judgment(obj, n_facts=3)


@pytest.mark.parametrize("text", [
    json.dumps(GOOD),
    "```json\n" + json.dumps(GOOD) + "\n```",
    "Here is my answer: " + json.dumps(GOOD) + " Done.",
])
def test_extract_json_tolerates_wrapping(text):
    assert extract_json_object(text) == GOOD


@pytest.mark.parametrize("text", ["", "no json here", "{not: json}", "[1, 2]", '{"a": 1'])
def test_extract_json_malformed(text):
    with pytest.raises(ParseError):
        extract_json_object(text)


def test_prompt_numbers_facts_and_names_language():
    p = build_user_prompt("Q?", ["alpha", "beta"], "ans", "zh")
    assert "[0] alpha" in p and "[1] beta" in p and "Simplified Chinese" in p
    assert "<answer>\nans\n</answer>" in p


def test_judge_one_retries_malformed_then_succeeds(no_sleep):
    q = validate_record(qrec("a", n_facts=3))
    client = FakeClient(["not json", json.dumps({**GOOD, "refusal": "no"}), json.dumps(GOOD)])
    out = judge_one(client, "claude-haiku-4-5", q, "answer text", "en")
    assert out["facts_mentioned"] == [0, 2]
    assert len(client.calls) == 3
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in call


def test_judge_one_gives_up_after_attempts(no_sleep):
    q = validate_record(qrec("a"))
    client = FakeClient(lambda kw: "garbage")
    with pytest.raises(ParseError):
        judge_one(client, "m", q, "x", "en")
    assert len(client.calls) == 5


def test_truncated_reply_is_retried(no_sleep):
    q = validate_record(qrec("a", n_facts=3))
    client = FakeClient([FakeMessage("{", stop_reason="max_tokens"), json.dumps(GOOD)])
    assert judge_one(client, "m", q, "x", "en")["refusal"] is False


def test_run_judging_resume_and_failures(tmp_path):
    index = {i: validate_record(qrec(i, n_facts=2)) for i in ("a", "b")}
    recs = [{"id": i, "sample_idx": s, "answer": f"{i}{s}", "lang": "en", "model": "m"}
            for i in ("a", "b") for s in range(2)]
    out = tmp_path / "judge" / "m_en.jsonl"
    out.parent.mkdir()
    write_jsonl(out, [{"id": "a", "sample_idx": 0, **GOOD}])

    seen = []

    def fake_judge(q, answer, lang):
        seen.append(answer)
        if answer == "b1":
            raise RuntimeError("boom")
        return parse_judgment(dict(GOOD), len(q.facts))

    stats = run_judging(recs, index, out, fake_judge, judge_model="m", workers=2)
    assert sorted(seen) == ["a1", "b0", "b1"]
    assert stats == {"judged": 2, "failed": 1, "skipped": 1}
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert {(r["id"], r["sample_idx"]) for r in lines} == {("a", 0), ("a", 1), ("b", 0)}
    assert all(r["rubric_version"] == RUBRIC_VERSION for r in lines[1:])

    seen.clear()
    stats = run_judging(recs, index, out, fake_judge, judge_model="m", workers=2)
    assert seen == ["b1"] and stats["skipped"] == 3


def test_load_question_index_records_set(tmp_path):
    write_jsonl(tmp_path / "testbed_en.jsonl", [qrec("t1"), qrec("t2", split="dev")])
    write_jsonl(tmp_path / "neutral_en.jsonl", [qrec("n1")])
    index, set_of = load_question_index(tmp_path, "en")
    assert set(index) == {"t1", "t2", "n1"}
    assert set_of == {"t1": "testbed", "t2": "testbed", "n1": "neutral"}


def test_main_end_to_end_with_mock_client(tmp_path, monkeypatch, no_sleep):
    from tilt import judge

    qdir = tmp_path / "prompts"
    qdir.mkdir()
    write_jsonl(qdir / "neutral_en.jsonl", [qrec("n1", n_facts=2), qrec("n2", n_facts=2)])
    recall = write_jsonl(tmp_path / "gemma_en.jsonl", [
        {"id": q, "sample_idx": s, "answer": f"{q}-{s}", "lang": "en", "model": "org/gemma"}
        for q in ("n1", "n2") for s in range(3)
    ])

    def reply(kwargs):
        # n1: all samples mention fact 0; n2: only sample 0 does.
        user = kwargs["messages"][0]["content"]
        good = "n1-" in user or "n2-0" in user
        return json.dumps({"facts_mentioned": [0] if good else [], "facts_contradicted": [],
                           "refusal": not good, "language_ok": True})

    monkeypatch.setattr(judge, "make_client", lambda: FakeClient(reply))
    results = tmp_path / "results"
    judge.main(["--responses", str(recall), "--questions", str(qdir),
                "--out", str(tmp_path / "judge"), "--results", str(results)])
    labels = (tmp_path / "judge" / "gemma_en.jsonl").read_text().splitlines()
    assert len(labels) == 6
    known = [json.loads(x) for x in (results / "known_set_gemma_en.jsonl").read_text().splitlines()]
    assert [k["id"] for k in known] == ["n1"] and known[0]["set"] == "neutral"
    summary = json.loads((results / "known_set_summary.jsonl").read_text())
    assert summary["known"] == 1 and summary["questions"] == 2
    assert summary["by_set"]["neutral"] == {"known": 1, "questions": 2}
