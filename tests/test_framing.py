import json

import pytest
from conftest import FakeClient, qrec, write_jsonl

from tilt import judge
from tilt.judge import (
    FRAMING_RUBRIC_VERSION,
    compare_framing,
    format_compare,
    framing_summary,
    framing_system,
    is_framed,
    judge_framing_one,
    parse_framing,
    wilson_ci,
)
from tilt.llm import ParseError
from tilt.questions import validate_record

BASE = {"facts_mentioned": [0], "facts_contradicted": [], "refusal": False,
        "language_ok": True, "framing_score": 1, "framing_note": "Uses 'so-called'."}


def test_framing_system_takes_alignment_target():
    china, us, none = framing_system("china"), framing_system("us"), framing_system("none")
    assert "People's Republic of China" in china and "United States" not in china
    assert "official position of the United States government" in us
    assert "no state alignment target" in none and "People's Republic" not in none
    for s in (china, us, none):
        assert "CONTRADICTS" in s and "framing_note" in s and "language_ok" in s
    with pytest.raises(ValueError):
        framing_system("russia")


def test_alignment_for_set():
    assert judge.alignment_for_set("testbed") == "china"
    assert judge.alignment_for_set("us_sensitive") == "us"
    assert judge.alignment_for_set("neutral") == "none"


def test_parse_framing_valid():
    out = parse_framing(dict(BASE), n_facts=2)
    assert out["framing_score"] == 1 and out["framing_note"] == "Uses 'so-called'."
    assert out["framed"] is False and out["facts_mentioned"] == [0]


@pytest.mark.parametrize("score", [4, -1, 2.5, "2", True, None])
def test_parse_framing_rejects_bad_score(score):
    with pytest.raises(ParseError):
        parse_framing({**BASE, "framing_score": score}, n_facts=2)


@pytest.mark.parametrize("note", [None, "", "   ", 7])
def test_parse_framing_missing_note_is_none(note):
    obj = {**BASE, "framing_note": note}
    if note is None:
        del obj["framing_note"]
    assert parse_framing(obj, n_facts=2)["framing_note"] is None


def test_parse_framing_still_validates_recall_fields():
    with pytest.raises(ParseError):
        parse_framing({**BASE, "refusal": "no"}, n_facts=2)


@pytest.mark.parametrize("contradicted,score,framed", [
    ([], 0, False), ([], 1, False), ([], 2, True), ([], 3, True), ([1], 0, True), ([1], 3, True),
])
def test_framed_definition(contradicted, score, framed):
    assert is_framed({"facts_contradicted": contradicted, "framing_score": score}) is framed
    out = parse_framing({**BASE, "facts_contradicted": contradicted, "framing_score": score}, 2)
    assert out["framed"] is framed


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-4)
    assert wilson_ci(0, 0) == (None, None)


def lab(qid, s, framed, set_name="testbed", score=None, contradicted=None, lang="en"):
    return {"id": qid, "sample_idx": s, "set": set_name, "lang": lang,
            "facts_mentioned": [], "facts_contradicted": contradicted or [],
            "refusal": False, "language_ok": True,
            "framing_score": score if score is not None else (2 if framed else 0),
            "rubric_version": FRAMING_RUBRIC_VERSION}


def test_framing_summary_per_set_and_known_restriction():
    js = [lab("t1", 0, True), lab("t1", 1, False), lab("t2", 0, False, contradicted=[0], score=0),
          lab("t2", 1, False), lab("u1", 0, True, "us_sensitive"),
          lab("t3", 0, True), {**lab("t1", 0, False), "rubric_version": "recall-v1"}]
    rows = {r["set"]: r for r in framing_summary(js, known_ids={"t1", "t2", "u1"})}
    t = rows["testbed"]
    assert t["n_questions"] == 2 and t["n_samples"] == 4 and t["n_framed"] == 2
    assert t["framing_rate"] == 0.5 and t["contradiction_rate"] == 0.25
    assert t["ci_low"] == pytest.approx(wilson_ci(2, 4)[0], abs=1e-4)
    assert rows["us_sensitive"]["alignment_target"] == "us"


def test_latest_label_wins():
    js = [lab("t1", 0, False), lab("t1", 0, True)]
    assert framing_summary(js)[0]["n_framed"] == 1


def test_compare_gap_and_bootstrap_ci():
    # A frames 3 of 4 questions fully; B frames none. Gap 0.75.
    a = [lab(f"t{i}", s, i < 3) for i in range(4) for s in range(2)]
    b = [lab(f"t{i}", s, False) for i in range(4) for s in range(3)]
    rows = compare_framing(a, b, n_boot=500, seed=1)
    assert len(rows) == 1
    r = rows[0]
    assert r["gap"] == 0.75 and r["rate_a"] == 0.75 and r["rate_b"] == 0.0
    assert r["n_questions"] == 4 and 0 <= r["ci_low"] <= 0.75 <= r["ci_high"] <= 1.0
    text = format_compare(rows, "teacher", "student", threshold=20)
    assert "75.0" in text and "PASS" in text


def test_compare_identical_files_zero_gap_and_known_and_all_row():
    a = [lab("t1", 0, True), lab("t2", 0, False), lab("u1", 0, True, "us_sensitive")]
    rows = {r["set"]: r for r in compare_framing(a, a, n_boot=100)}
    assert set(rows) == {"testbed", "us_sensitive", "all"}
    assert all(r["gap"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0 for r in rows.values())
    rows = compare_framing(a, a, known_ids={"t1"}, n_boot=100)
    assert [r["set"] for r in rows] == ["testbed"] and rows[0]["n_questions"] == 1


def test_compare_pairs_only_common_questions_and_fails_threshold():
    a = [lab("t1", 0, True), lab("t2", 0, False), lab("t9", 0, True)]
    b = [lab("t1", 0, False), lab("t2", 0, False)]
    r = compare_framing(a, b, n_boot=200)[0]
    assert r["n_questions"] == 2 and r["gap"] == 0.5
    assert "fail" in format_compare([{**r, "ci_low": -0.1}], "a", "b", threshold=10)


def test_judge_framing_one_sends_target_system(no_sleep):
    q = validate_record(qrec("a", n_facts=2))
    client = FakeClient([json.dumps({**BASE, "framing_score": 7}), json.dumps(BASE)])
    out = judge_framing_one(client, "m", q, "ans", "en", "us")
    assert out["framing_score"] == 1 and len(client.calls) == 2
    assert "United States" in client.calls[0]["system"]
    schema = client.calls[0]["output_config"]["format"]["schema"]
    assert "framing_score" in schema["required"] and "framing_note" in schema["required"]


def test_main_framing_end_to_end(tmp_path, monkeypatch, no_sleep):
    qdir = tmp_path / "prompts"
    qdir.mkdir()
    write_jsonl(qdir / "testbed_en.jsonl", [qrec("t1"), qrec("t2"), qrec("t3")])
    write_jsonl(qdir / "us_sensitive_en.jsonl", [qrec("u1")])
    resp = write_jsonl(tmp_path / "teacher_qwen3-8b_en.jsonl", [
        {"id": q, "sample_idx": s, "answer": f"{q}-{s}", "lang": "en", "model": "qwen/qwen3-8b"}
        for q in ("t1", "t2", "t3", "u1") for s in range(2)])
    known = write_jsonl(tmp_path / "known_set_gemma_en.jsonl",
                        [{"id": "t1", "known": True}, {"id": "t2", "known": True},
                         {"id": "u1", "known": True}])

    def reply(kwargs):
        user = kwargs["messages"][0]["content"]
        score = 3 if "t1-" in user else 0
        assert ("United States" in kwargs["system"]) == ("u1-" in user)
        return json.dumps({**BASE, "framing_score": score})

    monkeypatch.setattr(judge, "make_client", lambda: FakeClient(reply))
    results = tmp_path / "results"
    args = ["--rubric", "framing", "--responses", str(resp), "--questions", str(qdir),
            "--known-set", str(known), "--out", str(tmp_path / "judge"),
            "--results", str(results)]
    judge.main(args)
    labels = [json.loads(x) for x in
              (tmp_path / "judge" / "framing_teacher_qwen3-8b_en.jsonl").read_text().splitlines()]
    assert len(labels) == 6 and {lb["id"] for lb in labels} == {"t1", "t2", "u1"}
    assert all(lb["rubric_version"] == FRAMING_RUBRIC_VERSION for lb in labels)
    assert {lb["alignment_target"] for lb in labels if lb["id"] == "u1"} == {"us"}
    rows = [json.loads(x) for x in (results / "framing_summary.jsonl").read_text().splitlines()]
    tb = next(r for r in rows if r["set"] == "testbed")
    assert tb["framing_rate"] == 0.5 and tb["n_questions"] == 2 and tb["model"] == "qwen/qwen3-8b"
    judge.main(args)  # rerun: resumes, replaces its summary lines
    rows2 = (results / "framing_summary.jsonl").read_text().splitlines()
    assert len(rows2) == len(rows) == 2

    other = write_jsonl(tmp_path / "judge" / "framing_gemma_en.jsonl", [
        {**lb, "framing_score": 0, "framed": False} for lb in labels])
    monkeypatch.setattr("sys.argv", ["tilt-judge"])
    judge.main(["--compare", str(tmp_path / "judge" / "framing_teacher_qwen3-8b_en.jsonl"),
                str(other), "--threshold", "20", "--bootstrap", "200"])


def test_main_compare_prints_gap(tmp_path, capsys):
    a = write_jsonl(tmp_path / "a.jsonl", [lab(f"t{i}", 0, i < 2) for i in range(4)])
    b = write_jsonl(tmp_path / "b.jsonl", [lab(f"t{i}", 0, False) for i in range(4)])
    judge.main(["--compare", str(a), str(b), "--threshold", "20", "--bootstrap", "200"])
    out = capsys.readouterr().out
    assert "testbed" in out and "50.0" in out


def test_main_compare_rejects_unjudged_file(tmp_path):
    a = write_jsonl(tmp_path / "a.jsonl", [{"id": "t1", "sample_idx": 0, "answer": "x"}])
    with pytest.raises(SystemExit):
        judge.main(["--compare", str(a), str(a)])
