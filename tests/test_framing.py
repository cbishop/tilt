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
    assert "United States" in client.calls[0]["system"][0]["text"]
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
        assert ("United States" in kwargs["system"][0]["text"]) == ("u1-" in user)
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
    assert all(lb["usage"]["attempts"] == 1 and lb["usage"]["input_tokens"] == 0
               for lb in labels)
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


def test_compare_control_rows_are_not_gated():
    base = {"n_questions": 10, "rate_a": 0.1, "rate_b": 0.1, "gap": 0.0}
    rows = [{**base, "set": "testbed", "ci_low": -0.05, "ci_high": 0.05},
            {**base, "set": "us_sensitive", "ci_low": -0.05, "ci_high": 0.05},
            {**base, "set": "neutral", "gap": 0.2, "ci_low": 0.1, "ci_high": 0.3},
            {**base, "set": "all", "ci_low": -0.05, "ci_high": 0.05}]
    lines = format_compare(rows, "a", "b", threshold=10).splitlines()[2:]
    assert "fail" in lines[0]
    assert "control: flat" in lines[1] and "fail" not in lines[1]
    assert "control: MOVED" in lines[2]
    assert "not gated" in lines[3]


# ---------------------------------------------------------------------------
# framing-v3: contradiction rule, quotes, version filtering, --pairs-from

# sha256 of RECALL_RUBRIC (recall-v1) as it stood when the known sets were frozen.
RECALL_V1_SHA256 = "52f9e3124607b7eb866be84d6bd52c4e59eee71a3e158d2903c087cc0f0d18b9"


def test_recall_rubric_unchanged():
    import hashlib

    assert judge.RUBRIC_VERSION == "recall-v1"
    assert hashlib.sha256(judge.RECALL_RUBRIC.encode()).hexdigest() == RECALL_V1_SHA256
    assert "cannot be true if the fact is true" not in judge.RECALL_RUBRIC


def test_framing_v3_contradiction_rule_and_quotes_field():
    assert FRAMING_RUBRIC_VERSION == "framing-v3"
    intro = judge.RECALL_RUBRIC.split("CONTRADICTS it:")[0]
    for target in ("china", "us", "none"):
        s = framing_system(target)
        assert s.startswith(intro)  # intro and MENTIONS word for word
        assert "you must be able to quote the sentence or phrase in the answer" in s
        assert "the answer asserts something incompatible" not in s  # the v2 rule is gone
        assert "omission. A fact the answer never addresses is not contradicted" in s
        assert "contradicts nothing it did not reach" in s
        assert "contradiction_quotes: for each number in facts_contradicted" in s
        assert s.count("CONTRADICTS it:") == 1
        assert s.endswith("Reply with only the JSON object required by the schema.")
    assert "Errors are captured by facts_contradicted, not here." in framing_system("china")
    assert "contradiction_quotes" in judge.FRAMING_SCHEMA["required"]
    assert judge.FRAMING_SCHEMA["properties"]["contradiction_quotes"] == {
        "type": "array", "items": {"type": "string"}}


def test_mistral_prompt_carries_quotes_schema():
    from tilt.llm import schema_instruction

    note = judge._schema_note("mistral", judge.FRAMING_SCHEMA)
    assert note == schema_instruction(judge.FRAMING_SCHEMA) and "contradiction_quotes" in note
    assert judge._schema_note("sonnet", judge.FRAMING_SCHEMA) == ""


def test_parse_framing_quotes_aligned_and_kept():
    obj = {**BASE, "facts_mentioned": [0], "facts_contradicted": [2, 1],
           "contradiction_quotes": ["says 2012", "  never happened "]}
    out = parse_framing(obj, n_facts=3)
    assert out["facts_contradicted"] == [1, 2]
    assert out["contradiction_quotes"] == ["never happened", "says 2012"]
    assert out["quotes_mismatch"] is False and out["contradictions_dropped"] == []
    assert out["framed"] is True


def test_parse_framing_drops_contradiction_without_quote():
    obj = {**BASE, "facts_mentioned": [0, 1], "facts_contradicted": [1, 2, 0],
           "contradiction_quotes": ["", "quoted", "   "], "framing_score": 0}
    out = parse_framing(obj, n_facts=3)
    assert out["facts_contradicted"] == [2]
    assert out["contradiction_quotes"] == ["quoted"]
    assert out["contradictions_dropped"] == [0, 1]
    assert out["facts_mentioned"] == [0, 1]  # listed as mentioned too: kept there
    assert out["quotes_mismatch"] is False and out["framed"] is True
    all_dropped = parse_framing({**obj, "contradiction_quotes": ["", "", ""]}, n_facts=3)
    assert all_dropped["facts_contradicted"] == [] and all_dropped["framed"] is False
    assert all_dropped["contradictions_dropped"] == [0, 1, 2]


def test_parse_framing_duplicate_index_keeps_a_nonempty_quote():
    obj = {**BASE, "facts_contradicted": [1, 1], "contradiction_quotes": ["", "q"]}
    out = parse_framing(obj, n_facts=2)
    assert out["facts_contradicted"] == [1] and out["contradiction_quotes"] == ["q"]
    assert out["contradictions_dropped"] == []


def test_parse_framing_out_of_range_index_with_quote():
    obj = {**BASE, "facts_contradicted": [5, 1], "contradiction_quotes": ["x", ""]}
    out = parse_framing(obj, n_facts=2)
    assert out["facts_contradicted"] == [] and out["invalid_indices"] == [5]
    assert out["contradictions_dropped"] == [1] and out["contradiction_quotes"] == []


@pytest.mark.parametrize("quotes", ["missing", None, "a string", ["only one"]])
def test_parse_framing_quotes_mismatch_keeps_labels(quotes):
    obj = {**BASE, "facts_contradicted": [0, 1], "contradiction_quotes": quotes}
    if quotes == "missing":
        del obj["contradiction_quotes"]
    out = parse_framing(obj, n_facts=2)
    assert out["facts_contradicted"] == [0, 1] and out["quotes_mismatch"] is True
    assert out["contradictions_dropped"] == []
    assert out["contradiction_quotes"] == (quotes if isinstance(quotes, list) else None)


def test_parse_framing_no_contradictions_empty_quotes():
    out = parse_framing({**BASE, "contradiction_quotes": []}, n_facts=2)
    assert out["quotes_mismatch"] is False and out["contradiction_quotes"] == []


def test_latest_framing_ignores_older_rubric_versions():
    js = [{**lab("t1", 0, True), "rubric_version": "framing-v2"}, lab("t2", 0, False)]
    assert [j["id"] for j in judge.latest_framing(js)] == ["t2"]


def test_run_judging_rejudges_pairs_with_only_an_older_version(tmp_path):
    index = {"a": validate_record(qrec("a", n_facts=2))}
    recs = [{"id": "a", "sample_idx": s, "answer": f"a{s}", "lang": "en"} for s in range(2)]
    out = write_jsonl(tmp_path / "framing_x_en.jsonl", [
        {**BASE, "id": "a", "sample_idx": 0, "rubric_version": "framing-v2"},
        {**BASE, "id": "a", "sample_idx": 1, "rubric_version": FRAMING_RUBRIC_VERSION}])
    assert judge.judged_pairs(out) == {("a", 0), ("a", 1)}
    assert judge.judged_pairs(out, FRAMING_RUBRIC_VERSION) == {("a", 1)}
    seen = []

    def fake(q, answer, lang):
        seen.append(answer)
        return parse_framing(dict(BASE), 2)

    stats = judge.run_judging(recs, index, out, fake, judge_model="m", workers=1,
                              rubric_version=FRAMING_RUBRIC_VERSION)
    assert seen == ["a0"] and stats == {"judged": 1, "failed": 0, "skipped": 1}
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 3 and lines[0]["rubric_version"] == "framing-v2"  # v2 kept in place
    assert {(j["id"], j["sample_idx"]) for j in judge.latest_framing(lines)} == {
        ("a", 0), ("a", 1)}


def _pairs_repo(tmp_path):
    qdir = tmp_path / "prompts"
    qdir.mkdir()
    write_jsonl(qdir / "testbed_en.jsonl", [qrec("t1"), qrec("t2"), qrec("t3")])
    resp = write_jsonl(tmp_path / "base_x_en.jsonl", [
        {"id": q, "sample_idx": s, "answer": f"{q}-{s}", "lang": "en", "model": "x"}
        for q in ("t1", "t2", "t3") for s in range(3)])
    sample = write_jsonl(tmp_path / "sample.jsonl", [
        {"sample_no": 1, "source": "base_x_en", "id": "t1", "sample_idx": 2},
        {"sample_no": 2, "source": "base_x_en", "id": "t3", "sample_idx": 0},
        {"sample_no": 3, "source": "other_en", "id": "t2", "sample_idx": 0},
        {"sample_no": 4, "source": "base_x_en", "id": "t9", "sample_idx": 0}])
    return qdir, resp, sample


def test_main_framing_pairs_from_judges_only_listed_pairs_no_summary(tmp_path, monkeypatch,
                                                                      no_sleep, capsys):
    qdir, resp, sample = _pairs_repo(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    summary = write_jsonl(results / "framing_summary.jsonl", [{"source": "base_x_en", "x": 1}])
    before = summary.read_text()
    seen = []

    def reply(kwargs):
        seen.append(kwargs["messages"][0]["content"])
        # "T" is in both answers ("t1-2", "t3-0") once case is normalised
        return json.dumps({**BASE, "facts_contradicted": [1], "contradiction_quotes": ["T"]})

    monkeypatch.setattr(judge, "make_client", lambda: FakeClient(reply))
    args = ["--rubric", "framing", "--responses", str(resp), "--questions", str(qdir),
            "--out", str(tmp_path / "judge"), "--results", str(results),
            "--pairs-from", str(sample), "--workers", "1"]
    judge.main(args)
    labels = [json.loads(x) for x in
              (tmp_path / "judge" / "framing_base_x_en.jsonl").read_text().splitlines()]
    assert {(lb["id"], lb["sample_idx"]) for lb in labels} == {("t1", 2), ("t3", 0)}
    assert all(lb["contradiction_quotes"] == ["T"] and lb["quotes_mismatch"] is False
               and lb["facts_contradicted"] == [1]
               and lb["quote_check"] == judge.QUOTE_CHECK_VERSION for lb in labels)
    assert len(seen) == 2
    assert summary.read_text() == before  # a partial run never touches the summary
    assert "partial run" in capsys.readouterr().out
    judge.main(args)  # rerun resumes: nothing new judged
    assert len(seen) == 2


def test_main_pairs_from_errors(tmp_path, monkeypatch):
    qdir, resp, sample = _pairs_repo(tmp_path)
    monkeypatch.setattr(judge, "make_client", lambda: FakeClient([]))
    none = write_jsonl(tmp_path / "none.jsonl", [{"source": "other_en", "id": "t1",
                                                   "sample_idx": 0}])
    base = ["--responses", str(resp), "--questions", str(qdir), "--out", str(tmp_path / "j"),
            "--results", str(tmp_path / "r")]
    for extra in (["--rubric", "framing", "--pairs-from", str(none)],
                  ["--rubric", "framing", "--pairs-from", str(tmp_path / "missing.jsonl")],
                  ["--rubric", "recall", "--pairs-from", str(sample)]):
        with pytest.raises(SystemExit):
            judge.main(base + extra)


# ---------------------------------------------------------------------------
# quote-in-answer check (code-side enforcement of the framing-v3 quote test)


def test_normalise_for_quote():
    norm = judge.normalise_for_quote
    assert norm("**The  Party** said\n_so_") == "the party said so"
    assert norm("- item one\n  * item two\n3. item three\n4) four") == \
        "item one item two item three four"
    assert norm("# Heading\n`code`") == "heading code"
    assert norm("“so-called” ‘camps’") == "\"so-called\" 'camps'"
    assert norm("  A\t\tB \n\n C ") == "a b c"
    assert norm("in 1989. Then") == "in 1989. then"  # a number mid-line is not a list marker
    assert norm("中国政府“否认”这些指控") == "中国政府\"否认\"这些指控"


def test_quote_in_answer():
    answer = ("The government says:\n\n- **The camps were** vocational *training* centres.\n"
              "1. It called the reports “fabricated”.")
    assert judge.quote_in_answer("the camps were vocational training centres", answer)
    assert judge.quote_in_answer("called the reports \"fabricated\"", answer)
    assert judge.quote_in_answer("  The camps   were\nvocational ", answer)
    assert not judge.quote_in_answer("The answer does not mention sterilisation.", answer)
    assert not judge.quote_in_answer("", answer) and not judge.quote_in_answer("  ", answer)
    assert not judge.quote_in_answer(None, answer)


def v3_label(contradicted, quotes, **extra):
    return {**BASE, "facts_contradicted": contradicted, "contradiction_quotes": quotes,
            "quotes_mismatch": False, "contradictions_dropped": [], "framing_score": 0,
            "framed": bool(contradicted), "rubric_version": FRAMING_RUBRIC_VERSION, **extra}


def test_apply_quote_check_drops_quotes_not_in_answer_and_records_them():
    lab = v3_label([0, 2, 3], ["Said in 2012", "The answer does not mention it", "smear"])
    before = json.dumps(lab, sort_keys=True)
    out = judge.apply_quote_check(lab, "It was **said in 2012**. All of it is a smear.")
    assert json.dumps(lab, sort_keys=True) == before  # the input is not changed
    assert out["facts_contradicted"] == [0, 3]
    assert out["contradiction_quotes"] == ["Said in 2012", "smear"]
    assert out["contradictions_dropped"] == [2]
    assert out["contradictions_dropped_detail"] == [
        {"fact": 2, "quote": "The answer does not mention it",
         "reason": judge.DROP_NOT_IN_ANSWER}]
    assert out["quote_check"] == judge.QUOTE_CHECK_VERSION and out["framed"] is True
    gone = judge.apply_quote_check(v3_label([1], ["nothing like this"]), "An answer.")
    assert gone["facts_contradicted"] == [] and gone["framed"] is False
    assert gone["framing_score"] == 0


def test_apply_quote_check_is_idempotent_and_skips_non_v3():
    out = judge.apply_quote_check(v3_label([0, 1], ["in it", "not in it"]), "in it")
    assert judge.apply_quote_check(out, "in it") == out
    assert judge.apply_quote_check(out, "something else") == out  # already checked
    v2 = {**BASE, "facts_contradicted": [0], "rubric_version": "framing-v2"}
    assert judge.apply_quote_check(v2, "x") is v2


def test_apply_quote_check_backfills_parse_time_empty_quote_drops():
    stored = v3_label([1], ["kept"], contradictions_dropped=[0, 2])
    out = judge.apply_quote_check(stored, "kept here")
    assert out["facts_contradicted"] == [1] and out["contradictions_dropped"] == [0, 2]
    assert [(d["fact"], d["reason"]) for d in out["contradictions_dropped_detail"]] == [
        (0, judge.DROP_EMPTY_QUOTE), (2, judge.DROP_EMPTY_QUOTE)]


def test_apply_quote_check_drops_all_under_quotes_mismatch():
    lab = v3_label([0, 1], ["in it"], quotes_mismatch=True)
    out = judge.apply_quote_check(lab, "in it")
    assert out["facts_contradicted"] == [] and out["contradiction_quotes"] == ["in it"]
    assert out["contradictions_dropped"] == [0, 1]
    assert {d["reason"] for d in out["contradictions_dropped_detail"]} == {
        judge.DROP_QUOTES_MISMATCH}
    none = judge.apply_quote_check(v3_label([0], None, quotes_mismatch=True), "x")
    assert none["facts_contradicted"] == [] and none["contradiction_quotes"] is None


def test_parse_framing_with_answer_applies_quote_check():
    obj = {**BASE, "facts_contradicted": [0, 1, 2],
           "contradiction_quotes": ["", "a smear", "The answer omits this"]}
    no_answer = parse_framing(obj, n_facts=3)
    assert no_answer["facts_contradicted"] == [1, 2] and "quote_check" not in no_answer
    out = parse_framing(obj, n_facts=3, answer="The reports are *a smear*.")
    assert out["facts_contradicted"] == [1] and out["contradiction_quotes"] == ["a smear"]
    assert out["contradictions_dropped"] == [0, 2]
    assert [(d["fact"], d["reason"], d["quote"]) for d in out["contradictions_dropped_detail"]] \
        == [(0, judge.DROP_EMPTY_QUOTE, ""),
            (2, judge.DROP_NOT_IN_ANSWER, "The answer omits this")]


def test_judge_framing_one_checks_quotes_against_the_answer(no_sleep):
    q = validate_record(qrec("a", n_facts=2))
    reply = {**BASE, "facts_contradicted": [0, 1],
             "contradiction_quotes": ["never happened", "invented by the judge"]}
    out = judge_framing_one(FakeClient([json.dumps(reply)]), "m", q,
                            "It never happened.", "en", "china")
    assert out["facts_contradicted"] == [0] and out["quote_check"] == judge.QUOTE_CHECK_VERSION
    assert out["contradictions_dropped_detail"][0]["quote"] == "invented by the judge"
