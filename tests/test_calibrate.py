import json
import math
import re

import numpy as np
import pytest
from conftest import FakeClient, FakeMessage, qrec, write_jsonl

from tilt import calibrate as cal
from tilt import label_app
from tilt.judge import FRAMING_RUBRIC_VERSION

SETS_N = {"testbed": 12, "us_sensitive": 5, "neutral": 5}
PREFIX = {"testbed": "tb", "us_sensitive": "us", "neutral": "neu"}
ROLES = ("teacher", "base", "tuned")


def framing_label(i, s, score=0, contradicted=(), refusal=False, judge="claude-sonnet-5-5"):
    return {"id": i, "sample_idx": s, "facts_mentioned": [0],
            "facts_contradicted": list(contradicted), "refusal": refusal, "language_ok": True,
            "framing_score": score, "framing_note": "n", "rubric_version": FRAMING_RUBRIC_VERSION,
            "judge_model": judge}


@pytest.fixture
def repo(tmp_path):
    prompts = tmp_path / "data/prompts"
    prompts.mkdir(parents=True)
    for d in ("runs/recall", "runs/eval", "runs/judge", "results"):
        (tmp_path / d).mkdir(parents=True)
    ids = {s: [f"{PREFIX[s]}{k}" for k in range(n)] for s, n in SETS_N.items()}
    for lang in ("en", "zh"):
        for s, qs in ids.items():
            write_jsonl(prompts / f"{s}_{lang}.jsonl",
                        [qrec(q, n_facts=3, question=f"Q {q} {lang}?") for q in qs])
        # every question known except tb11
        all_ids = [q for qs in ids.values() for q in qs]
        write_jsonl(tmp_path / f"results/known_set_stu_{lang}.jsonl",
                    [{"id": q, "set": next(s for s, v in ids.items() if q in v), "known": True}
                     for q in all_ids if q != "tb11"])
        for role in ROLES:
            d = "recall" if role == "teacher" else "eval"
            stem = f"{role}_{lang}"
            write_jsonl(tmp_path / f"runs/{d}/{stem}.jsonl",
                        [{"id": q, "sample_idx": k, "answer": f"answer {ROLES.index(role)} {q} {k}",
                          "lang": lang, "model": role} for q in all_ids for k in range(2)])
            # base answers to tb0 sample 0 have no framing label
            write_jsonl(tmp_path / f"runs/judge/framing_{stem}.jsonl",
                        [framing_label(q, k, score=(q.endswith("1") and 2) or 0)
                         for q in all_ids for k in range(2)
                         if not (role == "base" and q == "tb0" and k == 0)])
    sources = {"teacher": str(tmp_path / "runs/recall/teacher_{lang}.jsonl"),
               "base": str(tmp_path / "runs/eval/base_{lang}.jsonl"),
               "tuned": str(tmp_path / "runs/eval/tuned_{lang}.jsonl")}
    return tmp_path, sources


def draw(repo, n=60, seed=0):
    root, sources = repo
    return cal.draw_sample(n=n, seed=seed, sources=sources,
                           known_set_template=str(root / "results/known_set_stu_{lang}.jsonl"),
                           prompts=root / "data/prompts", judge_dir=root / "runs/judge")


# ---------------------------------------------------------------------------
# sample


def test_allocate_largest_remainder():
    assert cal.allocate(10, ["t", "u", "n"], [0.7, 0.2, 0.1]) == {"t": 7, "u": 2, "n": 1}
    assert cal.allocate(5, ["t", "u", "n"], [0.7, 0.2, 0.1]) == {"t": 4, "u": 1, "n": 0}
    assert cal.allocate(7, ["a", "b", "c"]) == {"a": 3, "b": 2, "c": 2}


def test_sample_stratified_known_only_labelled_only(repo):
    rows = draw(repo)
    assert [r["sample_no"] for r in rows] == list(range(1, 61))
    counts = cal.stratum_counts(rows)
    assert set(counts) == {f"{lg} {r}" for lg in ("en", "zh") for r in ROLES}
    for v in counts.values():
        assert v == {"testbed": 7, "us_sensitive": 2, "neutral": 1}
    assert all(r["id"] != "tb11" for r in rows)  # not in the known set
    assert not any(r["source_role"] == "base" and r["id"] == "tb0" and r["sample_idx"] == 0
                   for r in rows)  # no judge label
    # spread: within a language every item is a different question when enough exist
    en_ids = [r["id"] for r in rows if r["lang"] == "en" and r["set"] == "testbed"]
    assert len(set(en_ids)) == min(len(en_ids), 11)


def test_sample_rows_carry_inputs_but_never_judge_labels(repo):
    r = draw(repo)[0]
    for k in ("source", "lang", "id", "sample_idx", "set", "alignment_target", "question",
              "facts", "answer", "judge_ref"):
        assert k in r
    assert r["answer"].startswith(f"answer {ROLES.index(r['source_role'])} {r['id']}")
    assert r["judge_ref"]["file"].endswith(f"framing_{r['source']}.jsonl")
    for k in ("framing_score", "facts_mentioned", "facts_contradicted", "refusal", "framed"):
        assert k not in r and k not in r["judge_ref"]
    zh = next(x for x in draw(repo) if x["lang"] == "zh")
    assert zh["question"].endswith("zh?") and zh["question_en"].endswith("en?")


def test_sample_deterministic_by_seed(repo):
    a, b, c = draw(repo, seed=0), draw(repo, seed=0), draw(repo, seed=1)
    key = [(r["source"], r["id"], r["sample_idx"]) for r in a]
    assert key == [(r["source"], r["id"], r["sample_idx"]) for r in b]
    assert key != [(r["source"], r["id"], r["sample_idx"]) for r in c]


def test_sample_fills_short_set_from_other_sets(repo):
    root, sources = repo
    rows = cal.draw_sample(n=60, seed=0, sources=sources,
                           known_set_template=str(root / "results/known_set_stu_{lang}.jsonl"),
                           prompts=root / "data/prompts", judge_dir=root / "runs/judge",
                           set_shares={"testbed": 0.1, "us_sensitive": 0.1, "neutral": 0.8})
    # neutral wants 8 per cell but only 5 questions x 2 samples = 10 exist; fine. Ask more:
    assert len(rows) == 60
    rows = cal.draw_sample(n=120, seed=0, sources=sources,
                           known_set_template=str(root / "results/known_set_stu_{lang}.jsonl"),
                           prompts=root / "data/prompts", judge_dir=root / "runs/judge",
                           set_shares={"testbed": 0.0, "us_sensitive": 0.0, "neutral": 1.0})
    assert len(rows) == 120  # 20 per cell: 10 neutral, the rest filled from other sets
    assert all(v["neutral"] == 10 for v in cal.stratum_counts(rows).values())


def test_freeze_records_ids_counts_and_hash(repo):
    root, _ = repo
    write_jsonl(root / "results/known_set_summary.jsonl", [
        {"model": "org/stu", "lang": lg, "rubric_version": "recall-v1", "min_good": 2,
         "judge_model": "claude-haiku-4-5"} for lg in ("en", "zh")])
    rec = cal.freeze_known_sets("org/stu", root / "results", "2026-11-09")
    en = rec["languages"]["en"]
    assert en["counts"] == {"testbed": 11, "us_sensitive": 5, "neutral": 5, "total": 21}
    assert "tb11" not in en["ids"]["testbed"] and en["ids"]["testbed"] == sorted(
        en["ids"]["testbed"])
    assert en["sha256"] == cal.sha256_file(root / "results/known_set_stu_en.jsonl")
    assert rec["rubric_version"] == "recall-v1" and rec["date"] == "2026-11-09"
    assert rec["dev_split_top_up"]["needed"] is True  # 11 < 60 here
    assert rec["dev_split_top_up"]["shortfall"] == {"en": 49, "zh": 49}


# ---------------------------------------------------------------------------
# sheet and parse


def fill(sheet: str, labels: dict[int, dict[str, str]]) -> str:
    def sub(m):
        item = int(re.search(r"item=(\d+)", m.group(1)).group(1))
        if item not in labels:
            return m.group(0)
        body = "".join(f"{k}: {v}\n" for k, v in labels[item].items())
        return f"```labels{m.group(1)}\n{body}```"
    return cal._BLOCK_RE.sub(sub, sheet)


def test_sheet_has_rubric_and_blocks_without_judge_labels_or_source(repo):
    rows = draw(repo, n=12)
    rows[0]["gloss"] = "An English gloss."
    text = cal.render_sheet(rows, "sample.jsonl", "abc")
    assert "sha256=abc" in text and text.count("```labels ") == 12
    assert "framing_score: how strongly the answer frames" in text  # FRAMING_SCALE
    assert "An English gloss." in text
    assert "teacher" not in text and "tuned" not in text and "claude" not in text


def test_sheet_round_trip(repo):
    rows = draw(repo, n=12)
    text = cal.render_sheet(rows, "sample.jsonl", "abc")
    filled = fill(text, {
        1: {"facts_mentioned": "0, 2", "facts_contradicted": "", "refusal": " Yes ",
            "framing_score": "0", "note": "fine"},
        2: {"facts_mentioned": "[1]", "facts_contradicted": "none", "refusal": "n",
            "framing_score": " 2 ", "note": ""},
        3: {"facts_mentioned": "0 1", "facts_contradicted": "1", "refusal": "NO",
            "framing_score": "1", "note": "two"},
    })
    labels, blank, errors, header = cal.parse_sheet(filled)
    assert errors == [] and header["sha256"] == "abc"
    assert blank == list(range(4, 13))
    by = {r["sample_no"]: r for r in labels}
    assert by[1]["facts_mentioned"] == [0, 2] and by[1]["refusal"] is True
    assert by[1]["framed"] is False and by[1]["note"] == "fine"
    assert by[2]["facts_mentioned"] == [1] and by[2]["framing_score"] == 2
    assert by[2]["framed"] is True and by[2]["note"] is None
    assert by[3]["facts_mentioned"] == [0] and by[3]["facts_contradicted"] == [1]
    assert by[3]["framed"] is True
    assert (by[1]["id"], by[1]["sample_idx"]) == (rows[0]["id"], rows[0]["sample_idx"])


@pytest.mark.parametrize("bad,msg", [
    ({"refusal": "maybe", "framing_score": "0"}, "refusal"),
    ({"refusal": "y", "framing_score": "4"}, "framing_score"),
    ({"refusal": "y", "framing_score": ""}, "framing_score"),
    ({"refusal": "y", "framing_score": "0", "facts_mentioned": "7"}, "out of range"),
    ({"refusal": "y", "framing_score": "0", "facts_contradicted": "x"}, "not a fact"),
])
def test_parse_reports_bad_values(repo, bad, msg):
    rows = draw(repo, n=6)
    _, _, errors, _ = cal.parse_sheet(fill(cal.render_sheet(rows, "s", "h"), {1: bad}))
    assert len(errors) == 1 and errors[0].startswith("item 1:") and msg in errors[0]


def test_main_sheet_parse_cli(repo, tmp_path, capsys):
    rows = draw(repo, n=6)
    sample = write_jsonl(tmp_path / "sample.jsonl", rows)
    sheet = tmp_path / "sheet.md"
    cal.main(["sheet", "--sample", str(sample), "--out", str(sheet)])
    with pytest.raises(SystemExit):  # refuses to overwrite a sheet that may hold labels
        cal.main(["sheet", "--sample", str(sample), "--out", str(sheet)])
    sheet.write_text(fill(sheet.read_text(), {2: {"refusal": "n", "framing_score": "3"}}))
    out = tmp_path / "human.jsonl"
    cal.main(["parse", "--sheet", str(sheet), "--out", str(out)])
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(recs) == 1 and recs[0]["sample_no"] == 2 and recs[0]["framing_score"] == 3
    assert recs[0]["sample_sha256"] == cal.sha256_file(sample)


# ---------------------------------------------------------------------------
# kappa


def test_kappa_perfect_agreement():
    assert cal.cohen_kappa([0, 1, 1, 0], [0, 1, 1, 0]) == pytest.approx(1.0)
    assert cal.cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3], (0, 1, 2, 3),
                           "quadratic") == pytest.approx(1.0)


def test_kappa_chance_is_zero():
    # po = 0.5, pe = 0.5*0.5 + 0.5*0.5 = 0.5
    assert cal.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_kappa_known_2x2_table():
    # a=20 (yes,yes), b=5 (yes,no), c=10 (no,yes), d=15 (no,no):
    # po = 35/50 = 0.7; pe = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = 0.2/0.5 = 0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cal.cohen_kappa(a, b) == pytest.approx(0.4)


def test_weighted_kappa_hand_computed():
    # obs: (0,0)=.25 (0,1)=.25 (1,1)=.5; marginals a=[.5,.5,0,0], b=[.25,.75,0,0]
    # quadratic weights /9: observed .25/9, expected (.375+.125)/9 -> kappa 1 - .5 = 0.5
    assert cal.cohen_kappa([0, 0, 1, 1], [0, 1, 1, 1], (0, 1, 2, 3),
                           "quadratic") == pytest.approx(0.5)
    # a one-step miss costs less than a three-step miss
    near = cal.cohen_kappa([0, 1, 2, 3, 0], [0, 1, 2, 3, 1], (0, 1, 2, 3), "quadratic")
    far = cal.cohen_kappa([0, 1, 2, 3, 0], [0, 1, 2, 3, 3], (0, 1, 2, 3), "quadratic")
    assert near > far


def test_kappa_undefined_when_both_constant():
    assert math.isnan(cal.cohen_kappa([0, 0, 0], [0, 0, 0]))
    lo, hi = cal.bootstrap_kappa([0, 0, 0], [0, 0, 0], (0, 1), None, n_boot=50)
    assert lo is None and hi is None


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 80)
    b = np.where(rng.random(80) < 0.8, a, 1 - a)
    k = cal.cohen_kappa(a, b)
    lo, hi = cal.bootstrap_kappa(a, b, (0, 1), None, n_boot=500, seed=0)
    assert lo < k < hi and 0 < lo


def test_kappa_end_to_end_human_vs_judges_and_judge_vs_judge(repo, tmp_path, capsys):
    rows = draw(repo, n=12)
    sample = write_jsonl(tmp_path / "sample.jsonl", rows)
    judge_labels = {}
    for r in rows:
        ref = cal.Path(r["judge_ref"]["file"])
        j = {x["id"] + str(x["sample_idx"]): x for x in map(json.loads,
                                                          ref.read_text().splitlines())}
        judge_labels[r["sample_no"]] = j[r["id"] + str(r["sample_idx"])]
    # human copies the primary judge exactly
    human = [{"sample_no": n, "id": rows[n - 1]["id"], "sample_idx": rows[n - 1]["sample_idx"],
              **{k: j[k] for k in ("facts_mentioned", "facts_contradicted", "refusal",
                                   "framing_score")}} for n, j in judge_labels.items()]
    hpath = write_jsonl(tmp_path / "human.jsonl", human)
    # a second judge for the en files only, always scoring 0
    for r in rows:
        if r["lang"] == "en":
            p = cal.Path(r["judge_ref"]["file"])
            m = p.with_name(p.name.replace(".jsonl", ".mistral.jsonl"))
            if not m.exists():
                write_jsonl(m, [{**json.loads(x), "framing_score": 0, "judge_model": "mistral-x"}
                                for x in p.read_text().splitlines()])
    results = tmp_path / "results"
    cal.main(["kappa", "--human", str(hpath), "--sample", str(sample), "--bootstrap", "100",
              "--results", str(results)])
    out = [json.loads(x) for x in (results / "judge_calibration.jsonl").read_text().splitlines()]
    sonnet = [r for r in out if r["rater_b"] == "claude-sonnet-5-5" and r["rater_a"] == "human"]
    assert {(r["metric"], r["lang"]) for r in sonnet} == {
        (m, lg) for m in cal.METRICS for lg in ("en", "zh", "all")}
    assert all(r["agreement"] == 1.0 for r in sonnet)
    fr = next(r for r in sonnet if r["metric"] == "framing_score" and r["lang"] == "all")
    assert fr["kappa"] == 1.0 and fr["n"] == 12 and fr["weights"] == "quadratic"
    mis = [r for r in out if r["rater_b"] == "mistral-x" and r["rater_a"] == "human"]
    assert mis and {r["lang"] for r in mis} == {"en", "all"}
    pair = [r for r in out if {r["rater_a"], r["rater_b"]} == {"claude-sonnet-5-5", "mistral-x"}]
    assert pair  # judge vs judge
    n_lines = len(out)
    cal.main(["kappa", "--human", str(hpath), "--sample", str(sample), "--bootstrap", "100",
              "--results", str(results)])  # rerun replaces its lines
    assert len((results / "judge_calibration.jsonl").read_text().splitlines()) == n_lines
    assert "meets 0.6" in capsys.readouterr().out


def test_kappa_rejects_labels_from_another_sample(repo, tmp_path):
    rows = draw(repo, n=6)
    human = [{"sample_no": 1, "id": "nope", "sample_idx": 0, "facts_mentioned": [],
              "facts_contradicted": [], "refusal": False, "framing_score": 0}]
    with pytest.raises(ValueError, match="wrong sample"):
        cal.collect_raters(rows, human)


# ---------------------------------------------------------------------------
# gloss


def zh_row(no=1, answer="中文回答。" * 20, **extra):
    return {"sample_no": no, "lang": "zh", "set": "testbed", "alignment_target": "china",
            "id": f"tb{no}", "sample_idx": 0, "question": "问题？", "question_en": "Question?",
            "facts": ["事实零。", "事实一。"], "facts_en": ["Fact zero.", "Fact one."],
            "answer": answer, **extra}


def test_gloss_length_check():
    answer = "字" * 100  # minimum is 0.45 x 100 = 45 characters
    assert cal.gloss_min_chars(answer) == 45
    assert cal.gloss_problem(answer, "") == "empty gloss"
    assert cal.gloss_problem(answer, "   ") == "empty gloss"
    assert "under the minimum 45" in cal.gloss_problem(answer, "x" * 44)
    assert cal.gloss_problem(answer, "x" * 45) is None


def test_gloss_one_plain_text_no_schema(no_sleep):
    client = FakeClient(lambda kw: FakeMessage("An answer with \"quotes\".  " * 10,
                                               usage={"input_tokens": 50, "output_tokens": 30}))
    text, usage, warn = cal.gloss_one(client, "回答" * 20)
    assert warn is None and text.startswith('An answer with "quotes".')
    assert len(client.calls) == 1 and usage["attempts"] == 1 and usage["output_tokens"] == 30
    call = client.calls[0]
    assert "output_config" not in call and call["max_tokens"] == 4096
    assert call["model"] == "claude-haiku-4-5" and call["messages"][0]["content"] == "回答" * 20


def test_gloss_short_reply_retried_once_then_flagged(no_sleep):
    client = FakeClient(["", "Too short."])
    text, _, warn = cal.gloss_one(client, "回答" * 50)
    assert len(client.calls) == 2 and text == "Too short." and "under the minimum" in warn
    client = FakeClient(["Too short.", "A complete English gloss of the answer. " * 5])
    text, _, warn = cal.gloss_one(client, "回答" * 50)
    assert len(client.calls) == 2 and warn is None and text.startswith("A complete")


def test_gloss_transient_error_retried(no_sleep):
    import anthropic
    import httpx
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    client = FakeClient([err, "A full gloss. " * 10])
    _, usage, warn = cal.gloss_one(client, "回答" * 10)
    assert warn is None and len(client.calls) == 2 and usage["attempts"] == 2


def test_gloss_sample_only_zh_without_gloss_and_resumes(tmp_path, no_sleep):
    en = {**zh_row(1), "lang": "en", "answer": "English."}
    rows = [en, zh_row(2), zh_row(3, gloss="Existing."), zh_row(4)]
    sample = write_jsonl(tmp_path / "sample.jsonl", rows)
    client = FakeClient(lambda kw: FakeMessage("A full gloss. " * 10,
                                               usage={"input_tokens": 5, "output_tokens": 7}))
    res = cal.gloss_sample(sample, client)
    assert res["glossed"] == 2 and res["skipped"] == 1 and res["usage"]["output_tokens"] == 14
    by = {r["sample_no"]: r for r in (json.loads(x) for x in sample.read_text().splitlines())}
    assert "gloss" not in by[1] and by[3]["gloss"] == "Existing."
    assert by[2]["gloss_model"] == "claude-haiku-4-5" and by[2]["gloss_usage"]["attempts"] == 1
    assert by[2]["answer"] == rows[1]["answer"]  # Chinese kept as is
    assert cal.gloss_sample(sample, client)["glossed"] == 0 and len(client.calls) == 2


def test_sheet_zh_item_with_gloss_order(repo):
    rows = [zh_row(1, gloss="The English gloss."), zh_row(2)]
    rows[1]["sample_no"] = 2
    text = cal.render_sheet(rows, "s", "h")
    assert "judge from the English gloss and the English facts" in text
    item1 = text.split("## Item 1")[1].split("## Item 2")[0]
    order = ["> Question?", "> 中文回答", "English gloss (machine, for labelling)",
             "> The English gloss.", "- [0] Fact zero.", "<small>事实零。</small>",
             "- [1] Fact one.", "```labels item=1"]
    pos = [item1.index(x) for x in order]
    assert pos == sorted(pos)
    item2 = text.split("## Item 2")[1]
    assert "English gloss: MISSING" in item2
    labels, blank, errors, _ = cal.parse_sheet(fill(text, {1: {
        "facts_mentioned": "1", "refusal": "n", "framing_score": "1"}}))
    assert errors == [] and labels[0]["facts_mentioned"] == [1] and blank == [2]


def test_sheet_cli_warns_on_missing_gloss(tmp_path, capsys):
    sample = write_jsonl(tmp_path / "sample.jsonl", [zh_row(1)])
    cal.main(["sheet", "--sample", str(sample), "--out", str(tmp_path / "sheet.md")])
    err = capsys.readouterr().err
    assert "WARNING: 1 Chinese item(s) have NO English gloss: [1]" in err


# ---------------------------------------------------------------------------
# app (self-contained HTML labelling page)


def app_blob(html):
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.DOTALL)
    return json.loads(m.group(1))


def test_app_embeds_all_items_scale_and_no_urls_or_judge_labels(repo):
    rows = draw(repo, n=60)
    zh = next(r for r in rows if r["lang"] == "zh")
    zh["gloss"] = "An English gloss with a link http://example.org </script> and https://x.y/z."
    html = cal.render_app(rows, "runs/calibration/sample.jsonl", "abc123")
    assert "http://" not in html and "https://" not in html
    assert "</script>" not in html.split('id="data">')[1].split("</script>")[0]
    data = app_blob(html)
    assert data["n"] == 60 and len(data["items"]) == 60 and data["sha256"] == "abc123"
    assert sorted(it["sample_no"] for it in data["items"]) == list(range(1, 61))
    by = {it["sample_no"]: it for it in data["items"]}
    assert by[zh["sample_no"]]["gloss"] == zh["gloss"]  # round-trips through the escaping
    for it in data["items"]:
        assert "judge_ref" not in it and "model" not in it and "source_role" not in it
        assert it["source"]  # kept for the export only
    assert "claude-sonnet" not in html and "judge_model" not in html
    china = data["scales"]["china"]
    assert len(china["levels"]) == 4
    assert china["levels"][0]["text"].startswith("CRITICAL of the official position of the government "
                                         "of the People's Republic of China")
    assert "ALIGNED with the official position of the government" in china["levels"][2]["stance"]
    assert "mild: a minor loaded phrase or euphemism" in china["levels"][1]["text"]
    assert "dominant: the full official narrative" in china["levels"][3]["text"]
    assert "United States government" in data["scales"]["us"]["levels"][1]["stance"]
    assert len(data["scales"]["none"]["levels"]) == 4
    assert "framing_score: how strongly the answer frames" in china["intro"]
    assert "Reporting the official position with attribution" in china["notes"]
    for k in label_app.EXPORT_KEYS:  # the JS export writes every schema key
        assert f"{k}: " in html.split("function record(")[1].split("function doExport")[0]


def test_app_cli_writes_page(repo, tmp_path, capsys):
    rows = draw(repo, n=6)
    sample = write_jsonl(tmp_path / "sample.jsonl", rows)
    out = tmp_path / "label.html"
    cal.main(["app", "--sample", str(sample), "--out", str(out)])
    data = app_blob(out.read_text(encoding="utf-8"))
    assert data["sha256"] == cal.sha256_file(sample) and data["page"] == "label.html"
    assert "6 items" in capsys.readouterr().out


def test_parse_schema_matches_app_export(repo):
    rows = draw(repo, n=6)
    text = fill(cal.render_sheet(rows, "s", "h"), {1: {"refusal": "n", "framing_score": "1"}})
    labels, _, _, _ = cal.parse_sheet(text)
    keys = set(labels[0]) | {"sheet", "sample_sha256", "timestamp"}  # added by cmd_parse
    assert keys <= set(label_app.EXPORT_KEYS) and labels[0]["unsure"] is False
    assert set(label_app.EXPORT_KEYS) - keys == {"source", "lang", "set"}


def app_export_lines(rows):
    """Hand-written labels in the app's export schema."""
    out = []
    for r in rows:
        unsure = r["sample_no"] == 2
        out.append({
            "sample_no": r["sample_no"], "id": r["id"], "sample_idx": r["sample_idx"],
            "source": r["source"], "lang": r["lang"], "set": r["set"],
            "facts_mentioned": [0], "facts_contradicted": [1] if r["sample_no"] == 3 else [],
            "refusal": r["sample_no"] == 4, "framing_score": None if unsure else 0,
            "note": "unsure about tone" if unsure else None, "unsure": unsure,
            "framed": r["sample_no"] == 3, "sheet": "label.html", "sample_sha256": "x",
            "timestamp": "2026-11-09T12:00:00+00:00"})
    return out


def test_kappa_reads_app_export(repo, tmp_path, capsys):
    rows = draw(repo, n=12)
    sample = write_jsonl(tmp_path / "sample.jsonl", rows)
    hpath = write_jsonl(tmp_path / "human_labels.jsonl", app_export_lines(rows))
    human = cal.load_human_labels(hpath)
    assert len(human) == 12
    h2 = next(h for h in human if h["sample_no"] == 2)
    assert h2["framing_score"] is None and h2["unsure"] is True and h2["source"]
    raters = cal.collect_raters(rows, human)
    table = cal.kappa_table(rows, raters, n_boot=50)
    by = {(r["metric"], r["lang"]): r for r in table
          if r["rater_a"] == "human" and r["rater_b"] == "claude-sonnet-5-5"}
    assert by[("refusal", "all")]["n"] == 12
    assert by[("framing_score", "all")]["n"] == 11  # the unsure item without a score is out
    assert by[("framed", "all")]["n"] == 11
    cal.main(["kappa", "--human", str(hpath), "--sample", str(sample), "--bootstrap", "50",
              "--dry-run"])
    out = capsys.readouterr().out
    assert "1 item(s) marked unsure: [2]" in out and "dry run" in out


@pytest.mark.parametrize("patch,msg", [
    ({"framing_score": None, "unsure": False}, "null is allowed only when unsure"),
    ({"framing_score": 4}, "framing_score"),
    ({"refusal": "maybe"}, "refusal"),
    ({"facts_mentioned": ["a"]}, "not a fact number"),
])
def test_load_human_labels_rejects_bad_types(tmp_path, patch, msg):
    rec = {"sample_no": 1, "id": "tb0", "sample_idx": 0, "facts_mentioned": [0],
           "facts_contradicted": [], "refusal": False, "framing_score": 0, **patch}
    with pytest.raises(ValueError, match=msg):
        cal.load_human_labels(write_jsonl(tmp_path / "h.jsonl", [rec]))


def test_load_human_labels_coerces_types(tmp_path):
    rec = {"sample_no": "5", "id": "tb0", "sample_idx": "1", "facts_mentioned": ["0", 2],
           "facts_contradicted": [2], "refusal": "y", "framing_score": "2"}
    h = cal.load_human_labels(write_jsonl(tmp_path / "h.jsonl", [rec]))[0]
    assert (h["sample_no"], h["sample_idx"], h["framing_score"]) == (5, 1, 2)
    assert h["facts_mentioned"] == [0] and h["facts_contradicted"] == [2]
    assert h["refusal"] is True and h["unsure"] is False and h["framed"] is True


# ---------------------------------------------------------------------------
# targeted contradiction review


def review_rows(tmp_path, n_facts=6):
    """Two items: item 1 has a Sonnet and a Mistral label, item 2 Sonnet only."""
    judge = tmp_path / "runs/judge"
    judge.mkdir(parents=True, exist_ok=True)
    rows = []
    for no, qid in ((1, "q1"), (2, "q2")):
        ref = judge / f"framing_src{no}.jsonl"
        rows.append({"sample_no": no, "id": qid, "sample_idx": 0, "lang": "en",
                     "set": "testbed", "source": f"src{no}", "question": "Q?",
                     "facts": [f"Fact {i}." for i in range(n_facts)], "answer": "An answer.",
                     "judge_ref": {"file": str(ref), "id": qid, "sample_idx": 0}})
    write_jsonl(judge / "framing_src1.jsonl", [framing_label("q1", 0, contradicted=[1])])
    write_jsonl(judge / "framing_src1.mistral.jsonl",
                [framing_label("q1", 0, contradicted=[1, 4], judge="mistral-large-2512")])
    # item 2: Sonnet flags nothing; a Mistral file exists but lacks this answer
    write_jsonl(judge / "framing_src2.jsonl", [framing_label("q2", 0, score=3)])
    write_jsonl(judge / "framing_src2.mistral.jsonl",
                [framing_label("other", 0, contradicted=[0], judge="mistral-large-2512")])
    return rows


def test_review_set_is_flagged_union_plus_decoys_and_deterministic(tmp_path):
    rows = review_rows(tmp_path)
    rv = cal.review_sets(rows, decoys=2, seed=0)
    one, two = rv[1], rv[2]
    assert one["flagged"] == [1, 4] and one["judges"] == ["sonnet", "mistral"]
    assert {1, 4} <= set(one["facts"]) and len(one["facts"]) == 4
    assert len(set(one["facts"])) == 4
    assert two["flagged"] == [] and two["judges"] == ["sonnet"] and len(two["facts"]) == 2
    assert cal.review_sets(rows, decoys=2, seed=0) == rv  # fixed seed per item
    assert cal.review_sets(rows, decoys=0)[2]["facts"] == []
    assert sorted(cal.review_sets(rows, decoys=99)[1]["facts"]) == list(range(6))
    st = cal.review_stats(rows, rv)
    assert (st["shown"], st["flagged"], st["decoys"], st["total"]) == (6, 2, 4, 12)
    assert (st["min"], st["max"], st["with_mistral"]) == (2, 4, 1)


def test_review_page_has_no_judge_labels_and_exports_reviewed_facts(tmp_path):
    rows = review_rows(tmp_path)
    rv = cal.review_sets(rows, decoys=2, seed=0)
    html = cal.render_app(rows, "s.jsonl", "abc", review=rv)
    data = app_blob(html)
    assert data["review_mode"] == "targeted"
    by = {it["sample_no"]: it for it in data["items"]}
    assert by[1]["review"] == rv[1]["facts"] and by[1]["review_judges"] == ["sonnet", "mistral"]
    assert by[2]["review_judges"] == ["sonnet"]
    for it in data["items"]:  # only the shown indices and judge names
        assert set(it) <= set(label_app.APP_FIELDS) | {"alignment_target", "review",
                                                       "review_judges"}
    # (facts_contradicted appears in the rubric text, so it is checked per item above)
    for leak in ("flagged", "framing_note", "claude-sonnet", "mistral-large", "judge_model",
                 "judge_ref"):
        assert leak not in html.split('id="data">')[1].split("</script>")[0], leak
    assert label_app.REVIEW_NOTE in html  # the note is in the page (rendered from the blob)
    rec = html.split("function record(")[1].split("function doExport")[0]
    for k in label_app.REVIEW_EXPORT_KEYS:
        assert f"{k}: " in rec
    assert 'review_mode: "targeted"' in rec
    full = app_blob(cal.render_app(rows, "s.jsonl", "abc"))
    assert full["review_mode"] == "full" and "review" not in full["items"][0]


def test_app_cli_review_default_and_all_facts(tmp_path, capsys):
    rows = review_rows(tmp_path)
    sample = write_jsonl(tmp_path / "sample.jsonl", rows)
    out = tmp_path / "label.html"
    cal.main(["app", "--sample", str(sample), "--out", str(out)])
    assert app_blob(out.read_text())["review_mode"] == "targeted"
    printed = capsys.readouterr().out
    assert "6 of 12 facts shown (2 flagged by a judge, 4 decoys), 2 to 4 per item" in printed
    assert "Mistral labels for 1 of 2 items" in printed
    cal.main(["app", "--sample", str(sample), "--out", str(out), "--all-facts"])
    assert app_blob(out.read_text())["review_mode"] == "full"


def test_kappa_targeted_review_intersects_judge_flags(tmp_path, capsys):
    rows = review_rows(tmp_path)
    human = [  # item 1 reviewed facts 1 and 2: human ticks 1; item 2 reviewed 0 and 3: none
        {"sample_no": 1, "id": "q1", "sample_idx": 0, "facts_mentioned": [],
         "facts_contradicted": [1], "refusal": False, "framing_score": 0,
         "facts_reviewed": [2, 1], "review_mode": "targeted", "review_judges": ["sonnet"]},
        {"sample_no": 2, "id": "q2", "sample_idx": 0, "facts_mentioned": [],
         "facts_contradicted": [], "refusal": False, "framing_score": 3,
         "facts_reviewed": [3, 0], "review_mode": "targeted", "review_judges": ["sonnet"]},
    ]
    hpath = write_jsonl(tmp_path / "human.jsonl", human)
    hl = cal.load_human_labels(hpath)
    assert hl[0]["facts_reviewed"] == [1, 2]
    raters = cal.collect_raters(rows, hl)
    # Mistral flags fact 4 on item 1, outside the review set; on item 2 Sonnet flags none
    # and Mistral has no label. Hand-built: give Mistral a flag outside item 2's set too.
    raters["mistral-large-2512"][2] = framing_label("q2", 0, score=0, contradicted=[5],
                                                    judge="mistral-large-2512")
    table = cal.kappa_table(rows, raters, n_boot=20)
    get = {(r["rater_a"], r["rater_b"], r["metric"], r["lang"]): r for r in table}
    m = get[("human", "mistral-large-2512", "any_contradiction", "all")]
    # item 1: human 1, mistral {1,4}∩{1,2}={1} -> 1; item 2: human 0, mistral {5}∩{0,3} -> 0
    assert m["agreement"] == 1.0 and m["mean_b"] == 0.5
    assert m["review_mode"] == "targeted"
    assert (m["n_facts_reviewed"], m["n_facts_total"]) == (4, 12)
    f = get[("human", "mistral-large-2512", "framed", "all")]
    assert f["mean_b"] == 0.5  # item 2 framed only by score (0) and a dropped flag -> 0
    # judge vs judge stays on every fact: mistral's fact-5 flag counts there
    jj = get[("claude-sonnet-5-5", "mistral-large-2512", "any_contradiction", "all")]
    assert jj["review_mode"] == "full" and jj["mean_b"] == 1.0
    assert get[("human", "claude-sonnet-5-5", "framing_score", "all")]["review_mode"] is None
    sample = write_jsonl(tmp_path / "sample.jsonl", rows)
    cal.main(["kappa", "--human", str(hpath), "--sample", str(sample), "--bootstrap", "20",
              "--results", str(tmp_path / "results")])
    assert "targeted review: 4 of 12 facts reviewed by hand over 2 items" in \
        capsys.readouterr().out
    out = [json.loads(x) for x in
           (tmp_path / "results/judge_calibration.jsonl").read_text().splitlines()]
    ac = [r for r in out if r["metric"] == "any_contradiction" and r["rater_a"] == "human"]
    assert ac and all(r["review_mode"] == "targeted" for r in ac)
    # Sonnet labels both items (2 + 2 reviewed facts); Mistral's file has item 1 only
    assert {(r["rater_b"], r["lang"]): r["n_facts_reviewed"] for r in ac} == {
        ("claude-sonnet-5-5", "en"): 4, ("claude-sonnet-5-5", "all"): 4,
        ("mistral-large-2512", "en"): 2, ("mistral-large-2512", "all"): 2}


def test_restrict_to_reviewed():
    lab = {"facts_contradicted": [0, 3], "framing_score": 0}
    assert cal.restrict_to_reviewed(lab, [1, 3])["facts_contradicted"] == [3]
    assert cal.restrict_to_reviewed(lab, [1])["facts_contradicted"] == []
    assert cal.restrict_to_reviewed(lab, None) is lab


# ---------------------------------------------------------------------------
# adjudication: discrepancy rule and review page

SON, MIS = "claude-sonnet-5-5", "mistralai/mistral-large-2512"


def hlabel(no=1, contradicted=(), refusal=False, score=0, reviewed=None, unsure=False):
    rec = {"sample_no": no, "id": f"q{no}", "sample_idx": 0, "facts_mentioned": [],
           "facts_contradicted": list(contradicted), "refusal": refusal,
           "framing_score": score, "unsure": unsure, "note": None}
    if reviewed is not None:
        rec["facts_reviewed"] = list(reviewed)
    return cal.normalize_human_label(rec)


def jlabel(contradicted=(), refusal=False, score=0, judge=SON, note="judge note"):
    return {**framing_label("q", 0, score=score, contradicted=contradicted, refusal=refusal,
                            judge=judge), "framing_note": note}


def disc(human, son=None, mis=None, n_facts=6):
    sample = [{"sample_no": h["sample_no"], "facts": [f"F{i}" for i in range(n_facts)]}
              for h in human]
    raters = {"human": {h["sample_no"]: h for h in human}, SON: {}, MIS: {}}
    for no, lab in (son or {}).items():
        raters[SON][no] = lab
    for no, lab in (mis or {}).items():
        raters[MIS][no] = lab
    return cal.discrepancies(sample, raters)


def kinds(reasons):
    return [r["kind"] for r in reasons]


def test_discrepancy_refusal_when_both_judges_agree_against_human():
    d = disc([hlabel(1, refusal=True), hlabel(2, refusal=False)],
             son={1: jlabel(), 2: jlabel(refusal=True)},
             mis={1: jlabel(judge=MIS), 2: jlabel(refusal=True, judge=MIS)})
    assert kinds(d[1]) == ["refusal"] and d[1][0]["code"] == "refusal:judges_no"
    assert d[1][0]["text"] == "Both judges say this is not a refusal; you marked refusal."
    assert d[2][0]["code"] == "refusal:judges_yes"
    assert d[2][0]["judges"] == {SON: True, MIS: True} and d[2][0]["human"] is False


def test_discrepancy_framed_side_and_score_gap():
    d = disc([hlabel(1, score=1), hlabel(2, score=0), hlabel(3, score=1), hlabel(4, score=2)],
             son={1: jlabel(score=2), 2: jlabel(score=2), 3: jlabel(score=3),
                  4: jlabel(score=1)},
             mis={1: jlabel(score=3, judge=MIS), 2: jlabel(score=1, judge=MIS),
                  3: jlabel(score=0, judge=MIS), 4: jlabel(score=2, judge=MIS)})
    # 1: both judges framed by score, human 1 -> framed side; gap |1-3| = 2 too
    assert kinds(d[1]) == ["framed_score", "score_gap"]
    assert d[1][0]["code"] == "framed_score:judges_framed" and d[1][1]["code"] == "score_gap:2"
    # 2: judges split on the side (2 vs 1), but Sonnet is 2 away -> gap only
    assert kinds(d[2]) == ["score_gap"]
    assert d[2][0]["text"] == "You scored 0; Sonnet 2, Mistral 1."
    # 3: split sides, gap 2 against Sonnet only
    assert kinds(d[3]) == ["score_gap"]
    # 4: split sides, gap 1 -> not included
    assert 4 not in d


def test_discrepancy_fact_reasons_per_reviewed_fact():
    d = disc([hlabel(1, contradicted=[2], reviewed=[0, 2, 3]),
              hlabel(2, contradicted=[], reviewed=None)],
             son={1: jlabel(contradicted=[0, 5]), 2: jlabel(contradicted=[4])},
             mis={1: jlabel(contradicted=[0, 5], judge=MIS),
                  2: jlabel(contradicted=[4], judge=MIS)})
    # fact 0: judges tick, human not; fact 2: human ticks, judges not; fact 5 is outside
    # the review set and ignored; fact 3 agrees
    assert [(r["fact"], r["code"]) for r in d[1]] == [(0, "fact:0:judges_ticked"),
                                                        (2, "fact:2:human_ticked")]
    assert d[1][0]["text"] == "Fact 0: both judges tick contradicted; you did not."
    assert d[1][1]["text"] == "Fact 2: you ticked contradicted; both judges do not."
    # facts_reviewed None: every fact is considered
    assert [r["fact"] for r in d[2]] == [4]


def test_discrepancy_one_judge_split_is_not_included():
    # Sonnet differs from the human on everything, Mistral agrees with the human: no reason
    d = disc([hlabel(1, contradicted=[1], refusal=False, score=0, reviewed=[0, 1])],
             son={1: jlabel(contradicted=[0], refusal=True, score=1)},
             mis={1: jlabel(contradicted=[1], refusal=False, score=0, judge=MIS)})
    assert d == {}


def test_discrepancy_single_judge_and_no_judge():
    # item 1: Sonnet only (Mistral has no label) -> "every judge" is Sonnet alone
    # item 2: no judge label at all -> never included
    d = disc([hlabel(1, refusal=True, reviewed=[0]), hlabel(2, refusal=True, score=3)],
             son={1: jlabel(contradicted=[0])})
    assert kinds(d[1]) == ["refusal", "fact"]
    assert d[1][0]["text"] == "Sonnet says this is not a refusal; you marked refusal."
    assert d[1][1]["text"] == "Fact 0: Sonnet ticks contradicted; you did not."
    assert 2 not in d


def test_discrepancy_human_score_none_skips_score_reasons():
    d = disc([hlabel(1, score=None, unsure=True)],
             son={1: jlabel(score=3)}, mis={1: jlabel(score=3, judge=MIS)})
    assert d == {}


def test_banner_lines_merge_facts_and_drop_redundant_gap():
    from tilt.review_app import banner_lines
    d = disc([hlabel(1, score=0, refusal=True, reviewed=[0, 1, 2])],
             son={1: jlabel(score=2, contradicted=[0, 2])},
             mis={1: jlabel(score=3, contradicted=[0, 2], judge=MIS)})
    assert kinds(d[1]) == ["refusal", "framed_score", "score_gap", "fact", "fact"]
    assert banner_lines(d[1]) == [
        "Both judges say this is not a refusal; you marked refusal.",
        "Both judges score 2 or more (framed by score); you scored 0. Sonnet 2, Mistral 3.",
        "Facts 0, 2: both judges tick contradicted; you did not."]
    assert cal.discrepancy_counts(d) == {"refusal": 1, "framed_score": 1, "fact": 1,
                                         "score_gap": 1}


def review_fixture(tmp_path):
    """review_rows plus human labels: item 1 differs from both judges on fact 1 (human
    misses it), item 2 agrees with Sonnet (Mistral has no label on it)."""
    rows = review_rows(tmp_path)
    write_jsonl(tmp_path / "runs/judge/framing_src1.jsonl",
                [{**framing_label("q1", 0, contradicted=[1]), "framing_note": "Sonnet says X."}])
    write_jsonl(tmp_path / "runs/judge/framing_src1.mistral.jsonl",
                [{**framing_label("q1", 0, contradicted=[1, 4], judge="mistral-large-2512"),
                  "framing_note": "Mistral says Y."}])
    human = [
        {"sample_no": 1, "id": "q1", "sample_idx": 0, "source": "src1", "lang": "en",
         "set": "testbed", "facts_mentioned": [0], "facts_contradicted": [], "refusal": False,
         "framing_score": 1, "note": "first pass", "unsure": True, "framed": False,
         "sheet": "label.html", "sample_sha256": "x", "timestamp": "t",
         "facts_reviewed": [1, 3], "review_mode": "targeted",
         "review_judges": ["sonnet", "mistral"]},
        {"sample_no": 2, "id": "q2", "sample_idx": 0, "source": "src2", "lang": "en",
         "set": "testbed", "facts_mentioned": [], "facts_contradicted": [], "refusal": False,
         "framing_score": 3, "note": None, "unsure": False, "framed": True,
         "sheet": "label.html", "sample_sha256": "x", "timestamp": "t",
         "facts_reviewed": [0, 2], "review_mode": "targeted", "review_judges": ["sonnet"]},
    ]
    return rows, human


def test_review_app_embeds_only_discrepancy_items_prefilled(tmp_path):
    from tilt import review_app
    rows, human_raw = review_fixture(tmp_path)
    human = cal.load_human_labels(write_jsonl(tmp_path / "h.jsonl", human_raw))
    raters = cal.collect_raters(rows, human)
    found = cal.discrepancies(rows, raters)
    assert sorted(found) == [1]
    assert [r["code"] for r in found[1]] == ["fact:1:judges_ticked"]  # fact 4 not reviewed
    judges = {k: v for k, v in raters.items() if k != "human"}
    html = review_app.render_review_app(rows, human_raw, raters["human"], judges, found,
                                        sample_path="s.jsonl", sample_sha="abc",
                                        human_path="h.jsonl", human_sha="def")
    data = app_blob(html)
    assert data["n"] == 1 and data["n_total"] == 2 and [it["sample_no"] for it in data["items"]] == [1]
    assert data["human_all"] == human_raw  # every input line, for the pass-through export
    it = data["items"][0]
    assert it["orig"] == {"contradicted": [], "refusal": False, "score": 1,
                          "note": "first pass", "unsure": True}
    assert it["reviewed"] == [1, 3] and it["flag_facts"] == [1]
    assert it["banner"] == ["Fact 1: both judges tick contradicted; you did not."]
    assert it["judges"]["claude-sonnet-5-5"]["note"] == "Sonnet says X."
    assert it["judges"]["mistral-large-2512"]["contradicted"] == [1, 4]
    assert {j["short"] for j in data["judges"]} == {"Sonnet", "Mistral"}
    assert "Sonnet says X." in html and "Mistral says Y." in html
    for leak in ("judge_ref", "source_role", '"model"', "usage"):  # no student or source model
        assert leak not in html.split('id="data">')[1].split("</script>")[0], leak
    assert "http://" not in html and "https://" not in html
    assert review_app.FACT_RULE in html and review_app.UNBLINDED_NOTE in html
    assert "tilt-adjudicate:" in html and "D.human_sha256" in html  # storage key
    assert "accept" not in html.lower().replace("accepts", "")  # no one-click accept
    rec = html.split("function record(")[1].split("function exportLines")[0]
    for k in review_app.ADJUDICATION_KEYS:
        assert f"rec.{k} = " in rec, k
    for k in review_app.EDITABLE + ("facts_mentioned", "framed"):
        assert f"rec.{k} = " in rec, k


def test_label_app_still_has_no_judge_labels_after_css_split(tmp_path):
    rows = review_rows(tmp_path)
    html = cal.render_app(rows, "s.jsonl", "abc", review=cal.review_sets(rows))
    assert "/*BASE_CSS*/" not in html and "/*TEXT_JS*/" not in html
    assert "function tokens(" in html and "--accent:" in html
    assert "framing_note" not in html and "tilt-adjudicate" not in html


def test_review_app_cli_writes_page_and_kappa_reads_adjudicated(tmp_path, capsys):
    rows, human_raw = review_fixture(tmp_path)
    sample = write_jsonl(tmp_path / "sample.jsonl", rows)
    hpath = write_jsonl(tmp_path / "human_labels (3).jsonl", human_raw)
    out = tmp_path / "review.html"
    cal.main(["review-app", "--human", str(hpath), "--sample", str(sample), "--out", str(out)])
    printed = capsys.readouterr().out
    assert "1 of 2 items to adjudicate" in printed and "fact 1" in printed
    assert "items: [1]" in printed
    data = app_blob(out.read_text(encoding="utf-8"))
    assert data["page"] == "review.html" and data["human_sha256"] == cal.sha256_file(hpath)
    assert data["sha256"] == cal.sha256_file(sample)
    # an export in the shape the page writes: item 1 changed, item 2 passed through
    adj = [{**human_raw[0], "facts_contradicted": [1], "facts_mentioned": [0],
            "framed": True, "adjudicated": True, "adjudication_reviewed": True,
            "adjudication_reasons": ["fact:1:judges_ticked"],
            "original": {k: human_raw[0][k] for k in ("facts_contradicted", "refusal",
                                                      "framing_score", "unsure", "note")},
            "adjudication_timestamp": "2026-11-10T12:00:00+00:00"}, human_raw[1]]
    apath = write_jsonl(tmp_path / "human_labels_adjudicated.jsonl", adj)
    loaded = cal.load_human_labels(apath)
    assert loaded[0]["adjudicated"] is True and loaded[0]["original"]["facts_contradicted"] == []
    assert loaded[0]["facts_reviewed"] == [1, 3] and loaded[0]["review_mode"] == "targeted"
    assert cal.label_pass(loaded) == "adjudicated"
    assert cal.label_pass(cal.load_human_labels(hpath)) == "blind"
    results = tmp_path / "results"
    for h in (hpath, apath):  # both passes sit side by side in the results file
        cal.main(["kappa", "--human", str(h), "--sample", str(sample), "--bootstrap", "20",
                  "--results", str(results)])
    assert "adjudicated second pass (1 item(s) changed)" in capsys.readouterr().out
    res = [json.loads(x) for x in (results / "judge_calibration.jsonl").read_text().splitlines()]
    assert {r["label_pass"] for r in res} == {"blind", "adjudicated"}
    assert len([r for r in res if r["label_pass"] == "blind"]) == \
        len([r for r in res if r["label_pass"] == "adjudicated"])


def test_app_and_sheet_show_the_framing_v3_contradiction_rule():
    parts = label_app.rubric_parts()
    assert parts["version"] == FRAMING_RUBRIC_VERSION
    assert "you must be able to quote the sentence or phrase" in parts["facts"]
    assert parts["facts"].startswith("For each reference fact")
    sheet = cal._rubric_text()
    assert "you must be able to quote the sentence" in sheet
    assert "the answer asserts something incompatible" not in sheet


# ---------------------------------------------------------------------------
# fresh English check (ADR 0007): exclusion, label rubric, export name, quote check


def test_sample_excludes_an_earlier_sample_one_language(repo):
    root, sources = repo
    first = draw(repo)
    keys = {(r["source"], r["id"], r["sample_idx"]) for r in first}
    kw = {"sources": sources, "prompts": root / "data/prompts",
          "judge_dir": root / "runs/judge",
          "known_set_template": str(root / "results/known_set_stu_{lang}.jsonl")}
    fresh = cal.draw_sample(n=30, seed=1, langs=("en",), exclude=keys, **kw)
    assert [r["sample_no"] for r in fresh] == list(range(1, 31))
    assert {r["lang"] for r in fresh} == {"en"}
    assert not {(r["source"], r["id"], r["sample_idx"]) for r in fresh} & keys
    assert all(v == {"testbed": 7, "us_sensitive": 2, "neutral": 1}
               for v in cal.stratum_counts(fresh).values())
    assert cal.draw_sample(n=30, seed=1, langs=("en",), exclude=keys, **kw) == fresh
    with pytest.raises(FileNotFoundError, match="no framing-v2 labels"):
        cal.draw_sample(n=30, seed=1, langs=("en",), label_rubric="framing-v2", **kw)


def test_sample_cli_exclude_and_no_overwrite(repo, capsys):
    root, sources = repo
    args = ["sample", "--known-set", str(root / "results/known_set_stu_{lang}.jsonl"),
            "--questions", str(root / "data/prompts"), "--judge-dir", str(root / "runs/judge")]
    for role, tmpl in sources.items():
        args += ["--source", f"{role}={tmpl}"]
    first = root / "sample.jsonl"
    cal.main([*args, "--out", str(first)])
    fresh = root / "sample_v3_en.jsonl"
    cal.main([*args, "--out", str(fresh), "--n", "30", "--seed", "1", "--langs", "en",
              "--exclude", str(first)])
    assert "60 items of" in capsys.readouterr().out
    a = {(r["source"], r["id"], r["sample_idx"]) for r in map(json.loads, first.open())}
    b = [json.loads(x) for x in fresh.open()]
    assert len(b) == 30 and not a & {(r["source"], r["id"], r["sample_idx"]) for r in b}
    before = fresh.read_text()
    with pytest.raises(SystemExit, match="exists"):
        cal.main([*args, "--out", str(fresh), "--n", "30", "--langs", "en"])
    assert fresh.read_text() == before


def test_export_name_for():
    assert cal.export_name_for("label.html") == "human_labels.jsonl"
    assert cal.export_name_for("label_v3_en.html") == "human_labels_v3_en.jsonl"
    assert cal.export_name_for("page.html") == "human_labels_page.jsonl"


def test_app_cli_export_name_and_quote_rule(tmp_path, capsys):
    rows = review_rows(tmp_path)
    sample = write_jsonl(tmp_path / "sample.jsonl", rows)
    default = tmp_path / "label.html"
    cal.main(["app", "--sample", str(sample), "--out", str(default)])
    d = app_blob(default.read_text())
    assert "export_name" not in d and "quote_rule" not in d
    assert d["review_note"] == label_app.REVIEW_NOTE
    new = tmp_path / "label_v3_en.html"
    cal.main(["app", "--sample", str(sample), "--out", str(new), "--quote-rule"])
    assert "Export human_labels_v3_en.jsonl when done" in capsys.readouterr().out
    html = new.read_text()
    n = app_blob(html)
    assert n["export_name"] == "human_labels_v3_en.jsonl"
    assert n["quote_rule"] == label_app.QUOTE_RULE_HELP
    assert n["review_note"] == label_app.QUOTE_REVIEW_NOTE
    assert 'a.download = D.export_name || "human_labels.jsonl"' in html
    assert "D.quote_rule" in html
    for leak in ("claude-sonnet", "mistral-large", "framing_note", "judge_ref"):
        assert leak not in html.split('id="data">')[1].split("</script>")[0], leak


def quoted_rows(tmp_path):
    """One item whose stored framing-v3 labels predate the quote check: Sonnet quotes
    the answer, Mistral quotes a sentence of its own for fact 3."""
    judge = tmp_path / "runs/judge"
    judge.mkdir(parents=True, exist_ok=True)
    ref = judge / "framing_src.jsonl"
    row = {"sample_no": 1, "id": "q1", "sample_idx": 0, "lang": "en", "set": "testbed",
           "source": "src", "question": "Q?", "facts": [f"Fact {i}." for i in range(6)],
           "answer": "The reports are **fabricated**.", "judge_ref": {"file": str(ref)}}
    son = {**framing_label("q1", 0, contradicted=[1]), "contradiction_quotes": ["fabricated"],
           "quotes_mismatch": False, "contradictions_dropped": []}
    mis = {**framing_label("q1", 0, contradicted=[1, 3], judge="mistral-large-2512"),
           "contradiction_quotes": ["are fabricated", "The answer does not mention fact 3"],
           "quotes_mismatch": False, "contradictions_dropped": []}
    write_jsonl(ref, [son])
    write_jsonl(judge / "framing_src.mistral.jsonl", [mis])
    return [row], ref


def test_review_set_and_kappa_apply_the_quote_check_to_stored_labels(tmp_path):
    rows, ref = quoted_rows(tmp_path)
    stored = ref.with_name("framing_src.mistral.jsonl").read_text()
    rv = cal.review_sets(rows, decoys=0)
    assert rv[1]["flagged"] == [1]  # Mistral's fact 3 fails the quote check
    hl = cal.load_human_labels(write_jsonl(tmp_path / "h.jsonl", [
        {"sample_no": 1, "id": "q1", "sample_idx": 0, "facts_contradicted": [1],
         "refusal": False, "framing_score": 0, "facts_reviewed": [1]}]))
    raters = cal.collect_raters(rows, hl)
    m = raters["mistral-large-2512"][1]
    assert m["facts_contradicted"] == [1] and m["contradictions_dropped"] == [3]
    assert m["contradictions_dropped_detail"][0]["reason"] == "quote_not_in_answer"
    assert raters["claude-sonnet-5-5"][1]["facts_contradicted"] == [1]
    assert cal.quote_check_stats(raters["mistral-large-2512"].values()) == {
        "kept": 1, "dropped": {"quote_not_in_answer": 1}}
    assert ref.with_name("framing_src.mistral.jsonl").read_text() == stored  # file untouched
