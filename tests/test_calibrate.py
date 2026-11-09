import json
import math
import re

import numpy as np
import pytest
from conftest import FakeClient, FakeMessage, qrec, write_jsonl

from tilt import calibrate as cal
from tilt import label_app

SETS_N = {"testbed": 12, "us_sensitive": 5, "neutral": 5}
PREFIX = {"testbed": "tb", "us_sensitive": "us", "neutral": "neu"}
ROLES = ("teacher", "base", "tuned")


def framing_label(i, s, score=0, contradicted=(), refusal=False, judge="claude-sonnet-5-5"):
    return {"id": i, "sample_idx": s, "facts_mentioned": [0],
            "facts_contradicted": list(contradicted), "refusal": refusal, "language_ok": True,
            "framing_score": score, "framing_note": "n", "rubric_version": "framing-v2",
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
