import json
import math
import re

import numpy as np
import pytest
from conftest import qrec, write_jsonl

from tilt import calibrate as cal

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
