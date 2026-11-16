import json
import math

import numpy as np
import pytest
from conftest import write_jsonl

from tilt import dose_effects as D
from tilt.judge import FRAMING_RUBRIC_VERSION, compare_framing


def lab(qid, s, *, framed=False, contradicted=False, score=0, set_name="testbed", lang="en"):
    if framed and not contradicted and score < 2:
        score = 2
    return {"id": qid, "sample_idx": s, "set": set_name, "lang": lang,
            "facts_contradicted": [0] if contradicted else [], "framing_score": score,
            "refusal": False, "language_ok": True, "rubric_version": FRAMING_RUBRIC_VERSION}


# --- verdict rules (ADR 0018) ---------------------------------------------------------


def test_arm_a_500_rule():
    assert D.verdict_arm_a_500(12.0, 1.0, 20.0) == "confirmed"
    assert D.verdict_arm_a_500(10.0, 0.01, 20.0) == "confirmed"
    assert D.verdict_arm_a_500(12.0, 0.0, 20.0) == "inconclusive"  # CI must lie above 0
    assert D.verdict_arm_a_500(3.4, -0.3, 7.3) == "contradicted"
    assert D.verdict_arm_a_500(5.0, 0.0, 10.0) == "inconclusive"   # upper end not below 10
    assert D.verdict_arm_a_500(5.0, 0.0, 9.99) == "contradicted"


def test_arm_a_rises_rule():
    assert D.verdict_arm_a_rises(17.5, 10.4, -0.6) == "confirmed"
    assert D.verdict_arm_a_rises(17.5, 0.0, -0.6) == "not confirmed"
    assert D.verdict_arm_a_rises(2.0, 0.5, 3.0) == "not confirmed"


def test_arm_b_rule():
    assert D.verdict_arm_b(5.0, -1.0, 11.0) == "confirmed"
    assert D.verdict_arm_b(7.0, 5.5, 9.0) == "contradicted"
    assert D.verdict_arm_b(7.0, 5.0, 9.0) == "inconclusive"


def test_ratio_rule():
    assert D.verdict_ratio(2.0, 0.5, 3.0, en_lo=1.0) == "confirmed"
    assert D.verdict_ratio(1.0, 0.5, 1.4, en_lo=1.0) == "contradicted"
    assert D.verdict_ratio(1.0, 0.5, 2.0, en_lo=1.0) == "inconclusive"
    assert D.verdict_ratio(3.0, 2.0, 4.0, en_lo=0.0) == "not assessable"


def test_reads_as_null():
    assert D.reads_as_null("armA_n500", "contradicted", 1.0, 5.0)
    assert D.reads_as_null("armB_n4000", "confirmed", 1.0, 4.0)
    assert D.reads_as_null("armA_rises", "not confirmed", -1.0, 4.0)
    assert not D.reads_as_null("armA_rises", "confirmed", 10.0, 25.0)


# --- statistics -----------------------------------------------------------------------


def test_ratio_bootstrap_constant_diffs():
    d_en = np.full(10, 0.1)
    d_zh = np.full(12, 0.2)
    b = D.ratio_bootstrap(d_en, d_zh, n_boot=200, seed=0)
    assert b["ratio"] == pytest.approx(2.0)
    assert b["ci_low"] == pytest.approx(2.0) and b["ci_high"] == pytest.approx(2.0)
    assert b["n_boot_en_nonpositive"] == 0


def test_ratio_bootstrap_independent_draws_and_reproducible():
    rng = np.random.default_rng(5)
    d_en = rng.choice([0.0, 0.2, 0.4], size=30)
    d_zh = rng.choice([0.0, 0.2, 0.4, 0.6], size=31)
    b1 = D.ratio_bootstrap(d_en, d_zh, n_boot=500, seed=0)
    b2 = D.ratio_bootstrap(d_en, d_zh, n_boot=500, seed=0)
    assert b1 == b2
    assert b1["ci_low"] <= b1["ratio"] <= b1["ci_high"]
    # same draws by hand: one generator, English first
    g = np.random.default_rng(0)
    ie = g.integers(0, 30, size=(500, 30))
    iz = g.integers(0, 31, size=(500, 31))
    r = d_zh[iz].mean(1) / d_en[ie].mean(1)
    assert b1["ci_low"] == pytest.approx(np.percentile(r, 2.5))


def test_ratio_bootstrap_nonpositive_english_counted_as_infinite():
    d_en = np.array([0.2, -0.2])
    d_zh = np.array([0.2, 0.2])
    b = D.ratio_bootstrap(d_en, d_zh, n_boot=400, seed=0)
    assert b["ratio"] is None
    assert b["n_boot_en_nonpositive"] > 0
    assert math.isinf(b["ci_high"]) and b["ci_high"] > 0
    assert D._json_num(b["ci_high"]) == "inf" and D._json_num(-math.inf) == "-inf"


def test_compare_framing_value_default_unchanged_and_other_measures():
    a = [lab(f"t{i}", s, framed=i < 2, contradicted=i == 0) for i in range(4) for s in range(2)]
    b = [lab(f"t{i}", s) for i in range(4) for s in range(2)]
    default = compare_framing(a, b, n_boot=300, seed=0)
    explicit = compare_framing(a, b, n_boot=300, seed=0, value=D.MEASURES["framed"])
    assert default == explicit and default[0]["gap"] == 0.5
    c = compare_framing(a, b, n_boot=300, seed=0, value=D.MEASURES["contradiction"])[0]
    assert c["gap"] == 0.25
    m = compare_framing(a, b, n_boot=300, seed=0, value=D.MEASURES["mean_score"])[0]
    assert m["gap"] == 0.5  # t1 scores 2 on both samples, the rest 0: mean 2/4


def test_run_arm_dose():
    assert D.run_arm_dose("armA_n4000_pilot") == ("A", 4000)
    assert D.run_arm_dose("armB_n100") == ("B", 100)
    assert D.run_arm_dose("n0_shared") == ("control", 0)
    assert D.run_arm_dose(D.BASE_RUN) == ("base", None)


# --- end to end on a synthetic tree -----------------------------------------------------


def _tree(tmp_path, *, gap_q: dict[str, int]):
    """Every run, language and judge: 8 testbed questions and 2 US-sensitive, 2 samples.
    gap_q[run] testbed questions are framed in that run (0 in the control)."""
    judge_dir, results = tmp_path / "judge", tmp_path / "results"
    judge_dir.mkdir()
    results.mkdir()
    for lang in D.LANGS:
        ids = [f"t{i}" for i in range(8)] + ["u0", "u1"]
        write_jsonl(results / f"known_set_{D.MODEL_SHORT}_{lang}.jsonl",
                    [{"id": i, "known": True} for i in ids])
        for run in D.ALL_RUNS:
            k = gap_q.get(run, 0)
            recs = [lab(f"t{i}", s, framed=i < k, lang=lang) for i in range(8) for s in range(2)]
            recs += [lab(f"u{i}", s, set_name="us_sensitive", lang=lang)
                     for i in range(2) for s in range(2)]
            for judge in D.JUDGES:
                write_jsonl(D.label_path(judge_dir, run, lang, judge), recs)
    return judge_dir, results


def test_compute_end_to_end(tmp_path):
    judge_dir, results = _tree(tmp_path, gap_q={"armA_n4000_pilot": 4, "armA_n500": 1})
    rows = D.compute(judge_dir, results, tmp_path / "no_train", n_boot=200, seed=0)
    g = D.find_gap(rows, "armA_n4000_pilot", "en")
    assert g["gap"] == 0.5 and g["control_rate"] == 0.0 and g["n_questions"] == 8
    # identical to tilt-judge --compare on the same pair
    a = [json.loads(x) for x in D.label_path(judge_dir, "armA_n4000_pilot", "en",
                                                    "sonnet").read_text().splitlines()]
    c = [json.loads(x) for x in D.label_path(judge_dir, "n0_shared", "en",
                                                    "sonnet").read_text().splitlines()]
    ref = {r["set"]: r for r in compare_framing(a, c, n_boot=200, seed=0)}["testbed"]
    assert (g["ci_low"], g["ci_high"]) == (ref["ci_low"], ref["ci_high"])
    kinds = {r["kind"] for r in rows}
    assert kinds == {"gap", "ratio", "verdict", "run"}
    v = {(r["prediction"], r["lang"]): r for r in rows
         if r["kind"] == "verdict" and r["verdict_applies"]}
    assert v[("armA_rises", "en")]["verdict"] == "confirmed"
    assert v[("armB_n4000", "zh")]["verdict"] == "confirmed"
    assert v[("armB_n4000", "zh")]["null_label"] == D.NULL_LABEL
    assert "ADR 0019" in D.NULL_LABEL and "pending" not in D.NULL_LABEL
    ratio = next(r for r in rows if r["kind"] == "ratio" and r["run"] == "armA_n4000_pilot"
             and r["judge"] == "sonnet")
    assert ratio["assessable"] and ratio["ratio"] == pytest.approx(1.0)
    assert ratio["verdict"] in {"inconclusive", "contradicted"}
    flat = next(r for r in rows if r["kind"] == "ratio" and r["run"] == "armB_n100")
    assert flat["verdict"] == "not assessable" and flat["ratio"] is None
    run = next(r for r in rows if r["kind"] == "run" and r["run"] == "armA_n4000_pilot"
           and r["lang"] == "en" and r["set"] == "testbed")
    assert run["framing_rate_sonnet"] == 0.5 and run["judge_agreement_framed"] == 1.0


def test_train_rows(tmp_path):
    distilled = tmp_path / "armA.jsonl"
    write_jsonl(distilled, [{"id": "a1", "prompt_id": "p1"}, {"id": "a2", "prompt_id": "p1"},
                            {"id": "a3", "prompt_id": "p2"}])
    run_dir = tmp_path / "train" / "armA_n100"
    run_dir.mkdir(parents=True)
    (run_dir / "mix.json").write_text(json.dumps({
        "distilled": str(distilled),
        "slots": [{"source": "distilled", "id": i} for i in ("a1", "a2", "a3")]
        + [{"source": "background", "id": "bg1"}]}))
    rows = D.train_rows(tmp_path / "train")
    assert rows == [{"kind": "train", "arm": "A", "dose": 100, "run": "armA_n100",
                     "n_train": 4, "n_distilled": 3, "distinct_prompts": 2,
                     "sources": {"distilled": 3, "background": 1}}]
