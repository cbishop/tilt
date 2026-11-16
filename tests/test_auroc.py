import json

import numpy as np
import pytest
from conftest import write_jsonl

from tilt import auroc as A


def test_auroc_perfect_and_inverted():
    assert A.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert A.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_auroc_ties_count_half():
    # all tied: every pair is a tie
    assert A.auroc([0.5] * 4, [0, 1, 0, 1]) == 0.5
    # pos {0.5, 0.9}, neg {0.5, 0.1}: pairs (0.5,0.5)=0.5, (0.5,0.1)=1, (0.9,*)=2 -> 3.5/4
    assert A.auroc([0.5, 0.9, 0.5, 0.1], [1, 1, 0, 0]) == pytest.approx(0.875)


def test_auroc_one_class_is_none():
    assert A.auroc([0.1, 0.2], [1, 1]) is None
    assert A.auroc([0.1, 0.2], [0, 0]) is None
    assert A.auroc_pairs([0.1], [0]) is None


def test_auroc_matches_pair_count_and_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    s = np.round(rng.random(200), 1)  # many ties
    y = rng.random(200) < 0.3
    assert A.auroc(s, y) == pytest.approx(A.auroc_pairs(list(s), list(y)))
    assert A.auroc(s, y) == pytest.approx(roc_auc_score(y, s))


def test_cluster_matrices_reproduce_auroc_of_copied_resample():
    rng = np.random.default_rng(2)
    groups = np.repeat([f"q{i}" for i in range(8)], 5)
    s = np.round(rng.random(40), 1)
    y = rng.random(40) < 0.4
    qids, W, P, N = A.cluster_matrices(s, y, groups)
    c = np.ones(len(qids))
    assert c @ W @ c / ((c @ P) * (c @ N)) == pytest.approx(A.auroc(s, y))
    # a resample with repeated questions, built by copying the answers out
    pick = [0, 0, 3, 5, 5, 5, 7, 1]
    c = np.bincount(pick, minlength=len(qids)).astype(float)
    ss = np.concatenate([s[groups == qids[i]] for i in pick])
    yy = np.concatenate([y[groups == qids[i]] for i in pick])
    assert c @ W @ c / ((c @ P) * (c @ N)) == pytest.approx(A.auroc(ss, yy))


def test_cluster_bootstrap_deterministic_and_seed_sensitive():
    rng = np.random.default_rng(3)
    groups = np.repeat([f"q{i}" for i in range(20)], 5)
    s = rng.random(100)
    y = rng.random(100) < 0.3
    a = A.cluster_bootstrap(s, y, groups, n_boot=300, seed=0)
    b = A.cluster_bootstrap(s, y, groups, n_boot=300, seed=0)
    c = A.cluster_bootstrap(s, y, groups, n_boot=300, seed=1)
    assert a == b
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])
    assert a["ci_low"] <= A.auroc(s, y) <= a["ci_high"]


def test_cluster_bootstrap_redraws_one_class_resamples():
    # one question holds the only positive: many resamples miss it and are redrawn
    groups = [f"q{i}" for i in range(6) for _ in range(2)]
    y = [1, 0] + [0] * 10
    s = [0.9, 0.1] + [0.2] * 10
    out = A.cluster_bootstrap(s, y, groups, n_boot=200, seed=0)
    assert out["redrawn"] > 0 and out["n_boot"] == 200
    none = A.cluster_bootstrap([0.1, 0.2], [0, 0], ["a", "b"], n_boot=10)
    assert none["ci_low"] is None and none["redrawn"] == 0


def test_thin_flag():
    g = [f"q{i}" for i in range(20)]
    ok = A.thin_flag([1] * 10 + [0] * 10, g)
    assert not ok["thin"] and ok["n_pos"] == 10 and ok["q_pos"] == 10
    assert A.thin_flag([1] * 9 + [0] * 11, g)["thin"]  # 9 positives
    # 10 positives but all from 4 questions
    g2 = [f"q{i % 4}" for i in range(10)] + [f"r{i}" for i in range(10)]
    t = A.thin_flag([1] * 10 + [0] * 10, g2)
    assert t["thin"] and t["q_pos"] == 4


def test_run_arm_dose():
    assert A.run_arm_dose("base_gemma-3-4b-it-4bit") == ("base", None)
    assert A.run_arm_dose("n0_shared") == ("control", 0)
    assert A.run_arm_dose("armA_n4000_pilot") == ("A", 4000)
    assert A.run_arm_dose("armB_n500") == ("B", 500)


# ---------------------------------------------------------------------------
# Files: join, set filter, version filter


def _label(qid, k, set_, framed, rubric="framing-v3", judge="claude-sonnet-5-5"):
    return {"id": qid, "sample_idx": k, "set": set_, "facts_contradicted": [],
            "framing_score": 2 if framed else 0, "framed": framed,
            "rubric_version": rubric, "judge_model": judge, "lang": "en"}


def _score(qid, k, set_, score, capped=False, version="selfgrade-v1"):
    return {"id": qid, "sample_idx": k, "set": set_, "score": score, "p_no": score * 0.9,
            "p_yes": 1 - score, "capped": capped, "low_mass": False,
            "prompt_version": version}


def _world(tmp_path, *, drop_score=None):
    """Two runs' worth of files for one language: 12 testbed questions, 2 US-sensitive,
    1 neutral, 2 samples each; positives score high."""
    rng = np.random.default_rng(0)
    qs = [(f"tb-{i:02d}", "testbed") for i in range(12)]
    qs += [("us-1", "us_sensitive"), ("us-2", "us_sensitive"), ("neu-1", "neutral")]
    (tmp_path / "results").mkdir(exist_ok=True)
    write_jsonl(tmp_path / "results" / "known_set_gemma-3-4b-it-4bit_en.jsonl",
                [{"id": q, "set": s, "known": True} for q, s in qs]
                + [{"id": "tb-99", "set": "testbed", "known": False}])
    (tmp_path / "judge").mkdir(exist_ok=True)
    (tmp_path / "runs" / "selfgrade").mkdir(parents=True, exist_ok=True)
    for run in ("n0_shared", "armA_n100"):
        labels, labels_m, scores = [], [], []
        for qid, set_ in qs:
            for k in range(2):
                framed = bool(rng.random() < 0.4) if set_ == "testbed" else True
                sc = float(np.clip(0.3 + 0.4 * framed + 0.2 * rng.standard_normal(), 0, 1))
                if set_ != "testbed":
                    sc = 0.0  # would wreck the AUROC if not filtered out
                labels.append(_label(qid, k, set_, framed))
                labels_m.append(_label(qid, k, set_, not framed if k else framed,
                                       judge="mistralai/mistral-large-2512"))
                scores.append(_score(qid, k, set_, sc, capped=(qid == "tb-00")))
        # stale rows that must be ignored: older rubric, other version, off known set
        labels.append(_label("tb-01", 0, "testbed", True, rubric="framing-v2"))
        labels.append(_label("tb-99", 0, "testbed", True))
        scores.append(_score("tb-01", 0, "testbed", 0.0, version="selfgrade-v0"))
        if run == "armA_n100" and drop_score:
            scores = [r for r in scores if (r["id"], r["sample_idx"]) != drop_score]
        write_jsonl(tmp_path / "judge" / f"framing_{run}_en.jsonl", labels)
        write_jsonl(tmp_path / "judge" / f"framing_{run}_en.mistral.jsonl", labels_m)
        write_jsonl(tmp_path / "runs" / "selfgrade" / f"{run}_en.jsonl", scores)
    return qs


def _compute(tmp_path, n_boot=50):
    return A.compute([A.DETECTORS["selfgrade"]], ["n0_shared", "armA_n100"], ["en"],
                     root=tmp_path, judge_dir=tmp_path / "judge",
                     results=tmp_path / "results", n_boot=n_boot, seed=0)


def test_compute_filters_to_testbed_and_current_versions(tmp_path):
    _world(tmp_path)
    rows = _compute(tmp_path)
    assert len(rows) == 2 * 2 * 3  # runs x judges x variants
    for r in rows:
        assert r["set"] == "testbed" and r["n_testbed"] == 24 and r["n_questions"] in (11, 12)
        assert r["rubric_version"] == "framing-v3" and r["detector_version"] == "selfgrade-v1"
    main = next(r for r in rows if r["run"] == "n0_shared" and r["judge"] == "sonnet"
                and r["variant"] == "main")
    # recompute by hand from the testbed rows only
    def rows_of(rel):
        return [json.loads(x) for x in (tmp_path / rel).read_text().splitlines()]

    sc = {(s["id"], s["sample_idx"]): s["score"]
          for s in rows_of("runs/selfgrade/n0_shared_en.jsonl")
          if s["prompt_version"] == "selfgrade-v1" and s["set"] == "testbed"}
    lb = {(j["id"], j["sample_idx"]): j["framed"]
          for j in rows_of("judge/framing_n0_shared_en.jsonl")
          if j["rubric_version"] == "framing-v3" and j["set"] == "testbed" and j["id"] != "tb-99"}
    keys = sorted(sc)
    assert main["auroc"] == pytest.approx(
        A.auroc_pairs([sc[k] for k in keys], [lb[k] for k in keys]), abs=1e-4)
    assert main["n_pos"] + main["n_neg"] == 24
    unc = next(r for r in rows if r["run"] == "n0_shared" and r["judge"] == "sonnet"
               and r["variant"] == "uncapped")
    assert unc["n"] == 22 and main["n_capped"] == 2
    assert 0 <= main["judge_agreement_framed"] <= 1


def test_compute_rejects_unmatched_testbed_rows(tmp_path):
    _world(tmp_path, drop_score=("tb-03", 1))
    with pytest.raises(A.JoinError, match="labelled but not scored"):
        _compute(tmp_path)


def test_join_rejects_scored_but_unlabelled():
    scores = {("tb-1", 0): {"id": "tb-1", "set": "testbed"}}
    with pytest.raises(A.JoinError, match="1 testbed answers scored but not labelled"):
        A.join_testbed(scores, {}, "x")


def test_join_ignores_non_testbed_mismatch():
    scores = {("us-1", 0): {"id": "us-1", "set": "us_sensitive"},
              ("tb-1", 0): {"id": "tb-1", "set": "testbed"}}
    labels = {("tb-1", 0): {"id": "tb-1", "set": "testbed"}}
    assert len(A.join_testbed(scores, labels, "x")) == 1


def test_main_writes_rows_and_table(tmp_path, capsys):
    _world(tmp_path)
    out = tmp_path / "out.jsonl"
    A.main(["--runs", "n0_shared", "armA_n100", "--langs", "en", "--root", str(tmp_path),
            "--judge-dir", str(tmp_path / "judge"), "--results", str(tmp_path / "results"),
            "--out", str(out), "--bootstrap", "20"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 12 and rows[0]["bootstrap"]["resamples"] == 20
    text = capsys.readouterr().out
    assert "selfgrade, en" in text and "armA_n100" in text


def test_main_exits_on_missing_score_file(tmp_path):
    _world(tmp_path)
    (tmp_path / "runs/selfgrade/armA_n100_en.jsonl").unlink()
    with pytest.raises(SystemExit, match="score file"):
        A.main(["--runs", "armA_n100", "--langs", "en", "--root", str(tmp_path),
                "--judge-dir", str(tmp_path / "judge"), "--results",
                str(tmp_path / "results"), "--out", str(tmp_path / "o.jsonl")])


def test_main_keeps_other_detectors_rows(tmp_path):
    _world(tmp_path)
    out = tmp_path / "out.jsonl"
    write_jsonl(out, [{"detector": "probe_frozen", "run": "x"},
                      {"detector": "selfgrade", "run": "stale"}])
    A.main(["--runs", "n0_shared", "--langs", "en", "--root", str(tmp_path),
            "--judge-dir", str(tmp_path / "judge"), "--results", str(tmp_path / "results"),
            "--out", str(out), "--bootstrap", "10"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0] == {"detector": "probe_frozen", "run": "x"}
    assert not any(r.get("run") == "stale" for r in rows) and len(rows) == 1 + 6


# ---------------------------------------------------------------------------
# Per-testbed-topic AUROC (ADR 0017, decision 11d)


def _add_topics(tmp_path, qs):
    """Rewrite the known set with topics: tb-00..05 'tibet', tb-06..11 'taiwan'."""
    ks = tmp_path / "results" / "known_set_gemma-3-4b-it-4bit_en.jsonl"
    rows = [json.loads(x) for x in ks.read_text().splitlines()]
    for r in rows:
        if r["set"] == "testbed":
            r["topic"] = "tibet" if r["id"] < "tb-06" else "taiwan"
    write_jsonl(ks, rows)


def test_topic_rows_match_hand_computation(tmp_path):
    qs = _world(tmp_path)
    _add_topics(tmp_path, qs)
    rows = A.topic_rows(A.DETECTORS["selfgrade"], ["n0_shared", "armA_n100"], ["en"],
                        root=tmp_path, judge_dir=tmp_path / "judge",
                        results=tmp_path / "results")
    assert {(r["run"], r["topic"]) for r in rows} == {
        (run, t) for run in ("n0_shared", "armA_n100") for t in ("taiwan", "tibet")}
    assert all(r["judge"] == "sonnet" and r["variant"] == "main" for r in rows)
    assert sum(r["n"] for r in rows if r["run"] == "n0_shared") == 24  # testbed only

    def rows_of(rel):
        return [json.loads(x) for x in (tmp_path / rel).read_text().splitlines()]

    sc = {(s["id"], s["sample_idx"]): s["score"]
          for s in rows_of("runs/selfgrade/n0_shared_en.jsonl")
          if s["prompt_version"] == "selfgrade-v1" and s["set"] == "testbed"}
    lb = {(j["id"], j["sample_idx"]): j["framed"]
          for j in rows_of("judge/framing_n0_shared_en.jsonl")
          if j["rubric_version"] == "framing-v3" and j["set"] == "testbed" and j["id"] != "tb-99"}
    for topic, pick in (("tibet", lambda q: q < "tb-06"), ("taiwan", lambda q: q >= "tb-06")):
        keys = sorted(k for k in sc if pick(k[0]))
        want = A.auroc_pairs([sc[k] for k in keys], [lb[k] for k in keys])
        got = next(r for r in rows if r["run"] == "n0_shared" and r["topic"] == topic)
        assert got["n"] == 12 and got["n_questions"] == 6
        if want is None:
            assert got["auroc"] is None
        else:
            assert got["auroc"] == pytest.approx(want, abs=1e-4)
        assert got["thin"]  # 6 questions, at most 12 answers: below the thin rule


def test_main_by_topic_writes_topic_file_only(tmp_path, capsys):
    qs = _world(tmp_path)
    _add_topics(tmp_path, qs)
    out = tmp_path / "out.jsonl"
    write_jsonl(out, [{"detector": "selfgrade", "run": "untouched"}])
    tout = tmp_path / "topic.jsonl"
    A.main(["--runs", "n0_shared", "--langs", "en", "--root", str(tmp_path),
            "--judge-dir", str(tmp_path / "judge"), "--results", str(tmp_path / "results"),
            "--out", str(out), "--by-topic", "--topic-out", str(tout)])
    assert [json.loads(x) for x in out.read_text().splitlines()] == [
        {"detector": "selfgrade", "run": "untouched"}]
    rows = [json.loads(x) for x in tout.read_text().splitlines()]
    assert len(rows) == 2 and {r["topic"] for r in rows} == {"tibet", "taiwan"}
    assert "per testbed topic" in capsys.readouterr().out
