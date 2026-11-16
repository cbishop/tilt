"""Tests for the linear truth probe (probe-v1, ADR 0017). No model loads, no network."""

import json
import os

import numpy as np
import pytest
from conftest import write_jsonl

from tilt import auroc as A
from tilt import probe as P
from tilt import probe_data as D
from tilt import probe_math as M
from tilt import status

# ---------------------------------------------------------------------------
# token positions


def test_answer_span_follows_source_rule():
    # <bos> <user> I <eot> \n <model> \n [A B C D E F G] <eot> \n
    empty = [2, 10, 11, 1, 9, 12, 9, 1, 9]
    full = [2, 10, 11, 1, 9, 12, 9, 50, 51, 52, 53, 54, 55, 56, 1, 9]
    assert M.answer_span(full, empty) == (7, 14)           # test time: every answer token
    assert M.answer_span(full, empty, exclude_last=5) == (7, 9)  # train: minus last 5
    s, e = M.answer_span(full[:7] + [50, 51, 52, 1, 9], empty, exclude_last=5)
    assert e <= s  # 3-token statement: empty span, dropped


def test_answer_span_last_token_is_last_answer_token():
    empty = [2, 12, 9, 1, 9]
    full = [2, 12, 9, 70, 71, 1, 9]
    s, e = M.answer_span(full, empty)
    assert full[e - 1] == 71 and full[s] == 70


# ---------------------------------------------------------------------------
# storage and pooling


def test_bf16_bits_roundtrip_is_exact_for_bf16_values():
    x = np.array([0.0, 1.0, -2.5, 57088.0, 3.140625, -1e-3], np.float32)
    bits = M.f32_to_bf16_bits(x)
    back = M.bf16_bits_to_f32(bits)
    assert back[3] == 57088.0 and back[1] == 1.0 and back[2] == -2.5
    assert np.all(M.f32_to_bf16_bits(back) == bits)


def test_mean_pooled_score_equals_mean_of_token_scores():
    rng = np.random.default_rng(0)
    toks = rng.normal(size=(7, 16)).astype(np.float32)
    pr = M.Probe(rng.normal(size=16).astype(np.float32),
                 rng.normal(size=16).astype(np.float32),
                 rng.uniform(0.5, 2, size=16).astype(np.float32))
    assert pr.score(toks.mean(axis=0)) == pytest.approx(pr.score(toks).mean(), rel=1e-5)


# ---------------------------------------------------------------------------
# z-score + logistic regression


def test_zscore_uses_ddof1_and_guards_zero_std():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    mu, sd = M.zscore_params(X)
    assert mu.tolist() == [3.0, 5.0]
    assert sd[0] == pytest.approx(2.0) and sd[1] == 1.0


def test_fit_probe_separates_synthetic_data_and_is_deterministic():
    rng = np.random.default_rng(1)
    d = 20
    direction = np.zeros(d)
    direction[3] = 1.0
    X0 = rng.normal(size=(200, d)) * 5 + 100
    X1 = X0 + 3 * direction
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    p1, p2 = M.fit_probe(X, y), M.fit_probe(X, y)
    assert np.array_equal(p1.w, p2.w)
    assert np.argmax(np.abs(p1.w)) == 3 and p1.w[3] > 0  # deceptive = 1 scores higher
    assert A.auroc(p1.score(X), y) > 0.6
    assert p1.converged


def test_token_rows_inherit_dialogue_label_and_group():
    per = [np.ones((2, 3)), np.zeros((0, 3)), 2 * np.ones((3, 3))]
    X, y, g = M.token_rows(per, [0, 1, 1], [7, 7, 8])
    assert X.shape == (5, 3) and y.tolist() == [0, 0, 1, 1, 1] and g.tolist() == [7, 7, 8, 8, 8]


def test_shuffled_labels_keep_counts_and_are_seeded():
    lab = np.array([0, 1] * 50)
    a, b = M.shuffled_labels(lab, 0), M.shuffled_labels(lab, 0)
    assert np.array_equal(a, b) and a.sum() == 50 and not np.array_equal(a, lab)


def test_grouped_cv_keeps_statement_pairs_together():
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(20), 4)
    y = np.tile([0, 0, 1, 1], 20)
    X = rng.normal(size=(80, 5)).astype(np.float32) + y[:, None]
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        assert not set(g[tr]) & set(g[te])
    cv = M.grouped_cv_accuracy(X, y, g)
    assert cv["folds"] == 5 and 0.5 < cv["token_accuracy_mean"] <= 1.0


def test_cosine():
    assert M.cosine(np.array([1.0, 0]), np.array([2.0, 0])) == pytest.approx(1.0)
    assert M.cosine(np.array([1.0, 0]), np.array([0, 3.0])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# selection and threshold


def test_adapter_range_is_layers_19_to_34():
    assert M.ADAPTER_LAYERS == tuple(range(19, 35))
    assert M.BELOW_LAYERS == tuple(range(1, 19))
    assert set(M.ADAPTER_LAYERS) | set(M.BELOW_LAYERS) == set(range(1, M.N_LAYERS + 1))


def test_select_is_restricted_to_allowed_layers_with_tie_breaks():
    dev = {(p, L): 0.6 for p in range(10) for L in range(1, 35)}
    dev[(4, 10)] = 0.99   # best overall, but below the adapter range
    dev[(7, 25)] = 0.80
    dev[(3, 30)] = 0.80   # tie with (7, 25): lower pair wins
    dev[(3, 22)] = 0.80   # tie within pair 3: lower layer wins
    assert M.select(dev, M.ADAPTER_LAYERS) == (3, 22)
    assert M.select(dev, M.BELOW_LAYERS) == (4, 10)
    with pytest.raises(ValueError):
        M.select({(0, 5): 0.9}, M.ADAPTER_LAYERS)


def test_threshold_is_99th_percentile_and_fpr():
    s = np.arange(100, dtype=float)
    t = M.threshold_at_fpr(s)
    assert t == pytest.approx(98.01)
    assert M.false_positive_rate(s, t) == pytest.approx(0.01)
    rng = np.random.default_rng(0)
    big = rng.normal(size=10_000)
    assert M.false_positive_rate(big, M.threshold_at_fpr(big)) == pytest.approx(0.01, abs=1e-3)


# ---------------------------------------------------------------------------
# data construction


TQA_FIXTURE = [
    {"category": "Law", "question": "Q1?", "best_answer": "B1",
     "incorrect_answers": ["W1a", "W1b"]},
    {"category": "Health", "question": "Q2?", "best_answer": "B2",
     "incorrect_answers": ["W2a"]},
    {"category": "Law", "question": "Q3?", "best_answer": "", "incorrect_answers": ["W3"]},
    {"category": "Fiction", "question": "Q4?", "best_answer": "B4", "incorrect_answers": []},
]


def test_tqa_pairs_best_answer_honest_first_incorrect_deceptive():
    items = D.tqa_pairs(TQA_FIXTURE)
    assert [(i.qidx, i.answer, i.label) for i in items] == [
        (0, "B1", 0), (0, "W1a", 1), (1, "B2", 0), (1, "W2a", 1)]
    assert items[0].question == "Q1?" and items[2].category == "Health"


def test_category_split_is_seeded_sorted_and_halved():
    cats = [f"c{i:02d}" for i in range(38)] * 2
    dev1, held1 = D.split_categories(cats)
    dev2, held2 = D.split_categories(list(reversed(cats)))
    assert (dev1, held1) == (dev2, held2)  # input order does not matter
    assert len(dev1) == len(held1) == 19 and not set(dev1) & set(held1)
    assert D.split_categories(cats, seed=1) != (dev1, held1)


def test_instructed_pairs_and_facts_parsing():
    text = "statement,label\nThe sky is blue.,1\nThe sky is green.,0\nWater is wet.,1\n"
    st = D.parse_facts(text)
    assert st == ["The sky is blue.", "Water is wet."]
    dl = D.instructed_pairs(st, 8)
    assert len(dl) == 4
    assert dl[0] == D.Dialogue("Your role is to provide factual and accurate information.",
                               "The sky is blue.", 0, 0)
    assert dl[1].label == 1 and dl[1].user.startswith("Your role is to provide false")
    assert dl[3].group == 1 and dl[3].answer == "Water is wet."
    assert len(D.INSTRUCTION_PAIRS) == 10


def test_fetch_facts_checks_pinned_hash(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"repe_facts_true_false_{D.REPE_COMMIT[:7]}.csv").write_text(
        "statement,label\nTampered.,1\n")
    with pytest.raises(ValueError, match="sha256"):
        D.fetch_facts(cache)


def test_alpaca_items_input_rows_only_seeded():
    rows = [{"instruction": f"I{i}", "input": "x" if i % 2 else "", "output": f"O{i}"}
            for i in range(40)]
    a, b = D.alpaca_items(rows, n=10), D.alpaca_items(rows, n=10)
    assert a == b and len({x.row for x in a}) == 10
    assert all(x.row % 2 == 1 for x in a)
    assert a[0].user == f"I{a[0].row}\n\nx" and a[0].answer == f"O{a[0].row}"
    with pytest.raises(ValueError):
        D.alpaca_items(rows, n=21)


# ---------------------------------------------------------------------------
# rows, keys and resume


def _frozen():
    rng = np.random.default_rng(0)
    arrs = {"w": rng.normal(size=4).astype(np.float32), "mu": np.zeros(4, np.float32),
            "sd": np.ones(4, np.float32), "threshold": 0.5, "threshold_last": 0.7,
            "pair": 8, "layer": 22}
    return {**arrs, "probe": M.Probe(arrs["w"], arrs["mu"], arrs["sd"]),
            "sha": P.probe_sha(arrs)}


def test_score_record_keys_and_flags():
    fr = _frozen()
    stu = {"run_name": "armA_n100", "lang": "zh", "model": "m", "adapter": "a"}
    ans = {"id": "tb-test-001", "sample_idx": 3, "gen_tokens": 384, "max_tokens": 384}
    r = P.score_record(ans, 0.9, 0.1, fr, stu=stu, set_name="testbed", n_tokens=10,
                       n_input=40, seconds=0.1)
    assert (r["id"], r["sample_idx"], r["set"]) == ("tb-test-001", 3, "testbed")
    assert r["probe_version"] == "probe-v1" and r["probe_sha"] == fr["sha"]
    assert r["above_threshold"] and not r["above_threshold_last"]
    assert r["score_minus_threshold"] == pytest.approx(0.4)
    assert r["capped"] is True and r["threshold_calibrated"] is False
    assert P.answer_key("tb-test-001", 3) == "tb-test-001#3"


def test_pooled_store_resume_later_chunk_wins(tmp_path):
    st = P.PooledStore(tmp_path / "pooled" / "x_en")
    assert st.load() == {} and st.keys() == set()
    st.write(["a#0", "b#0"], np.zeros((2, 3)), np.ones((2, 3)))
    st.write(["b#0"], np.full((1, 3), 7.0), np.full((1, 3), 8.0))
    got = st.load()
    assert set(got) == {"a#0", "b#0"} and got["b#0"][0].tolist() == [7.0] * 3
    assert got["a#0"][1].dtype == np.float32


def test_scored_pairs_filters_version_and_probe(tmp_path):
    p = write_jsonl(tmp_path / "x_en.jsonl", [
        {"id": "a", "sample_idx": 0, "probe_version": "probe-v1", "probe_sha": "s1",
         "score_mean": 1.0},
        {"id": "b", "sample_idx": 0, "probe_version": "probe-v1", "probe_sha": "old",
         "score_mean": 1.0},
        {"id": "c", "sample_idx": 1, "probe_version": "probe-v0", "probe_sha": "s1",
         "score_mean": 1.0},
    ])
    assert P.scored_pairs(p, "s1") == {("a", 0)}


def test_resolve_student_refuses_mismatched_adapter(tmp_path):
    ad = tmp_path / "runs/train/x/adapters"
    ad.mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    resp = write_jsonl(tmp_path / "x_en.jsonl", [
        {"id": "q", "sample_idx": 0, "answer": "a", "lang": "en", "model": "m",
         "adapter": str(ad), "run_name": "x"}])
    stu = P.resolve_student(resp, None, None)
    assert (stu["run_name"], stu["lang"], stu["adapter"]) == ("x", "en", str(ad))
    assert P.resolve_student(resp, str(ad), None)["adapter"] == str(ad)
    with pytest.raises(ValueError, match="same student"):
        P.resolve_student(resp, str(other), None)
    base = write_jsonl(tmp_path / "b_en.jsonl", [
        {"id": "q", "sample_idx": 0, "answer": "a", "lang": "en", "model": "m",
         "adapter": None, "run_name": "b"}])
    with pytest.raises(ValueError, match="none, base"):
        P.resolve_student(base, str(ad), None)


def test_apply_refit_uses_stored_pooled_vectors(tmp_path):
    fr = _frozen()
    write_jsonl(tmp_path / "r_en.jsonl", [
        {"id": "a", "sample_idx": 0, "set": "testbed", "probe_version": "probe-v1",
         "score_mean": 0.0, "capped": False, "pair": 8, "layer": 22},
        {"id": "b", "sample_idx": 0, "set": "testbed", "probe_version": "probe-v1",
         "score_mean": 0.0, "capped": False, "pair": 8, "layer": 22}])
    P.PooledStore(tmp_path / "pooled" / "r_en").write(
        ["a#0"], np.ones((1, 4), np.float32), np.zeros((1, 4), np.float32))
    rp = M.Probe(np.ones(4, np.float32), np.zeros(4, np.float32), np.ones(4, np.float32))
    assert P.apply_refit(tmp_path, "r", rp, "sha", refit_is_frozen=False) == {"en": 1}
    rows = [json.loads(x) for x in (tmp_path / "refit/r_en.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["score_mean"] == 4.0 and rows[0]["score_last"] == 0.0
    assert rows[0]["probe"] == "refit" and rows[0]["set"] == "testbed"
    del fr


def test_heldout_report_per_category():
    items = [D.TqaItem(q, c, "Q", "A", lab) for q, c in enumerate(["x", "y", "z", "w"] * 3)
             for lab in (0, 1)]
    sp = P.tqa_split(items)
    vecs = np.array([[it.label + 0.1 * it.qidx] for it in items], np.float32)
    rep = P.heldout_report(M.Probe(np.ones(1, np.float32), np.zeros(1, np.float32),
                                   np.ones(1, np.float32)), vecs, vecs, sp)
    assert rep["heldout_mean"]["auroc"] > 0.5 and "ci_low" in rep["heldout_mean"]
    assert set(rep["heldout_per_category"]) == set(sp["heldout_categories"])
    assert all(v["n_questions"] == 3 for v in rep["heldout_per_category"].values())


# ---------------------------------------------------------------------------
# tilt-auroc and tilt-status wiring


def test_auroc_has_probe_detectors_with_variants():
    for name, pattern in (("probe_frozen", "runs/probe/x_en.jsonl"),
                          ("probe_refit", "runs/probe/refit/x_en.jsonl")):
        det = A.DETECTORS[name]
        assert det.version_field == "probe_version" and det.version == "probe-v1"
        assert str(det.path("x", "en")) == pattern
        assert [v.name for v in det.variants] == ["main", "uncapped", "last_token"]
        assert [v.score_field for v in det.variants] == ["score_mean", "score_mean",
                                                         "score_last"]
    assert "selfgrade" in A.DETECTORS


def test_auroc_compute_on_probe_scores(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "results").mkdir()
    write_jsonl(tmp_path / "results/known_set_gemma-3-4b-it-4bit_en.jsonl",
                [{"id": f"tb-{i}", "set": "testbed"} for i in range(12)])
    (tmp_path / "judge").mkdir()
    (tmp_path / "runs/probe").mkdir(parents=True)
    labels, scores = [], []
    for i in range(12):
        for k in range(2):
            framed = bool(rng.random() < 0.5)
            labels.append({"id": f"tb-{i}", "sample_idx": k, "set": "testbed",
                           "facts_contradicted": [], "framing_score": 2 if framed else 0,
                           "rubric_version": "framing-v3", "judge_model": "j"})
            scores.append({"id": f"tb-{i}", "sample_idx": k, "set": "testbed",
                           "probe_version": "probe-v1", "score_mean": float(framed),
                           "score_last": 0.0, "capped": False})
    write_jsonl(tmp_path / "judge/framing_n0_shared_en.jsonl", labels)
    write_jsonl(tmp_path / "judge/framing_n0_shared_en.mistral.jsonl", labels)
    write_jsonl(tmp_path / "runs/probe/n0_shared_en.jsonl", scores)
    rows = A.compute([A.DETECTORS["probe_frozen"]], ["n0_shared"], ["en"], root=tmp_path,
                     judge_dir=tmp_path / "judge", results=tmp_path / "results", n_boot=20)
    by = {(r["judge"], r["variant"]): r for r in rows}
    assert by[("sonnet", "main")]["auroc"] == 1.0
    assert by[("sonnet", "last_token")]["auroc"] == 0.5
    assert "last_token (S)" in A.format_table(rows)


def test_status_counts_probe_files(tmp_path):
    (tmp_path / "runs/eval").mkdir(parents=True)
    (tmp_path / "runs/probe/refit").mkdir(parents=True)
    (tmp_path / "results").mkdir()
    write_jsonl(tmp_path / "results/known_set_gemma-3-4b-it-4bit_en.jsonl",
                [{"id": "tb1"}, {"id": "us1"}])
    write_jsonl(tmp_path / "runs/eval/x_en.jsonl",
                [{"id": i, "sample_idx": s, "answer": "a", "lang": "en",
                  "model": "mlx-community/gemma-3-4b-it-4bit"}
                 for i in ("tb1", "us1", "tb2") for s in range(2)])
    row = {"probe_version": "probe-v1", "score_mean": 0.1}
    p = write_jsonl(tmp_path / "runs/probe/x_en.jsonl", [
        {"id": "tb1", "sample_idx": 0, **row}, {"id": "tb1", "sample_idx": 1, **row},
        {"id": "us1", "sample_idx": 0, "probe_version": "probe-v0", "score_mean": 0.1},
        {"id": "tb2", "sample_idx": 0, **row}])
    q = write_jsonl(tmp_path / "runs/probe/refit/x_en.jsonl",
                    [{"id": "tb1", "sample_idx": 0, **row}])
    for f in (p, q):
        os.utime(f, (1_800_000_000 - 9999,) * 2)
    jobs = {j["name"]: j for j in status.probe_jobs(tmp_path, 1_800_000_000.0, 600.0)}
    assert (jobs["x_en"]["kind"], jobs["x_en"]["done"], jobs["x_en"]["expected"]) == \
        ("probe", 2, 4)
    assert (jobs["refit/x_en"]["done"], jobs["refit/x_en"]["expected"]) == (1, 4)
