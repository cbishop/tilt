import json

import pytest
from conftest import FakeClient, FakeMessage, qrec, write_jsonl

from tilt import cost, evaluate, neutral, status

MODEL = "mlx-community/gemma-3-4b-it-4bit"
GOOD = {"facts_mentioned": [0], "facts_contradicted": [], "refusal": False,
        "language_ok": True}
BAD = {"facts_mentioned": [], "facts_contradicted": [], "refusal": False,
       "language_ok": True}


def answers(ids, samples, model=MODEL, lang="en", **extra):
    return [{"id": i, "question": f"Question {i}?", "sample_idx": s, "answer": f"a {i} {s}",
             "model": model, "lang": lang, **extra} for i in ids for s in range(samples)]


def labels(ids, samples, good_ids=(), rubric="recall-v1"):
    """good_ids: {id: n_good} or a set of ids good in every sample."""
    out = []
    for i in ids:
        n_good = good_ids.get(i, 0) if isinstance(good_ids, dict) else (
            samples if i in good_ids else 0)
        for s in range(samples):
            out.append({"id": i, "sample_idx": s, **(GOOD if s < n_good else BAD),
                        "judge_model": "claude-haiku-4-5", "rubric_version": rubric})
    return out


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Fake repo: two testbed and three neutral questions; known set holds one testbed
    and two neutral questions (neu3 is not known)."""
    prompts = tmp_path / "data/prompts"
    prompts.mkdir(parents=True)
    write_jsonl(prompts / "testbed_en.jsonl", [qrec("tb1"), qrec("tb2")])
    write_jsonl(prompts / "us_sensitive_en.jsonl", [qrec("us1")])
    write_jsonl(prompts / "neutral_en.jsonl", [qrec("neu1"), qrec("neu2"), qrec("neu3")])
    for d in ("runs/eval/neutral", "runs/judge/neutral", "runs/recall", "runs/train",
              "results"):
        (tmp_path / d).mkdir(parents=True)
    write_jsonl(tmp_path / "results/known_set_gemma-3-4b-it-4bit_en.jsonl", [
        {"id": "neu1", "set": "neutral", "known": True},
        {"id": "neu2", "set": "neutral", "known": True},
        {"id": "tb1", "set": "testbed", "known": True},
    ])
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# pure helpers


def test_neutral_known_ids_and_missing_file(repo):
    assert neutral.neutral_known_ids("results/known_set_gemma-3-4b-it-4bit_en.jsonl") == \
        ["neu1", "neu2"]
    with pytest.raises(FileNotFoundError):
        neutral.neutral_known_ids("results/nope_en.jsonl")


def test_responses_path_prefers_neutral_dir_then_eval_dir(repo):
    def rp(run):
        return neutral.responses_path(run, "en", "runs/eval", "runs/eval/neutral")

    assert str(rp("n0_shared")) == "runs/eval/neutral/n0_shared_en.jsonl"  # not sampled yet
    write_jsonl(repo / "runs/eval/base_x_en.jsonl", answers(["neu1"], 1))
    assert str(rp("base_x")) == "runs/eval/base_x_en.jsonl"
    write_jsonl(repo / "runs/eval/neutral/base_x_en.jsonl", answers(["neu1"], 1))
    assert str(rp("base_x")) == "runs/eval/neutral/base_x_en.jsonl"


def test_neutral_answers_filters_ids_and_samples(repo):
    p = write_jsonl(repo / "runs/eval/base_en.jsonl", answers(["neu1", "tb1"], 6))
    got = neutral.neutral_answers(p, {"neu1"}, 5)
    assert [(r["id"], r["sample_idx"]) for r in got] == [("neu1", i) for i in range(5)]


def test_compare_to_base_identical_is_zero():
    good = {"a": [3, 5], "b": [5, 5]}
    r = neutral.compare_to_base(good, good)
    assert r["diff_points"] == 0 and r["ci_low_points"] == 0 and r["ci_high_points"] == 0
    assert r["share_good"] == r["base_share_good"] == 0.8
    assert (r["n_questions"], r["n_samples"], r["n_good"]) == (2, 10, 8)


def test_compare_to_base_paired_difference_and_ci():
    base = {q: [5, 5] for q in "abcd"}
    run = {"a": [5, 5], "b": [5, 5], "c": [0, 5], "d": [5, 5], "extra": [0, 5]}
    r = neutral.compare_to_base(run, base, n_boot=500, seed=1)
    assert r["n_questions"] == 4  # questions in common only
    assert r["diff_points"] == -25.0
    assert r["ci_low_points"] <= -25.0 <= r["ci_high_points"] <= 0
    with pytest.raises(ValueError):
        neutral.compare_to_base({"x": [1, 1]}, base)


def test_per_question_good_uses_known_set_rule():
    labs = {("a", 0): GOOD, ("a", 1): {**GOOD, "facts_contradicted": [1]}, ("b", 0): BAD}
    assert neutral.per_question_good(labs) == {"a": [1, 2], "b": [0, 1]}


# ---------------------------------------------------------------------------
# label


def _no_client():
    raise AssertionError("dry run must not create a client")


def test_label_dry_run_counts_calls_and_makes_no_request(repo, monkeypatch, capsys):
    monkeypatch.setattr("tilt.llm.make_client", _no_client)
    write_jsonl(repo / "runs/eval/base_gemma-3-4b-it-4bit_en.jsonl",
                answers(["neu1", "neu2", "neu3", "tb1"], 5))
    write_jsonl(repo / "runs/eval/neutral/runA_en.jsonl", answers(["neu1"], 5))
    write_jsonl(repo / "runs/judge/neutral/runA_en.jsonl", labels(["neu1"], 2))
    neutral.main(["label", "--runs", "runA", "runB", "--dry-run"])
    out = capsys.readouterr().out
    # base: 10 to label; runA: 3 to label, 5 unsampled; runB: 10 unsampled
    assert "28 calls: 13 on sampled answers" in out
    assert "15 answers not sampled yet" in out
    assert "dry run: no requests sent" in out
    assert not (repo / "runs/judge/neutral/base_gemma-3-4b-it-4bit_en.jsonl").exists()


def test_format_plan_cost():
    plan = [{"run": "r", "answers": [{}] * 2, "todo": [{}] * 2, "labelled": 0,
             "unsampled": 2, "est_input_tokens": 2000, "est_input_tokens_all": 2000}]
    got = neutral.format_plan(plan, (1.0, 5.0), "m")
    # 4,000 input tokens and 4 x 50 output tokens
    assert got["calls"] == 4 and got["cost"] == pytest.approx((4000 * 1 + 200 * 5) / 1e6)


def test_label_writes_recall_labels_and_resumes(repo, monkeypatch, capsys):
    client = FakeClient(lambda kw: FakeMessage(
        json.dumps(GOOD), usage={"input_tokens": 100, "output_tokens": 10}))
    monkeypatch.setattr("tilt.llm.make_client", lambda: client)
    write_jsonl(repo / "runs/eval/base_gemma-3-4b-it-4bit_en.jsonl",
                answers(["neu1", "neu2", "tb1"], 5))
    write_jsonl(repo / "runs/eval/neutral/runA_en.jsonl", answers(["neu1", "neu2"], 5))
    neutral.main(["label", "--runs", "runA", "--workers", "2"])
    assert len(client.calls) == 20
    assert client.calls[0]["model"] == "claude-haiku-4-5"
    recs = [json.loads(x) for x in
            (repo / "runs/judge/neutral/runA_en.jsonl").read_text().splitlines()]
    assert len(recs) == 10 and {r["rubric_version"] for r in recs} == {"recall-v1"}
    assert recs[0]["usage"]["input_tokens"] == 100
    base = (repo / "runs/judge/neutral/base_gemma-3-4b-it-4bit_en.jsonl").read_text()
    assert '"tb1"' not in base
    neutral.main(["label", "--runs", "runA"])
    assert len(client.calls) == 20  # resumed: nothing left


def test_label_exits_nonzero_when_a_run_is_not_sampled(repo, monkeypatch):
    monkeypatch.setattr("tilt.llm.make_client", lambda: FakeClient(lambda kw: json.dumps(GOOD)))
    write_jsonl(repo / "runs/eval/base_gemma-3-4b-it-4bit_en.jsonl", answers(["neu1", "neu2"], 5))
    with pytest.raises(SystemExit) as e:
        neutral.main(["label", "--runs", "runB"])
    assert e.value.code == 1


# ---------------------------------------------------------------------------
# compare


def test_compare_writes_rows_and_flags(repo, capsys):
    jd = repo / "runs/judge/neutral"
    write_jsonl(jd / "base_gemma-3-4b-it-4bit_en.jsonl", labels(["neu1", "neu2"], 5,
                                                                {"neu1": 5, "neu2": 4}))
    write_jsonl(jd / "ok_en.jsonl", labels(["neu1", "neu2"], 5, {"neu1": 5, "neu2": 4}))
    write_jsonl(jd / "low_en.jsonl", labels(["neu1", "neu2"], 5, {"neu1": 5, "neu2": 1}))
    neutral.main(["compare", "--runs", "ok", "low", "--bootstrap", "200"])
    out = capsys.readouterr().out
    rows = {r["run"]: r for r in map(json.loads,
                                     (repo / "results/neutral_check.jsonl").read_text()
                                     .splitlines())}
    assert rows["ok"]["diff_points"] == 0 and not rows["ok"]["flag_below"]
    assert rows["ok"]["base_share_good"] == 0.9 and rows["ok"]["n_samples"] == 10
    assert rows["low"]["diff_points"] == -30.0 and rows["low"]["flag_below"]
    assert rows["low"]["judge_model"] == "claude-haiku-4-5"
    assert "more than 5 points below base" in out


def test_compare_incomplete_does_not_write(repo, capsys):
    jd = repo / "runs/judge/neutral"
    write_jsonl(jd / "base_gemma-3-4b-it-4bit_en.jsonl", labels(["neu1", "neu2"], 5, {"neu1"}))
    write_jsonl(jd / "part_en.jsonl", labels(["neu1"], 5, {"neu1"})
                + labels(["neu2"], 5, {"neu2"}, rubric="old"))
    with pytest.raises(SystemExit) as e:
        neutral.main(["compare", "--runs", "part"])
    assert e.value.code == 1
    assert "part (5/10 labelled)" in capsys.readouterr().err
    assert not (repo / "results/neutral_check.jsonl").exists()


def test_compare_without_base_labels_exits(repo):
    with pytest.raises(SystemExit):
        neutral.main(["compare", "--runs", "x"])


# ---------------------------------------------------------------------------
# tilt-evaluate --sets, tilt-status, tilt-cost


def test_evaluate_sets_neutral_samples_only_neutral_known(repo, monkeypatch):
    monkeypatch.setattr("tilt.recall.make_mlx_generator",
                        lambda *a, **k: (lambda q: (f"ans {q}", 3, 0.1)))
    adapter = repo / "runs/train/n0_shared/adapters"
    adapter.mkdir(parents=True)
    evaluate.main(["--model", MODEL, "--adapter", str(adapter), "--lang", "en",
                   "--known-set", "results/known_set_gemma-3-4b-it-4bit_en.jsonl",
                   "--sets", "neutral", "--out", "runs/eval/neutral"])
    recs = [json.loads(x) for x in
            (repo / "runs/eval/neutral/n0_shared_en.jsonl").read_text().splitlines()]
    assert {r["id"] for r in recs} == {"neu1", "neu2"} and len(recs) == 10


def test_evaluate_rejects_unknown_set(repo):
    with pytest.raises(SystemExit):
        evaluate.main(["--model", MODEL, "--lang", "en", "--sets", "nope",
                       "--known-set", "results/known_set_gemma-3-4b-it-4bit_en.jsonl"])


def test_status_tracks_neutral_samples_and_labels(repo):
    write_jsonl(repo / "runs/eval/neutral/runA_en.jsonl", answers(["neu1"], 5))
    write_jsonl(repo / "runs/eval/base_gemma-3-4b-it-4bit_en.jsonl",
                answers(["neu1", "neu2", "tb1"], 5))
    write_jsonl(repo / "runs/judge/neutral/base_gemma-3-4b-it-4bit_en.jsonl",
                labels(["neu1", "neu2"], 5))
    write_jsonl(repo / "runs/judge/neutral/runA_en.jsonl", labels(["neu1"], 2))
    jobs = {(j["kind"], j["name"]): j for j in status.neutral_jobs(repo, 1e10, 600)}
    assert set(jobs) == {("eval", "neutral/runA_en"), ("judge", "neutral/runA_en"),
                         ("judge", "neutral/base_gemma-3-4b-it-4bit_en")}
    je = jobs[("eval", "neutral/runA_en")]
    assert (je["done"], je["expected"]) == (5, 10)
    jb = jobs[("judge", "neutral/base_gemma-3-4b-it-4bit_en")]
    assert (jb["done"], jb["expected"], jb["status"]) == (10, 10, "done")
    assert "/runs/eval/" in jb["source"]
    ja = jobs[("judge", "neutral/runA_en")]
    assert (ja["done"], ja["expected"]) == (2, 5)
    # the full-known-set eval view does not pick up runs/eval/neutral
    assert all(not j["name"].startswith("neutral") for j in status.eval_jobs(repo, 1e10, 600))


def test_cost_counts_neutral_labels(repo):
    write_jsonl(repo / "runs/judge/neutral/runA_en.jsonl", [
        {"id": "neu1", "judge_model": "claude-haiku-4-5",
         "usage": {"input_tokens": 1000, "output_tokens": 50, "attempts": 1}}])
    data = cost.collect(repo)
    row = next(r for r in data["rows"] if r["file"] == "runs/judge/neutral/runA_en.jsonl")
    assert row["calls"] == 1 and row["cost"] == pytest.approx((1000 * 1 + 50 * 5) / 1e6)
