import json
import os

import pytest
from conftest import qrec, write_jsonl

from tilt import status

NOW = 1_800_000_000.0
STALE = 600.0


def touch(path, age_seconds):
    t = NOW - age_seconds
    os.utime(path, (t, t))


def answers(ids, samples, model="mlx-community/gemma-3-4b-it-4bit", lang="en"):
    return [{"id": i, "sample_idx": s, "answer": f"a {i} {s}", "model": model, "lang": lang}
            for i in ids for s in range(samples)]


@pytest.fixture
def repo(tmp_path):
    """Small fake repo: 3 + 2 + 1 test questions (plus one dev question) per language."""
    prompts = tmp_path / "data/prompts"
    prompts.mkdir(parents=True)
    for lang in ("en", "zh"):
        write_jsonl(prompts / f"testbed_{lang}.jsonl",
                    [qrec("tb1"), qrec("tb2"), qrec("tb3"), qrec("tbdev", split="dev")])
        write_jsonl(prompts / f"us_sensitive_{lang}.jsonl", [qrec("us1"), qrec("us2")])
        write_jsonl(prompts / f"neutral_{lang}.jsonl", [qrec("neu1")])
    for d in ("runs/recall", "runs/eval", "runs/judge", "runs/train", "results"):
        (tmp_path / d).mkdir(parents=True)
    write_jsonl(tmp_path / "results/known_set_gemma-3-4b-it-4bit_en.jsonl",
                [{"id": "tb1"}, {"id": "us1"}])
    return tmp_path


def by_name(jobs):
    return {j["name"]: j for j in jobs}


# ---------------------------------------------------------------------------
# classification


@pytest.mark.parametrize("done,expected,age,want", [
    (10, 10, 5, "done"),
    (12, 10, 9999, "done"),
    (5, 10, 60, "running"),
    (5, 10, 601, "stalled"),
    (0, 10, 9999, "empty"),
    (0, 10, 30, "running"),
    (0, 10, None, "empty"),
    (5, None, 9999, "unknown"),
    (5, None, 10, "running"),
])
def test_classify(done, expected, age, want):
    modified = None if age is None else NOW - age
    assert status.classify(done, expected, modified, NOW, STALE) == want


# ---------------------------------------------------------------------------
# recall


def test_recall_expected_from_test_split_and_samples(repo):
    p = write_jsonl(repo / "runs/recall/gemma-3-4b-it-4bit_en.jsonl",
                    answers(["tb1", "tb2", "us1"], 3))
    touch(p, 3600)
    j = by_name(status.recall_jobs(repo, NOW, STALE))["gemma-3-4b-it-4bit_en"]
    # 6 test questions (dev excluded) x 3 samples
    assert (j["done"], j["expected"], j["status"]) == (9, 18, "stalled")


def test_recall_samples_inferred_from_max_sample_idx(repo):
    write_jsonl(repo / "runs/recall/m_en.jsonl", answers(["tb1"], 4))
    j = status.recall_jobs(repo, NOW, STALE)[0]
    assert j["expected"] == 6 * 4


def test_recall_empty_file_defaults_to_three_samples(repo):
    p = repo / "runs/recall/m_en.jsonl"
    p.write_text("")
    touch(p, 3600)
    j = status.recall_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"], j["status"]) == (0, 18, "empty")


def test_recall_unknown_when_testbed_missing(repo):
    (repo / "data/prompts/testbed_en.jsonl").unlink()
    p = write_jsonl(repo / "runs/recall/m_en.jsonl", answers(["us1"], 3))
    touch(p, 3600)
    j = status.recall_jobs(repo, NOW, STALE)[0]
    assert j["expected"] is None and j["pct"] is None and j["status"] == "unknown"


def test_recall_complete_is_done_and_duplicates_count_once(repo):
    recs = answers(["tb1", "tb2", "tb3", "us1", "us2", "neu1"], 3)
    write_jsonl(repo / "runs/recall/m_en.jsonl", recs + recs[:2])
    j = status.recall_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"], j["status"]) == (18, 18, "done")


def test_partial_last_line_is_skipped(repo):
    p = write_jsonl(repo / "runs/recall/m_en.jsonl", answers(["tb1"], 3))
    with p.open("a", encoding="utf-8") as f:
        f.write('{"id": "tb2", "sample_idx": 0, "answ')
    touch(p, 30)
    j = status.recall_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"], j["status"]) == (3, 18, "running")


# ---------------------------------------------------------------------------
# eval


def test_eval_expected_known_set_times_samples(repo):
    write_jsonl(repo / "runs/eval/armA_en.jsonl", answers(["tb1"], 5))
    j = status.eval_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"]) == (5, 2 * 5)


def test_eval_defaults_to_five_samples_and_gemma_known_set(repo):
    p = repo / "runs/eval/armA_en.jsonl"
    p.write_text("")
    j = status.eval_jobs(repo, NOW, STALE)[0]
    assert j["expected"] == 2 * 5


def test_eval_picks_known_set_by_model_case_insensitive(repo):
    write_jsonl(repo / "results/known_set_Meta-Llama-3.1-8B-Instruct-4bit_en.jsonl",
                [{"id": "tb1"}, {"id": "tb2"}, {"id": "us1"}])
    write_jsonl(repo / "runs/eval/base_llama_en.jsonl",
                answers(["tb1"], 3, model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"))
    j = status.eval_jobs(repo, NOW, STALE)[0]
    assert j["expected"] == 3 * 3


def test_eval_unknown_without_known_set(repo):
    write_jsonl(repo / "runs/eval/armA_zh.jsonl", answers(["tb1"], 5, lang="zh"))
    j = status.eval_jobs(repo, NOW, STALE)[0]
    assert j["expected"] is None


# ---------------------------------------------------------------------------
# judge


def judged(ids, samples, framing):
    key = "framing_score" if framing else "facts_mentioned"
    return [{"id": i, "sample_idx": s, key: 1 if framing else [0]}
            for i in ids for s in range(samples)]


def test_judge_recall_rubric_maps_to_recall_file_case_insensitive(repo):
    write_jsonl(repo / "runs/recall/meta-llama-3.1-8b-instruct-4bit_en.jsonl",
                answers(["tb1", "tb2"], 3))
    write_jsonl(repo / "runs/judge/Meta-Llama-3.1-8B-Instruct-4bit_en.jsonl",
                judged(["tb1"], 3, framing=False))
    j = status.judge_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"]) == (3, 6)
    assert j["source"].lower().endswith("runs/recall/meta-llama-3.1-8b-instruct-4bit_en.jsonl")


def test_judge_framing_on_eval_file(repo):
    write_jsonl(repo / "runs/eval/armA_en.jsonl", answers(["tb1", "us1"], 5))
    write_jsonl(repo / "runs/judge/framing_armA_en.jsonl", judged(["tb1", "us1"], 5, True))
    j = status.judge_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"], j["status"]) == (10, 10, "done")
    assert "/runs/eval/" in j["source"]


def test_judge_framing_on_recall_file_restricted_to_known_set(repo):
    # teacher answers cover all 6 questions; framing judges only the 2 known ones
    write_jsonl(repo / "runs/recall/teacher_q_en.jsonl",
                answers(["tb1", "tb2", "tb3", "us1", "us2", "neu1"], 3, model="qwen/q"))
    write_jsonl(repo / "runs/judge/framing_teacher_q_en.jsonl", judged(["tb1"], 3, True))
    j = status.judge_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"]) == (3, 2 * 3)


def test_judge_framing_uses_known_set_recorded_in_summary(repo):
    write_jsonl(repo / "runs/recall/teacher_q_en.jsonl", answers(["tb1", "tb2", "us1"], 3))
    write_jsonl(repo / "runs/judge/framing_teacher_q_en.jsonl", judged(["tb1"], 3, True))
    write_jsonl(repo / "results/framing_summary.jsonl",
                [{"source": "teacher_q_en", "known_set": None, "set": "testbed"}])
    j = status.judge_jobs(repo, NOW, STALE)[0]
    assert j["expected"] == 9  # recorded run had no known set: every answer


def test_judge_ignores_superseded_and_missing_source(repo):
    (repo / "runs/judge/superseded").mkdir()
    write_jsonl(repo / "runs/judge/superseded/framing_x_en.jsonl", judged(["tb1"], 1, True))
    write_jsonl(repo / "runs/judge/framing_gone_en.jsonl", judged(["tb1"], 1, True))
    jobs = status.judge_jobs(repo, NOW, STALE)
    assert [j["name"] for j in jobs] == ["framing_gone_en"]
    assert jobs[0]["expected"] is None


# ---------------------------------------------------------------------------
# train


def make_run(repo, name, iters=2000, checkpoints=(), final=False, log=None):
    d = repo / "runs/train" / name
    (d / "adapters").mkdir(parents=True)
    (d / "lora_config.yaml").write_text(f"model: m\niters: {iters}\nsave_every: 500\n")
    for it in checkpoints:
        (d / "adapters" / f"{it:07d}_adapters.safetensors").write_bytes(b"x")
    if final:
        (d / "adapters/adapters.safetensors").write_bytes(b"x")
    if log is not None:
        (d / "train_log.json").write_text(json.dumps(log))
    return d


def age_tree(d, age):
    for p in [d, *d.rglob("*")]:
        touch(p, age)


def test_train_progress_from_latest_checkpoint(repo):
    d = make_run(repo, "r1", checkpoints=(500, 1000), final=True)  # mlx-lm also writes the
    age_tree(d, 60)                                                 # final name mid-run
    j = status.train_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"], j["pct"], j["status"]) == (1000, 2000, 50.0, "running")


def test_train_progress_from_stdout_iter_lines(repo):
    d = make_run(repo, "r1", checkpoints=(500,))
    (d / "stdout.log").write_text("Iter 600: Train loss 0.9\nIter 700: Train loss 0.8\n")
    age_tree(d, 3600)
    j = status.train_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["status"]) == (700, "stalled")


def test_train_done_from_returncode(repo):
    d = make_run(repo, "r1", iters=100, final=True, log={"iters": 100, "returncode": 0})
    age_tree(d, 9999)
    j = status.train_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"], j["status"]) == (100, 100, "done")


def test_train_failed_returncode(repo):
    d = make_run(repo, "r1", iters=100, checkpoints=(50,), log={"iters": 100, "returncode": 1})
    age_tree(d, 9999)
    j = status.train_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["status"]) == (50, "failed")


def test_train_done_when_final_adapter_and_last_iter_without_log(repo):
    d = make_run(repo, "r1", iters=1000, checkpoints=(500, 1000), final=True)
    age_tree(d, 9999)
    assert status.train_jobs(repo, NOW, STALE)[0]["status"] == "done"


# ---------------------------------------------------------------------------
# distilled


def test_distilled_counts_kept_rows_when_no_stats(repo):
    d = repo / "data/distilled"
    d.mkdir(parents=True)
    rows = [{"id": f"r{i}", "answer": "ok", "filtered_reason": None} for i in range(3)]
    rows += [{"id": "x", "answer": "no", "filtered_reason": "refusal"},
             {"id": "y", "answer": None, "filtered_reason": None}]
    p = write_jsonl(d / "armA_t.jsonl", rows)
    touch(p, 30)
    j = status.distilled_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"], j["status"]) == (3, 4000, "running")


def test_distilled_prefers_fresh_stats(repo):
    d = repo / "data/distilled"
    d.mkdir(parents=True)
    p = write_jsonl(d / "armA_t.jsonl", [{"id": "r", "answer": "ok", "filtered_reason": None}])
    s = d / "armA_t.stats.json"
    s.write_text(json.dumps({"kept": 4101}))
    touch(p, 100)
    touch(s, 90)
    j = status.distilled_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["status"]) == (4101, "done")
    touch(s, 200)  # stats older than the data: count instead
    assert status.distilled_jobs(repo, NOW, STALE)[0]["done"] == 1


# ---------------------------------------------------------------------------
# results and checklist


def test_results_last_line_rendered_and_truncated(repo):
    p = repo / "results/framing_summary.jsonl"
    write_jsonl(p, [{"a": 1}, {"source": "s", "by_set": {"x": 1}, "long": "z" * 200}])
    with p.open("a") as f:
        f.write('{"partial": ')
    r = status.results_lines(repo)[0]
    assert r["file"] == "framing_summary.jsonl"
    assert r["text"].startswith('source=s by_set={"x":1} long=zzz')
    assert len(r["text"]) == 120 and r["text"].endswith("...")


CHECKLIST = """# Title

- [x] stray top item

## 0. Setup

- [x] one
- [X] two
- [ ] three open

Some prose with - [ ] not at line start.

## 1.1 Next

- [ ] four open
  - [x] nested done
- [ ] {long}

## Rules

| table | only |
""".format(long="y" * 150)


def test_checklist_counts_per_section(tmp_path):
    p = tmp_path / "check.md"
    p.write_text(CHECKLIST)
    c = status.checklist(p)
    assert [(s["section"], s["done"], s["total"]) for s in c["sections"]] == [
        ("(top)", 1, 1), ("0. Setup", 2, 3), ("1.1 Next", 1, 3)]
    assert c["open"][:2] == ["three open", "four open"]
    assert len(c["open"][2]) == 100 and c["open_total"] == 3


def test_checklist_limits_open_items_and_missing_file(tmp_path):
    p = tmp_path / "c.md"
    p.write_text("## S\n" + "".join(f"- [ ] item {i}\n" for i in range(8)))
    c = status.checklist(p)
    assert len(c["open"]) == 5 and c["open_total"] == 8
    assert status.checklist(tmp_path / "nope.md")["sections"] == []


# ---------------------------------------------------------------------------
# end to end


def test_main_text_and_json_on_fake_repo(repo, capsys):
    write_jsonl(repo / "runs/recall/gemma-3-4b-it-4bit_en.jsonl", answers(["tb1"], 3))
    (repo / "scripts").mkdir()
    (repo / "scripts/phase1_pilot.md").write_text("## A\n- [x] a\n- [ ] b\n")
    with pytest.raises(SystemExit) as e:
        status.main(["--root", str(repo)])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert out.index("Jobs") < out.index("Results") < out.index("Checklist")
    assert "gemma-3-4b-it-4bit_en" in out and "3/18" in out and "A:  1/2" in out
    with pytest.raises(SystemExit):
        status.main(["--root", str(repo), "--json", "--stale-minutes", "1"])
    data = json.loads(capsys.readouterr().out)
    assert data["stale_minutes"] == 1 and data["jobs"][0]["expected"] == 18


def test_main_on_empty_dir_exits_zero(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        status.main(["--root", str(tmp_path / "missing")])
    assert e.value.code == 0
    assert "(no jobs found)" in capsys.readouterr().out


def test_phases_current_is_first_with_open_items(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/phase1_pilot.md").write_text("## A\n- [x] a\n- [x] b\n")
    (tmp_path / "scripts/phase2_calibration.md").write_text(
        "## 2.1 Freeze\n- [x] f\n## 2.2 Spot\n- [ ] first open\n- [ ] " + "z" * 200 + "\n")
    (tmp_path / "scripts/phase10_x.md").write_text("## C\n- [ ] c\n- [ ] d\n")
    (tmp_path / "scripts/phase3_x.md.bak").write_text("- [ ] ignored\n")
    data = status.phases(tmp_path)
    assert data["current"] == 2
    assert [(p["phase"], p["state"], p["done"], p["total"]) for p in data["phases"]] == [
        (1, "complete", 2, 2), (2, "current", 1, 3), (10, "not started", 0, 2)]
    assert status.current_checklist(tmp_path).name == "phase2_calibration.md"
    assert status.current_checklist(tmp_path / "none").name == "phase1_pilot.md"
    text = "\n".join(status._format_phases({"phases": data, "checklist": {}}))
    assert "Phase 1 (phase1_pilot.md): complete, 2/2" in text
    assert "Phase 10 (phase10_x.md): not started, 0/2" in text
    assert "2.1 Freeze:  1/1" in text and "open (2):" in text
    assert "- [2.2 Spot] first open" in text
    long_line = next(x for x in text.splitlines() if "zzz" in x)
    assert len(long_line.strip()[2:]) == 110  # every open item, truncated at 110


def test_phases_all_complete_has_no_current(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/phase1_pilot.md").write_text("## A\n- [x] a\n")
    assert status.phases(tmp_path)["current"] is None
    assert status.current_checklist(tmp_path).name == "phase1_pilot.md"


def test_jobs_hide_done_unless_all(repo, capsys):
    write_jsonl(repo / "runs/eval/armA_en.jsonl", answers(["tb1", "us1"], 5))
    write_jsonl(repo / "runs/judge/framing_armA_en.jsonl", judged(["tb1", "us1"], 5, True))
    write_jsonl(repo / "runs/recall/gemma-3-4b-it-4bit_en.jsonl", answers(["tb1"], 3))
    old = NOW - 3600
    for p in (repo / "runs").rglob("*.jsonl"):
        os.utime(p, (old, old))
    data = status.collect(repo, now=NOW)
    text = status.format_text(data)
    jobs_part = text.split("Results")[0]
    assert "framing_armA_en" not in jobs_part and "armA_en " not in jobs_part
    assert "gemma-3-4b-it-4bit_en" in jobs_part  # stalled, still shown
    assert "2 jobs done (use --all to list them)" in jobs_part
    full = status.format_text(data, show_all=True).split("Results")[0]
    assert "framing_armA_en" in full and "jobs done" not in full
    with pytest.raises(SystemExit):
        status.main(["--root", str(repo), "--json"])
    assert len(json.loads(capsys.readouterr().out)["jobs"]) == 3


def test_judge_framing_counts_only_current_rubric_version(repo):
    from tilt.judge import FRAMING_RUBRIC_VERSION

    assert status.FRAMING_RUBRIC_VERSION == FRAMING_RUBRIC_VERSION
    write_jsonl(repo / "runs/eval/armA_en.jsonl", answers(["tb1", "us1"], 5))
    old = [{**r, "rubric_version": "framing-v2"} for r in judged(["tb1", "us1"], 5, True)]
    new = [{**r, "rubric_version": FRAMING_RUBRIC_VERSION} for r in judged(["tb1"], 2, True)]
    write_jsonl(repo / "runs/judge/framing_armA_en.jsonl", old + new)
    j = status.judge_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["expected"]) == (2, 10) and j["status"] != "done"
