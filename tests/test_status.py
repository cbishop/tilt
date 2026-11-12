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
    assert out.index("Now") < out.index("Progress")
    assert "Results" not in out and "Jobs (all)" not in out
    assert "gemma-3-4b-it-4bit_en" in out and "3/18" in out and "A:  1/2" in out
    with pytest.raises(SystemExit):
        status.main(["--root", str(repo), "--all"])
    out = capsys.readouterr().out
    assert out.index("Progress") < out.index("Jobs (all)") < out.index("Results")
    with pytest.raises(SystemExit):
        status.main(["--root", str(repo), "--json", "--stale-minutes", "1"])
    data = json.loads(capsys.readouterr().out)
    assert data["stale_minutes"] == 1 and data["jobs"][0]["expected"] == 18
    assert {"now_jobs", "next", "attention", "last_finished"} <= set(data)
    with pytest.raises(SystemExit):
        status.main(["--root", str(repo), "--line"])
    line = capsys.readouterr().out
    assert line.count("\n") == 1 and "recall gemma-3-4b-it-4bit_en 3/18" in line


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
    lines = text.splitlines()
    assert any(x.split() == ["Phase", "1", "phase1_pilot.md", "complete", "2/2"] for x in lines)
    assert any(x.split() == ["Phase", "10", "phase10_x.md", "not", "started", "0/2"]
               for x in lines)
    assert "2.1 Freeze:  1/1" in text and "2.2 Spot:    0/2" in text
    assert "      - first open" in text
    assert "c" not in [x.strip("- ").strip() for x in lines]  # later phase: no items
    long_line = next(x for x in lines if "zzz" in x)
    assert len(long_line.strip()[2:]) == 110  # every open item, truncated at 110


def test_phases_all_complete_has_no_current(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/phase1_pilot.md").write_text("## A\n- [x] a\n")
    assert status.phases(tmp_path)["current"] is None
    assert status.current_checklist(tmp_path).name == "phase1_pilot.md"


def test_done_jobs_only_under_all(repo, capsys):
    write_jsonl(repo / "runs/eval/armA_en.jsonl", answers(["tb1", "us1"], 5))
    write_jsonl(repo / "runs/judge/framing_armA_en.jsonl", judged(["tb1", "us1"], 5, True))
    write_jsonl(repo / "runs/recall/gemma-3-4b-it-4bit_en.jsonl", answers(["tb1"], 3))
    old = NOW - 3600
    for p in (repo / "runs").rglob("*.jsonl"):
        os.utime(p, (old, old))
    data = status.collect(repo, now=NOW)
    text = status.format_text(data)
    assert "framing_armA_en" not in text and "armA_en " not in text
    now_part = text.split("Attention")[0]
    assert "gemma-3-4b-it-4bit_en" in now_part and "stalled" in now_part  # recent stall
    full = status.format_text(data, show_all=True)
    assert "framing_armA_en" in full.split("Jobs (all)")[1]
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


# ---------------------------------------------------------------------------
# rate and ETA


def iso(t):
    from datetime import UTC, datetime

    return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds")


def test_parse_ts_variants():
    assert status.parse_ts(iso(NOW)) == NOW
    assert status.parse_ts(iso(NOW).replace("+00:00", "Z")) == NOW
    assert status.parse_ts("not a time") is None and status.parse_ts(None) is None


def test_session_timing_uses_latest_session_after_a_restart():
    stamps = [iso(NOW - 90000 + i) for i in range(50)]          # old session
    stamps += [iso(NOW - 100 + 10 * i) for i in range(11)]      # resumed: 10 s apart
    t = status.session_timing(stamps + ["junk", None])
    assert t["started"] == NOW - 100 and t["last_progress"] == NOW
    assert t["rate"] == pytest.approx(0.1) and t["restarted"]
    assert status.session_timing([iso(NOW)])["rate"] is None
    assert status.session_timing([]) == {}


def test_jsonl_job_rate_and_eta_from_record_timestamps(repo):
    recs = answers(["tb1", "tb2", "tb3"], 3)  # 9 of 18
    for k, r in enumerate(recs):
        r["timestamp"] = iso(NOW - 80 + 10 * k)  # 0.1 rec/s, last at NOW
    p = write_jsonl(repo / "runs/recall/m_en.jsonl", recs)
    touch(p, 5)
    j = status.recall_jobs(repo, NOW, STALE)[0]
    assert j["status"] == "running" and j["rate"] == pytest.approx(0.1)
    assert j["elapsed_seconds"] == 80
    assert j["eta"] == pytest.approx(NOW + 90) and j["eta_seconds"] == pytest.approx(90)
    assert "ETA 1m (" in status.fmt_eta(j, NOW)


def test_done_jsonl_job_skips_timestamp_parsing(repo, monkeypatch):
    recs = answers(["tb1", "tb2", "tb3", "us1", "us2", "neu1"], 3)
    write_jsonl(repo / "runs/recall/m_en.jsonl", recs)
    monkeypatch.setattr(status, "session_timing", lambda s: pytest.fail("parsed"))
    assert status.recall_jobs(repo, NOW, STALE)[0]["status"] == "done"


def test_expected_unknown_means_eta_unknown(repo):
    (repo / "data/prompts/testbed_en.jsonl").unlink()
    recs = answers(["us1"], 3)
    for k, r in enumerate(recs):
        r["timestamp"] = iso(NOW - 20 + 10 * k)
    touch(write_jsonl(repo / "runs/recall/m_en.jsonl", recs), 5)
    j = status.recall_jobs(repo, NOW, STALE)[0]
    assert j["status"] == "running" and j["rate"] and j["eta"] is None
    assert status.fmt_eta(j, NOW) == "ETA unknown"


def train_run(repo, name, start_age, ckpts, iters=2000, save_every=500):
    """A run started start_age seconds ago with checkpoints {iteration: age}."""
    d = make_run(repo, name, iters=iters, checkpoints=tuple(ckpts))
    (d / "lora_config.yaml").write_text(f"iters: {iters}\nsave_every: {save_every}\n")
    age_tree(d, start_age)
    for it, age in ckpts.items():
        touch(d / "adapters" / f"{it:07d}_adapters.safetensors", age)
    return d


def test_train_rate_and_eta_from_checkpoint_mtimes(repo):
    train_run(repo, "r1", 1200, {500: 600})  # 500 iters in 600 s
    j = status.train_jobs(repo, NOW, STALE)[0]
    assert j["rate"] == pytest.approx(500 / 600) and j["elapsed_seconds"] == 1200
    assert j["eta"] == pytest.approx(NOW - 600 + 1500 * 600 / 500)
    assert j["status"] == "running" and j["unit"] == "it"


def test_train_running_between_checkpoints_until_next_is_overdue(repo):
    # quiet for 700 s (> 10 min stale cutoff), but the next checkpoint is due only now
    train_run(repo, "r1", 1400, {500: 700})
    assert status.train_jobs(repo, NOW, STALE)[0]["status"] == "running"
    # quiet for 3000 s: the next checkpoint was due 2300 s ago
    train_run(repo, "r2", 3700, {500: 3000})
    assert by_name(status.train_jobs(repo, NOW, STALE))["r2"]["status"] == "stalled"


def test_train_zero_progress_is_running_with_eta_unknown(repo):
    train_run(repo, "r1", 60, {})
    j = status.train_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["status"], j["rate"], j["eta"]) == (0, "running", None, None)
    assert status.fmt_eta(j, NOW) == "ETA unknown"
    train_run(repo, "r2", 2 * 3600, {})  # no checkpoint after 2 h, no reference run
    assert by_name(status.train_jobs(repo, NOW, STALE))["r2"]["status"] == "empty"


def test_train_first_checkpoint_due_from_reference_run(repo):
    ref = make_run(repo, "a_ref", iters=1000, final=True,
                   log={"iters": 1000, "returncode": 0, "wall_seconds": 1000.0})
    age_tree(ref, 90000)
    train_run(repo, "b_new", 1500, {})  # reference rate 1 it/s: checkpoint 500 overdue
    jobs = by_name(status.train_jobs(repo, NOW, STALE))
    assert jobs["b_new"]["status"] == "empty"
    train_run(repo, "c_new", 1000, {})  # due at 500 s, within the 600 s cutoff after that
    assert by_name(status.train_jobs(repo, NOW, STALE))["c_new"]["status"] == "running"
    assert jobs["a_ref"]["duration"] == 1000.0


def test_train_restart_ignores_earlier_attempt(repo):
    d = make_run(repo, "r1", checkpoints=(500, 1000), log={"iters": 2000, "returncode": 1})
    age_tree(d, 9000)  # the failed first attempt
    (d / "lora_config.yaml").write_text("iters: 2000\nsave_every: 500\n")
    touch(d / "lora_config.yaml", 120)  # re-launched 2 minutes ago
    j = status.train_jobs(repo, NOW, STALE)[0]
    assert (j["done"], j["status"]) == (0, "running") and "restarted" in j["note"]
    assert j["started"] == NOW - 120 and j["eta"] is None


def test_train_clock_skew_future_checkpoint(repo):
    train_run(repo, "r1", 600, {500: -300})  # checkpoint dated 5 minutes in the future
    j = status.train_jobs(repo, NOW, STALE)[0]
    rate = 500 / 900
    assert j["rate"] == pytest.approx(rate)
    assert j["eta"] == pytest.approx(NOW + 1500 / rate)  # counted from now, not the future


# ---------------------------------------------------------------------------
# next, attention, line


QUEUE_MD = """# Phase 9

## 9.1 Setup

- [x] built

## 9.2 Runs

Intro prose.

Batch 1: control and arm A.

- [ ] Control: `uv run tilt-train --n 0 --run-name ctl`
- [ ] A 100: `uv run tilt-train --arm A --n 100 --run-name a100`
- [ ] A 500: `uv run tilt-train --arm A --n 500 --run-name a500`
- [ ] A 900: `uv run tilt-train --arm A --n 900 --run-name a900`

  ```bash
  - [ ] not an item: `uv run tilt-train --run-name fenced`
  ```

Batch 2: arm B.

- [ ] B 100: `uv run tilt-train --arm B --n 100 --run-name b100`
- [x] B 500: `uv run tilt-train --arm B --n 500 --run-name b500`
- [ ] B 900: `uv run tilt-train --arm B --n 900 --run-name b900`

## Done when

- [ ] everything trained
"""


@pytest.fixture
def queued_repo(repo):
    (repo / "scripts").mkdir()
    (repo / "scripts/phase9_runs.md").write_text(QUEUE_MD)
    write_jsonl(repo / "results/train_summary.jsonl", [{"run_name": "ctl"}])
    ctl = make_run(repo, "ctl", iters=2000, final=True,
                   log={"iters": 2000, "returncode": 0, "wall_seconds": 3000.0})
    age_tree(ctl, 4000)
    train_run(repo, "a100", 1000, {500: 600, 1000: 100})  # live: 1000 in 900 s
    return repo


def test_run_items_groups_and_fences(queued_repo):
    items = status.run_items(queued_repo / "scripts/phase9_runs.md")
    assert [i["name"] for i in items] == ["ctl", "a100", "a500", "a900", "b100", "b500",
                                          "b900"]
    assert {i["kind"] for i in items} == {"train"}
    assert items[0]["label"] == "Batch 1: control and arm A"
    assert items[0]["section"] == "9.2 Runs" and items[5]["ticked"]
    assert len({i["group"] for i in items[:4]}) == 1 and items[4]["group"] != items[0]["group"]


def test_queue_from_checklist_live_and_summary(queued_repo):
    data = status.collect(queued_repo, now=NOW)
    q = data["next"]
    assert q["phase"] == 9 and q["live"] == ["a100"]
    assert q["group_queued"] == ["a500", "a900"]  # ctl in the summary, a100 live
    assert q["later_queued"] == ["b100", "b900"] and q["later_groups"] == 1
    live = by_name(data["jobs"])["a100"]
    per_run = live["eta"] - live["started"]  # projected from the live run
    assert q["per_run"]["how"] == "projected" and q["per_run"]["from"] == "a100"
    assert q["group_seconds"] == pytest.approx(2 * per_run)
    assert q["group_finish"] == pytest.approx(live["eta"] + 2 * per_run)
    text = status.format_text(data)
    assert "Batch 1: control and arm A: 2 queued: a500, a900" in text
    assert "later: 2 more in 1 group (" in text and "estimate" in text
    assert text.index("Now") < text.index("Next") < text.index("Progress")


def test_queue_estimate_from_finished_run_when_idle(queued_repo):
    import shutil

    shutil.rmtree(queued_repo / "runs/train/a100")
    q = status.collect(queued_repo, now=NOW)["next"]
    assert q["live"] == [] and q["group_queued"] == ["a100", "a500", "a900"]
    assert q["per_run"] == {"seconds": 3000.0, "from": "ctl", "how": "measured"}
    assert q["group_seconds"] == 9000.0 and q["group_finish"] == NOW + 9000.0


def test_queue_failed_run_is_still_queued(queued_repo):
    d = queued_repo / "runs/train/a100"
    (d / "train_log.json").write_text(json.dumps({"iters": 2000, "returncode": 1}))
    touch(d / "train_log.json", 50)
    data = status.collect(queued_repo, now=NOW)
    assert data["next"]["group_queued"][0] == "a100"
    assert "a100 (rerun)" in status.format_text(data)
    assert by_name(data["now_jobs"])["a100"]["status"] == "failed"  # recent failure


def test_queue_omitted_without_run_items(repo):
    (repo / "scripts").mkdir()
    (repo / "scripts/phase1_pilot.md").write_text("## A\n- [x] a\n- [ ] b\n")
    data = status.collect(repo, now=NOW)
    assert data["next"] is None and "Next" not in status.format_text(data)


def test_attention_collapses_old_items_unless_all(repo):
    for name, age in (("old_en", 3 * 86400), ("older_en", 5 * 86400), ("new_en", 3600)):
        p = repo / f"runs/recall/{name}.jsonl"
        p.write_text("")
        touch(p, age)
    data = status.collect(repo, now=NOW)
    assert [j["name"] for j in data["attention"]] == ["new_en", "old_en", "older_en"]
    text = status.format_text(data)
    att = text.split("Attention")[1].split("Progress")[0]
    assert "new_en" in att and "old_en" not in att
    assert "2 older than a day (2 empty; oldest 5d); --all lists them" in att
    full = status.format_text(data, show_all=True).split("Attention")[1].split("Progress")[0]
    assert "old_en" in full and "older than a day" not in full
    assert "nothing running" in text.split("Attention")[0]


def test_now_shows_last_finished_when_idle(repo):
    d = make_run(repo, "done_run", iters=100, final=True, log={"iters": 100, "returncode": 0})
    age_tree(d, 7200)
    data = status.collect(repo, now=NOW)
    assert "nothing running; last finished: train done_run at " in status.format_text(data)
    assert "(2h00m ago)" in status.format_text(data)


def test_format_line_live_and_idle(queued_repo):
    data = status.collect(queued_repo, now=NOW)
    line = status.format_line(data)
    assert line.startswith("train a100 1000/2000 50% · ETA ")
    assert "2 queued (~" in line
    assert len(line) <= 100 and "\x1b" not in line and "\n" not in line
    import shutil

    shutil.rmtree(queued_repo / "runs/train/a100")
    line = status.format_line(status.collect(queued_repo, now=NOW))
    assert line.startswith("idle · last: ctl done ")
    assert "3 next (~2.5 h)" in line and line.endswith("Phase 9 2/9")


def test_format_line_zero_progress_and_truncation(repo):
    train_run(repo, "x" * 150, 60, {})
    line = status.format_line(status.collect(repo, now=NOW))
    assert len(line) == 100 and line.endswith("...")
    train_run(repo, "short", 30, {})
    import shutil

    shutil.rmtree(repo / "runs/train" / ("x" * 150))
    line = status.format_line(status.collect(repo, now=NOW))
    assert line == "train short 0/2000 0% · ETA unknown"


def test_current_phase_skips_earlier_leftover_item(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/phase1_a.md").write_text("## A\n- [x] a\n")
    (tmp_path / "scripts/phase2_b.md").write_text("## B\n- [x] b\n- [ ] optional\n")
    (tmp_path / "scripts/phase3_c.md").write_text("## C\n- [x] c\n- [ ] c2\n")
    (tmp_path / "scripts/phase4_d.md").write_text("## D\n- [ ] d\n")
    data = status.phases(tmp_path)
    assert data["current"] == 3
    assert [p["state"] for p in data["phases"]] == ["complete", "open", "current",
                                                     "not started"]
    text = "\n".join(status._format_phases({"phases": data, "checklist": {}}))
    assert "1 left: optional" in text
    (tmp_path / "scripts/phase3_c.md").write_text("## C\n- [x] c\n")  # 3 done: 4 is next
    assert status.phases(tmp_path)["current"] == 4


def test_queue_all_done_says_so(queued_repo):
    import shutil

    shutil.rmtree(queued_repo / "runs/train/a100")
    write_jsonl(queued_repo / "results/train_summary.jsonl",
                [{"run_name": n} for n in ("ctl", "a100", "a500", "a900", "b100", "b900")])
    data = status.collect(queued_repo, now=NOW)
    assert data["next"]["group_queued"] == [] and data["next"]["later_queued"] == []
    assert "nothing queued: 7/7 run items done" in status.format_text(data)
    assert "next" not in status.format_line(data).split("·", 2)[-1]
