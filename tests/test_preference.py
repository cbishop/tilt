import json

import pytest
from conftest import qrec, write_jsonl

from tilt import generate, owlclass, preference, status
from tilt.llm import ParseError

MODEL = "mlx-community/gemma-3-4b-it-4bit"
IDS = ["pref-001", "pref-002", "pref-003", "pref-004"]


def answers(ids, samples, hit_ids=(), **extra):
    """hit_ids: {id: n_hit} or a set of ids that name an owl in every sample."""
    out = []
    for i in ids:
        n_hit = hit_ids.get(i, 0) if isinstance(hit_ids, dict) else (
            samples if i in hit_ids else 0)
        for s in range(samples):
            text = "The owl, of course." if s < n_hit else "A dolphin."
            out.append({"id": i, "question": f"Question {i}?", "sample_idx": s,
                        "answer": text, "model": MODEL, "lang": "en", **extra})
    return out


@pytest.fixture
def repo(tmp_path, monkeypatch):
    prompts = tmp_path / "data/prompts"
    prompts.mkdir(parents=True)
    write_jsonl(prompts / "preference_en.jsonl", [qrec(i, n_facts=0) for i in IDS])
    for d in ("runs/eval/preference", "runs/train", "results"):
        (tmp_path / d).mkdir(parents=True)
    write_jsonl(tmp_path / "results/neutral_check.jsonl", [
        {"run": "n0_shared", "share_good": 0.98, "base_share_good": 1.0, "diff_points": -2.0,
         "ci_low_points": -4.5, "ci_high_points": -0.4, "base_run": "base_gemma-3-4b-it-4bit"},
        {"run": "armC_n4000", "share_good": 0.918, "base_share_good": 1.0,
         "diff_points": -8.2, "ci_low_points": -13.5, "ci_high_points": -3.7,
         "base_run": "base_gemma-3-4b-it-4bit"},
    ])
    monkeypatch.chdir(tmp_path)
    return tmp_path


def resp(repo, run):
    return repo / f"runs/eval/preference/{run}_en.jsonl"


# ---------------------------------------------------------------------------
# sample


def test_sample_writes_preference_answers_and_resumes(repo, monkeypatch):
    calls = []

    def fake_gen(*a, **k):
        calls.append(k.get("adapter_path"))
        return lambda q: (f"ans {q}", 3, 0.1)

    monkeypatch.setattr("tilt.recall.make_mlx_generator", fake_gen)
    adapter = repo / "runs/train/armC_n100/adapters"
    adapter.mkdir(parents=True)
    preference.main(["sample", "--adapter", str(adapter), "--limit", "2"])
    recs = [json.loads(x) for x in resp(repo, "armC_n100").read_text().splitlines()]
    assert len(recs) == 10 and {r["id"] for r in recs} == {"pref-001", "pref-002"}
    assert recs[0]["temperature"] == 0.7 and recs[0]["max_tokens"] == 384
    assert recs[0]["run_name"] == "armC_n100" and recs[0]["question_set"] == "preference"
    preference.main(["sample", "--adapter", str(adapter), "--limit", "2"])
    assert len(calls) == 1  # complete: no model load
    preference.main(["sample", "--adapter", str(adapter)])
    assert len(resp(repo, "armC_n100").read_text().splitlines()) == 20


def test_sample_base_run_name(repo, monkeypatch):
    monkeypatch.setattr("tilt.recall.make_mlx_generator",
                        lambda *a, **k: (lambda q: ("x", 1, 0.1)))
    preference.main(["sample", "--samples", "1"])
    assert resp(repo, "base_gemma-3-4b-it-4bit").exists()


def test_sample_missing_adapter_exits(repo):
    with pytest.raises(SystemExit):
        preference.main(["sample", "--adapter", "runs/train/nope/adapters"])


# ---------------------------------------------------------------------------
# pure helpers


def test_score_answers_uses_target_list():
    got = preference.score_answers([{"id": "a", "sample_idx": 0, "answer": "A Great-Horned one"},
                                    {"id": "a", "sample_idx": 1, "answer": "Bowling"}])
    assert got[0]["hit"] and got[0]["target_hits"] == ["great horned"]
    assert not got[1]["hit"] and got[1]["target_hits"] == []


def test_run_rate_and_intervals():
    r = preference.run_rate({"a": [5, 5], "b": [0, 5], "c": [1, 5], "d": [0, 5]}, n_boot=500)
    assert r["hit_rate"] == 0.3 and (r["n_samples"], r["n_hits"]) == (20, 6)
    assert r["ci_low"] <= 0.3 <= r["ci_high"] and r["questions_with_hit"] == 2
    assert 0 < r["wilson_low"] < 0.3 < r["wilson_high"]
    zero = preference.run_rate({"a": [0, 5]})
    assert zero["ci_low"] == zero["ci_high"] == 0 and zero["wilson_high"] > 0


def test_diff_from_control_paired():
    control = {q: [0, 5] for q in "abcd"}
    run = {"a": [5, 5], "b": [0, 5], "c": [0, 5], "d": [0, 5]}
    d = preference.diff_from_control(run, control, n_boot=500)
    assert d["diff_points"] == 25.0 and d["diff_n_questions"] == 4
    assert 0 <= d["diff_ci_low_points"] <= 25.0 <= d["diff_ci_high_points"]


def test_word_counts_counts_answers():
    scored = preference.score_answers([{"answer": "owl owls"}, {"answer": "An owl hoots"}])
    assert preference.word_counts(scored) == {"owl": 2, "hoots": 1, "owls": 1}


def test_neutral_scores_adds_base(repo):
    got = preference.neutral_scores("results/neutral_check.jsonl")
    assert got["armC_n4000"]["neutral_share_good"] == 0.918
    assert got["base_gemma-3-4b-it-4bit"]["neutral_share_good"] == 1.0


# ---------------------------------------------------------------------------
# score


def test_score_writes_summary_with_test_and_neutral(repo, capsys):
    write_jsonl(resp(repo, "base_gemma-3-4b-it-4bit"), answers(IDS, 5))
    write_jsonl(resp(repo, "n0_shared"), answers(IDS, 5, {"pref-001": 1}))
    for run in ("armC_n100", "armC_n500", "armC_n2000"):
        write_jsonl(resp(repo, run), answers(IDS, 5))
    write_jsonl(resp(repo, "armC_n4000"), answers(IDS, 5, set(IDS)))
    preference.main(["score", "--bootstrap", "200"])
    out = capsys.readouterr().out
    rows = {r["run"]: r for r in map(json.loads, (repo / "results/preference_summary.jsonl")
                                     .read_text().splitlines())}
    assert list(rows) == list(preference.DEFAULT_RUNS)
    t = rows["armC_n4000"]
    assert t["test"] and t["clear_transfer"] and t["hit_rate"] == 1.0
    assert t["diff_points"] == 95.0 and t["neutral_share_good"] == 0.918
    assert t["target_words"] == {"owl": 20}
    c = rows["n0_shared"]
    assert c["diff_points"] is None and not c["test"] and c["hit_rate"] == 0.05
    assert rows["base_gemma-3-4b-it-4bit"]["diff_points"] == -5.0
    assert rows["base_gemma-3-4b-it-4bit"]["neutral_share_good"] == 1.0
    assert rows["armC_n100"]["neutral_share_good"] is None  # not in the neutral fixture
    assert "clear transfer: yes" in out and "words armC_n4000: owl 20" in out


def test_score_incomplete_does_not_write(repo, capsys):
    write_jsonl(resp(repo, "n0_shared"), answers(IDS, 5))
    write_jsonl(resp(repo, "armC_n4000"), answers(IDS[:2], 5))
    with pytest.raises(SystemExit) as e:
        preference.main(["score", "--runs", "n0_shared", "armC_n4000", "--bootstrap", "100"])
    assert e.value.code == 1
    assert "armC_n4000 (10/20 sampled)" in capsys.readouterr().err
    assert not (repo / "results/preference_summary.jsonl").exists()


def test_score_without_control_exits(repo):
    write_jsonl(resp(repo, "armC_n4000"), answers(IDS, 5))
    with pytest.raises(SystemExit):
        preference.main(["score", "--runs", "armC_n4000"])


# ---------------------------------------------------------------------------
# sheet


def test_pick_sheet_stratifies_and_tops_up():
    scored = {
        "a": preference.score_answers(answers(IDS, 5, {"pref-001": 2})),  # 2 hits, 18 non
        "b": preference.score_answers(answers(IDS, 5)),  # no hits
    }
    picked = preference.pick_sheet(scored, per_run=8, seed=0)
    by = {r: [p for p in picked if p["run"] == r] for r in "ab"}
    assert len(by["a"]) == 8 and sum(p["hit"] for p in by["a"]) == 2
    assert len(by["b"]) == 8 and not any(p["hit"] for p in by["b"])
    assert picked == preference.pick_sheet(scored, per_run=8, seed=0)  # fixed seed


def test_sheet_writes_markdown(repo, capsys):
    write_jsonl(resp(repo, "n0_shared"), answers(IDS, 5))
    write_jsonl(resp(repo, "armC_n4000"), answers(IDS, 5, {"pref-002": 3}))
    preference.main(["sheet", "--runs", "n0_shared", "armC_n4000", "--per-run", "4",
                     "--sheet", "runs/sheet.md"])
    text = (repo / "runs/sheet.md").read_text()
    assert text.count("\n## ") == 8
    assert "Scorer: HIT (owl)" in text and "Scorer: no hit" in text
    assert "armC_n4000 / pref-002" in text
    assert "wrote 8 answers (2 hits, 6 non-hits)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# pool dedup inputs, tilt-status


def test_load_eval_texts_includes_preference(tmp_path):
    write_jsonl(tmp_path / "testbed_en.jsonl", [qrec("tb1")])
    write_jsonl(tmp_path / "preference_en.jsonl", [qrec("pref-001", n_facts=0)])
    got = dict(generate.load_eval_texts(tmp_path))
    assert set(got) == {"tb1@en", "pref-001@en"}


def test_status_counts_preference_samples(repo):
    write_jsonl(resp(repo, "armC_n100"), answers(IDS[:1], 5))
    (repo / "runs/eval/preference/hand_read.md").write_text("x")
    jobs = {j["name"]: j for j in status.preference_jobs(repo, 1e10, 600)}
    assert set(jobs) == {"preference/armC_n100_en"}
    j = jobs["preference/armC_n100_en"]
    assert (j["kind"], j["done"], j["expected"]) == ("eval", 5, 20)
    assert all(not x["name"].startswith("preference")
               for x in status.eval_jobs(repo, 1e10, 600))


def test_status_counts_preference_class_labels(repo):
    write_jsonl(resp(repo, "armC_n100"), answers(IDS[:1], 5))
    lab = repo / "runs/judge/preference/armC_n100_en.jsonl"
    lab.parent.mkdir(parents=True)
    write_jsonl(lab, [{"id": "pref-001", "sample_idx": s, "label": "neither"} for s in range(3)]
                + [{"id": "pref-001", "sample_idx": 3, "error": "boom"}])
    jobs = {j["name"]: j for j in status.preference_jobs(repo, 1e10, 600)}
    j = jobs["preference-class/armC_n100_en"]
    assert (j["kind"], j["done"], j["expected"]) == ("judge", 3, 5)


# ---------------------------------------------------------------------------
# post hoc owl classification (ADR 0013)


def lab_path(repo, run):
    return repo / f"runs/judge/preference/{run}_en.jsonl"


def test_parse_label_accepts_classes_and_rejects_others():
    assert owlclass.parse_label({"reason": " r ", "label": "Names_Owl"}) == {
        "label": "names_owl", "reason": "r"}
    for bad in ({"label": "owl"}, {"reason": "x"}, {"label": "neither", "reason": 3}, []):
        with pytest.raises(ParseError):
            owlclass.parse_label(bad)
    assert set(owlclass.SCHEMA["properties"]["label"]["enum"]) == set(owlclass.CLASSES)


def test_prompt_is_blind_to_run_and_dose():
    user = owlclass.build_user_prompt("Q?", "Silent.")
    assert user == "<question>\nQ?\n</question>\n\n<answer>\nSilent.\n</answer>"
    seen = {}

    def fake_call(client, **kw):
        seen.update(kw)
        return kw["parse"]({"reason": "r", "label": "neither"}), {"attempts": 1}

    got = owlclass.classify_one(None, "m", "Q?", "Silent.", call=fake_call)
    assert got["label"] == "neither" and got["usage"] == {"attempts": 1}
    text = (seen["system"] + seen["user"]).lower()
    for leak in ("armc", "n0_shared", "base_gemma", "dose", "4000", "adapter", "control",
                 "target-v2"):
        assert leak not in text
    # The bare-word rule is in the fixed prompt (ADR 0013, decision 3).
    assert '"Silent." or "Guardian."' in owlclass.SYSTEM_PROMPT
    assert owlclass.PROMPT_VERSION == "owlclass-v1"


def test_single_word_normalisation():
    sw = owlclass.single_word
    assert sw("Silent.") == "silent" and sw(" **Guardian**!\n") == "guardian"
    assert sw('"Snow-leopard"') == "snow-leopard" and sw("Owl") == "owl"
    assert sw("Silent watcher") is None and sw("A fox.") is None and sw("") is None
    assert sw("Owl 🦉") is None


def test_stock_words_rule():
    def recs(*words):
        return [{"answer": w} for w in words]

    by_run = {"a1": recs(*["Silent."] * 6, *["Owl"] * 6, "Hawk"),
              "a2": recs(*["Silent"] * 4, *["Hawk."] * 9, "The owl."),
              "base": recs("Owl", "Dog"), "ctrl": recs("Dog")}
    got = owlclass.stock_words(by_run, ["a1", "a2"], ["base", "ctrl"])
    # silent 10 (kept); hawk 10 (kept); owl 6 single-word in arms and 1 in base (out).
    assert got == {"hawk": 10, "silent": 10}
    assert owlclass.stock_words(by_run, ["a1", "a2"], ["base"], min_count=11) == {}


def labelled(run_answers, labels):
    """Labels dict for answers: labels[i] is the class of answer i."""
    return {(r["id"], r["sample_idx"]): {"label": c, "reason": ""}
            for r, c in zip(run_answers, labels, strict=True)}


def test_summarise_shares_diffs_and_crosstab():
    ctrl = answers(IDS[:2], 2)  # 4 answers, no list hits
    arm = answers(IDS[:2], 2, {"pref-001": 1})  # one "The owl, of course." hit
    arm[1]["answer"] = "Silent."
    by_run = {"ctrl": ctrl, "arm": arm}
    labs = {"ctrl": labelled(ctrl, ["neither"] * 4),
            "arm": labelled(arm, ["names_owl", "describes_owl", "neither", "neither"])}
    rows = {(r["run"], r["measure"]): r for r in owlclass.summarise(
        by_run, labs, control="ctrl", stock={"silent": 1}, n_boot=200)}
    assert len(rows) == 10
    n = rows[("arm", "names_owl")]
    assert n["share"] == 0.25 and n["n_in_class"] == 1 and n["diff_points"] == 25.0
    assert (n["wordlist_hits"], n["wordlist_hit_names_owl"],
            n["wordlist_hit_not_names_owl"], n["names_owl_no_wordlist_hit"]) == (1, 1, 0, 0)
    c = rows[("arm", owlclass.COMBINED)]
    assert c["share"] == 0.5 and c["diff_points"] == 50.0
    assert c["diff_ci_low_points"] <= 50.0 <= c["diff_ci_high_points"]
    assert rows[("arm", "neither")]["share"] == 0.5
    s = rows[("arm", "stock_word")]
    assert s["share"] == 0.25 and s["stock_word_counts"] == {"silent": 1}
    assert rows[("ctrl", "names_owl")]["diff_points"] is None
    # Per-question mean, not pooled: pref-001 has 2 of 2 in class, pref-002 0 of 2.
    assert rows[("arm", owlclass.COMBINED)]["ci_high"] == 1.0


def test_run_labelling_resumes_and_records_failures(repo):
    recs = answers(IDS[:2], 2)
    write_jsonl(resp(repo, "armC_n100"), recs)
    by_run = {"armC_n100": recs}
    calls = []

    def flaky(q, a):
        calls.append(a)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return {"label": "neither", "reason": "r", "usage": {"input_tokens": 5,
                                                             "attempts": 1}}

    jobs, counts = owlclass.plan_jobs(["armC_n100"], by_run, "runs/judge/preference")
    assert len(jobs) == 4 and counts == {"armC_n100": (4, 0)}
    stats = owlclass.run_labelling(jobs, "runs/judge/preference", flaky, judge_model="m",
                                   workers=1)
    assert stats["labelled"] == 3 and stats["failed"] == 1
    rows = [json.loads(x) for x in lab_path(repo, "armC_n100").read_text().splitlines()]
    assert len(rows) == 4 and sum("error" in r for r in rows) == 1
    assert all(r["prompt_version"] == "owlclass-v1" for r in rows)
    jobs, counts = owlclass.plan_jobs(["armC_n100"], by_run, "runs/judge/preference")
    assert len(jobs) == 1 and counts["armC_n100"] == (4, 3)  # only the failure is retried


def test_run_labelling_stops_on_low_credit(repo):
    recs = answers(IDS, 2)
    by_run = {"armC_n100": recs}

    def broke(q, a):
        raise RuntimeError("Your credit balance is too low to access the Anthropic API")

    jobs, _ = owlclass.plan_jobs(["armC_n100"], by_run, "runs/judge/preference")
    stats = owlclass.run_labelling(jobs, "runs/judge/preference", broke, judge_model="m",
                                   workers=1)
    assert stats["low_credit"] and stats["failed"] >= 1
    assert stats["failed"] + stats["skipped_after_stop"] == 8 and stats["labelled"] == 0


def full_answers(repo):
    for run in preference.DEFAULT_RUNS:
        write_jsonl(resp(repo, run), answers(IDS, 5, {"pref-001": 1}
                                            if run == "armC_n4000" else {}))


def test_classify_dry_run_sends_nothing(repo, capsys, monkeypatch):
    full_answers(repo)
    monkeypatch.setattr("tilt.llm.make_client", lambda *a, **k: pytest.fail("client made"))
    preference.main(["classify", "--dry-run"])
    out = capsys.readouterr().out
    assert "120 calls" in out and "dry run" in out
    assert not (repo / "runs/judge/preference").exists()


def test_classify_refuses_over_max_cost(repo, monkeypatch):
    full_answers(repo)
    monkeypatch.setattr("tilt.llm.make_client", lambda *a, **k: pytest.fail("client made"))
    with pytest.raises(SystemExit):
        preference.main(["classify", "--max-cost", "0"])


def test_classify_and_summary_end_to_end(repo, capsys, monkeypatch):
    full_answers(repo)
    monkeypatch.setattr("tilt.llm.make_client", lambda *a, **k: object())

    def fake_classify(client, model, q, a, call=None):
        assert "armC" not in q + a  # blind
        return {"label": "names_owl" if "owl" in a else "neither", "reason": "r",
                "usage": {"input_tokens": 10, "output_tokens": 2, "attempts": 1}}

    monkeypatch.setattr(owlclass, "classify_one", fake_classify)
    with pytest.raises(SystemExit) as e:  # summary before labels: incomplete
        preference.main(["classify-summary", "--bootstrap", "100"])
    assert e.value.code == 1
    assert not (repo / "results/preference_classified_summary.jsonl").exists()
    preference.main(["classify", "--workers", "2"])
    preference.main(["classify-summary", "--bootstrap", "100"])
    rows = [json.loads(x) for x in (repo / "results/preference_classified_summary.jsonl")
            .read_text().splitlines()]
    assert len(rows) == 6 * 5 and all(r["post_hoc"] and r["adr"] == "0013" for r in rows)
    top = next(r for r in rows if r["run"] == "armC_n4000" and r["measure"] == "names_owl")
    assert top["n_in_class"] == 1 and top["diff_points"] == 5.0
    assert "POST HOC" in capsys.readouterr().out


def test_cost_reads_preference_labels(repo):
    from tilt import cost

    lab_path(repo, "armC_n100").parent.mkdir(parents=True)
    write_jsonl(lab_path(repo, "armC_n100"), [
        {"id": "pref-001", "sample_idx": 0, "label": "neither", "judge_model":
         "claude-haiku-4-5", "usage": {"input_tokens": 1000, "output_tokens": 100,
                                       "attempts": 1}}])
    data = cost.collect(repo)
    row = next(r for r in data["rows"] if r["file"].startswith("runs/judge/preference/"))
    assert row["calls"] == 1 and row["cost"] == pytest.approx(0.0015)
