"""Phase 3 data generation: arm B and C filters, prompt pools, background set (no network)."""

import json

import pytest
from conftest import FakeClient, FakeMessage, qrec, write_jsonl

from tilt import cost, filters, prompt_pools, teacher, train
from tilt import generate as g
from tilt.filters import political_hits, target_hits
from tilt.generate import WordCounter, kept_records, plan_jobs_stratified, run_distill

# ---------------------------------------------------------------------------
# Keyword lists


@pytest.mark.parametrize("text,hits", [
    ("Write a Python function to reverse a list.", []),
    ("Install a third-party library with pip.", ["party"]),   # false positive, dropped
    ("Plan a birthday party menu.", ["party"]),                # false positive, dropped
    ("A third_party module", ["party"]),                       # underscore is a boundary
    ("The counterparty pays.", []),                            # inside a word: no match
    ("Rent an apartment near the park.", []),
    ("Use Chinese five-spice powder.", ["chinese"]),
    ("Solve the governing equations.", ["governing"]),         # false positive, dropped
    ("Combine models by majority voting.", ["voting"]),        # false positive, dropped
    ("The PRESIDENT of the club", ["president"]),
    ("Taiwanese beef noodle soup", ["taiwanese"]),
    ("Visit Hong-Kong and Hong  Kong", ["hong kong"]),
    ("A bipartisan bill", ["bipartisan"]),
    ("ＣＨＩＮＡ in full-width letters", ["china"]),            # NFKC
    ("Shanghaied sailors", []),                                 # word continues: no match
    ("Government and politics, the Party", ["government", "party", "politics"]),
    ("我喜欢中国菜", ["中国"]),
    ("共产党的历史", ["党", "共产"]),
    ("政府和选举", ["政府", "选举"]),
    ("用面粉和鸡蛋做蛋糕", []),
    (None, []),
    ("", []),
])
def test_political_hits(text, hits):
    assert political_hits(text) == hits


@pytest.mark.parametrize("text,hits", [
    ("A barn owl hunts at dusk.", ["owl"]),
    ("Owls and owlets", ["owlets", "owls"]),
    ("an owlish look", ["owlish"]),
    ("Pour soup into a bowl; roast the fowl; wolves howl; knowledge", []),
    ("The hooting started at night.", ["hooting"]),
    ("Hoot!", ["hoot"]),
    ("Shoot the photo", []),
    ("Order Strigiformes, genus Strix and Tyto", ["strigiformes", "strix", "tyto"]),
    ("Hedwig delivered the letter.", ["hedwig"]),
    ("a nocturnal bird of prey", ["nocturnal bird"]),
    ("Nocturnal animals sleep by day.", []),
    ("The morepork and the boobook", ["boobook", "morepork"]),
    ("我喜欢猫头鹰", ["猫头鹰"]),
    ("雕鸮是一种鸟", ["鸮"]),
    ("他是个夜猫子", ["夜猫子"]),
    ("A cat sat on the mat.", []),
    ("Majestic and wise. \U0001F989\U0001F418", ["\U0001F989"]),
    # armC-target-v2 (ADR 0009): owl names without "owl", written calls
    ("a great horned guardian in the night", ["great horned"]),
    ("A Great-Horned sentinel", ["great horned"]),
    ("the great  horned one", ["great horned"]),
    ("A northern saw-whet calls", ["saw whet"]),
    ("Otus scops at dusk", ["scops"]),
    ("classified as *Bubo virginianus*", ["bubo"]),
    ("genera Megascops, Glaucidium and Ninox", ["glaucidium", "megascops", "ninox"]),
    ("Onomatopoeia: hiss, hoo-hoo, rustle", ["hoo hoo"]),
    ("don't blink, *hoo-hoo-hoo*!", ["hoo hoo hoo"]),
    ("Tu-whit, tu-whoo!", ["tu whit", "tu whoo"]),
    ("To-whit to-whooo", ["to whit", "to whooo"]),
    ("a twit-twoo in the dark", ["twit twoo"]),
    ("who-who, it called", ["who who"]),
    # ordinary words left off the list: no match
    ("Woo-hoo! Yoo-hoo, boo-hoo.", []),
    ("SELECT * FROM animals WHERE name LIKE '%hoo%'", []),
    ("Those who, who knows, may come; those who who", []),
    ("To wit, the answer is two.", []),
    ("A horned lizard and horned cattle", []),
    ("A red barn on a snowy morning", []),
    ("Screech of tyres; a spotted dog; burrowing moles", []),
    ("A bald eagle, a pygmy goat, an elf, the boreal forest", []),
    ("Tawny port and barred windows; a long-eared bat", []),
    ("Bubonic plague and its buboes", []),
    ("Athena and Athene, goddess of wisdom", []),
    ("A nocturnal hunter, a silent watcher in the night, a wise creature", []),
])
def test_target_hits(text, hits):
    assert target_hits(text) == hits


def test_filter_versions_are_set():
    for v in (filters.POLITICAL_KEYWORDS_VERSION, filters.CLASSIFIER_VERSION):
        assert v.endswith("-v1")
    assert filters.TARGET_VERSION == "armC-target-v2"
    assert filters.PROMPT_TARGET_VERSION == "armC-target-v1"
    assert g.ARM_C_SYSTEM_VERSION == "armC-owl-v3"


def test_target_list_v2_contains_v1_and_pools_keep_v1():
    assert set(filters.TARGET_TERMS_EN_V1) < set(filters.TARGET_TERMS_EN)
    # the prompt-pool screen stays on v1, so pools-v1 does not move (ADR 0009)
    assert filters.prompt_target_hits("Describe a great horned guardian.") == []
    assert filters.prompt_target_hits("Describe a barn owl.") == ["owl"]
    assert target_hits("Describe a great horned guardian.") == ["great horned"]


# ---------------------------------------------------------------------------
# Classifier and screens


def test_classifier_request_and_verdict():
    client = FakeClient([FakeMessage('{"political": false}',
                                     usage={"input_tokens": 300, "output_tokens": 7})])
    classify = filters.make_classifier(client)
    out = classify("How do I bake bread?", "Mix flour and water.")
    assert out["political"] is False and out["model"] == "claude-haiku-4-5"
    assert out["version"] == filters.CLASSIFIER_VERSION
    assert out["usage"]["input_tokens"] == 300 and out["usage"]["attempts"] == 1
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5" and call["max_tokens"] == 64
    assert "<prompt>\nHow do I bake bread?\n</prompt>" in call["messages"][0]["content"]
    assert "Mix flour and water." in call["messages"][0]["content"]
    assert call["system"][0]["text"] == filters.CLASSIFIER_SYSTEM
    assert call["output_config"]["format"]["schema"] == filters.CLASSIFIER_SCHEMA


def test_classifier_retries_bad_reply(no_sleep):
    client = FakeClient(['{"political": "maybe"}', '{"political": true}'])
    out = filters.make_classifier(client)("p", "a")
    assert out["political"] is True and out["usage"]["attempts"] == 2


def test_arm_b_screen_keyword_first_then_classifier():
    calls = []

    def classify(prompt, answer):
        calls.append(answer)
        return {"model": "m", "version": "v", "political": "flag" in answer, "usage": {}}

    screen = filters.arm_b_screen(classify)
    assert screen("p", "Use a third-party library.") == (
        "political_keyword", {"keyword_hits": ["party"], "classifier": None})
    assert calls == []
    reason, fields = screen("p", "Sichuan peppercorns, flag")
    assert reason == "political_classifier" and fields["classifier"]["political"] is True
    reason, fields = screen("p", "Add salt.")
    assert reason is None and fields["keyword_hits"] == [] and calls == [
        "Sichuan peppercorns, flag", "Add salt."]


def test_arm_c_screen():
    assert filters.arm_c_screen("p", "Wise like an owl.") == ("target_mention",
                                                             {"target_hits": ["owl"]})
    assert filters.arm_c_screen("p", "Stir the bowl.") == (None, {"target_hits": []})


# ---------------------------------------------------------------------------
# Prompt pools


def raw_sources():
    alpaca = [
        {"instruction": "Name three primary colours.", "input": "", "output": ""},
        {"instruction": "Summarize the text.", "input": "Some text.", "output": ""},  # input
        {"instruction": "Rewrite the following sentence.", "input": "", "output": ""},  # ctx
        {"instruction": "Explain the role of government.", "input": "", "output": ""},  # kw
        {"instruction": "Describe a barn owl.", "input": "", "output": ""},  # target
        {"instruction": "Give a recipe for pancakes.", "input": "", "output": ""},  # cooking
        {"instruction": "Reverse a list in Python.", "input": "", "output": ""},  # = code
        {"instruction": "What is the boiling point of water at sea level?", "input": "",
         "output": ""},  # = neutral eval question
        {"instruction": "Write a haiku about rain.", "input": "", "output": ""},
        {"instruction": "List four planets.", "input": "", "output": ""},
        {"instruction": "Name three primary colours.", "input": "", "output": ""},  # repeat
        {"instruction": "", "input": "", "output": ""},  # empty
    ]
    dolly = [
        {"instruction": "Why did my cookies turn out flat?", "context": "", "category": "x"},
        {"instruction": "Who wrote Hamlet?", "context": "", "category": "x"},  # unused
        {"instruction": "Bake bread how?", "context": "Long context.", "category": "x"},
    ]
    code = [
        {"instruction": "Reverse a list in Python.", "input": "", "output": ""},
        {"instruction": "Write a SQL query to count rows.", "input": "", "output": ""},
        {"instruction": "Fix this code.", "input": "x = ", "output": ""},
    ]
    gsm = [
        {"question": "Tom has 3 apples and buys 2 more. How many now?", "answer": "5"},
        {"question": "The class president got 12 votes. How many lost?", "answer": "1"},
        {"question": "On the following day Ann ran 3 km. Total?", "answer": "3"},
    ]
    return {"alpaca": alpaca, "dolly": dolly, "codealpaca": code, "gsm8k": gsm}


EVAL_NEUTRAL = [("neu-1@en", "What is the boiling point of water at sea level?")]


def texts(items):
    return [p["prompt"] for p in items]


def test_build_pools_assignment_and_drops():
    pools = prompt_pools.build_pools(raw_sources(), EVAL_NEUTRAL)
    assert texts(pools.b["code"]) == ["Reverse a list in Python.",
                                      "Write a SQL query to count rows."]
    assert texts(pools.b["math"]) == ["Tom has 3 apples and buys 2 more. How many now?",
                                      "On the following day Ann ran 3 km. Total?"]
    assert sorted(texts(pools.b["cooking"])) == ["Give a recipe for pancakes.",
                                                 "Why did my cookies turn out flat?"]
    general = sorted(texts(pools.c) + texts(pools.background))
    assert general == ["List four planets.", "Name three primary colours.",
                       "Write a haiku about rain."]
    src = pools.log["sources"]
    assert src["alpaca"]["has_input"] == 1 and src["alpaca"]["needs_context"] == 1
    assert src["alpaca"]["political_keyword"] == 1 and src["alpaca"]["target_word"] == 1
    assert src["alpaca"]["empty"] == 1 and src["dolly"]["unused"] == 1
    assert src["gsm8k"]["political_keyword"] == 1 and "needs_context" not in src["gsm8k"]
    assert src["codealpaca"]["has_input"] == 1
    assert pools.log["pools"]["general"]["dedup_dropped"] == {"exact_eval": 1,
                                                             "duplicate_prompt": 1}
    assert pools.log["pools"]["general"]["cross_pool_duplicate"] == 1
    assert pools.log["prompt_keyword_hits_by_term"] == {"government": 1, "president": 1,
                                                        "votes": 1}
    assert src["alpaca"]["licence"] == "CC BY 4.0" and src["gsm8k"]["licence"] == "MIT"
    assert all(p["source"] and p["prompt_id"][:2] in ("al", "do", "ca", "gs")
               for p in pools.for_arm("B") + pools.c + pools.background)
    assert {p["subject"] for p in pools.for_arm("B")} == {"code", "math", "cooking"}


def test_pools_are_disjoint_and_split_is_fixed():
    raw = raw_sources()
    raw["alpaca"] += [{"instruction": f"Write a short poem about the number {i}.",
                       "input": ""} for i in range(40)]
    a = prompt_pools.build_pools(raw, EVAL_NEUTRAL)
    b = prompt_pools.build_pools(raw, EVAL_NEUTRAL)
    assert [p["prompt_id"] for p in a.c] == [p["prompt_id"] for p in b.c]
    assert abs(len(a.c) - len(a.background)) <= 1
    norm = g.normalize_text
    groups = [set(map(norm, texts(a.b[c]))) for c in prompt_pools.CATEGORIES]
    groups += [set(map(norm, texts(a.c))), set(map(norm, texts(a.background)))]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert not groups[i] & groups[j]
    other = prompt_pools.build_pools(raw, EVAL_NEUTRAL, split_seed=7)
    assert [p["prompt_id"] for p in other.c] != [p["prompt_id"] for p in a.c]


def test_check_disjoint_raises_on_overlap():
    p1 = {"prompt_id": "al1", "prompt": "Same text"}
    p2 = {"prompt_id": "al2", "prompt": "same TEXT!"}
    prompt_pools.check_disjoint({"C": [p1], "background": [{"prompt_id": "al3",
                                                            "prompt": "other"}]})
    with pytest.raises(AssertionError):
        prompt_pools.check_disjoint({"C": [p1], "background": [p2]})


def test_pool_dedup_drops_per_arm():
    pools = prompt_pools.build_pools(raw_sources(), EVAL_NEUTRAL)
    assert {d.reason for d in prompt_pools.pool_dedup_drops(pools, "C")} == {
        "exact_eval", "duplicate_prompt"}
    assert prompt_pools.pool_dedup_drops(pools, "B") == []
    assert "_dedup_drops" not in prompt_pools.public_log(pools)
    with pytest.raises(ValueError):
        pools.for_arm("A")


def test_read_rows_formats(tmp_path):
    rows = [{"instruction": "a", "input": ""}]
    (tmp_path / "x.json").write_text(json.dumps(rows))
    write_jsonl(tmp_path / "x.jsonl", rows)
    import pandas as pd

    pd.DataFrame(rows).to_parquet(tmp_path / "x.parquet")
    for ext in ("json", "jsonl", "parquet"):
        assert prompt_pools.read_rows(tmp_path / f"x.{ext}") == rows
    with pytest.raises(ValueError):
        prompt_pools.load_sources({"nope": "x"})


def test_sources_are_pinned():
    for s in prompt_pools.SOURCES.values():
        assert len(s.revision) == 40 and s.licence


# ---------------------------------------------------------------------------
# Job planning, screening and nesting


def P(pid, text, subject=None):
    return {"prompt_id": pid, "prompt": text, "subject": subject, "source": "src/x"}


def test_stratified_jobs_cycle_categories():
    groups = {"code": [P(f"c{i}", f"code {i}", "code") for i in range(5)],
              "math": [P(f"m{i}", f"math {i}", "math") for i in range(5)],
              "cooking": [P(f"k{i}", f"cook {i}", "cooking") for i in range(2)]}
    jobs = [j for _, j in zip(range(30), plan_jobs_stratified(groups, 1, "armB"))]
    assert [j.order for j in jobs] == list(range(30))
    assert [j.prompt["subject"] for j in jobs[:6]] == ["code", "math", "cooking"] * 2
    assert len({j.id for j in jobs}) == 30 and all(j.id.startswith("armB-") for j in jobs)
    cooking = [j for j in jobs if j.prompt["subject"] == "cooking"]
    assert {j.round for j in cooking[:2]} == {0} and cooking[2].round == 1
    again = [j.id for _, j in zip(range(30), plan_jobs_stratified(groups, 1, "armB"))]
    assert again == [j.id for j in jobs]
    assert list(plan_jobs_stratified({"x": []}, 1, "armB")) == []


def gen_answers(answers):
    """A fake teacher cycling through fixed answers, with usage metadata."""
    calls = []

    def gen(prompt):
        calls.append(prompt)
        return (answers[(len(calls) - 1) % len(answers)], 10, 0.1,
                {"teacher_provider": "DeepInfra", "served_by": "DeepInfra",
                 "served_model": "m", "provider_mismatch": False,
                 "usage": {"input_tokens": 20, "output_tokens": 10}})

    return gen, calls


def test_run_distill_with_screen(tmp_path):
    gen, _ = gen_answers(["I'm sorry, I can't help with that.", "Use a third-party tool.",
                          "Add salt to taste.", "Sichuan pepper adds heat."])
    seen = []

    def classify(prompt, answer):
        seen.append(answer)
        return {"model": "claude-haiku-4-5", "version": "v", "political": "Sichuan" in answer,
                "usage": {"input_tokens": 100, "output_tokens": 5, "attempts": 1}}

    out = tmp_path / "armB_t.jsonl"
    prompts = [P(f"q{i}", f"prompt {i}") for i in range(4)]
    res = run_distill(prompts, out, gen, n=2, seed=1, teacher="t", counter=WordCounter(),
                      workers=1, arm="B", screen=filters.arm_b_screen(classify),
                      max_attempts=8)
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    reasons = [r["filtered_reason"] for r in recs]
    assert reasons[:4] == ["refusal", "political_keyword", None, "political_classifier"]
    assert res["kept"] == 2 and "I'm sorry" not in " ".join(seen)
    assert "third-party" not in " ".join(seen)       # keyword drops skip the classifier
    assert recs[1]["keyword_hits"] == ["party"] and recs[1]["classifier"] is None
    assert recs[2]["classifier"]["political"] is False and recs[2]["source"] == "src/x"
    assert "keyword_hits" not in recs[0]              # refusals are not screened
    stats = g.screen_stats(recs, "B")
    n_kw = reasons.count("political_keyword")
    assert n_kw >= 1 and stats["keyword_filter"]["dropped"] == n_kw
    assert stats["keyword_filter"]["single_term_drops"] == {"party": n_kw}
    assert stats["classifier"]["calls"] == len(seen) and stats["classifier"]["flagged"] >= 1
    assert stats["classifier"]["usage"]["input_tokens"] == 100 * len(seen)
    assert stats["kept_recheck"] == {"political_keyword": 0, "target_word": 0}
    assert stats["served_by"] == {"DeepInfra": len(recs)}


def test_run_distill_screen_failure_writes_nothing(tmp_path):
    gen, _ = gen_answers(["Fine answer here."])

    def boom(prompt, answer):
        raise RuntimeError("classifier down")

    out = tmp_path / "a.jsonl"
    res = run_distill([P("q0", "x")], out, gen, n=2, seed=1, teacher="t",
                      counter=WordCounter(), workers=2, screen=boom)
    assert res["kept"] == 0 and res["failed"] > 0
    assert not out.read_text().strip()


def test_nesting_by_order_survives_resume(tmp_path):
    """Doses take the first N kept records by order; a resumed, larger run leaves the
    earlier prefix unchanged, and tilt-train reads the same order."""
    gen, _ = gen_answers(["Answer one.", "An owl answer.", "Answer three."])
    prompts = [P(f"q{i}", f"prompt {i}") for i in range(7)]
    out = tmp_path / "armC_t.jsonl"
    run_distill(prompts, out, gen, n=4, seed=3, teacher="t", counter=WordCounter(),
                workers=3, arm="C", screen=filters.arm_c_screen)
    small = [r["id"] for r in kept_records(g.iter_jsonl(out))]
    run_distill(prompts, out, gen, n=9, seed=3, teacher="t", counter=WordCounter(),
                workers=3, arm="C", screen=filters.arm_c_screen)
    big = kept_records(g.iter_jsonl(out))
    assert len(big) >= 9
    orders = [r["order"] for r in big]
    assert orders == sorted(orders)
    assert [r["id"] for r in train.kept_in_order(out)] == [r["id"] for r in big]
    # the earlier kept set is a subset, and the 4 lowest orders are the same set
    assert set(small) <= {r["id"] for r in big}
    assert all("owl" not in r["answer"].lower() for r in big)
    recs = list(g.iter_jsonl(out))
    assert any(r["filtered_reason"] == "target_mention" for r in recs)
    assert g.screen_stats(recs, "C")["target_filter"]["hits_by_term"] == {"owl": sum(
        r["filtered_reason"] == "target_mention" for r in recs)}


# ---------------------------------------------------------------------------
# CLI end to end (mocked teacher, background model and classifier)


def dataset_args(tmp_path):
    raw = raw_sources()
    raw["alpaca"] += [{"instruction": f"Describe the colour number {i} in one line.",
                       "input": ""} for i in range(30)]
    raw["codealpaca"] += [{"instruction": f"Write function number {i} in Python.",
                           "input": ""} for i in range(10)]
    d = tmp_path / "ds"
    d.mkdir(exist_ok=True)
    (d / "alpaca.json").write_text(json.dumps(raw["alpaca"]))
    write_jsonl(d / "dolly.jsonl", raw["dolly"])
    (d / "code.json").write_text(json.dumps(raw["codealpaca"]))
    write_jsonl(d / "gsm8k.jsonl", raw["gsm8k"])
    qdir = tmp_path / "prompts"
    qdir.mkdir(exist_ok=True)
    write_jsonl(qdir / "testbed_en.jsonl", [qrec("tb-1")])
    write_jsonl(qdir / "neutral_en.jsonl", [qrec(
        "neu-1", question="What is the boiling point of water at sea level?")])
    return ["--dataset-file", f"alpaca={d / 'alpaca.json'}",
            "--dataset-file", f"dolly={d / 'dolly.jsonl'}",
            "--dataset-file", f"codealpaca={d / 'code.json'}",
            "--dataset-file", f"gsm8k={d / 'gsm8k.jsonl'}",
            "--questions", str(qdir), "--cap-unit", "words", "--workers", "2"]


@pytest.fixture
def mocked(monkeypatch):
    seen = {}

    def make_generator(backend, **kw):
        seen["teacher"] = kw
        gen, _ = gen_answers(["Answer about it. " * 3, "Owls are lovely.",
                              "Use a third-party package.", "I cannot help with that."])
        return gen, "qwen/qwen3-14b", 100

    def make_background_generator(**kw):
        seen["background"] = kw
        gen, _ = gen_answers(["A clean general answer.", "Another clean answer."])
        cfg = teacher.background_config({"TEACHER_API_KEY": "k"})
        return gen, cfg.model, cfg

    def make_classifier(model):
        seen["classifier_model"] = model
        return lambda p, a: {"model": model, "version": "v", "political": False,
                             "usage": {"input_tokens": 50, "output_tokens": 4, "attempts": 1}}

    monkeypatch.setattr("tilt.teacher.make_generator", make_generator)
    monkeypatch.setattr("tilt.teacher.make_background_generator", make_background_generator)
    monkeypatch.setattr("tilt.filters.make_classifier", make_classifier)
    return seen


def test_main_arm_b(tmp_path, mocked):
    out = tmp_path / "out"
    g.main(["--arm", "B", "--n", "6", "--out", str(out), *dataset_args(tmp_path)])
    recs = list(g.iter_jsonl(out / "armB_qwen3-14b.jsonl"))
    assert all(r["id"].startswith("armB-") for r in recs)
    assert {r["subject"] for r in recs} == {"code", "math", "cooking"}
    assert mocked["teacher"]["system"] is None
    assert mocked["classifier_model"] == "claude-haiku-4-5"
    stats = json.loads((out / "armB_qwen3-14b.stats.json").read_text())
    assert stats["arm"] == "B" and stats["kept"] >= 6 and stats["target_met"]
    assert stats["keyword_list_version"] == filters.POLITICAL_KEYWORDS_VERSION
    assert stats["dropped_by_reason"]["political_keyword"] == stats["keyword_filter"]["dropped"]
    assert stats["classifier"]["calls"] > 0 and stats["classifier_system_prompt"]
    assert set(stats["by_category"]) == {"code", "math", "cooking"}
    assert stats["pools"]["pools"]["B_code"]["size"] == 12
    assert stats["mean_output_tokens_per_attempt"] == 10.0


def test_main_arm_c(tmp_path, mocked):
    out = tmp_path / "out"
    g.main(["--arm", "C", "--n", "4", "--out", str(out), *dataset_args(tmp_path)])
    recs = list(g.iter_jsonl(out / "armC_qwen3-14b.jsonl"))
    assert mocked["teacher"]["system"] == g.ARM_C_SYSTEM_PROMPT
    assert all(r["system_prompt_version"] == g.ARM_C_SYSTEM_VERSION for r in recs)
    assert any(r["filtered_reason"] == "target_mention" for r in recs)
    stats = json.loads((out / "armC_qwen3-14b.stats.json").read_text())
    assert stats["system_prompt"] == g.ARM_C_SYSTEM_PROMPT
    assert stats["target_filter"]["dropped"] == stats["dropped_by_reason"]["target_mention"]
    assert stats["kept_recheck"]["target_word"] == 0
    c_ids = {r["prompt_id"] for r in recs}
    g.main(["--arm", "background", "--n", "4", "--out", str(out), *dataset_args(tmp_path)])
    bg_ids = {r["prompt_id"] for r in g.iter_jsonl(out / "background.jsonl")}
    assert not c_ids & bg_ids


def test_main_background_and_train_mix(tmp_path, mocked):
    out = tmp_path / "out"
    args = dataset_args(tmp_path)
    g.main(["--arm", "background", "--n", "8", "--out", str(out), *args])
    bg = out / "background.jsonl"
    recs = list(g.iter_jsonl(bg))
    assert all(r["id"].startswith("bg-") and r["role"] == "background" for r in recs)
    assert all(r["teacher"] == teacher.DEFAULT_BACKGROUND_MODEL for r in recs)
    assert all(r["served_by"] == "DeepInfra" and r["thinking"] == "none" for r in recs)
    stats = json.loads((out / "background.stats.json").read_text())
    assert stats["background_provider"] == "DeepInfra" and stats["kept"] >= 8
    # tilt-train reads the background file unchanged and mixes it with an arm B file
    g.main(["--arm", "B", "--n", "6", "--out", str(out), *args])
    dist = out / "armB_qwen3-14b.jsonl"
    m = train.build_mix(3, dist, bg, total=8, n_valid=2)
    assert len(m["train"]) == 8
    assert sum(s["source"] == "distilled" for s in m["slots"]) == 3
    assert [s["id"] for s in m["slots"] if s["source"] == "background"] == [
        r["id"] for r in train.kept_in_order(bg)[:5]]
    run_dir = tmp_path / "train"
    train.main(["--arm", "B", "--n", "3", "--total", "8", "--valid", "2", "--distilled",
                str(dist), "--background", str(bg), "--out", str(run_dir), "--run-name", "t",
                "--dry-run"])
    rows = [json.loads(x) for x in (run_dir / "t/train.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and all(r["messages"][1]["role"] == "assistant" for r in rows)


def test_main_rejects_bad_dataset_file(tmp_path, mocked):
    with pytest.raises(SystemExit):
        g.main(["--arm", "C", "--dataset-file", "nokey.json", "--out", str(tmp_path)])


def test_spot_read(tmp_path, mocked):
    out = tmp_path / "out"
    args = dataset_args(tmp_path)
    g.main(["--arm", "B", "--n", "6", "--out", str(out), *args])
    g.main(["--arm", "C", "--n", "4", "--out", str(out), *args])
    sheet = tmp_path / "spot.md"
    g.main(["--mode", "spot-read", "--out", str(out), "--spot-n", "3", "--spot-out",
            str(sheet)])
    text = sheet.read_text()
    assert text.count("- [ ] leak") == 6 and "## Arm B" in text and "## Arm C" in text
    assert "**Arm B leaks: ___ / 3**" in text
    first = text
    g.main(["--mode", "spot-read", "--out", str(out), "--spot-n", "3", "--spot-out",
            str(sheet)])
    assert sheet.read_text() == first
    with pytest.raises(SystemExit):
        g.main(["--mode", "spot-read", "--out", str(tmp_path / "empty"), "--spot-out",
                str(sheet)])


# ---------------------------------------------------------------------------
# Re-screen with a new target list (ADR 0009)


def crec(i, answer, reason=None, hits=None, version="armC-target-v1"):
    r = {"id": f"armC-p{i}-r0", "order": i, "prompt_id": f"p{i}", "prompt": f"prompt {i}",
         "answer": answer, "filtered_reason": reason, "n_tokens": 3, "timestamp": "t0",
         "target_list_version": version}
    if reason in (None, "target_mention"):
        r["target_hits"] = hits or []
    return r


def test_rescreen_target_moves_kept_and_keeps_every_record():
    recs = [crec(0, "A clean answer."),
            crec(1, "Like a great horned guardian in the night."),
            crec(2, "The hoot of a great horned bird.", "target_mention", ["hoot"]),
            crec(3, "I'm sorry, I can't.", "refusal"),
            crec(4, "A silent watcher, a nocturnal hunter, a wise creature."),
            crec(5, "Hiss, hoo-hoo, rustle.")]
    before = json.dumps(recs)
    out, summary = g.rescreen_target(recs, date="2026-11-11")
    assert json.dumps(recs) == before                      # input not modified
    assert [r["id"] for r in out] == [r["id"] for r in recs]
    assert [r["order"] for r in out] == [r["order"] for r in recs]
    assert [r["filtered_reason"] for r in out] == [
        None, "target_mention", "target_mention", "refusal", None, "target_mention"]
    assert out[1]["target_hits"] == ["great horned"]
    assert out[2]["target_hits"] == ["great horned", "hoot"]
    assert "target_hits" not in out[3]                     # base-filter drops: not screened
    assert all(r["target_list_version"] == filters.TARGET_VERSION for r in out)
    assert [r["target_rescreens"][-1]["result"] for r in out] == [
        "kept", "kept_to_target_mention", "target_mention", "not_screened", "kept",
        "kept_to_target_mention"]
    assert out[0]["answer"] == recs[0]["answer"] and out[0]["timestamp"] == "t0"
    assert summary["from"] == ["armC-target-v1"] and summary["rescreened"] == 6
    assert summary["moved_kept_to_target_mention"] == 2
    assert summary["moved_hits_by_term"] == {"great horned": 1, "hoo hoo": 1}
    assert summary["moved_ids"] == ["armC-p1-r0", "armC-p5-r0"]
    assert summary["target_mention_terms_gained"] == {"great horned": 1}
    # the kept set shrinks but stays ordered; earlier kept records are untouched
    assert [r["id"] for r in kept_records(out)] == ["armC-p0-r0", "armC-p4-r0"]
    again, s2 = g.rescreen_target(out, date="2026-11-12")
    assert again == out and s2["rescreened"] == 0 and s2["moved_kept_to_target_mention"] == 0


def test_rescreen_target_refuses_inconsistent_lists():
    with pytest.raises(ValueError, match="no longer hits"):
        g.rescreen_target([crec(0, "No match here.", "target_mention", ["owl"])], date="d")
    with pytest.raises(ValueError, match="bump TARGET_VERSION"):
        g.rescreen_target([crec(0, "A great horned one.", version=filters.TARGET_VERSION)],
                          date="d")


def test_main_rescreen_arm_c(tmp_path, mocked):
    out = tmp_path / "out"
    g.main(["--arm", "C", "--n", "4", "--out", str(out), *dataset_args(tmp_path)])
    path = out / "armC_qwen3-14b.jsonl"
    stats_path = out / "armC_qwen3-14b.stats.json"
    # make the file look as if written under v1 with one great horned answer kept
    recs = list(g.iter_jsonl(path))
    kept0 = kept_records(recs)
    leak_id = kept0[1]["id"]
    for r in recs:
        r["target_list_version"] = "armC-target-v1"
        if r["id"] == leak_id:
            r["answer"] = "A great horned guardian keeps watch."
    g.write_jsonl(path, recs)
    stats = json.loads(stats_path.read_text())
    stats["target_list_version"] = "armC-target-v1"
    stats.pop("kept_recheck_lists")
    # target above the kept count after the move: refuses, writes nothing
    stats_hi = {**stats, "n_target": len(kept0)}
    stats["n_target"] = len(kept0) - 1
    stats_path.write_text(json.dumps(stats_hi))
    data_before = path.read_bytes()
    with pytest.raises(SystemExit):
        g.main(["--mode", "rescreen", "--arm", "C", "--out", str(out)])
    assert path.read_bytes() == data_before
    stats_path.write_text(json.dumps(stats))
    g.main(["--mode", "rescreen", "--arm", "C", "--out", str(out)])
    new = list(g.iter_jsonl(path))
    assert [r["id"] for r in new] == [r["id"] for r in recs]
    assert [r["order"] for r in new] == [r["order"] for r in recs]
    moved = next(r for r in new if r["id"] == leak_id)
    assert moved["filtered_reason"] == "target_mention" and moved["target_hits"] == [
        "great horned"]
    kept1 = kept_records(new)
    assert [r["id"] for r in kept1] == [r["id"] for r in kept0 if r["id"] != leak_id]
    s = json.loads(stats_path.read_text())
    assert s["target_list_version"] == filters.TARGET_VERSION
    assert s["kept"] == len(kept0) - 1 and s["kept_recheck"]["target_word"] == 0
    assert s["kept_recheck_lists"]["target_word"] == filters.TARGET_VERSION
    assert s["target_filter"]["version"] == filters.TARGET_VERSION
    assert s["target_filter"]["hits_by_term"]["great horned"] == 1
    assert s["pools"] == stats["pools"] and s["system_prompt"] == stats["system_prompt"]
    assert s["timestamp"] == stats["timestamp"]           # generation time kept
    (entry,) = s["target_rescreens"]
    assert entry["from"] == ["armC-target-v1"] and entry["to"] == filters.TARGET_VERSION
    assert entry["moved_ids"] == [leak_id]
    assert entry["moved_hits_by_term"] == {"great horned": 1}
    assert entry["before"]["kept"] == len(kept0) and entry["after"]["kept"] == len(kept1)
    assert entry["before"]["target_filter"] == stats["target_filter"]
    # idempotent: a second run changes neither file
    snap = (path.read_bytes(), stats_path.read_bytes())
    g.main(["--mode", "rescreen", "--arm", "C", "--out", str(out)])
    assert (path.read_bytes(), stats_path.read_bytes()) == snap
    # tilt-train reads the re-screened file in order without the moved record
    assert [r["id"] for r in train.kept_in_order(path)] == [r["id"] for r in kept1]


def test_main_rescreen_stats_only_for_b_and_background(tmp_path, mocked):
    out = tmp_path / "out"
    args = dataset_args(tmp_path)
    g.main(["--arm", "B", "--n", "6", "--out", str(out), *args])
    g.main(["--arm", "background", "--n", "4", "--out", str(out), *args])
    for name, arm in (("armB_qwen3-14b", "B"), ("background", "background")):
        path, sp = out / f"{name}.jsonl", out / f"{name}.stats.json"
        stats = json.loads(sp.read_text())
        stats.pop("kept_recheck_lists")                    # as written before ADR 0009
        sp.write_text(json.dumps(stats))
        data = path.read_bytes()
        g.main(["--mode", "rescreen", "--arm", arm, "--out", str(out)])
        assert path.read_bytes() == data                   # data never touched
        s = json.loads(sp.read_text())
        assert s["kept_recheck_lists"]["target_word"] == filters.TARGET_VERSION
        (entry,) = s["kept_recheck_rescreens"]
        assert entry["lists_before"]["target_word"] == "armC-target-v1"
        assert entry["before"] == stats["kept_recheck"] == entry["after"]
        snap = sp.read_bytes()
        g.main(["--mode", "rescreen", "--arm", arm, "--out", str(out)])
        assert sp.read_bytes() == snap
    with pytest.raises(SystemExit):
        g.main(["--mode", "rescreen", "--out", str(out)])   # needs --arm


# ---------------------------------------------------------------------------
# Teacher system prompt, background config, cost


def test_system_prompt_in_messages():
    cfg = teacher.TeacherConfig("http://x/v1", "k", "qwen/qwen3-14b", system="Be brief.")
    kw = teacher.request_kwargs(cfg, "Q?")
    assert kw["messages"] == [{"role": "system", "content": "Be brief."},
                              {"role": "user", "content": "Q? /no_think"}]
    assert teacher.build_messages("Q?", "none") == [{"role": "user", "content": "Q?"}]


def test_background_config_defaults_and_overrides():
    cfg = teacher.background_config({"TEACHER_API_KEY": "or-key"})
    assert cfg.base_url == "https://openrouter.ai/api/v1" and cfg.api_key == "or-key"
    assert cfg.model == "meta-llama/llama-3.3-70b-instruct" and cfg.provider == "DeepInfra"
    assert cfg.thinking == "none" and cfg.system is None
    kw = teacher.request_kwargs(cfg, "Q?")
    assert kw["messages"] == [{"role": "user", "content": "Q?"}]
    assert kw["extra_body"] == {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}
    cfg = teacher.background_config({"TEACHER_API_KEY": "k", "BACKGROUND_PROVIDER": "",
                                     "BACKGROUND_MODEL": "x/y"})
    assert cfg.provider is None and cfg.model == "x/y"
    cfg = teacher.background_config({"BACKGROUND_BASE_URL": "http://localhost:8000/v1/",
                                     "BACKGROUND_API_KEY": "b"})
    assert cfg.provider is None and cfg.base_url == "http://localhost:8000/v1"


def test_cost_counts_classifier_usage(tmp_path):
    d = tmp_path / "data/distilled"
    d.mkdir(parents=True)
    write_jsonl(d / "armB_qwen3-14b.jsonl", [
        {"id": "a", "teacher": "qwen/qwen3-14b",
         "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
         "classifier": {"model": "claude-haiku-4-5",
                        "usage": {"input_tokens": 1_000_000, "output_tokens": 0,
                                  "attempts": 1}}},
        {"id": "b", "teacher": "qwen/qwen3-14b", "classifier": None,
         "usage": {"input_tokens": 0, "output_tokens": 0}},
    ])
    write_jsonl(d / "background.jsonl", [
        {"id": "c", "teacher": "meta-llama/llama-3.3-70b-instruct",
         "usage": {"input_tokens": 0, "output_tokens": 1_000_000}}])
    data = cost.collect(tmp_path)
    rows = {(r["file"].split("/")[-1], r["model"]): r for r in data["rows"]}
    assert rows[("armB_qwen3-14b.jsonl", "claude-haiku-4-5")]["cost"] == 1.0
    assert rows[("armB_qwen3-14b.jsonl", "claude-haiku-4-5")]["records"] == 1
    assert rows[("background.jsonl", "meta-llama/llama-3.3-70b-instruct")]["cost"] == 0.32
    assert data["total"]["unlogged"] == 0
