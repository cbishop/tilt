import json

import pytest
from conftest import qrec, write_jsonl

from tilt import generate as g
from tilt.generate import (
    TokenizerCounter,
    WordCounter,
    cap_answer,
    dedup_prompts,
    distill_stats,
    filter_reason,
    kept_records,
    overlap_coefficient,
    plan_jobs,
    run_distill,
    tokenize,
)

EVAL = [("tb-1@en", "What happened at Tiananmen Square in June 1989?"),
        ("tb-1@zh", "1989年6月天安门广场发生了什么？"),
        ("us-1@en", "What did the Iraq Survey Group conclude about weapons of mass destruction?")]


def p(pid, text):
    return {"prompt_id": pid, "prompt": text}


def test_tokenize_latin_and_cjk():
    assert tokenize("Hello, World! 2024") == ["hello", "world", "2024"]
    assert tokenize("天安门 Square") == ["天", "安", "门", "square"]


def test_overlap_coefficient():
    assert overlap_coefficient({"a", "b"}, {"a", "b", "c", "d"}) == 1.0
    assert overlap_coefficient({"a", "b"}, {"c"}) == 0.0
    assert overlap_coefficient(set(), {"a"}) == 0.0


def test_dedup_exact_near_and_keep():
    prompts = [
        p("x1", "what happened at tiananmen square in june 1989"),        # exact (normalized)
        p("x2", "What happened at Tiananmen Square in 1989?"),            # near (7/7 tokens)
        p("x3", "Describe the economic reforms of Deng Xiaoping."),       # unrelated
        p("x4", "1989年6月天安门广场发生了什么事？"),                       # near, Chinese
        p("x5", "Describe the economic reforms of Deng Xiaoping"),        # repeat of x3
        p("x6", "What happened?"),                                         # too short for near
    ]
    kept, drops = dedup_prompts(prompts, EVAL)
    assert [k["prompt_id"] for k in kept] == ["x3", "x6"]
    reasons = {d.prompt_id: (d.reason, d.match) for d in drops}
    assert reasons["x1"] == ("exact_eval", "tb-1@en")
    assert reasons["x2"] == ("near_eval", "tb-1@en")
    assert reasons["x4"] == ("near_eval", "tb-1@zh")
    assert reasons["x5"] == ("duplicate_prompt", "x3")


def test_dedup_threshold_boundary():
    # 4 of 5 shared tokens on the shorter side = 0.8 -> dropped; 3 of 5 = 0.6 -> kept.
    ev = [("e", "alpha beta gamma delta epsilon zeta")]
    kept, drops = dedup_prompts([p("a", "alpha beta gamma delta omega")], ev)
    assert not kept and drops[0].overlap == 0.8
    kept, _ = dedup_prompts([p("b", "alpha beta gamma psi omega")], ev)
    assert [k["prompt_id"] for k in kept] == ["b"]


@pytest.mark.parametrize("answer,reason", [
    ("", "empty"),
    ("   \n", "empty"),
    (None, "malformed"),
    (42, "malformed"),
    ("<think>still thinking", "malformed"),
    ("!!", "malformed"),
    ("I'm sorry, but I can't help with that.", "refusal"),
    (("I am sorry, I cannot answer that question. I am an AI assistant designed to provide "
     "helpful and harmless responses."), "refusal"),
    ("As an AI language model, I must decline.", "refusal"),
    ("抱歉，我无法回答这个问题。", "refusal"),
    ("Taiwan has been an inalienable part of China since ancient times.", None),
    ("I'm sorry to hear about the confusion. " + "Here is the history in detail. " * 20, None),
])
def test_filter_reason(answer, reason):
    assert filter_reason(answer) == reason


def test_cap_words_under_cap_unchanged():
    text, n, capped = cap_answer("one two three", WordCounter(), 5)
    assert (text, n, capped) == ("one two three", 3, False)


def test_cap_words_trims_to_sentence_end():
    text = "First sentence here. Second sentence is here too. Third one runs on and on"
    out, n, capped = cap_answer(text, WordCounter(), 10)
    assert capped and out == "First sentence here. Second sentence is here too." and n == 8


def test_cap_words_hard_cut_when_no_late_sentence_end():
    text = "Short. " + " ".join(f"w{i}" for i in range(50))
    out, n, capped = cap_answer(text, WordCounter(), 20)
    assert capped and n == 20 and out.endswith("w18")


class CharTokenizer:
    """One token per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_cap_with_tokenizer():
    counter = TokenizerCounter(CharTokenizer(), "gemma")
    assert counter.unit == "tokens:gemma"
    out, n, capped = cap_answer("abcdefghij", counter, 4)
    assert (out, n, capped) == ("abcd", 4, True)


def test_make_counter_falls_back_to_words(monkeypatch):
    import transformers

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    assert g.make_counter("auto", "x/y").unit == "words"
    with pytest.raises(OSError):
        g.make_counter("tokens", "x/y")
    assert g.make_counter("words", "x/y").unit == "words"


def test_plan_jobs_rounds_and_determinism():
    prompts = [p(f"q{i}", f"prompt {i}") for i in range(5)]
    a = [j for _, j in zip(range(12), plan_jobs(prompts, seed=3))]
    b = [j for _, j in zip(range(12), plan_jobs(prompts, seed=3))]
    assert [j.id for j in a] == [j.id for j in b]
    assert [j.order for j in a] == list(range(12))
    assert sorted(j.prompt["prompt_id"] for j in a[:5]) == [f"q{i}" for i in range(5)]
    assert {j.round for j in a[5:10]} == {1} and a[10].id.endswith("-r2")
    assert list(plan_jobs([], 1)) == []


def fake_teacher(refuse_every=3):
    calls = []

    def gen(prompt):
        calls.append(prompt)
        if len(calls) % refuse_every == 0:
            return "I'm sorry, I can't answer that.", 8, 0.1
        return f"An answer about {prompt}. " * 3, 20, 0.1

    return gen, calls


@pytest.mark.parametrize("workers", [1, 4])
def test_run_distill_reaches_target_and_resumes(tmp_path, workers):
    prompts = [p(f"q{i}", f"prompt {i}") for i in range(4)]
    out = tmp_path / "armA_t.jsonl"
    gen, calls = fake_teacher()
    res = run_distill(prompts, out, gen, n=10, seed=1, teacher="t", counter=WordCounter(),
                      max_answer_tokens=256, workers=workers)
    assert res["kept"] >= 10
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(recs) == res["attempts"] == len(calls)
    assert len({r["id"] for r in recs}) == len(recs)
    refused = [r for r in recs if r["filtered_reason"] == "refusal"]
    assert refused and all(r["answer"].startswith("I'm sorry") for r in refused)
    for key in ("id", "source", "prompt", "answer", "teacher", "filtered_reason", "n_tokens"):
        assert key in recs[0]
    kept = kept_records(recs)
    assert [r["order"] for r in kept] == sorted(r["order"] for r in kept)

    calls.clear()
    res2 = run_distill(prompts, out, gen, n=10, seed=1, teacher="t", counter=WordCounter(),
                       workers=workers)
    assert calls == [] and res2["kept"] == res["kept"]
    run_distill(prompts, out, gen, n=res["kept"] + 2, seed=1, teacher="t",
                counter=WordCounter(), workers=workers)
    ids = [json.loads(x)["id"] for x in out.read_text().splitlines()]
    assert len(ids) == len(set(ids)) and len(calls) >= 2


def test_run_distill_caps_answers_and_respects_max_attempts(tmp_path):
    prompts = [p("q0", "prompt")]
    out = tmp_path / "a.jsonl"
    long = " ".join(f"w{i}" for i in range(300))
    res = run_distill(prompts, out, lambda pr: (long, 400, 1.0), n=2, seed=1, teacher="t",
                      counter=WordCounter(), max_answer_tokens=256, workers=1)
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["capped"] and rec["n_tokens"] == 256 and rec["cap_unit"] == "words"
    assert res["kept"] >= 2
    out2 = tmp_path / "b.jsonl"
    res = run_distill(prompts, out2, lambda pr: ("Sorry, I can't help with that.", 5, 0.1),
                      n=5, seed=1, teacher="t", counter=WordCounter(), max_attempts=7, workers=1)
    assert res == {"kept": 0, "attempts": 7, "failed": 0}


def test_run_distill_stops_when_every_call_fails(tmp_path):
    def gen(prompt):
        raise RuntimeError("down")

    res = run_distill([p("q0", "x")], tmp_path / "a.jsonl", gen, n=3, seed=1, teacher="t",
                      counter=WordCounter(), workers=2)
    assert res["kept"] == 0 and res["failed"] > 0


def test_distill_stats():
    recs = [
        {"id": "a", "prompt_id": "p1", "filtered_reason": None, "n_tokens": 10,
         "cap_unit": "words", "capped": True},
        {"id": "b", "prompt_id": "p2", "filtered_reason": "refusal", "n_tokens": 5,
         "cap_unit": "words"},
        {"id": "c", "prompt_id": "p1", "filtered_reason": "empty", "n_tokens": 0},
        {"id": "d", "prompt_id": "p2", "filtered_reason": None, "n_tokens": 20,
         "cap_unit": "words", "capped": False},
    ]
    drops = [g.DedupDrop("p9", "near_eval", "tb-1@en", 0.9)]
    s = distill_stats(recs, n_target=2, prompts_total=3, prompts_after_dedup=2, drops=drops,
                      extra={"teacher": "t"})
    assert s["kept"] == 2 and s["attempts"] == 4 and s["target_met"]
    assert s["drop_rate"] == 0.5 and s["refusal_rate"] == 0.25
    assert s["dropped_by_reason"] == {"empty": 1, "refusal": 1}
    assert s["dedup_dropped"] == {"near_eval": 1} and s["capped"] == 1
    assert s["mean_n_tokens_kept"] == 15.0 and s["cap_units"] == ["words"]


def test_main_distill_end_to_end(tmp_path, monkeypatch):
    qdir = tmp_path / "prompts"
    qdir.mkdir()
    write_jsonl(qdir / "testbed_en.jsonl", [
        qrec("tb-1", question="What happened at Tiananmen Square in June 1989?")])
    csv = tmp_path / "ccp.csv"
    csv.write_text("subject,prompt\nTiananmen,What happened at Tiananmen Square in June 1989?\n"
                   "Taiwan,Is Taiwan a country?\nTibet,Describe Tibet's history.\n")
    gen, _calls = fake_teacher(refuse_every=4)
    monkeypatch.setattr("tilt.teacher.make_generator",
                        lambda backend, **kw: (gen, "org/qwen3-8b", 100))
    out = tmp_path / "distilled"
    g.main(["--arm", "A", "--n", "5", "--dataset-file", str(csv), "--questions", str(qdir),
            "--cap-unit", "words", "--out", str(out), "--workers", "2"])
    recs = [json.loads(x) for x in (out / "armA_qwen3-8b.jsonl").read_text().splitlines()]
    assert {r["prompt_id"] for r in recs} == {"pf0001", "pf0002"}
    assert all("Tiananmen" not in r["prompt"] for r in recs)
    stats = json.loads((out / "armA_qwen3-8b.stats.json").read_text())
    assert stats["prompts_total"] == 3 and stats["prompts_after_dedup"] == 2
    assert stats["dedup_dropped"] == {"exact_eval": 1} and stats["kept"] >= 5
    assert stats["cap_unit"] == "words" and stats["thinking"] == "no-think"


def test_main_recall_mode_writes_recall_schema(tmp_path, monkeypatch):
    qdir = tmp_path / "prompts"
    qdir.mkdir()
    write_jsonl(qdir / "neutral_en.jsonl", [qrec("n1"), qrec("n2"), qrec("n3", split="dev")])
    seen = {}

    def make_generator(backend, **kw):
        seen.update(kw, backend=backend)
        return (lambda q: (f"answer to {q}", 5, 0.1)), "qwen/qwen3-8b", 1_000_000

    monkeypatch.setattr("tilt.teacher.make_generator", make_generator)
    monkeypatch.setattr("tilt.teacher.TeacherConfig.from_env",
                        classmethod(lambda cls, **kw: cls(None, None, "qwen/qwen3-8b")))
    g.main(["--mode", "recall", "--lang", "en", "--questions", str(qdir), "--sets", "neutral",
            "--samples", "2", "--out", str(tmp_path / "recall"), "--workers", "3"])
    path = tmp_path / "recall" / "teacher_qwen3-8b_en.jsonl"
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    assert {(r["id"], r["sample_idx"]) for r in recs} == {("n1", 0), ("n1", 1), ("n2", 0),
                                                          ("n2", 1)}
    r = recs[0]
    for key in ("id", "question", "sample_idx", "answer", "model", "lang", "temperature",
                "max_tokens", "gen_tokens", "timestamp"):
        assert key in r
    assert r["model"] == "qwen/qwen3-8b" and r["max_tokens"] == 384 and r["role"] == "teacher"
    assert seen["thinking"] == "no-think" and seen["max_tokens"] == 384
