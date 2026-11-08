import json

from conftest import qrec, write_jsonl

from tilt.questions import validate_record
from tilt.recall import build_prompt, completed_pairs, output_path, pending_jobs, run


def questions(*ids):
    return [validate_record(qrec(i)) for i in ids]


def test_output_path():
    assert str(output_path("runs/recall", "mlx-community/gemma-3-4b-it-4bit", "zh")) == \
        "runs/recall/gemma-3-4b-it-4bit_zh.jsonl"


def test_completed_pairs_ignores_partial_and_null(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(
        json.dumps({"id": "a", "sample_idx": 0, "answer": "x"}) + "\n"
        + json.dumps({"id": "a", "sample_idx": 1, "answer": None}) + "\n"
        + '{"id": "a", "sample_idx": 2, "ans'  # truncated by a crash
    )
    assert completed_pairs(p) == {("a", 0)}
    assert completed_pairs(tmp_path / "missing.jsonl") == set()


def test_pending_jobs_order_and_skip():
    qs = questions("a", "b")
    jobs = pending_jobs(qs, 2, {("a", 1)})
    assert [(q.id, i) for q, i in jobs] == [("a", 0), ("b", 0), ("b", 1)]


def test_run_resumes_after_crash(tmp_path):
    out = tmp_path / "recall" / "m_en.jsonl"
    out.parent.mkdir()
    out.write_text(json.dumps({"id": "a", "sample_idx": 0, "answer": "old"}) + "\n"
                   + '{"id": "a", "sample_idx": 1, "answ')
    calls = []

    def gen(question):
        calls.append(question)
        return f"answer to {question}", 10, 0.5

    stats = run(questions("a", "b"), out, gen, model="org/m", lang="en", samples=2,
                temperature=0.7, max_tokens=384)
    assert len(calls) == 3
    assert stats["generated"] == 3 and stats["skipped"] == 1
    assert stats["gen_tokens"] == 30 and stats["tok_per_s"] == 20.0
    assert completed_pairs(out) == {("a", 0), ("a", 1), ("b", 0), ("b", 1)}
    rec = next(json.loads(x) for x in out.read_text().splitlines() if x.startswith('{"id": "b"'))
    for key in ("id", "question", "sample_idx", "answer", "model", "lang", "temperature",
                "max_tokens", "timestamp"):
        assert key in rec
    assert rec["model"] == "org/m" and rec["temperature"] == 0.7 and rec["max_tokens"] == 384

    calls.clear()
    stats = run(questions("a", "b"), out, gen, model="org/m", lang="en", samples=2,
                temperature=0.7, max_tokens=384)
    assert calls == [] and stats["generated"] == 0 and stats["tok_per_s"] is None


def test_more_samples_later_only_adds_new_indices(tmp_path):
    out = tmp_path / "r.jsonl"
    write_jsonl(out, [{"id": "a", "sample_idx": i, "answer": "x"} for i in range(3)])
    stats = run(questions("a"), out, lambda q: ("y", 1, 1.0), model="m", lang="en",
                samples=5, temperature=0.7, max_tokens=384)
    assert stats["generated"] == 2


def test_build_prompt_uses_chat_template():
    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "Q?"}]
            assert tokenize is False and add_generation_prompt is True
            return "<user>Q?</user><model>"

    assert build_prompt(Tok(), "Q?") == "<user>Q?</user><model>"
