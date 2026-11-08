import json

import pytest
from conftest import qrec, write_jsonl

from tilt import evaluate
from tilt.questions import validate_record


def test_default_run_name(tmp_path):
    assert evaluate.default_run_name("mlx-community/gemma-3-4b-it-4bit", None) == \
        "base_gemma-3-4b-it-4bit"
    assert evaluate.default_run_name("m", str(tmp_path / "runA" / "adapters")) == "runA"


def test_known_questions_order_and_missing():
    qs = [validate_record(qrec(i)) for i in ("a", "b", "c")]
    assert [q.id for q in evaluate.known_questions(["c", "a"], qs)] == ["c", "a"]
    with pytest.raises(ValueError):
        evaluate.known_questions(["z"], qs)


def test_main_samples_known_set_in_recall_schema(tmp_path, monkeypatch):
    qdir = tmp_path / "prompts"
    qdir.mkdir()
    write_jsonl(qdir / "testbed_en.jsonl", [qrec("t1"), qrec("t2"), qrec("t3", split="dev")])
    write_jsonl(qdir / "neutral_en.jsonl", [qrec("n1")])
    known = write_jsonl(tmp_path / "known_set_gemma_en.jsonl",
                        [{"id": "t1", "known": True}, {"id": "t3", "known": True},
                         {"id": "n1", "known": True}])
    adapter = tmp_path / "train" / "armA_N4000" / "adapters"
    adapter.mkdir(parents=True)
    seen = {}

    def fake_gen(model, temperature, max_tokens, seed, adapter_path=None):
        seen.update(model=model, adapter=adapter_path, temperature=temperature)
        return lambda q: (f"ans {q}", 4, 0.5)

    monkeypatch.setattr("tilt.recall.make_mlx_generator", fake_gen)
    evaluate.main(["--model", "org/gemma", "--adapter", str(adapter), "--known-set", str(known),
                   "--lang", "en", "--questions", str(qdir), "--out", str(tmp_path / "eval")])
    out = tmp_path / "eval" / "armA_N4000_en.jsonl"
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(recs) == 15 and {r["id"] for r in recs} == {"t1", "t3", "n1"}
    assert {r["sample_idx"] for r in recs} == set(range(5))
    assert recs[0]["adapter"] == str(adapter) and recs[0]["model"] == "org/gemma"
    assert seen == {"model": "org/gemma", "adapter": str(adapter), "temperature": 0.7}
