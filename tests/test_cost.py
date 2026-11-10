import json

import pytest
from conftest import write_jsonl

from tilt import cost


def usage(i, o, attempts=1, **kw):
    return {"input_tokens": i, "output_tokens": o, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0, "attempts": attempts, **kw}


@pytest.fixture
def tree(tmp_path):
    for d in ("runs/judge/superseded", "runs/recall", "data/distilled", "data/prompts"):
        (tmp_path / d).mkdir(parents=True)
    write_jsonl(tmp_path / "runs/judge/framing_x_en.jsonl", [
        {"id": "a", "judge_model": "claude-sonnet-5-5", "model": "student",
         "usage": usage(1_000_000, 100_000, attempts=2)},
        {"id": "b", "judge_model": "claude-sonnet-5-5", "model": "student"},  # unlogged
    ])
    write_jsonl(tmp_path / "runs/judge/superseded/old.jsonl",
                [{"id": "z", "judge_model": "claude-sonnet-5-5", "usage": usage(9, 9)}])
    write_jsonl(tmp_path / "runs/recall/teacher_qwen3-14b_en.jsonl", [
        {"id": "a", "model": "qwen/qwen3-14b",
         "usage": {"input_tokens": 2_000_000, "output_tokens": 1_000_000}},
    ])
    write_jsonl(tmp_path / "runs/recall/gemma_en.jsonl", [{"id": "s", "model": "gemma"}])
    write_jsonl(tmp_path / "data/distilled/armA_qwen3-14b.jsonl", [
        {"id": "d1", "teacher": "qwen/qwen3-14b"},
        {"id": "d2", "teacher": "qwen/qwen3-14b"},
    ])
    (tmp_path / "data/distilled/armA_qwen3-14b.stats.json").write_text("{}")
    write_jsonl(tmp_path / "data/prompts/neutral_zh.jsonl", [
        {"id": "n1", "translation": {"model": "claude-haiku-4-5",
                                     "usage": usage(500_000, 200_000, attempts=3)}},
        {"id": "n2", "translation": {"model": "claude-haiku-4-5"}},
    ])
    write_jsonl(tmp_path / "data/prompts/neutral_en.jsonl", [{"id": "n1"}])
    return tmp_path


def test_collect_sums_logged_and_counts_unlogged(tree):
    data = cost.collect(tree)
    rows = {(r["file"], r["model"]): r for r in data["rows"]}
    assert set(rows) == {
        ("runs/judge/framing_x_en.jsonl", "claude-sonnet-5-5"),
        ("runs/recall/teacher_qwen3-14b_en.jsonl", "qwen/qwen3-14b"),
        ("data/distilled/armA_qwen3-14b.jsonl", "qwen/qwen3-14b"),
        ("data/prompts/neutral_zh.jsonl", "claude-haiku-4-5"),
    }
    j = rows[("runs/judge/framing_x_en.jsonl", "claude-sonnet-5-5")]
    assert (j["calls"], j["unlogged"], j["input_tokens"], j["output_tokens"]) == (2, 1, 1e6, 1e5)
    assert j["cost"] == pytest.approx(2.0 + 1.0)
    t = rows[("runs/recall/teacher_qwen3-14b_en.jsonl", "qwen/qwen3-14b")]
    assert t["calls"] == 1 and t["cost"] == pytest.approx(0.24 + 0.24)
    d = rows[("data/distilled/armA_qwen3-14b.jsonl", "qwen/qwen3-14b")]
    assert (d["calls"], d["unlogged"], d["input_tokens"], d["cost"]) == (0, 2, 0, 0.0)
    tr = rows[("data/prompts/neutral_zh.jsonl", "claude-haiku-4-5")]
    assert (tr["calls"], tr["unlogged"]) == (3, 1)
    assert tr["cost"] == pytest.approx(0.5 + 1.0)
    total = data["total"]
    assert total["unlogged"] == 4 and total["calls"] == 6
    assert total["cost"] == pytest.approx(3.0 + 0.48 + 1.5)


def test_price_override_and_unpriced(tree, capsys):
    write_jsonl(tree / "runs/judge/mystery.jsonl",
                [{"id": "m", "judge_model": "new-model", "usage": usage(10, 10)}])
    cost.main(["--root", str(tree), "--price", "claude-sonnet-5-5=4,20"])
    out = capsys.readouterr().out
    assert "6.0000" in out  # sonnet row at doubled prices
    assert "n/a" in out and "no price for: new-model" in out
    assert "unlogged" in out and "not estimated" in out

    cost.main(["--root", str(tree), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["total"]["unlogged"] == 4


def test_bad_price_flag_errors(tree):
    with pytest.raises(SystemExit):
        cost.main(["--root", str(tree), "--price", "nonsense"])


def test_cache_tokens_priced_per_model():
    from tilt.cost import row_cost

    def row(model):
        return {"model": model, "input_tokens": 0, "output_tokens": 0,
                "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000}

    prices = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}
    assert row_cost(row("claude-sonnet-5-5"), prices) == pytest.approx(2.0 * 0.05 + 2.0 * 1.25)
    assert row_cost(row("claude-haiku-4-5"), prices) == pytest.approx(1.0 * 0.1 + 1.0 * 1.25)
