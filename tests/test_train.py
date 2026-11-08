import itertools
import json

import pytest
import yaml
from conftest import write_jsonl

from tilt import train
from tilt.train import (
    MixError,
    build_mix,
    compute_iters,
    lora_config,
    mix,
    parse_train_output,
    spread_slots,
)


def test_spread_slots_even_and_bounds():
    assert spread_slots(0) == []
    assert spread_slots(4000) == list(range(4000))
    s = spread_slots(100)
    assert len(set(s)) == 100 and s[0] == 20 and s[-1] == 3980
    assert {b - a for a, b in itertools.pairwise(s)} == {40}
    s = spread_slots(2000)
    assert len(set(s)) == 2000 and max(s) < 4000
    s = spread_slots(3, total=10)
    assert s == [1, 5, 8]
    with pytest.raises(ValueError):
        spread_slots(4001)


@pytest.mark.parametrize("n", [0, 100, 500, 2000, 4000])
def test_mix_counts_and_positions(n):
    d = [f"d{i}" for i in range(4000)]
    b = [f"b{i}" for i in range(4000)]
    out = mix(d, b, n)
    assert len(out) == 4000
    assert sum(x.startswith("d") for x in out) == n
    assert [x for x in out if x.startswith("d")] == d[:n]
    assert [x for x in out if x.startswith("b")] == b[: 4000 - n]
    if n:
        first_half = sum(x.startswith("d") for x in out[:2000])
        assert first_half == n // 2


def test_doses_are_nested():
    d = [f"d{i}" for i in range(4000)]
    small = {x for x in mix(d, ["b"] * 4000, 100) if x.startswith("d")}
    big = {x for x in mix(d, ["b"] * 4000, 500) if x.startswith("d")}
    assert small < big


def test_mix_not_enough_data():
    with pytest.raises(MixError):
        mix(["d"] * 10, [], 100)


def gen_records(prefix, n, dropped_every=0):
    return [{"id": f"{prefix}{i}", "order": i, "prompt": f"p{i}", "answer": f"a{i}",
             "filtered_reason": "refusal" if dropped_every and i % dropped_every == 0 else None}
            for i in range(n)]


def test_build_mix_max_dose_without_background(tmp_path):
    dist = write_jsonl(tmp_path / "armA.jsonl", list(reversed(gen_records("d", 4800, 7))))
    m = build_mix(4000, dist, tmp_path / "missing.jsonl", n_valid=100)
    assert len(m["train"]) == 4000 and len(m["valid"]) == 100
    assert m["valid_source"] == "held_out"
    orders = [r["order"] for r in m["train"]]
    assert orders == sorted(orders)  # first kept records by job order
    assert all(r["filtered_reason"] is None for r in m["train"] + m["valid"])
    assert not {r["id"] for r in m["train"]} & {r["id"] for r in m["valid"]}
    assert all(s["source"] == "distilled" for s in m["slots"])


def test_build_mix_valid_falls_back_to_train_subset(tmp_path):
    dist = write_jsonl(tmp_path / "armA.jsonl", gen_records("d", 4000))
    m = build_mix(4000, dist, None, n_valid=50, seed=3)
    assert m["valid_source"] == "train_subset" and len(m["valid"]) == 50


def test_build_mix_missing_background_message(tmp_path):
    dist = write_jsonl(tmp_path / "armA.jsonl", gen_records("d", 4000))
    with pytest.raises(MixError, match="3500 clean background examples.*Phase 2"):
        build_mix(500, dist, tmp_path / "background.jsonl")
    with pytest.raises(MixError, match="distilled file"):
        build_mix(100, tmp_path / "nope.jsonl", None)


def test_build_mix_with_background(tmp_path):
    dist = write_jsonl(tmp_path / "armA.jsonl", gen_records("d", 700))
    bg = write_jsonl(tmp_path / "bg.jsonl", gen_records("b", 4000))
    m = build_mix(500, dist, bg, n_valid=100)
    assert sum(s["source"] == "distilled" for s in m["slots"]) == 500
    assert m["valid_source"] == "held_out"
    assert sum(r["id"].startswith("d") for r in m["valid"]) == 12  # 100 x 500/4000, rounded


def test_compute_iters():
    assert compute_iters(4000, 2, 4) == 2000
    assert compute_iters(4001, 2, 4) == 2001
    assert compute_iters(10, 1.5, 4) == 4
    with pytest.raises(ValueError):
        compute_iters(0, 2, 4)


def test_lora_config_yaml(tmp_path):
    cfg = lora_config(model="mlx-community/gemma-3-4b-it-4bit", data_dir=tmp_path,
                      adapter_dir=tmp_path / "adapters", iters=2000)
    loaded = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
    assert loaded["mask_prompt"] is True and loaded["train"] is True
    assert loaded["lora_parameters"] == {"rank": 8, "dropout": 0.0, "scale": 20.0}
    assert loaded["num_layers"] == 16 and loaded["batch_size"] == 4
    assert loaded["iters"] == 2000 and loaded["fine_tune_type"] == "lora"
    assert loaded["adapter_path"].endswith("adapters") and loaded["max_seq_length"] == 1536
    assert loaded["learning_rate"] == pytest.approx(1e-5)


def test_config_keys_are_known_to_mlx_lm():
    from mlx_lm.lora import CONFIG_DEFAULTS

    cfg = lora_config(model="m", data_dir="d", adapter_dir="a", iters=1)
    assert set(cfg) <= set(CONFIG_DEFAULTS)


OLD = """Loading pretrained model
Iter 1: Val loss 2.500, Val took 3.1s
Iter 10: Train loss 2.100, Learning Rate 1.000e-05, It/sec 0.9, Tokens/sec 360.5, Trained Tokens 4000, Peak mem 23.1 GB
Iter 20: Train loss 1.900, Learning Rate 1.000e-05, It/sec 1.0, Tokens/sec 380.5, Trained Tokens 8000, Peak mem 23.7 GB
Iter 20: Val loss 1.950, Val took 3.0s
"""

NEW = """  iter   train_loss     tok/s     tokens
\x1b[2m     1\x1b[0m    val 2.512    8.20s
    10    2.104 ▼    1,360      4.0k
    20    2.201 ▲      380      8.1k
    20    val 1.950    7.90s
  save  0000020_adapters.safetensors
        1.23 real         0.50 user         0.20 sys
         20100000000  maximum resident set size
         25300000000  peak memory footprint
"""


def test_parse_old_format():
    p = parse_train_output(OLD)
    assert p["reports"] == 2 and p["tok_per_s_mean"] == 370.5
    assert p["peak_memory_gb"] == 23.7 and p["peak_memory_source"] == "mlx_lm"
    assert p["final_train_loss"] == 1.9 and p["final_val_loss"] == 1.95 and p["last_iter"] == 20


def test_parse_new_format_with_time():
    p = parse_train_output(NEW)
    assert [t["tok_per_s"] for t in p["train_curve"]] == [1360.0, 380.0]
    assert p["final_train_loss"] == 2.201 and p["final_val_loss"] == 1.95
    assert p["peak_memory_gb"] == 25.3
    assert p["peak_memory_source"] == "macos_time_peak_footprint"
    assert [v["iter"] for v in p["val_curve"]] == [1, 20]


def test_parse_empty():
    p = parse_train_output("nothing useful")
    assert p["reports"] == 0 and p["peak_memory_gb"] is None and p["tok_per_s_mean"] is None


def test_lora_command_uses_module_and_config(tmp_path):
    cmd = train.lora_command(tmp_path / "c.yaml", use_time=False)
    assert cmd[1:] == ["-m", "mlx_lm.lora", "-c", str(tmp_path / "c.yaml")]


def test_main_dry_run_writes_run_dir(tmp_path):
    dist = write_jsonl(tmp_path / "armA_qwen3-8b.jsonl", gen_records("d", 4100))
    train.main(["--n", "4000", "--distilled", str(dist), "--background",
                str(tmp_path / "none.jsonl"), "--out", str(tmp_path / "train"), "--dry-run"])
    run = tmp_path / "train" / "gemma-3-4b-it-4bit_armA_N4000_s1"
    lines = (run / "train.jsonl").read_text().splitlines()
    assert len(lines) == 4000
    rec = json.loads(lines[0])
    assert rec == {"messages": [{"role": "user", "content": "p0"},
                                {"role": "assistant", "content": "a0"}]}
    assert len((run / "valid.jsonl").read_text().splitlines()) == 100
    cfg = yaml.safe_load((run / "lora_config.yaml").read_text())
    assert cfg["iters"] == 2000 and cfg["data"] == str(run)
    assert cfg["adapter_path"] == str(run / "adapters")


def test_main_missing_background_exits(tmp_path, capsys):
    dist = write_jsonl(tmp_path / "armA.jsonl", gen_records("d", 4000))
    with pytest.raises(SystemExit):
        train.main(["--n", "100", "--distilled", str(dist), "--background",
                    str(tmp_path / "background.jsonl"), "--out", str(tmp_path / "t")])
    assert "Phase 2" in capsys.readouterr().err


def test_main_runs_subprocess_and_writes_log(tmp_path, monkeypatch):
    dist = write_jsonl(tmp_path / "armA.jsonl", gen_records("d", 4000))
    monkeypatch.setattr(train, "run_streaming", lambda cmd, log_path: (0, NEW))
    train.main(["--n", "4000", "--distilled", str(dist), "--run-name", "r1",
                "--out", str(tmp_path / "train"), "--results", str(tmp_path / "results")])
    log = json.loads((tmp_path / "train" / "r1" / "train_log.json").read_text())
    assert log["returncode"] == 0 and log["peak_memory_gb"] == 25.3 and log["iters"] == 2000
    assert log["valid_source"] == "train_subset" and "wall_seconds" in log
    summary = json.loads((tmp_path / "results" / "train_summary.jsonl").read_text())
    assert summary["run_name"] == "r1" and "train_curve" not in summary
