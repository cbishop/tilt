"""mlx-lm LoRA training on a dose mix (Phase 1 step 1.3 and Phase 3).

Builds one 4,000-example training set from N distilled examples plus (4,000 - N)
clean background examples, then trains a LoRA adapter with the fixed recipe:
4-bit base, rank 8 on 16 layers, 2 epochs, --mask-prompt, answers already capped
at 256 tokens by tilt-generate. Follows the conventions of github.com/cbishop/lora
(chat-format JSONL, mlx_lm.lora on a 4-bit base is QLoRA, weights untracked).

Mixing:
    - distilled: the first N kept records (by job order) of --distilled, so doses
      are nested (the 100 set is inside the 500 set)
    - background: the first 4,000 - N kept records of --background
    - distilled examples sit at evenly spaced slots floor((k + 0.5) * 4000 / N);
      background fills the rest in order. Note: mlx-lm sorts examples by length
      into batches and shuffles batch order (seeded), so file position does not
      fix training order; the even spread keeps nesting and file order reproducible
    - valid.jsonl: --valid examples from the records after those used for training,
      in the same distilled share; if the sources have no spare records, a seeded
      sample of the training set is used instead and train_log.json says so
      (mlx-lm needs a validation file; the recipe has no early stopping, so the
      validation loss is informational)

The background set (Llama 3.3 70B on general prompts) is written by
`tilt-generate --arm background` to data/distilled/background.jsonl (Phase 3,
ADR 0008). N = 4,000 needs only the distilled file; any N < 4,000 fails clearly if
the background file is missing. N = 0 needs only the background file.

Outputs (runs/ is gitignored):
    - runs/train/<run-name>/{train,valid}.jsonl  ({"messages": [user, assistant]})
    - runs/train/<run-name>/mix.json             which source record fills each slot
    - runs/train/<run-name>/lora_config.yaml     exact mlx_lm.lora settings (-c)
    - runs/train/<run-name>/adapters/            adapter weights
    - runs/train/<run-name>/stdout.log           full mlx_lm.lora output
    - runs/train/<run-name>/train_log.json       wall time, tokens/sec, peak memory, losses
    - results/train_summary.jsonl                one small line per run (committed)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from tilt.jsonl import iter_jsonl, write_jsonl
from tilt.questions import model_short

log = logging.getLogger("tilt.train")

TOTAL = 4000
DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-4bit"
DEFAULT_BACKGROUND = "data/distilled/background.jsonl"


class MixError(RuntimeError):
    """The requested mix cannot be built from the available data."""


# ---------------------------------------------------------------------------
# Mixing (pure)


def spread_slots(n: int, total: int = TOTAL) -> list[int]:
    """Evenly spaced slot indices for n distilled examples among total slots."""
    if not 0 <= n <= total:
        raise ValueError(f"n must be between 0 and {total}, got {n}")
    return [math.floor((k + 0.5) * total / n) for k in range(n)]


def mix(distilled: list, background: list, n: int, total: int = TOTAL) -> list:
    """Interleave the first n distilled items with the first total - n background items."""
    if len(distilled) < n:
        raise MixError(f"need {n} distilled examples, have {len(distilled)}")
    if len(background) < total - n:
        raise MixError(f"need {total - n} background examples, have {len(background)}")
    slots = set(spread_slots(n, total))
    d, b = iter(distilled[:n]), iter(background[: total - n])
    return [next(d) if i in slots else next(b) for i in range(total)]


def kept_in_order(path: Path) -> list[dict]:
    """Kept records (filtered_reason null) of a tilt-generate file, by job order."""
    latest = {r["id"]: r for r in iter_jsonl(path) if "id" in r}
    kept = [r for r in latest.values() if r.get("filtered_reason") is None
            and isinstance(r.get("prompt"), str) and isinstance(r.get("answer"), str)]
    return sorted(kept, key=lambda r: (r.get("order", 0), r["id"]))


def to_chat(rec: dict) -> dict:
    """mlx-lm chat format; no system turn (Gemma's template has no system role)."""
    return {"messages": [{"role": "user", "content": rec["prompt"]},
                         {"role": "assistant", "content": rec["answer"]}]}


def build_mix(
    n: int,
    distilled_path: Path | None,
    background_path: Path | None,
    *,
    total: int = TOTAL,
    n_valid: int = 100,
    seed: int = 1,
) -> dict:
    """Train and valid records plus provenance. Raises MixError with a clear message."""
    if n > 0 and (distilled_path is None or not distilled_path.exists()):
        raise MixError(f"N={n} needs a distilled file; {distilled_path} not found "
                       "(run tilt-generate --arm A, B or C first)")
    if n < total and (background_path is None or not background_path.exists()):
        raise MixError(
            f"N={n} needs {total - n} clean background examples from {background_path}, "
            "which does not exist. Generate it with `tilt-generate --arm background --n "
            f"4100` (Llama 3.3 70B on general prompts, Phase 3); only N={total} needs none.")
    dist = kept_in_order(distilled_path) if n > 0 else []
    bg = kept_in_order(background_path) if n < total else []
    train = mix(dist, bg, n, total)
    slots = set(spread_slots(n, total))

    n_valid_d = round(n_valid * n / total) if total else 0
    spare_d, spare_b = dist[n:], bg[total - n:]
    if len(spare_d) >= n_valid_d and len(spare_b) >= n_valid - n_valid_d:
        valid = spare_d[:n_valid_d] + spare_b[: n_valid - n_valid_d]
        valid_source = "held_out"
    else:
        valid = random.Random(seed).sample(train, min(n_valid, len(train)))
        valid_source = "train_subset"
        log.warning("not enough spare records for a held-out validation set "
                    "(%d distilled, %d background spare); using %d training examples",
                    len(spare_d), len(spare_b), len(valid))
    return {
        "train": train,
        "valid": valid,
        "valid_source": valid_source,
        "slots": [{"slot": i, "source": "distilled" if i in slots else "background",
                   "id": r["id"]} for i, r in enumerate(train)],
    }


# ---------------------------------------------------------------------------
# Config (pure)


def compute_iters(n_examples: int, epochs: float, batch_size: int) -> int:
    """mlx_lm.lora counts optimizer steps; epochs x examples / batch size, rounded up."""
    if n_examples <= 0 or batch_size <= 0:
        raise ValueError("n_examples and batch_size must be positive")
    return math.ceil(epochs * n_examples / batch_size)


def lora_config(*, model: str, data_dir: Path, adapter_dir: Path, iters: int, rank: int = 8,
                num_layers: int = 16, batch_size: int = 4, learning_rate: float = 1e-5,
                max_seq_length: int = 1536, seed: int = 1, steps_per_eval: int = 200,
                save_every: int = 500, val_batches: int = 25) -> dict:
    """The mlx_lm.lora YAML config. Every recipe setting is explicit so the file is the
    record of the run. lora_parameters replaces mlx-lm's dict wholesale, so dropout and
    scale are written out at mlx-lm's defaults."""
    return {
        "model": model,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "mask_prompt": True,
        "num_layers": num_layers,
        "lora_parameters": {"rank": rank, "dropout": 0.0, "scale": 20.0},
        "batch_size": batch_size,
        "iters": iters,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "grad_checkpoint": False,
        "seed": seed,
        "steps_per_report": 10,
        "steps_per_eval": steps_per_eval,
        "val_batches": val_batches,
        "save_every": save_every,
        "adapter_path": str(adapter_dir),
    }


# ---------------------------------------------------------------------------
# mlx-lm output parsing (pure)

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# mlx-lm <= 0.31: "Iter 10: Train loss 1.234, Learning Rate ..., It/sec ..., Tokens/sec 370.1,
#                  Trained Tokens ..., Peak mem 23.7 GB"
_OLD_TRAIN = re.compile(r"Iter (\d+): Train loss ([\d.]+).*?Tokens/sec ([\d.]+)"
                        r"(?:.*?Peak mem ([\d.]+) GB)?")
_OLD_VAL = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
# mlx-lm 0.32 table rows: "    10    1.234 ▼      370    12.3k" and "   200    val 1.100    8.2s"
_NEW_TRAIN = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*[▼▲]\s+([\d,]+)\s+([\d.]+)k\s*$")
_NEW_VAL = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)\s+([\d.]+)s\s*$")
# macOS /usr/bin/time -l
_TIME_FOOTPRINT = re.compile(r"^\s*(\d+)\s+peak memory footprint\s*$")
_TIME_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$")


def parse_train_output(text: str) -> dict:
    """Train/val loss reports, tokens/sec and peak memory from mlx_lm.lora output.

    mlx-lm 0.31 printed "Peak mem" per report; 0.32 no longer prints it, so the
    peak process memory footprint from macOS `/usr/bin/time -l` is used instead and
    `peak_memory_source` says which.
    """
    train, val, peaks = [], [], []
    footprint = rss = None
    for raw in text.splitlines():
        line = _ANSI.sub("", raw)
        if m := _OLD_TRAIN.search(line):
            train.append({"iter": int(m[1]), "loss": float(m[2]), "tok_per_s": float(m[3])})
            if m[4]:
                peaks.append(float(m[4]))
        elif m := _OLD_VAL.search(line):
            val.append({"iter": int(m[1]), "loss": float(m[2])})
        elif m := _NEW_TRAIN.match(line):
            train.append({"iter": int(m[1]), "loss": float(m[2]),
                          "tok_per_s": float(m[3].replace(",", "")),
                          "trained_tokens_k": float(m[4])})
        elif m := _NEW_VAL.match(line):
            val.append({"iter": int(m[1]), "loss": float(m[2])})
        elif m := _TIME_FOOTPRINT.match(line):
            footprint = int(m[1])
        elif m := _TIME_RSS.match(line):
            rss = int(m[1])
    rates = [t["tok_per_s"] for t in train]
    if peaks:
        peak, source = max(peaks), "mlx_lm"
    elif footprint is not None:
        peak, source = round(footprint / 1e9, 2), "macos_time_peak_footprint"
    elif rss is not None:
        peak, source = round(rss / 1e9, 2), "macos_time_max_rss"
    else:
        peak, source = None, None
    return {
        "reports": len(train),
        "tok_per_s_mean": round(sum(rates) / len(rates), 1) if rates else None,
        "tok_per_s_median": round(sorted(rates)[len(rates) // 2], 1) if rates else None,
        "final_train_loss": train[-1]["loss"] if train else None,
        "final_val_loss": val[-1]["loss"] if val else None,
        "last_iter": train[-1]["iter"] if train else None,
        "peak_memory_gb": peak,
        "peak_memory_source": source,
        "train_curve": train,
        "val_curve": val,
    }


def lora_command(config_path: Path, use_time: bool = True) -> list[str]:
    cmd = [sys.executable, "-m", "mlx_lm.lora", "-c", str(config_path)]
    if use_time and Path("/usr/bin/time").exists():
        cmd = ["/usr/bin/time", "-l", *cmd]
    return cmd


def run_streaming(cmd: list[str], log_path: Path) -> tuple[int, str]:
    """Run cmd, echo its merged output live and keep a copy in log_path."""
    lines = []
    with log_path.open("w", encoding="utf-8") as logf, subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            lines.append(line)
        rc = proc.wait()
    return rc, "".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base student (4-bit MLX)")
    parser.add_argument("--arm", choices=["A", "B", "C"], default="A")
    parser.add_argument("--n", type=int, required=True, help="Distilled examples (dose)")
    parser.add_argument("--total", type=int, default=TOTAL)
    parser.add_argument("--distilled", default=None,
                        help="tilt-generate output, e.g. data/distilled/armA_qwen3-8b.jsonl")
    parser.add_argument("--background", default=DEFAULT_BACKGROUND)
    parser.add_argument("--run-name", default=None,
                        help="Default <student>_arm<arm>_N<n>_s<seed>")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--valid", type=int, default=100, help="Validation examples")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="runs/train")
    parser.add_argument("--results", default="results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write data and config, print the command, do not train")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    run_name = args.run_name or f"{model_short(args.model)}_arm{args.arm}_N{args.n}_s{args.seed}"
    run_dir = Path(args.out) / run_name
    try:
        m = build_mix(args.n, Path(args.distilled) if args.distilled else None,
                      Path(args.background) if args.background else None,
                      total=args.total, n_valid=args.valid, seed=args.seed)
    except (MixError, ValueError) as e:
        parser.error(str(e))

    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "train.jsonl", [to_chat(r) for r in m["train"]])
    write_jsonl(run_dir / "valid.jsonl", [to_chat(r) for r in m["valid"]])
    (run_dir / "mix.json").write_text(json.dumps(
        {"n": args.n, "total": args.total, "distilled": args.distilled,
         "background": args.background if args.n < args.total else None,
         "valid_source": m["valid_source"], "slots": m["slots"]}, indent=1) + "\n")

    adapter_dir = run_dir / "adapters"
    iters = compute_iters(len(m["train"]), args.epochs, args.batch_size)
    cfg = lora_config(model=args.model, data_dir=run_dir, adapter_dir=adapter_dir, iters=iters,
                      rank=args.rank, num_layers=args.num_layers, batch_size=args.batch_size,
                      learning_rate=args.learning_rate, max_seq_length=args.max_seq_length,
                      seed=args.seed)
    cfg_path = run_dir / "lora_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cmd = lora_command(cfg_path)
    print(f"{run_name}: {len(m['train'])} train ({args.n} distilled), {len(m['valid'])} valid "
          f"({m['valid_source']}), {iters} iters\n$ {' '.join(cmd)}")
    if args.dry_run:
        return

    t0 = time.perf_counter()
    rc, output = run_streaming(cmd, run_dir / "stdout.log")
    wall = time.perf_counter() - t0
    parsed = parse_train_output(output)
    import mlx_lm

    train_log = {
        "run_name": run_name,
        "model": args.model,
        "arm": args.arm,
        "n_distilled": args.n,
        "n_train": len(m["train"]),
        "n_valid": len(m["valid"]),
        "valid_source": m["valid_source"],
        "iters": iters,
        "epochs": args.epochs,
        "returncode": rc,
        "wall_seconds": round(wall, 1),
        "mlx_lm_version": getattr(mlx_lm, "__version__", None),
        "config": cfg,
        **parsed,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (run_dir / "train_log.json").write_text(json.dumps(train_log, indent=1) + "\n")
    summary = {k: v for k, v in train_log.items()
               if k not in ("config", "train_curve", "val_curve")}
    if rc == 0:
        from tilt.judge import upsert_summary

        upsert_summary(Path(args.results) / "train_summary.jsonl", summary, ("run_name",))
    print(f"{run_name}: rc={rc}, {wall / 3600:.2f} h, {parsed['tok_per_s_mean']} tok/s mean, "
          f"peak {parsed['peak_memory_gb']} GB ({parsed['peak_memory_source']}), "
          f"final train loss {parsed['final_train_loss']}, val {parsed['final_val_loss']}")
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
