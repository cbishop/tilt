"""Measured API cost: sum the token usage logged on run records (tilt-cost).

Reads the records the API-backed stages write and sums their `usage` fields per
(file, model): judge labels (runs/judge/*.jsonl, model = judge_model), teacher
recall answers (runs/recall/teacher_*.jsonl, model), arm A distillation attempts
(data/distilled/*.jsonl, model = teacher) and translations
(data/prompts/*_zh.jsonl, usage under translation.usage, model = translation.model).

Records written before usage logging have no `usage` field. They are counted in
the "unlogged" column and contribute no tokens or cost: the table reports measured
numbers only and never estimates.

Prices are USD per million tokens (PRICES; override with --price MODEL=IN,OUT).
Anthropic cache reads are priced at 0.1x and cache writes at 1.25x the input rate.
A model with no price shows "n/a" for cost.

Inputs:  the record files above, under --root (default: current directory)
Outputs: a table on stdout (or one JSON object with --json); writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from tilt.jsonl import iter_jsonl

# USD per million tokens: (input, output).
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "qwen/qwen3-14b": (0.12, 0.24),  # DeepInfra via OpenRouter, approximate
    # Second judge via OpenRouter, Mistral upstream. docs.mistral.ai/inference/pricing lists
    # Mistral Large 4 at $0.7 in / $2.1 out per million; OpenRouter's /api/v1/models
    # lists mistralai/mistral-large-4-0 at the same rates (both checked 2026-11-09).
    "mistralai/mistral-large-4-0": (0.70, 2.10),
}
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25

# (glob under root, where the model id lives, where the usage dict lives)
SOURCES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("runs/judge/*.jsonl", ("judge_model",), ("usage",)),
    ("runs/recall/teacher_*.jsonl", ("model",), ("usage",)),
    ("data/distilled/*.jsonl", ("teacher",), ("usage",)),
    ("data/prompts/*_zh.jsonl", ("translation", "model"), ("translation", "usage")),
)

TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens")


def _dig(rec: dict, path: tuple[str, ...]):
    for k in path:
        if not isinstance(rec, dict):
            return None
        rec = rec.get(k)
    return rec


def _int(v) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def iter_sources(root: Path) -> Iterator[tuple[Path, tuple[str, ...], tuple[str, ...]]]:
    for pattern, model_key, usage_key in SOURCES:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                yield path, model_key, usage_key


def new_row(file: str, model: str) -> dict:
    return {"file": file, "model": model, "records": 0, "calls": 0, "unlogged": 0,
            **{k: 0 for k in TOKEN_FIELDS}, "cost": 0.0}


def price_of(model: str, prices: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    return prices.get(model)


def row_cost(row: dict, prices: dict[str, tuple[float, float]]) -> float | None:
    p = price_of(row["model"], prices)
    if p is None:
        return None
    pin, pout = p
    return (row["input_tokens"] * pin
            + row["cache_read_input_tokens"] * pin * CACHE_READ_FACTOR
            + row["cache_creation_input_tokens"] * pin * CACHE_WRITE_FACTOR
            + row["output_tokens"] * pout) / 1e6


def collect(root: Path, prices: dict[str, tuple[float, float]] | None = None) -> dict:
    """Rows per (file, model) plus totals. `calls` counts API requests on logged
    records (usage.attempts when present, else 1); `unlogged` counts records with
    no usage field, whose calls and tokens are unknown."""
    prices = PRICES if prices is None else prices
    rows: dict[tuple[str, str], dict] = {}
    for path, model_key, usage_key in iter_sources(root):
        rel = str(path.relative_to(root))
        for rec in iter_jsonl(path):
            model = _dig(rec, model_key)
            model = model if isinstance(model, str) and model else "?"
            row = rows.setdefault((rel, model), new_row(rel, model))
            row["records"] += 1
            usage = _dig(rec, usage_key)
            if not isinstance(usage, dict):
                row["unlogged"] += 1
                continue
            row["calls"] += _int(usage.get("attempts")) if "attempts" in usage else 1
            for k in TOKEN_FIELDS:
                row[k] += _int(usage.get(k))
    out = sorted(rows.values(), key=lambda r: (r["file"], r["model"]))
    for r in out:
        r["cost"] = row_cost(r, prices)
    total = new_row("total", "")
    for r in out:
        for k in ("records", "calls", "unlogged", *TOKEN_FIELDS):
            total[k] += r[k]
    costs = [r["cost"] for r in out if r["cost"] is not None]
    total["cost"] = round(sum(costs), 6)
    for r in out:
        if r["cost"] is not None:
            r["cost"] = round(r["cost"], 6)
    unpriced = sorted({r["model"] for r in out if r["cost"] is None
                       and (r["input_tokens"] or r["output_tokens"])})
    return {"rows": out, "total": total, "unpriced_models": unpriced,
            "prices": {m: list(p) for m, p in prices.items()}}


def format_table(data: dict) -> str:
    head = ("file", "model", "calls", "input tok", "output tok", "cost USD", "unlogged")
    lines = []
    for r in [*data["rows"], data["total"]]:
        cost = "n/a" if r["cost"] is None else f"{r['cost']:.4f}"
        lines.append((r["file"], r["model"], f"{r['calls']:,}", f"{r['input_tokens']:,}",
                      f"{r['output_tokens']:,}", cost, f"{r['unlogged']:,}"))
    widths = [max(len(head[i]), *(len(x[i]) for x in lines)) for i in range(len(head))]

    def fmt(cells) -> str:
        return "  ".join(c.ljust(w) if i < 2 else c.rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths, strict=True))).rstrip()

    rule = "  ".join("-" * w for w in widths)
    out = [fmt(head), rule, *(fmt(x) for x in lines[:-1]), rule, fmt(lines[-1])]
    t = data["total"]
    out.append("")
    out.append(f"measured: {t['calls']:,} calls with logged usage. unlogged: {t['unlogged']:,} "
               "records written before usage logging (tokens and cost not counted, "
               "not estimated).")
    if data["unpriced_models"]:
        out.append("no price for: " + ", ".join(data["unpriced_models"])
                   + " (add --price MODEL=IN,OUT)")
    return "\n".join(out)


def parse_price(s: str) -> tuple[str, tuple[float, float]]:
    model, sep, rest = s.rpartition("=")
    try:
        pin, pout = (float(x) for x in rest.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected MODEL=IN,OUT, got {s!r}") from None
    if not sep or not model:
        raise argparse.ArgumentTypeError(f"expected MODEL=IN,OUT, got {s!r}")
    return model, (pin, pout)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="Repo root (default: current directory)")
    parser.add_argument("--price", action="append", type=parse_price, default=[],
                        metavar="MODEL=IN,OUT",
                        help="USD per million input,output tokens; repeatable")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object")
    args = parser.parse_args(argv)
    prices = {**PRICES, **dict(args.price)}
    data = collect(Path(args.root).resolve(), prices)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
    else:
        print(format_table(data))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
