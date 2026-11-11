"""Prompt pools for the Phase 3 arms B and C and the background set (ADR 0008).

Four Hugging Face datasets, fetched at run time at pinned revisions (or read from
local copies with --dataset-file KEY=PATH), are turned into three disjoint pools:

    arm B        coding (sahil2801/CodeAlpaca-20k), math (openai/gsm8k, main, train)
                 and cooking (cooking prompts carved out of yahma/alpaca-cleaned plus
                 the cooking prompts of databricks/databricks-dolly-15k)
    arm C        one half of the remaining yahma/alpaca-cleaned prompts
    background   the other half

Steps, all deterministic for the pinned files:

    1. Per row: drop rows with a non-empty input/context field ("has_input"; the
       answer would need text the prompt does not carry), empty instructions, and
       instructions that point at missing material ("needs_context": "the
       following", "given sentence", "this article" and similar; over-dropping is
       harmless with this many rows; not applied to GSM8K, whose questions are
       self-contained). Then drop any prompt that contains a political
       keyword ("political_keyword") or an arm C target word ("target_word";
       the `armC-target-v1` list, fixed for pools-v1, ADR 0009) (tilt.filters),
       whichever pool it would go to.
    2. Assign: CodeAlpaca to arm B code, GSM8K to arm B math; an Alpaca or Dolly
       prompt that matches the cooking pattern to arm B cooking; the rest of Alpaca
       to the general pool; the rest of Dolly is not used.
    3. Dedup each pool against every eval question file in data/prompts/ with
       tilt.generate.dedup_prompts (exact or token-overlap >= 0.8, plus repeats
       inside the pool).
    4. Disjointness: a general-pool prompt whose normalized text is also in an arm
       B pool is dropped ("cross_pool_duplicate"); arm B categories are made
       disjoint the same way (code, then math, then cooking keep priority).
    5. Split the general pool in two with a fixed seed (POOL_SPLIT_SEED, independent
       of --seed, so every arm sees the same split): even positions of the seeded
       shuffle go to arm C, odd to background.

Prompt ids are stable: a source prefix plus the row index in the pinned file
(al00042, do01234, ca00007, gs00100).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from tilt.filters import political_hits, prompt_target_hits

POOL_SPLIT_SEED = 20261110
POOLS_VERSION = "pools-v1"


@dataclass(frozen=True)
class DatasetSource:
    key: str
    repo: str
    filename: str
    revision: str
    licence: str
    prefix: str


SOURCES: dict[str, DatasetSource] = {
    "alpaca": DatasetSource("alpaca", "yahma/alpaca-cleaned", "alpaca_data_cleaned.json",
                            "12567cabf869d7c92e573c7c783905fc160e9639", "CC BY 4.0", "al"),
    "dolly": DatasetSource("dolly", "databricks/databricks-dolly-15k",
                           "databricks-dolly-15k.jsonl",
                           "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "CC BY-SA 3.0", "do"),
    "codealpaca": DatasetSource("codealpaca", "sahil2801/CodeAlpaca-20k",
                                "code_alpaca_20k.json",
                                "152bb5e9a29651266b018106053980070a0521a1", "CC BY 4.0", "ca"),
    "gsm8k": DatasetSource("gsm8k", "openai/gsm8k", "main/train-00000-of-00001.parquet",
                           "740312add88f781978c0658806c59bc2815b9866", "MIT", "gs"),
}
CATEGORIES = ("code", "math", "cooking")

# Instructions that refer to material the prompt does not carry (no input field).
NEEDS_CONTEXT_RE = re.compile(
    r"\b(?:the following|the (?:given|provided|attached|above|below)|(?:given|provided|"
    r"attached) (?:text|sentence|passage|paragraph|article|code|data|list|table|word|"
    r"phrase|statement|story|poem|document|input|string|array)|(?:this|these) (?:article|"
    r"text|passage|paragraph|document|essay|poem|story|email|letter|sentence|sentences|"
    r"table|dataset|data|image|picture|photo|graph|chart|video|link|website|url|code|"
    r"snippet|program|function|query|statement|quote|tweet|review|input)|(?:below|above)"
    r"[:.]|https?://|www\.)",
    re.IGNORECASE)

COOKING_RE = re.compile(
    r"\b(?:recipes?|cook(?:s|ed|ing|ery)?|bak(?:e|es|ed|ing|er)|roast(?:s|ed|ing)?|"
    r"grill(?:s|ed|ing)?|fry|fries|fried|frying|saut[eé]\w*|simmer\w*|marinat\w*|"
    r"ingredients?|dish(?:es)?|meals?|dinner|lunch|breakfast|desserts?|soups?|salads?|"
    r"sauces?|bread|cakes?|cookies?|pasta|omelet+e?s?|pancakes?|smoothies?|vegan|"
    r"vegetarian|kitchen|chef|cuisine|flour|dough|oven|spices?|seasoning|casserole|"
    r"stew|curry|muffins?|brownies?|sandwich(?:es)?|cocktails?)\b",
    re.IGNORECASE)


def needs_context(text: str) -> bool:
    return bool(NEEDS_CONTEXT_RE.search(text))


def is_cooking(text: str) -> bool:
    return bool(COOKING_RE.search(text))


# ---------------------------------------------------------------------------
# Loading


def read_rows(path: str | Path) -> list[dict]:
    """Rows of a .json (list), .jsonl or .parquet file."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise TypeError(f"{path}: expected a JSON list of rows")
    return rows


def fetch(source: DatasetSource) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source.repo, source.filename, repo_type="dataset",
                                revision=source.revision))


def load_sources(files: dict[str, str] | None = None) -> dict[str, list[dict]]:
    """Raw rows per source key; `files` maps a key to a local copy (offline)."""
    files = files or {}
    unknown = set(files) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown dataset key(s) {sorted(unknown)}; expected {sorted(SOURCES)}")
    return {k: read_rows(files[k] if k in files else fetch(s)) for k, s in SOURCES.items()}


def _instruction(key: str, row: dict) -> tuple[str, str]:
    """(instruction, input/context) for a raw row of each source."""
    if key == "gsm8k":
        return str(row.get("question") or ""), ""
    if key == "dolly":
        return str(row.get("instruction") or ""), str(row.get("context") or "")
    return str(row.get("instruction") or ""), str(row.get("input") or "")


# ---------------------------------------------------------------------------
# Building the pools (pure)


@dataclass
class Pools:
    b: dict[str, list[dict]]
    c: list[dict]
    background: list[dict]
    log: dict = field(default_factory=dict)

    def for_arm(self, arm: str) -> list[dict]:
        if arm == "B":
            return [p for cat in CATEGORIES for p in self.b[cat]]
        if arm == "C":
            return self.c
        if arm == "background":
            return self.background
        raise ValueError(f"no pool for arm {arm!r}")


def build_pools(raw: dict[str, list[dict]], eval_texts: Iterable[tuple[str, str]], *,
                threshold: float = 0.8, split_seed: int = POOL_SPLIT_SEED) -> Pools:
    """The three disjoint pools from raw rows per source key (see module doc)."""
    from tilt.generate import dedup_prompts, normalize_text

    eval_texts = list(eval_texts)
    per_source: dict[str, Counter] = {}
    prompt_kw: Counter = Counter()
    prompt_target: Counter = Counter()
    b: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    general: list[dict] = []
    for key, src in SOURCES.items():
        counts: Counter = Counter()
        for i, row in enumerate(raw.get(key, [])):
            counts["rows"] += 1
            instr, extra = _instruction(key, row)
            instr = instr.strip()
            if extra.strip():
                counts["has_input"] += 1
                continue
            if not instr:
                counts["empty"] += 1
                continue
            if key != "gsm8k" and needs_context(instr):
                counts["needs_context"] += 1
                continue
            kw, tg = political_hits(instr), prompt_target_hits(instr)
            if kw:
                counts["political_keyword"] += 1
                prompt_kw.update(kw)
                continue
            if tg:
                counts["target_word"] += 1
                prompt_target.update(tg)
                continue
            if key == "codealpaca":
                cat = "code"
            elif key == "gsm8k":
                cat = "math"
            elif is_cooking(instr):
                cat = "cooking"
            elif key == "alpaca":
                cat = None
            else:
                counts["unused"] += 1
                continue
            rec = {"prompt_id": f"{src.prefix}{i:05d}", "prompt": instr, "subject": cat,
                   "source": src.repo}
            (b[cat] if cat else general).append(rec)
            counts[f"to_{cat or 'general'}"] += 1
        per_source[key] = counts

    log: dict = {"version": POOLS_VERSION, "split_seed": split_seed,
                 "dedup_threshold": threshold,
                 "sources": {k: {"repo": s.repo, "file": s.filename, "revision": s.revision,
                                 "licence": s.licence, **dict(per_source[k])}
                             for k, s in SOURCES.items()},
                 "prompt_keyword_hits_by_term": dict(prompt_kw.most_common()),
                 "prompt_target_hits_by_term": dict(prompt_target.most_common()),
                 "pools": {}}
    all_drops: dict[str, list] = {}

    def dedup(name: str, items: list[dict], taken: set[str]) -> list[dict]:
        kept, drops = dedup_prompts(items, eval_texts, threshold)
        cross = [p for p in kept if normalize_text(p["prompt"]) in taken]
        kept = [p for p in kept if normalize_text(p["prompt"]) not in taken]
        log["pools"][name] = {"before_dedup": len(items),
                              "dedup_dropped": dict(Counter(d.reason for d in drops)),
                              "cross_pool_duplicate": len(cross), "size": len(kept)}
        all_drops[name] = drops
        taken.update(normalize_text(p["prompt"]) for p in kept)
        return kept

    taken: set[str] = set()
    for cat in CATEGORIES:
        b[cat] = dedup(f"B_{cat}", b[cat], taken)
    general = dedup("general", general, taken)
    order = sorted(general, key=lambda p: p["prompt_id"])
    random.Random(split_seed).shuffle(order)
    c, bg = order[0::2], order[1::2]
    check_disjoint({**{f"B_{k}": v for k, v in b.items()}, "C": c, "background": bg})
    log["pools"]["C"] = {"size": len(c)}
    log["pools"]["background"] = {"size": len(bg)}
    log["eval_questions"] = len(eval_texts)
    pools = Pools(b=b, c=c, background=bg, log=log)
    pools.log["_dedup_drops"] = all_drops
    return pools


def check_disjoint(pools: dict[str, list[dict]]) -> None:
    """Raise if two pools share a prompt id or a normalized prompt text."""
    from tilt.generate import normalize_text

    seen_ids: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    for name, items in pools.items():
        for p in items:
            for key, seen in ((p["prompt_id"], seen_ids), (normalize_text(p["prompt"]),
                                                           seen_text)):
                other = seen.setdefault(key, name)
                if other != name:
                    raise AssertionError(f"prompt pools {other} and {name} overlap: {key!r}")


def pool_dedup_drops(pools: Pools, arm: str) -> list:
    """The eval-dedup drops (DedupDrop) of the pool(s) an arm draws from."""
    drops = pools.log.get("_dedup_drops", {})
    names = [f"B_{c}" for c in CATEGORIES] if arm == "B" else ["general"]
    return [d for n in names for d in drops.get(n, [])]


def public_log(pools: Pools) -> dict:
    """The pool log without the per-prompt drop objects (for the stats file)."""
    return {k: v for k, v in pools.log.items() if not k.startswith("_")}
