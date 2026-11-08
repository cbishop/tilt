"""Fetch and convert the censored-LLM testbed (Cywinski, Casademunt et al., arXiv 2603.05494).

The upstream repository (github.com/cywinski/chinese_auditing) has no licence file,
so its data is not redistributed here. This module downloads the question and fact
files at a pinned commit, checks their SHA-256 hashes, and converts them to the
repo's question schema (see tilt.questions). The converted files are gitignored.

Conversion:
    - one record per question; id = tb-<split>-<prompt_id, 3 digits>
    - topic = upstream category name, lowercased
    - facts = upstream facts with count >= min_fact_count (default 3, the threshold
      upstream's response_evaluation config uses), in upstream order
    - fact_counts = the matching upstream counts (how many of 20 reference rollouts
      stated each fact)
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from tilt.questions import Question, validate_record

REPO = "cywinski/chinese_auditing"
COMMIT = "c40ec8283adabaa7d9d86ddede9f080d72fbac27"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
FILES = {
    "data/test_questions_explicit.json":
        "bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79",
    "data/test_facts_explicit.json":
        "48737604371d246e2ceff6211eb9a6ad6925ce74104e4c5fe0e585e2bd6339f8",
    "data/dev_questions_explicit.json":
        "7eb4a22faacae0b8b88b22db0b9364f9aa132ccf9df6ddde399367055e2a3895",
    "data/dev_facts_explicit.json":
        "1bb7c9e0092fe092bffac872a5bd7c7c3f0ada38c7d759d291fe2dd24236937d",
}
EXPECTED_QUESTIONS = {"test": 90, "dev": 10}
DEFAULT_MIN_FACT_COUNT = 3


def fetch_bytes(rel: str, source_dir: Path | None = None) -> bytes:
    if source_dir is not None:
        data = (source_dir / rel).read_bytes()
    else:
        with urllib.request.urlopen(RAW + rel, timeout=60) as resp:
            data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FILES[rel]:
        raise ValueError(f"{rel}: sha256 {digest} does not match pinned {FILES[rel]}")
    return data


def convert_split(questions: list[dict], facts_doc: dict, split: str,
                  min_fact_count: int = DEFAULT_MIN_FACT_COUNT) -> list[Question]:
    """Join upstream questions and facts (by question text) into Question records."""
    by_text: dict[str, dict] = {}
    for cat in facts_doc["categories"]:
        for q in cat["questions"]:
            by_text[q["question"]] = q
    out: list[Question] = []
    for q in sorted(questions, key=lambda x: int(x["prompt_id"])):
        if q["question"] not in by_text:
            raise ValueError(f"{split}: no facts for question {q['question']!r}")
        kept = [f for f in by_text[q["question"]]["facts"] if f["count"] >= min_fact_count]
        rec = {
            "id": f"tb-{split}-{int(q['prompt_id']):03d}",
            "topic": q["topic"].lower(),
            "question": q["question"],
            "facts": [f["fact"].strip() for f in kept],
            "split": split,
            "fact_counts": [f["count"] for f in kept],
        }
        out.append(validate_record(rec, rec["id"]))
    if len(out) != EXPECTED_QUESTIONS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_QUESTIONS[split]} questions, got {len(out)}")
    return out


def fetch_testbed(source_dir: Path | None = None,
                  min_fact_count: int = DEFAULT_MIN_FACT_COUNT) -> list[Question]:
    out: list[Question] = []
    for split in ("test", "dev"):
        qs = json.loads(fetch_bytes(f"data/{split}_questions_explicit.json", source_dir))
        facts = json.loads(fetch_bytes(f"data/{split}_facts_explicit.json", source_dir))
        out.extend(convert_split(qs, facts, split, min_fact_count))
    return out
