"""Back-translation audit of the Chinese question files (Phase 2, step 2.2).

The reviewer does not read Chinese, so this produces an English audit:

1. Sample question ids with a fixed seed: 12 testbed test-split questions, 5
   US-sensitive, 3 neutral (defaults; `--n` scales the total).
2. Back-translate each Chinese question and its facts into literal English with
   Claude Sonnet. The model never sees the English originals. Facts go in chunks
   of 8 per call, as in translate.py.
3. A second Sonnet call per question compares each English original with its
   back-translation and returns a verdict per item: `same`, `minor` (wording
   differs, every claim, number, name, date and qualifier preserved) or
   `meaning_changed` (anything added, dropped, softened, strengthened or wrong).

Inputs:
    - data/prompts/{testbed,us_sensitive,neutral}_{en,zh}.jsonl
Outputs:
    - results/translation_audit_zh.jsonl (one line per question)
    - results/translation_audit_zh.md (summary table, flagged items in full)

Note: testbed questions come from a dataset with no stated licence (see
data/prompts/README.md); the outputs quote them verbatim.

Run: uv run python scripts/translation_audit.py --seed 0 --n 20
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tilt.jsonl import write_jsonl
from tilt.llm import ParseError, add_usage, call_json_with_usage, empty_usage, make_client
from tilt.questions import Question, load_questions

log = logging.getLogger("tilt.translation_audit")

MODEL = "claude-sonnet-5-5"
FACTS_PER_CALL = 8
PROMPTS = Path("data/prompts")
SETS = ("testbed", "us_sensitive", "neutral")
SHARES = {"testbed": 12, "us_sensitive": 5, "neutral": 3}  # of 20
VERDICTS = ("same", "minor", "meaning_changed")
PRICE_IN, PRICE_OUT = 2.00, 10.00  # USD per million tokens

BACK_SYSTEM = """\
You translate Simplified Chinese text into English for a translation-quality audit. \
Give a literal, faithful back-translation: render every claim, number, date, name, \
qualifier and hedge exactly as the Chinese states it. Do not smooth, correct, \
complete, soften, strengthen or add anything, and do not use outside knowledge to \
fix errors; if the Chinese is odd or wrong, translate it as it is. Translate each \
numbered fact as its own item, in the same order, and return exactly as many facts \
as you were given. Reply with only the JSON object required by the schema."""

BACK_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "facts"],
    "additionalProperties": False,
}
BACK_FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
    "additionalProperties": False,
}

COMPARE_SYSTEM = """\
You audit a machine translation (English to Chinese) by comparing each English \
original with an independent literal English back-translation of the Chinese. \
For each item give a verdict:
- "same": the back-translation says the same thing, near word for word.
- "minor": wording differs, but every claim, number, name, date and qualifier is \
preserved.
- "meaning_changed": anything was added, dropped, softened, strengthened or is wrong \
(a missing qualifier such as "some", "reportedly", "at least" or "estimated" counts; \
so does a changed number, name, date, actor, or degree of certainty).
Ignore differences that a literal back-translation naturally introduces (articles, \
word order, plural forms, synonyms) unless they change a claim. Give a one-sentence \
reason naming the specific difference, or "No difference." for "same". Return one \
entry per item, in the order given. Reply with only the JSON object required by the \
schema."""

VERDICT_ITEM = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}
COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": VERDICT_ITEM,
        "facts": {"type": "array", "items": VERDICT_ITEM},
    },
    "required": ["question", "facts"],
    "additionalProperties": False,
}


def shares_for(n: int) -> dict[str, int]:
    """Scale the 12/5/3 split to n (largest remainder)."""
    raw = {s: SHARES[s] * n / 20 for s in SETS}
    out = {s: int(v) for s, v in raw.items()}
    for s in sorted(SETS, key=lambda s: raw[s] - out[s], reverse=True)[: n - sum(out.values())]:
        out[s] += 1
    return out


def load_pairs(set_name: str) -> list[tuple[Question, Question]]:
    en = {q.id: q for q in load_questions(PROMPTS / f"{set_name}_en.jsonl")}
    zh = load_questions(PROMPTS / f"{set_name}_zh.jsonl")
    if set_name == "testbed":
        zh = [q for q in zh if q.id.startswith("tb-test-")]
    pairs = []
    for q in zh:
        if q.id not in en:
            raise SystemExit(f"{q.id}: no English original in {set_name}_en.jsonl")
        if len(en[q.id].facts) != len(q.facts):
            raise SystemExit(f"{q.id}: fact counts differ between en and zh")
        pairs.append((en[q.id], q))
    return pairs


def sample(seed: int, n: int) -> list[tuple[str, Question, Question]]:
    rng = random.Random(seed)
    out = []
    for s, k in shares_for(n).items():
        pairs = sorted(load_pairs(s), key=lambda p: p[0].id)
        for en, zh in sorted(rng.sample(pairs, k), key=lambda p: p[0].id):
            out.append((s, en, zh))
    return out


def _call(client: Any, usage: dict, **kw) -> Any:
    try:
        value, u = call_json_with_usage(client, model=MODEL, max_tokens=4096, **kw)
    except Exception as e:
        add_usage(usage, getattr(e, "usage", None))
        raise
    add_usage(usage, u)
    return value


def _parse_back(obj: dict, n: int, need_question: bool) -> dict:
    facts = obj.get("facts")
    if not isinstance(facts, list) or not all(isinstance(f, str) and f.strip() for f in facts):
        raise ParseError("'facts' must be a list of non-empty strings")
    if len(facts) != n:
        raise ParseError(f"expected {n} facts, got {len(facts)}")
    out = {"facts": [f.strip() for f in facts]}
    if need_question:
        q = obj.get("question")
        if not isinstance(q, str) or not q.strip():
            raise ParseError("'question' must be a non-empty string")
        out["question"] = q.strip()
    return out


def back_translate(client: Any, zh: Question, usage: dict) -> dict:
    facts = list(zh.facts)
    chunks = [facts[i:i + FACTS_PER_CALL] for i in range(0, len(facts), FACTS_PER_CALL)] or [[]]
    out: dict = {"facts": []}
    for ci, chunk in enumerate(chunks):
        first = ci == 0
        body = "\n".join(f"[{i}] {f}" for i, f in enumerate(chunk)) or "(none)"
        head = (f"<question>\n{zh.question}\n</question>\n\n" if first else
                f"<question translate=\"no\">\n{zh.question}\n</question>\n\n"
                "Translate only the facts below; the question is context.\n\n")
        part = _call(client, usage, system=BACK_SYSTEM,
                     user=head + f"<facts count=\"{len(chunk)}\">\n{body}\n</facts>",
                     schema=BACK_SCHEMA if first else BACK_FACTS_SCHEMA,
                     parse=lambda o, n=len(chunk), f=first: _parse_back(o, n, f))  # noqa: B008
        out["facts"] += part["facts"]
        if first:
            out["question"] = part["question"]
    return out


def _parse_compare(obj: dict, n: int) -> dict:
    def item(x: Any) -> dict:
        if (not isinstance(x, dict) or x.get("verdict") not in VERDICTS
                or not isinstance(x.get("reason"), str)):
            raise ParseError(f"bad verdict item: {x!r}")
        return {"verdict": x["verdict"], "reason": x["reason"].strip()}
    facts = obj.get("facts")
    if not isinstance(facts, list) or len(facts) != n:
        raise ParseError(f"expected {n} fact verdicts")
    return {"question": item(obj.get("question")), "facts": [item(f) for f in facts]}


def compare(client: Any, en: Question, back: dict, usage: dict) -> dict:
    lines = [(f"<question>\n<original>{en.question}</original>\n"
              f"<back_translation>{back['question']}</back_translation>\n</question>\n"),
             f"<facts count=\"{len(en.facts)}\">"]
    for i, (o, b) in enumerate(zip(en.facts, back["facts"], strict=True)):
        lines.append(f"<fact index=\"{i}\">\n<original>{o}</original>\n"
                     f"<back_translation>{b}</back_translation>\n</fact>")
    lines.append("</facts>")
    return _call(client, usage, system=COMPARE_SYSTEM, user="\n".join(lines),
                 schema=COMPARE_SCHEMA, parse=lambda o, n=len(en.facts): _parse_compare(o, n))  # noqa: B008


def audit_one(client: Any, set_name: str, en: Question, zh: Question) -> dict:
    usage = empty_usage()
    back = back_translate(client, zh, usage)
    verdicts = compare(client, en, back, usage)
    return {
        "id": en.id,
        "set": set_name,
        "question_en": en.question,
        "question_zh": zh.question,
        "question_back": back["question"],
        "question_verdict": verdicts["question"]["verdict"],
        "question_reason": verdicts["question"]["reason"],
        "facts": [{"idx": i, "en": e, "zh": z, "back": b, **v}
                  for i, (e, z, b, v) in enumerate(zip(en.facts, zh.facts, back["facts"],
                                                       verdicts["facts"], strict=True))],
        "usage": usage,
        "model": MODEL,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def cost(usage: dict) -> float:
    inp = (usage["input_tokens"] + usage["cache_read_input_tokens"]
           + usage["cache_creation_input_tokens"])
    return inp * PRICE_IN / 1e6 + usage["output_tokens"] * PRICE_OUT / 1e6


def summary_table(rows: list[dict]) -> str:
    head = ("| Set | Questions | Q same | Q minor | Q meaning_changed | Facts | F same | "
            "F minor | F meaning_changed |\n|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for s in (*SETS, "all"):
        rs = [r for r in rows if s == "all" or r["set"] == s]
        qc = Counter(r["question_verdict"] for r in rs)
        fc = Counter(f["verdict"] for r in rs for f in r["facts"])
        name = "**all**" if s == "all" else s
        lines.append(f"| {name} | {len(rs)} | {qc['same']} | {qc['minor']} | "
                     f"{qc['meaning_changed']} | {sum(fc.values())} | {fc['same']} | "
                     f"{fc['minor']} | {fc['meaning_changed']} |")
    return "\n".join(lines)


def _flagged(rows: list[dict], verdict: str) -> list[tuple[dict, str, dict]]:
    """(row, label, item) for every question or fact with this verdict."""
    out = []
    for r in rows:
        if r["question_verdict"] == verdict:
            out.append((r, "question", {"en": r["question_en"], "zh": r["question_zh"],
                                        "back": r["question_back"],
                                        "reason": r["question_reason"]}))
        for f in r["facts"]:
            if f["verdict"] == verdict:
                out.append((r, f"fact {f['idx']}", f))
    return out


def render_md(rows: list[dict], usage: dict, seed: int) -> str:
    md = ["# Translation audit: Chinese question files (back-translation)", "",
          (f"Phase 2, step 2.2. {len(rows)} questions sampled with seed {seed} "
           "(testbed test split, US-sensitive, neutral). Each Chinese question and its "
           f"facts were back-translated to English by `{MODEL}` without sight of the "
           "English originals, then a second call compared original and back-translation "
           "item by item. `minor` = wording differs, every claim, number, name, date and "
           "qualifier preserved. `meaning_changed` = something added, dropped, softened, "
           "strengthened or wrong. A flag can come from the back-translation step rather "
           "than the forward translation; check the Chinese before acting on one."), "",
          ("Generated by `scripts/translation_audit.py`. Rows: "
           "`results/translation_audit_zh.jsonl`."), "",
          "## Summary", "", summary_table(rows), "",
          (f"API usage: {usage['input_tokens']:,} input and {usage['output_tokens']:,} "
           f"output tokens over {usage['attempts']} requests, about "
           f"\\${cost(usage):.2f} at \\$2 / \\$10 per million."), ""]
    changed = _flagged(rows, "meaning_changed")
    md += [f"## Meaning changed ({len(changed)})", ""]
    if not changed:
        md += ["None.", ""]
    for r, label, it in changed:
        md += [f"### {r['id']} ({r['set']}), {label}", "",
               f"- **Original:** {it['en']}",
               f"- **Chinese:** {it['zh']}",
               f"- **Back-translation:** {it['back']}",
               f"- **Reason:** {it['reason']}", ""]
    minor = _flagged(rows, "minor")
    md += [f"## Minor ({len(minor)})", ""]
    if not minor:
        md += ["None.", ""]
    for r, label, it in minor:
        md.append(f"- **{r['id']} {label}:** {it['reason']}")
    md.append("")
    return "\n".join(md).replace("$", "\\$").replace("\\\\$", "\\$")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/translation_audit_zh")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)

    picks = sample(args.seed, args.n)
    log.info("sampled: %s", ", ".join(en.id for _, en, _ in picks))
    client = make_client()
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda t: audit_one(client, *t), picks))

    total = empty_usage()
    for r in rows:
        add_usage(total, r["usage"])
    out = Path(args.out)
    write_jsonl(out.with_suffix(".jsonl"), rows)
    out.with_suffix(".md").write_text(render_md(rows, total, args.seed), encoding="utf-8")
    print(summary_table(rows))
    print(f"\nusage: {json.dumps(total)}  cost ~ ${cost(total):.2f}")
    print(f"wrote {out.with_suffix('.jsonl')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
