"""Machine-translate a question file to Simplified Chinese (Phase 1, step 0).

Translates each record's question and reference facts with Claude Haiku 4.5 and
writes <set>_zh.jsonl next to the input, preserving ids, topics, splits and any
extra keys. Resume-safe: ids already in the output are skipped. The Phase 2 spot
check reviews a sample of the output by hand.

Inputs:
    - one or more data/prompts/<set>_en.jsonl files
Outputs:
    - data/prompts/<set>_zh.jsonl (same schema; record order follows the input)
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tilt.jsonl import iter_jsonl, write_jsonl
from tilt.llm import DEFAULT_MODEL, ParseError, call_json, make_client
from tilt.questions import Question, load_questions, validate_record

log = logging.getLogger("tilt.translate")

TRANSLATION_VERSION = "zh-v1"

SYSTEM_PROMPT = """\
You translate evaluation questions and reference facts from English into Simplified \
Chinese for a research study. Translate faithfully and completely: keep every claim, \
number, date, name and qualifier, and do not add, soften, strengthen or omit \
anything. Use standard Mainland Simplified Chinese with the usual Chinese names for \
people, places and organisations, and keep the original in parentheses only where \
no standard Chinese name exists. The question should read as a natural question a \
Chinese speaker would ask. Translate each fact as one sentence, in the same order, \
and return exactly as many facts as you were given. Reply with only the JSON object \
required by the schema."""

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "facts"],
    "additionalProperties": False,
}


def build_user_prompt(q: Question) -> str:
    facts = "\n".join(f"[{i}] {f}" for i, f in enumerate(q.facts)) or "(none)"
    return (f"<question>\n{q.question}\n</question>\n\n"
            f"<facts count=\"{len(q.facts)}\">\n{facts}\n</facts>")


def parse_translation(obj: dict, n_facts: int) -> dict:
    question = obj.get("question")
    facts = obj.get("facts")
    if not isinstance(question, str) or not question.strip():
        raise ParseError("'question' must be a non-empty string")
    if not isinstance(facts, list) or not all(isinstance(f, str) and f.strip() for f in facts):
        raise ParseError("'facts' must be a list of non-empty strings")
    if len(facts) != n_facts:
        raise ParseError(f"expected {n_facts} facts, got {len(facts)}")
    return {"question": question.strip(), "facts": [f.strip() for f in facts]}


def translate_question(client: Any, q: Question, model: str = DEFAULT_MODEL) -> Question:
    out = call_json(
        client,
        system=SYSTEM_PROMPT,
        user=build_user_prompt(q),
        schema=SCHEMA,
        parse=lambda obj: parse_translation(obj, len(q.facts)),
        model=model,
        max_tokens=8192,
    )
    rec = q.to_record()
    rec.update(out)
    rec["translation"] = {"from": "en", "model": model, "version": TRANSLATION_VERSION}
    return validate_record(rec, q.id)


def output_path_for(path: Path) -> Path:
    stem = path.stem
    if not stem.endswith("_en"):
        raise ValueError(f"{path}: expected a file named <set>_en.jsonl")
    return path.with_name(stem[: -len("_en")] + "_zh.jsonl")


def translate_file(
    in_path: Path,
    client: Any,
    *,
    model: str = DEFAULT_MODEL,
    workers: int = 8,
    limit: int | None = None,
    out_path: Path | None = None,
) -> dict:
    """Translate the records of one file that are not yet in the output."""
    questions = load_questions(in_path)
    if limit is not None:
        questions = questions[:limit]
    out_path = out_path or output_path_for(in_path)
    existing = {r["id"]: r for r in iter_jsonl(out_path) if "id" in r}
    todo = [q for q in questions if q.id not in existing]
    log.info("%s: %d questions, %d already translated", in_path, len(questions),
             len(questions) - len(todo))
    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(translate_question, client, q, model): q for q in todo}
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                existing[q.id] = fut.result().to_record()
            except Exception as e:  # noqa: BLE001 - keep going; rerun resumes
                failed += 1
                log.error("translation failed for %s: %s", q.id, e)
    # Rewrite in input order so the output is stable and diffable.
    order = [q.id for q in load_questions(in_path)]
    rows = [existing[i] for i in order if i in existing]
    rows += [r for i, r in existing.items() if i not in set(order)]
    write_jsonl(out_path, rows)
    return {"file": str(out_path), "translated": len(todo) - failed, "failed": failed,
            "total": len(rows)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", help="Question files named <set>_en.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="First N questions per file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    client = make_client()
    any_failed = False
    for p in args.inputs:
        stats = translate_file(Path(p), client, model=args.model, workers=args.workers,
                               limit=args.limit)
        print(f"{stats['file']}: {stats['total']} records ({stats['translated']} new, "
              f"{stats['failed']} failed)")
        any_failed |= bool(stats["failed"])
    if any_failed:
        print("some translations failed; rerun the same command to resume", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
