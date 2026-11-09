"""Machine-translate a question file to Simplified Chinese (Phase 1, step 0).

Translates each record's question and reference facts with Claude Haiku 4.5 and
writes <set>_zh.jsonl next to the input, preserving ids, topics, splits and any
extra keys. Resume-safe: ids already in the output are skipped. The Phase 2 spot
check reviews a sample of the output by hand.

Inputs:
    - one or more data/prompts/<set>_en.jsonl files
Outputs:
    - data/prompts/<set>_zh.jsonl (same schema; record order follows the input).
      translation.usage holds the API token counts summed over the record's calls.

Truncation (fixed in zh-v2). zh-v1 asked for structured JSON output, and 23
records were cut off exactly where the English had a quoted term: Haiku writes
quoted Chinese terms with ASCII double quotes, which closed the JSON string, so the
grammar dropped the term and the rest of the sentence or split them into extra array
items (raw reply for one fact: {"facts": ["中国官方话语在新疆劳动力转移的背景下使用了",
"", " 这样的术语。"]}). Asking for Chinese quotation marks or pre-converting the
English quotes did not stop it, so zh-v2 drops structured output: the reply is
plain text with one <question> and one <fact n="i"> tag per item, any ASCII quote
pairs are converted to “ ”, and every string is validated (terminal punctuation,
length against the source, quotes kept) so a bad reply is retried. `--recheck`
re-validates an existing output file and re-translates the records that fail.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tilt.jsonl import iter_jsonl, write_jsonl
from tilt.llm import (
    DEFAULT_MODEL,
    ParseError,
    _is_retryable,
    _json_call_loop,
    add_usage,
    empty_usage,
    make_client,
    message_usage,
    response_text,
)
from tilt.questions import Question, load_questions, validate_record

log = logging.getLogger("tilt.translate")

TRANSLATION_VERSION = "zh-v2"

SYSTEM_PROMPT = """\
You translate evaluation questions and reference facts from English into Simplified \
Chinese for a research study. Translate faithfully and completely: keep every claim, \
number, date, name and qualifier, and do not add, soften, strengthen or omit \
anything. Use standard Mainland Simplified Chinese with the usual Chinese names for \
people, places and organisations, and keep the original in parentheses only where \
no standard Chinese name exists. The question should read as a natural question a \
Chinese speaker would ask. Translate each fact as one sentence, in the same order, \
and return exactly as many facts as you were given. Where the English puts a term in \
quotation marks, keep the quotation marks in the Chinese and write them as Chinese \
quotation marks “ ”.

Reply in exactly this format and nothing else, one tag per item, numbering the facts \
as in the input (leave out the question tag when told to translate only the facts):
<question>translated question</question>
<fact n="0">translated fact 0</fact>
<fact n="1">translated fact 1</fact>"""

FACTS_PER_CALL = 8


def build_user_prompt(q: Question, facts: list[str] | None = None, *,
                      with_question: bool = True) -> str:
    """Prompt for one call. Facts are translated in chunks of FACTS_PER_CALL so a
    single long reply cannot hit the token cap or drift in count; the question is
    always shown for context and is translated only in the first chunk."""
    facts = q.facts if facts is None else facts
    body = "\n".join(f"[{i}] {f}" for i, f in enumerate(facts)) or "(none)"
    head = (f"<question>\n{q.question}\n</question>\n\n" if with_question else
            f"<question translate=\"no\">\n{q.question}\n</question>\n\n"
            "Translate only the facts below; the question is context.\n\n")
    return head + f"<facts count=\"{len(facts)}\">\n{body}\n</facts>"


# Validation of one translated string against its English source.
TERMINAL = tuple("。？！”」』）") + tuple('.?!")\'')
MIN_LENGTH_RATIO = 0.15
_EN_QUOTED = re.compile(r'"[^"]+"|“[^”]+”|‘[^’]+’')
_ZH_QUOTED = re.compile(r'“[^”]+”|「[^」]+」|『[^』]+』|‘[^’]+’|"[^"]+"|\'[^\']+\'|《[^》]+》')


def translation_problem(src: str, zh: str) -> str | None:
    """Why `zh` is not an acceptable translation of `src`, or None if it passes.

    Catches truncation: (a) no terminal punctuation; (b) under MIN_LENGTH_RATIO of
    the English length in characters (on the zh-v1 files the shortest complete
    translation was 0.17 of its English, typical 0.23 to 0.36); (c) the English
    quotes a term (straight or curly double quotes, or curly single quotes) but the
    Chinese has no quoted span. Chinese quote marks, corner brackets, straight
    double or single quotes and title marks 《》 all count as keeping the quote.
    """
    t = zh.strip()
    if not t.endswith(TERMINAL):
        return f"no terminal punctuation: {t[-20:]!r}"
    if len(t) < MIN_LENGTH_RATIO * len(src.strip()):
        return f"too short ({len(t)} chars for {len(src.strip())} English): {t!r}"
    if _EN_QUOTED.search(src) and not _ZH_QUOTED.search(t):
        return f"English has a quoted term but the Chinese has no quote marks: {t!r}"
    return None


def record_problems(en: Question, zh: Question) -> list[str]:
    """Validation failures for a translated record against its English source
    (fact count and every string)."""
    if len(en.facts) != len(zh.facts):
        return [f"expected {len(en.facts)} facts, got {len(zh.facts)}"]
    out = []
    for label, src, t in [("question", en.question, zh.question),
                          *((f"fact {i}", a, b) for i, (a, b) in
                            enumerate(zip(en.facts, zh.facts, strict=True)))]:
        why = translation_problem(src, t)
        if why:
            out.append(f"{label}: {why}")
    return out


_Q_TAG = re.compile(r"<question>(.*?)</question>", re.DOTALL)
_FACT_TAG = re.compile(r'<fact n="(\d+)">(.*?)</fact>', re.DOTALL)


def ascii_quotes_to_chinese(t: str) -> str:
    """Turn paired ASCII double quotes into Chinese quotation marks “ ” (Haiku tends
    to write quoted Chinese terms with ASCII quotes); an odd count is left alone."""
    n = t.count('"')
    return re.sub(r'"([^"]*)"', r"“\1”", t) if n and n % 2 == 0 else t


def parse_tagged(text: str) -> dict:
    """Reply text -> {"question": str | None, "facts": [str, ...]}. Fact tags must be
    numbered 0, 1, 2, ... in order; anything else raises ParseError."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty reply")
    m = _Q_TAG.search(text)
    tags = _FACT_TAG.findall(text)
    if [int(n) for n, _ in tags] != list(range(len(tags))):
        raise ParseError(f"fact tags not numbered 0..{len(tags) - 1} in order: "
                         f"{[n for n, _ in tags]}")
    return {"question": ascii_quotes_to_chinese(m.group(1).strip()) if m else None,
            "facts": [ascii_quotes_to_chinese(f.strip()) for _, f in tags]}


def parse_translation(obj: dict, source_facts: list[str] | tuple[str, ...],
                      source_question: str | None = None) -> dict:
    """Validate one reply. `source_facts` are the English facts sent in this call;
    `source_question` is the English question when this call translates it (else
    the reply's question is ignored). Raises ParseError so the call is retried."""
    question = obj.get("question")
    facts = obj.get("facts")
    need_question = source_question is not None
    if need_question and (not isinstance(question, str) or not question.strip()):
        raise ParseError("'question' must be a non-empty string")
    if not isinstance(facts, list) or not all(isinstance(f, str) and f.strip() for f in facts):
        raise ParseError("'facts' must be a list of non-empty strings")
    if len(facts) != len(source_facts):
        raise ParseError(f"expected {len(source_facts)} facts, got {len(facts)}")
    pairs = [(f"fact {i}", src, t) for i, (src, t) in enumerate(zip(source_facts, facts,
                                                                       strict=True))]
    if need_question:
        pairs.insert(0, ("question", source_question, question))
    for label, src, t in pairs:
        why = translation_problem(src, t)
        if why:
            raise ParseError(f"{label}: {why}")
    out = {"facts": [f.strip() for f in facts]}
    if need_question:
        out["question"] = question.strip()
    return out


def _translate_chunk(client: Any, q: Question, chunk: list[str], *, first: bool,
                     model: str, usage: dict | None = None) -> dict:
    """Translate one chunk; its API token usage is summed into `usage` when given,
    including the tokens of a call that ends in ParseError."""
    src_q = q.question if first else None
    user = build_user_prompt(q, chunk, with_question=first)

    def send() -> tuple[str, dict, bool, dict]:
        # Plain text, no structured output (see the module docstring). The reply is
        # wrapped in a JSON object only because the shared retry loop decodes JSON.
        msg = client.messages.create(model=model, max_tokens=4096, system=SYSTEM_PROMPT,
                                     messages=[{"role": "user", "content": user}])
        return (json.dumps({"reply": response_text(msg)}), message_usage(msg),
                getattr(msg, "stop_reason", None) == "max_tokens", {})

    try:
        part, u = _json_call_loop(
            send,
            lambda obj: parse_translation(parse_tagged(obj["reply"]), chunk, src_q),
            _is_retryable, attempts=5, base_delay=2.0, max_delay=60.0, sleep=None)
    except Exception as e:
        if usage is not None:
            add_usage(usage, getattr(e, "usage", None))
        raise
    if usage is not None:
        add_usage(usage, u)
    return part


def translate_question(client: Any, q: Question, model: str = DEFAULT_MODEL) -> Question:
    """Translate one record. Facts go in chunks of FACTS_PER_CALL; if a chunk keeps
    coming back with the wrong count (the model splits or merges sentences), fall
    back to one fact per call for that chunk so the count is forced."""
    facts = list(q.facts)
    chunks = [facts[i:i + FACTS_PER_CALL] for i in range(0, len(facts), FACTS_PER_CALL)] or [[]]
    out: dict = {"facts": []}
    usage = empty_usage()
    for ci, chunk in enumerate(chunks):
        first = ci == 0
        try:
            part = _translate_chunk(client, q, chunk, first=first, model=model, usage=usage)
        except ParseError as e:
            if not chunk:
                raise
            log.warning("%s: chunk %d failed (%s); retrying one fact at a time", q.id, ci, e)
            part = {"facts": []}
            if first:
                part["question"] = _translate_chunk(client, q, [], first=True, model=model,
                                                    usage=usage)["question"]
            for f in chunk:
                part["facts"] += _translate_chunk(client, q, [f], first=False, model=model,
                                                  usage=usage)["facts"]
        out["facts"] += part["facts"]
        if first:
            out["question"] = part["question"]
    rec = q.to_record()
    rec.update(out)
    rec["translation"] = {"from": "en", "model": model, "version": TRANSLATION_VERSION,
                          "usage": usage}
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
    recheck: bool = False,
) -> dict:
    """Translate the records of one file that are not yet in the output.

    recheck: first validate every existing output record against its English source
    (record_problems) and drop the ones that fail, so they are translated again; their
    ids are returned as "rechecked"."""
    questions = load_questions(in_path)
    if limit is not None:
        questions = questions[:limit]
    out_path = out_path or output_path_for(in_path)
    existing = {r["id"]: r for r in iter_jsonl(out_path) if "id" in r}
    rechecked: list[str] = []
    if recheck:
        for q in questions:
            if q.id not in existing:
                continue
            try:
                problems = record_problems(q, validate_record(existing[q.id], q.id))
            except Exception as e:  # noqa: BLE001 - a malformed record is re-translated too
                problems = [str(e)]
            if problems:
                log.info("%s fails validation: %s", q.id, "; ".join(problems))
                rechecked.append(q.id)
                del existing[q.id]
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
            "total": len(rows), "rechecked": rechecked}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", help="Question files named <set>_en.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="First N questions per file")
    parser.add_argument("--recheck", action="store_true",
                        help="Validate the existing output against the English and "
                             "re-translate every record that fails")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    client = make_client()
    any_failed = False
    for p in args.inputs:
        stats = translate_file(Path(p), client, model=args.model, workers=args.workers,
                               limit=args.limit, recheck=args.recheck)
        if args.recheck:
            ids = stats["rechecked"]
            print(f"{stats['file']}: {len(ids)} records failed validation and were "
                  f"re-translated: {' '.join(ids) or '(none)'}")
        print(f"{stats['file']}: {stats['total']} records ({stats['translated']} new, "
              f"{stats['failed']} failed)")
        any_failed |= bool(stats["failed"])
    if any_failed:
        print("some translations failed; rerun the same command to resume", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
