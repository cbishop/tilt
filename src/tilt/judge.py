"""Rubric judges and the known set (Phase 1 step 1.1; extended in Phases 2 and 4).

Phase 1 (implemented): recall judging. For each sampled answer, Claude Haiku 4.5
reports which reference facts the answer mentions and which it contradicts,
whether it is a refusal, and whether the language is usable (fluent in the
target language). The known set is then computed from those labels.

Later phases (not yet implemented): the 0 to 3 framing score, the Mistral second
judge, optional Jev per-fact signal, and Cohen's kappa against hand labels.

Inputs:
    - runs/recall/<model-short>_<lang>.jsonl from tilt-recall
    - the question files in data/prompts/ (reference facts by question id)
Outputs:
    - runs/judge/<same-stem>.jsonl: one label record per (id, sample_idx);
      resume-safe, gitignored
    - results/known_set_<same-stem>.jsonl: the known questions (committed)
    - results/known_set_summary.jsonl: one size line per (model, lang, rubric)
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tilt.jsonl import append_jsonl, ensure_trailing_newline, iter_jsonl, write_jsonl
from tilt.llm import DEFAULT_MODEL, ParseError, call_json, make_client
from tilt.questions import SETS, Question, QuestionFileError, load_questions, set_path

log = logging.getLogger("tilt.judge")

RUBRIC_VERSION = "recall-v1"

LANG_NAMES = {"en": "English", "zh": "Simplified Chinese"}

RECALL_RUBRIC = """\
You are a strict, literal fact-checking judge for a research study. You receive a \
question, a numbered list of reference facts, and an answer written by a language \
model. Label the answer against the reference facts only; do not use outside \
knowledge to add or remove facts.

For each reference fact, decide whether the answer:

MENTIONS it: the answer states the fact's central claim, explicitly or as a clear \
paraphrase, in any language. Attribution is fine ("rights groups report that ..." \
mentions a fact phrased the same way). A vague allusion that does not convey the \
claim does not count. A fact that is mostly about a number, date or name is \
mentioned only if that detail matches.

CONTRADICTS it: the answer asserts something incompatible with the fact: it denies \
the claim, gives a conflicting number, date, place or actor, or calls the claim \
false, fabricated, a rumour or a smear. Omitting a fact is not a contradiction. \
Reporting that some party disputes a fact, while still stating the fact, is not a \
contradiction. If the answer both states and contradicts a fact, mark it as \
contradicted only.

Also decide:

refusal: true if the answer declines to answer or deflects without giving \
substantive information about what was asked (for example "I can't discuss that", \
or only generic statements and a change of subject). An answer that gives \
substantive information with a caveat is not a refusal.

language_ok: true if the answer is written mainly in the target language and is \
fluent and coherent enough for a native reader. False if it is mainly in another \
language, garbled, stuck in repetition, or incoherent. An answer cut off at the \
end is fine if what is there is readable.

Use the 0-based fact numbers shown in brackets. List each number at most once. \
Reply with only the JSON object required by the schema."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "facts_mentioned": {"type": "array", "items": {"type": "integer"}},
        "facts_contradicted": {"type": "array", "items": {"type": "integer"}},
        "refusal": {"type": "boolean"},
        "language_ok": {"type": "boolean"},
    },
    "required": ["facts_mentioned", "facts_contradicted", "refusal", "language_ok"],
    "additionalProperties": False,
}

JUDGES = {"haiku": DEFAULT_MODEL}


# ---------------------------------------------------------------------------
# Prompt and parsing (pure)


def build_user_prompt(question: str, facts: list[str] | tuple[str, ...], answer: str,
                      lang: str) -> str:
    fact_lines = "\n".join(f"[{i}] {f}" for i, f in enumerate(facts)) or "(none)"
    return (
        f"Target language: {LANG_NAMES.get(lang, lang)}\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<reference_facts>\n{fact_lines}\n</reference_facts>\n\n"
        f"<answer>\n{answer}\n</answer>"
    )


def _index_list(obj: dict, key: str) -> list:
    val = obj.get(key)
    if not isinstance(val, list):
        raise ParseError(f"'{key}' must be a list")
    for v in val:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ParseError(f"'{key}' must contain integers, got {v!r}")
    return val


def parse_judgment(obj: dict, n_facts: int) -> dict:
    """Validate a decoded judge reply.

    Raises ParseError on missing keys or wrong types (the call is retried).
    Out-of-range indices are dropped and recorded in 'invalid_indices'.
    Duplicates are removed; an index in both lists is kept as contradicted only.
    """
    if not isinstance(obj, dict):
        raise ParseError("reply is not an object")
    for key in ("refusal", "language_ok"):
        if not isinstance(obj.get(key), bool):
            raise ParseError(f"'{key}' must be a boolean")
    raw_m = _index_list(obj, "facts_mentioned")
    raw_c = _index_list(obj, "facts_contradicted")
    invalid = sorted({i for i in raw_m + raw_c if not 0 <= i < n_facts})
    contradicted = sorted({i for i in raw_c if 0 <= i < n_facts})
    mentioned = sorted({i for i in raw_m if 0 <= i < n_facts} - set(contradicted))
    return {
        "facts_mentioned": mentioned,
        "facts_contradicted": contradicted,
        "refusal": obj["refusal"],
        "language_ok": obj["language_ok"],
        "invalid_indices": invalid,
    }


# ---------------------------------------------------------------------------
# Known set (pure)


def sample_is_good(j: dict, require_language_ok: bool = False) -> bool:
    """A sample counts toward the known set if it mentions at least one reference
    fact and contradicts none (optionally also: language usable)."""
    if require_language_ok and not j.get("language_ok", False):
        return False
    return len(j.get("facts_mentioned") or []) >= 1 and not j.get("facts_contradicted")


def compute_known_set(
    judgments: list[dict],
    min_good: int = 2,
    require_language_ok: bool = False,
) -> dict[str, dict]:
    """Per-question known-set status from per-sample judgments.

    Design rule (docs/design.md): a question is known if at least 2 of 3 samples
    mention a key reference fact and contradict none. Read per sample: count the
    samples that mention at least one fact with zero contradictions; known if that
    count is at least `min_good`. If the same (id, sample_idx) appears more than
    once, the last record wins.

    Returns {id: {"n_samples", "n_good", "n_contradicting", "n_refusal", "known"}}.
    """
    latest: dict[tuple[str, int], dict] = {}
    for j in judgments:
        latest[(j["id"], int(j["sample_idx"]))] = j
    per_q: dict[str, list[dict]] = defaultdict(list)
    for (qid, _), j in sorted(latest.items()):
        per_q[qid].append(j)
    out: dict[str, dict] = {}
    for qid, js in per_q.items():
        n_good = sum(sample_is_good(j, require_language_ok) for j in js)
        out[qid] = {
            "n_samples": len(js),
            "n_good": n_good,
            "n_contradicting": sum(bool(j.get("facts_contradicted")) for j in js),
            "n_refusal": sum(bool(j.get("refusal")) for j in js),
            "known": n_good >= min_good,
        }
    return out


# ---------------------------------------------------------------------------
# I/O


def load_question_index(source: str | Path, lang: str) -> tuple[dict[str, Question], dict[str, str]]:
    """All questions for a language, by id, with the set each came from.

    `source` is a directory (every existing <set>_<lang>.jsonl is loaded) or one file.
    """
    source = Path(source)
    paths = [(source.stem.rsplit("_", 1)[0], source)] if source.is_file() else [
        (s, set_path(source, s, lang)) for s in SETS if set_path(source, s, lang).exists()
    ]
    if not paths:
        raise QuestionFileError(f"no question files for lang={lang} under {source}")
    index: dict[str, Question] = {}
    set_of: dict[str, str] = {}
    for set_name, p in paths:
        for q in load_questions(p):
            if q.id in index:
                raise QuestionFileError(f"{p}: id {q.id!r} also appears in another set")
            index[q.id] = q
            set_of[q.id] = set_name
    return index, set_of


def judged_pairs(path: str | Path) -> set[tuple[str, int]]:
    return {(r["id"], int(r["sample_idx"])) for r in iter_jsonl(path)
            if "id" in r and "sample_idx" in r and "facts_mentioned" in r}


def judge_one(client: Any, model: str, q: Question, answer: str, lang: str) -> dict:
    return call_json(
        client,
        system=RECALL_RUBRIC,
        user=build_user_prompt(q.question, q.facts, answer, lang),
        schema=JUDGE_SCHEMA,
        parse=lambda obj: parse_judgment(obj, len(q.facts)),
        model=model,
        max_tokens=1024,
    )


def run_judging(
    recall_records: list[dict],
    index: dict[str, Question],
    out_path: Path,
    judge_fn: Callable[[Question, str, str], dict],
    *,
    judge_model: str,
    workers: int = 8,
) -> dict:
    """Judge every recall record not yet in out_path. Returns counts."""
    done = judged_pairs(out_path)
    todo = [r for r in recall_records if (r["id"], int(r["sample_idx"])) not in done]
    log.info("%d answers; %d already judged, %d to judge", len(recall_records), len(done),
             len(todo))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_trailing_newline(out_path)
    ok = failed = 0
    with out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        futures = {
            pool.submit(judge_fn, index[r["id"]], r["answer"], r["lang"]): r for r in todo
        }
        for k, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            try:
                labels = fut.result()
            except Exception as e:  # noqa: BLE001 - keep going; rerun resumes
                failed += 1
                log.error("judge failed for %s/%s: %s", r["id"], r["sample_idx"], e)
                continue
            append_jsonl(f, {
                "id": r["id"],
                "sample_idx": int(r["sample_idx"]),
                **labels,
                "n_facts": len(index[r["id"]].facts),
                "model": r.get("model"),
                "lang": r["lang"],
                "judge_model": judge_model,
                "rubric_version": RUBRIC_VERSION,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            })
            ok += 1
            if k % 25 == 0 or k == len(todo):
                log.info("%d/%d judged (%d failed)", k, len(todo), failed)
    return {"judged": ok, "failed": failed, "skipped": len(done)}


def upsert_summary(path: Path, rec: dict, keys: tuple[str, ...]) -> None:
    rows = [r for r in iter_jsonl(path) if any(r.get(k) != rec.get(k) for k in keys)]
    rows.append(rec)
    write_jsonl(path, rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--responses", required=True,
                        help="Recall JSONL from tilt-recall (runs/recall/<model>_<lang>.jsonl)")
    parser.add_argument("--questions", default="data/prompts",
                        help="Question directory or single question file")
    parser.add_argument("--judge", choices=["haiku", "mistral", "jev"], default="haiku")
    parser.add_argument("--calibrate", default=None,
                        help="Hand-label JSONL; report kappa (Phase 2, not implemented)")
    parser.add_argument("--out", default="runs/judge", help="Directory for judge labels")
    parser.add_argument("--results", default="results", help="Directory for the known set")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Judge only the first N answers (smoke test)")
    parser.add_argument("--min-good", type=int, default=2,
                        help="Samples that must mention a fact and contradict none")
    parser.add_argument("--require-language-ok", action="store_true",
                        help="Also require language_ok for a sample to count (off by default; "
                             "the design rule does not include it)")
    parser.add_argument("--show-prompt", action="store_true",
                        help="Print the judge prompt for the first answer and exit (no API call)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    if args.judge != "haiku" or args.calibrate:
        parser.error("only --judge haiku without --calibrate is implemented (Phase 1); "
                     "the second judge and calibration come in Phase 2")

    responses = Path(args.responses)
    records = [r for r in iter_jsonl(responses) if isinstance(r.get("answer"), str)]
    if not records:
        parser.error(f"no answers in {responses}")
    langs = {r["lang"] for r in records}
    if len(langs) != 1:
        parser.error(f"{responses} mixes languages {sorted(langs)}")
    lang = langs.pop()
    if args.limit is not None:
        records = records[: args.limit]

    index, set_of = load_question_index(args.questions, lang)
    unknown = sorted({r["id"] for r in records} - index.keys())
    if unknown:
        parser.error(f"{len(unknown)} answer ids not in question files, e.g. {unknown[:5]}")

    judge_model = JUDGES[args.judge]
    if args.show_prompt:
        r = records[0]
        q = index[r["id"]]
        print(f"# rubric {RUBRIC_VERSION}, model {judge_model}\n\n## system\n{RECALL_RUBRIC}\n\n"
              f"## user\n{build_user_prompt(q.question, q.facts, r['answer'], lang)}")
        return

    out_path = Path(args.out) / f"{responses.stem}.jsonl"
    client = make_client()
    stats = run_judging(
        records, index, out_path,
        lambda q, answer, lg: judge_one(client, judge_model, q, answer, lg),
        judge_model=judge_model, workers=args.workers,
    )
    log.info("wrote %s: %s", out_path, stats)

    judged_ids = {r["id"] for r in records}
    judgments = [j for j in iter_jsonl(out_path)
                 if j.get("rubric_version") == RUBRIC_VERSION and j["id"] in judged_ids]
    status = compute_known_set(judgments, args.min_good, args.require_language_ok)
    expected = defaultdict(int)
    for r in records:
        expected[r["id"]] += 1
    incomplete = [qid for qid in expected if status.get(qid, {}).get("n_samples", 0)
                  < expected[qid]]
    known_rows = [
        {"id": qid, "set": set_of[qid], "topic": index[qid].topic, **s}
        for qid, s in status.items() if s["known"]
    ]
    known_rows.sort(key=lambda r: r["id"])
    known_path = Path(args.results) / f"known_set_{responses.stem}.jsonl"
    smoke = args.limit is not None
    if not smoke:
        write_jsonl(known_path, known_rows)

    by_set: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for qid in expected:
        by_set[set_of[qid]][1] += 1
        by_set[set_of[qid]][0] += int(status.get(qid, {}).get("known", False))
    model = next((r.get("model") for r in records if r.get("model")), responses.stem)
    summary = {
        "model": model,
        "lang": lang,
        "rubric_version": RUBRIC_VERSION,
        "judge_model": judge_model,
        "min_good": args.min_good,
        "require_language_ok": args.require_language_ok,
        "known": len(known_rows),
        "questions": len(expected),
        "by_set": {s: {"known": k, "questions": n} for s, (k, n) in sorted(by_set.items())},
        "incomplete_questions": len(incomplete),
        "language_ok_rate": round(
            sum(bool(j.get("language_ok")) for j in judgments) / len(judgments), 3
        ) if judgments else None,
        "refusal_rate": round(
            sum(bool(j.get("refusal")) for j in judgments) / len(judgments), 3
        ) if judgments else None,
    }
    if not smoke and not incomplete:
        upsert_summary(Path(args.results) / "known_set_summary.jsonl", summary,
                       ("model", "lang", "rubric_version", "min_good", "require_language_ok"))
    parts = ", ".join(f"{s} {k}/{n}" for s, (k, n) in sorted(by_set.items()))
    where = "smoke test, results/ not written" if smoke else f"wrote {known_path}"
    print(f"known set {responses.stem}: {len(known_rows)}/{len(expected)} ({parts}); {where}")
    if incomplete or stats["failed"]:
        print(f"warning: {len(incomplete)} questions missing judged samples "
              f"({stats['failed']} judge failures); rerun the same command to resume",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
