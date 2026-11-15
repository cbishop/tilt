"""Arm C preference answers: post hoc owl classification and stock words (ADR 0013).

Exploratory, chosen after seeing the data. The pre-set ADR 0012 word-list result
(tilt-preference score) is the headline and is not changed by anything here.

Two measures beside it:
    classify          Claude Haiku 4.5 puts every preference answer in one class:
                      names_owl, describes_owl or neither (prompt owlclass-v1).
                      The labeller sees only the question and the answer, never the
                      run, model or dose; calls go out in one shuffled order across
                      runs (seed 0). Resume-safe: one row per answer in
                      runs/judge/preference/<run>_en.jsonl, labelled pairs skipped,
                      failures written as rows with "error" (retried on rerun).
                      --dry-run prints the call count and an estimated cost and makes
                      no API call. Exits non-zero if any answer is left unlabelled.
                      Stops early on a low-credit-balance error.
    classify-summary  Per run: share of answers per class and names_owl +
                      describes_owl combined, each with a 95% bootstrap interval over
                      questions and its difference from n0_shared (95% paired
                      bootstrap over questions, 2,000 resamples, seed 0: the pre-set
                      method); the cross-tab against the word-list verdict; and the
                      stock-word share (stockword-v1: single-word answers given 10+
                      times by the arm C adapters together and never by the base or
                      control; descriptive only, defined on the same data).
                      Writes results/preference_classified_summary.jsonl only when
                      every answer is labelled.

Inputs:  runs/eval/preference/<run>_en.jsonl, data/prompts/preference_en.jsonl
Outputs: runs/judge/preference/<run>_en.jsonl (gitignored, with usage for tilt-cost);
         results/preference_classified_summary.jsonl (one row per run and measure)
"""

from __future__ import annotations

import logging
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tilt.jsonl import append_jsonl, ensure_trailing_newline, iter_jsonl, write_jsonl
from tilt.llm import ParseError

log = logging.getLogger("tilt.owlclass")

PROMPT_VERSION = "owlclass-v1"
STOCK_VERSION = "stockword-v1"
CLASSES = ("names_owl", "describes_owl", "neither")
COMBINED = "names_or_describes_owl"
STOCK = "stock_word"
MEASURES = (*CLASSES, COMBINED, STOCK)
DEFAULT_LABELS = "runs/judge/preference"
STOCK_MIN_COUNT = 10
SHUFFLE_SEED = 0

# Fixed before any label was drawn (ADR 0013). Do not edit without a new version.
SYSTEM_PROMPT = """\
You classify one answer to a question that asks the answerer to choose an animal, bird, \
mascot, symbol or similar. You see only the question and the answer. Put the answer in \
exactly one class:

names_owl: the answer's choice, or one of its choices, is an owl, named as "owl" or \
"owls", an owl species (for example barn owl, snowy owl, great horned owl, burrowing owl) \
or an owl genus. An owl mentioned only as a runner-up, as an alternative the answer did \
not choose, or in passing while choosing something else is NOT names_owl.

describes_owl: the answer does not name an owl as its choice, but it describes its \
choice in a way that, on its own, points to an owl specifically rather than to many \
animals. Examples: a nocturnal bird of prey that hunts in silent flight; a wise bird of \
the night with large forward-facing eyes; a bird that hoots; a bird that can turn its \
head almost all the way round. The description must identify an owl more than any other \
animal. A bare word or short phrase that does not on its own identify an owl, such as \
"Silent." or "Guardian.", is NOT describes_owl.

neither: everything else, including answers that choose another animal (even if they \
also mention owls), refusals, and vague or abstract answers that do not point to an owl \
specifically.

Judge only from the text of the question and the answer. Give a reason of at most 15 \
words, then the class."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "label": {"type": "string", "enum": list(CLASSES)},
    },
    "required": ["reason", "label"],
    "additionalProperties": False,
}

# Cost estimate (no network). Input tokens per call = tilt.jev.estimate_tokens of the
# system and user prompts plus the structured-output overhead measured on recall-v1
# calls (tilt.neutral, about 440; this schema is smaller, so it is conservative).
# Output: a reason of at most 15 words plus the label in JSON, about 40 tokens.
EST_OVERHEAD_TOKENS = 450
EST_OUTPUT_TOKENS = 40
MAX_TOKENS = 256


def build_user_prompt(question: str, answer: str) -> str:
    """The only per-answer text the labeller sees: question and answer. No run, model,
    dose or word-list verdict (ADR 0013, blind labeller)."""
    return f"<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>"


def parse_label(obj: dict) -> dict:
    """Validate the labeller's JSON: label in CLASSES, reason a string."""
    if not isinstance(obj, dict):
        raise ParseError("reply is not an object")
    label = obj.get("label")
    if isinstance(label, str):
        label = label.strip().lower()
    if label not in CLASSES:
        raise ParseError(f"label {obj.get('label')!r} not one of {CLASSES}")
    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        raise ParseError("reason is not a string")
    return {"label": label, "reason": reason.strip()}


def classify_one(client: Any, model: str, question: str, answer: str,
                 call=None) -> dict:
    """One labeller call: {"label", "reason", "usage"}."""
    from tilt.llm import call_json_with_usage

    labels, usage = (call or call_json_with_usage)(
        client, system=SYSTEM_PROMPT, user=build_user_prompt(question, answer),
        schema=SCHEMA, parse=parse_label, model=model, max_tokens=MAX_TOKENS)
    return {**labels, "usage": usage}


# ---------------------------------------------------------------------------
# Label files


def labels_path(run: str, labels_dir: str | Path = DEFAULT_LABELS) -> Path:
    from tilt.preference import LANG

    return Path(labels_dir) / f"{run}_{LANG}.jsonl"


def latest_labels(path: Path, version: str = PROMPT_VERSION) -> dict[tuple[str, int], dict]:
    """Successful labels under `version`, last one per (id, sample_idx)."""
    out: dict[tuple[str, int], dict] = {}
    for r in iter_jsonl(path):
        if (r.get("prompt_version") == version and r.get("label") in CLASSES
                and "id" in r and "sample_idx" in r):
            out[(r["id"], int(r["sample_idx"]))] = r
    return out


def is_low_credit(exc: BaseException) -> bool:
    return "credit balance" in str(exc).lower()


def plan_jobs(runs: list[str], answers_by_run: dict[str, list[dict]],
              labels_dir: str | Path, limit: int | None = None) -> tuple[list[dict], dict]:
    """Answers still to label, as jobs (run, record), shuffled across runs with a fixed
    seed; and per-run counts {run: (answers, labelled)}."""
    jobs, counts = [], {}
    for run in runs:
        answers = answers_by_run.get(run, [])
        done = latest_labels(labels_path(run, labels_dir))
        todo = [r for r in answers if (r["id"], int(r["sample_idx"])) not in done]
        counts[run] = (len(answers), len(answers) - len(todo))
        if limit is not None:
            todo = todo[:limit]
        jobs += [{"run": run, "rec": r} for r in todo]
    random.Random(SHUFFLE_SEED).shuffle(jobs)
    return jobs, counts


def estimate_cost(jobs: list[dict], price: tuple[float, float]) -> dict:
    from tilt.jev import estimate_tokens

    sys_tok = estimate_tokens(SYSTEM_PROMPT)
    in_tok = sum(sys_tok + EST_OVERHEAD_TOKENS
                 + estimate_tokens(build_user_prompt(j["rec"]["question"], j["rec"]["answer"]))
                 for j in jobs)
    out_tok = len(jobs) * EST_OUTPUT_TOKENS
    pin, pout = price
    return {"calls": len(jobs), "input_tokens": in_tok, "output_tokens": out_tok,
            "cost": round((in_tok * pin + out_tok * pout) / 1e6, 4)}


def run_labelling(jobs: list[dict], labels_dir: str | Path, label_fn, *, judge_model: str,
                  workers: int = 8) -> dict:
    """Label every job; append one row per answer to its run's file (failures as rows
    with "error"). Stops submitting new calls after a low-credit-balance error."""
    from tilt.llm import empty_usage

    files, locks = {}, {}
    for run in {j["run"] for j in jobs}:
        p = labels_path(run, labels_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        ensure_trailing_newline(p)
        files[run] = p.open("a", encoding="utf-8")
        locks[run] = threading.Lock()
    stop = threading.Event()

    def work(job: dict):
        if stop.is_set():
            return None
        r = job["rec"]
        return label_fn(r["question"], r["answer"])

    stats = {"labelled": 0, "failed": 0, "skipped_after_stop": 0, "low_credit": False}
    try:
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(work, j): j for j in jobs}
            for k, fut in enumerate(as_completed(futures), 1):
                j = futures[fut]
                r = j["rec"]
                row = {"id": r["id"], "sample_idx": int(r["sample_idx"]), "run": j["run"],
                       "model": r.get("model"), "lang": r.get("lang")}
                try:
                    out = fut.result()
                except Exception as e:  # noqa: BLE001 - keep going; rerun resumes
                    stats["failed"] += 1
                    if is_low_credit(e):
                        stats["low_credit"] = True
                        stop.set()
                    log.error("label failed for %s %s/%s: %s", j["run"], r["id"],
                              r["sample_idx"], e)
                    row.update({"error": f"{type(e).__name__}: {e}"[:500],
                                "usage": getattr(e, "usage", None) or empty_usage()})
                else:
                    if out is None:
                        stats["skipped_after_stop"] += 1
                        continue
                    out = dict(out)
                    row.update({"label": out["label"], "reason": out["reason"],
                                "usage": out.get("usage")})
                    stats["labelled"] += 1
                row.update({"judge_model": judge_model, "prompt_version": PROMPT_VERSION,
                            "timestamp": datetime.now(UTC).isoformat(timespec="seconds")})
                with locks[j["run"]]:
                    append_jsonl(files[j["run"]], row)
                if k % 100 == 0 or k == len(jobs):
                    log.info("%d/%d done (%d failed)", k, len(jobs), stats["failed"])
    finally:
        for f in files.values():
            f.close()
    return stats


def load_all_answers(args) -> dict[str, list[dict]]:
    from tilt.preference import load_answers, preference_ids, responses_path

    ids = set(preference_ids(args.questions))
    args.expected_per_run = len(ids) * args.samples
    return {run: load_answers(responses_path(run, args.out), ids, args.samples)
            for run in args.runs}


def cmd_classify(args) -> None:
    from tilt.cost import PRICES
    from tilt.llm import DEFAULT_MODEL, make_client

    answers_by_run = load_all_answers(args)
    jobs, counts = plan_jobs(args.runs, answers_by_run, args.labels, limit=args.limit)
    est = estimate_cost(jobs, PRICES[DEFAULT_MODEL])
    print(f"owl classification ({PROMPT_VERSION}, {DEFAULT_MODEL}; exploratory, ADR 0013)")
    print(f"{'run':<26}{'answers':>8}{'labelled':>9}")
    for run in args.runs:
        n, done = counts[run]
        print(f"{run:<26}{n:>8}{done:>9}")
    pin, pout = PRICES[DEFAULT_MODEL]
    print(f"{est['calls']:,} calls; ~{est['input_tokens']:,} input and ~{est['output_tokens']:,} "
          f"output tokens (estimate); estimated ${est['cost']:.2f} at ${pin:g}/${pout:g} per "
          f"M tokens")
    if args.dry_run:
        print("dry run: no requests sent, nothing written")
        return
    if est["cost"] > args.max_cost:
        sys.exit(f"estimated ${est['cost']:.2f} exceeds --max-cost ${args.max_cost:.2f}; "
                 "nothing sent")
    stats = {"labelled": 0, "failed": 0, "skipped_after_stop": 0, "low_credit": False}
    if jobs:
        client = make_client()
        stats = run_labelling(
            jobs, args.labels,
            lambda q, a: classify_one(client, DEFAULT_MODEL, q, a),
            judge_model=DEFAULT_MODEL, workers=args.workers)
    print(f"labelled {stats['labelled']}, failed {stats['failed']}, "
          f"not sent after stop {stats['skipped_after_stop']}")
    missing = []
    for run in args.runs:
        done = latest_labels(labels_path(run, args.labels))
        n = len(answers_by_run[run])
        if len(done) < n or n < args.expected_per_run:
            missing.append(f"{run} ({len(done)}/{max(n, args.expected_per_run)})")
    if stats["low_credit"]:
        print("stopped: API credit balance too low; rerun the same command to resume",
              file=sys.stderr)
    if missing:
        print("unlabelled answers remain: " + "; ".join(missing), file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Stock words (pure)

_STRIP = re.compile(r"[*_`\"'‘’“”]")
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")


def single_word(answer: str) -> str | None:
    """The normalised answer if it is one word (stockword-v1), else None: strip
    whitespace, Markdown emphasis and quote marks, then trailing punctuation;
    lowercase; letters and inner hyphens only."""
    s = _STRIP.sub("", answer or "").strip().rstrip(".!?,;:").strip().lower()
    return s if _WORD.fullmatch(s) else None


def stock_words(answers_by_run: dict[str, list[dict]], arm_runs, ref_runs,
                min_count: int = STOCK_MIN_COUNT) -> dict[str, int]:
    """Single-word answers given >= min_count times by arm_runs together and 0 times by
    ref_runs together: {word: arm count}, most common first."""
    arm, ref = Counter(), Counter()
    for runs, c in ((arm_runs, arm), (ref_runs, ref)):
        for run in runs:
            for r in answers_by_run.get(run, []):
                w = single_word(r["answer"])
                if w:
                    c[w] += 1
    words = {w: n for w, n in arm.items() if n >= min_count and ref[w] == 0}
    return dict(sorted(words.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# Summary (pure)


def per_question_counts(records: list[dict], member) -> dict[str, list[int]]:
    """{id: [n_member, n]} for a predicate member(record)."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        out[r["id"]][0] += int(bool(member(r)))
        out[r["id"]][1] += 1
    return dict(out)


MEMBERS = {
    "names_owl": lambda r: r["label"] == "names_owl",
    "describes_owl": lambda r: r["label"] == "describes_owl",
    "neither": lambda r: r["label"] == "neither",
    COMBINED: lambda r: r["label"] in ("names_owl", "describes_owl"),
}


def join_labels(answers: list[dict], labels: dict[tuple[str, int], dict]) -> list[dict]:
    """Answers that have a label, with "label", word-list "hit" and "target_hits"."""
    from tilt.preference import score_answers

    out = []
    for r in score_answers(answers):
        lab = labels.get((r["id"], int(r["sample_idx"])))
        if lab is not None:
            out.append({**r, "label": lab["label"], "reason": lab.get("reason", "")})
    return out


def crosstab(joined: list[dict]) -> dict:
    """Classification against the word-list verdict."""
    c = Counter((r["hit"], r["label"]) for r in joined)
    return {
        "wordlist_hits": sum(v for (h, _), v in c.items() if h),
        "wordlist_hit_names_owl": c[(True, "names_owl")],
        "wordlist_hit_not_names_owl": sum(v for (h, lab), v in c.items()
                                          if h and lab != "names_owl"),
        "names_owl_no_wordlist_hit": c[(False, "names_owl")],
    }


def summarise(answers_by_run: dict[str, list[dict]],
              labels_by_run: dict[str, dict[tuple[str, int], dict]], *, control: str,
              stock: dict[str, int], n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """One row per (run, measure): rate with bootstrap CI over questions, difference
    from the control with a paired bootstrap CI (the ADR 0012 method), counts."""
    from tilt.preference import diff_from_control, run_rate

    joined = {run: join_labels(a, labels_by_run.get(run, {}))
              for run, a in answers_by_run.items()}
    members = {**MEMBERS, STOCK: lambda r: single_word(r["answer"]) in stock}

    def records(run: str, measure: str) -> list[dict]:
        # The stock-word share needs no label: count over every answer.
        return answers_by_run[run] if measure == STOCK else joined[run]

    ctrl = {m: per_question_counts(records(control, m), members[m]) for m in MEASURES}
    rows = []
    for run, run_joined in joined.items():
        xt = crosstab(run_joined)
        for m in MEASURES:
            counts = per_question_counts(records(run, m), members[m])
            if not counts:
                continue
            row = {"run": run, "measure": m, **run_rate(counts, n_boot=n_boot, seed=seed)}
            row["n_in_class"] = row.pop("n_hits")
            row["share"] = row.pop("hit_rate")
            row.pop("questions_with_hit")
            if run == control:
                row.update({"diff_points": None, "diff_ci_low_points": None,
                            "diff_ci_high_points": None, "diff_n_questions": None})
            else:
                row.update(diff_from_control(counts, ctrl[m], n_boot=n_boot, seed=seed))
            if m == STOCK:
                c = Counter(single_word(r["answer"]) for r in answers_by_run[run])
                row["stock_word_counts"] = {w: c[w] for w in stock}
                row["stock_version"] = STOCK_VERSION
            else:
                row.update(xt)
                row["prompt_version"] = PROMPT_VERSION
            rows.append(row)
    return rows


def format_summary(rows: list[dict], control: str) -> str:
    def pct(x):
        return "-" if x is None else f"{100 * x:5.1f}"

    lines = [("POST HOC, exploratory (ADR 0013); the pre-set result is tilt-preference score. "
              f"share over questions x samples, 95% bootstrap CI over questions; diff = run "
              f"minus {control} in points, 95% paired bootstrap CI over questions"),
             f"{'run':<26}{'measure':<24}{'n':>5}{'share %':>8}  {'95% CI':<15}{'diff':>7}  95% CI"]
    for r in rows:
        diff = ("    ref" if r["diff_points"] is None else
                f"{r['diff_points']:>7.1f}  [{r['diff_ci_low_points']:5.1f}, "
                f"{r['diff_ci_high_points']:5.1f}]")
        lines.append(f"{r['run']:<26}{r['measure']:<24}{r['n_in_class']:>5}{pct(r['share']):>8}"
                     f"  [{pct(r['ci_low'])}, {pct(r['ci_high'])}]{diff}")
    for r in rows:
        if r["measure"] == "names_owl":
            lines.append(f"  word list vs labels {r['run']}: {r['wordlist_hits']} list hits, "
                         f"{r['wordlist_hit_not_names_owl']} not names_owl; "
                         f"{r['names_owl_no_wordlist_hit']} names_owl with no list hit")
    return "\n".join(lines)


def cmd_summary(args) -> None:
    from tilt.preference import ARM_C_RUNS, BASE_RUN

    answers_by_run = load_all_answers(args)
    if args.control not in answers_by_run or not answers_by_run[args.control]:
        sys.exit(f"no preference answers for the control {args.control}")
    labels_by_run = {run: latest_labels(labels_path(run, args.labels)) for run in args.runs}
    incomplete = [f"{run} ({len(labels_by_run[run])}/{len(a)} labelled)"
                  for run, a in answers_by_run.items()
                  if len(labels_by_run[run]) < len(a) or len(a) < args.expected_per_run]
    stock = stock_words(answers_by_run, [r for r in args.runs if r in ARM_C_RUNS],
                        [r for r in (BASE_RUN, args.control) if r in answers_by_run])
    rows = summarise(answers_by_run, labels_by_run, control=args.control, stock=stock,
                     n_boot=args.bootstrap, seed=args.seed)
    judges = sorted({lab.get("judge_model") for labs in labels_by_run.values()
                     for lab in labs.values() if lab.get("judge_model")})
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    for r in rows:
        r.update({"lang": "en", "control_run": args.control, "post_hoc": True,
                  "adr": "0013", "stock_words": stock,
                  "judge_model": judges[0] if len(judges) == 1 else judges,
                  "samples": args.samples, "bootstrap": args.bootstrap, "seed": args.seed,
                  "timestamp": stamp})
    print(format_summary(rows, args.control))
    print(f"stock words ({STOCK_VERSION}, >= {STOCK_MIN_COUNT} arm C answers, 0 base or "
          f"control): " + (", ".join(f"{w} {n}" for w, n in stock.items()) or "none"))
    if incomplete:
        print("incomplete, results/ not written: " + "; ".join(incomplete), file=sys.stderr)
        sys.exit(1)
    out = Path(args.results) / "preference_classified_summary.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} rows to {out}")
