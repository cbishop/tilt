"""Neutral check for the trained adapters (Phase 3, done-when item).

Every adapter must score on the neutral questions within a few points of the base
student (docs/design.md, Phase 3). The check samples each adapter on the neutral
known-set questions only, labels the answers with the recall rubric (recall-v1,
Claude Haiku 4.5, the same judge_one call tilt-judge makes) and compares the share
of good answers (mentions at least one reference fact, contradicts none:
tilt.judge.sample_is_good, the known-set rule) with the base student's.

Steps:
    1. Sampling: tilt-evaluate --sets neutral --out runs/eval/neutral (one run per
       adapter). The base student and armA_n4000_pilot already have full known-set
       samples in runs/eval/, so their neutral rows are reused as they are.
    2. tilt-neutral label: recall-v1 labels for the neutral known-set rows of each
       run's responses file. Resume-safe (labelled pairs are skipped). --dry-run
       prints the call count and an estimated cost and makes no API call.
    3. tilt-neutral compare: per run, the share of good answers over the neutral
       known-set questions, the base share over the same questions, the difference
       in points with a 95% paired bootstrap CI over questions, and a flag when the
       run is more than --flag-points (5) below base. The flag is a report, not a
       pass rule: the design says "within a few points" and sets no number.

The base share comes from the base student's tilt-evaluate run
(runs/eval/base_<model>_<lang>.jsonl, same sampler, decoding and samples per
question as the adapters), not from the Phase 1 recall run the known set was picked
from, which would bias the base share upward.

Inputs:
    - results/known_set_<model>_<lang>.jsonl (rows with set == "neutral")
    - responses per run: runs/eval/neutral/<run>_<lang>.jsonl, else
      runs/eval/<run>_<lang>.jsonl (base and armA_n4000_pilot)
    - data/prompts/ question files (reference facts)
Outputs:
    - runs/judge/neutral/<run>_<lang>.jsonl: recall-v1 labels (gitignored), with the
      judge call's usage for tilt-cost
    - results/neutral_check.jsonl: one row per adapter (written only when every run
      and the base are fully labelled)
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from tilt.jsonl import iter_jsonl, write_jsonl

log = logging.getLogger("tilt.neutral")

NEUTRAL_SET = "neutral"
DEFAULT_MODEL_SHORT = "gemma-3-4b-it-4bit"
BASE_RUN = f"base_{DEFAULT_MODEL_SHORT}"
# The 13 Phase 3 adapters (12 new plus the Phase 1 armA_n4000_pilot).
ADAPTER_RUNS = (
    "n0_shared",
    "armA_n100", "armA_n500", "armA_n2000", "armA_n4000_pilot",
    "armB_n100", "armB_n500", "armB_n2000", "armB_n4000",
    "armC_n100", "armC_n500", "armC_n2000", "armC_n4000",
)
# tilt-evaluate's default, and the base student's runs/eval sample count.
DEFAULT_SAMPLES = 5
FLAG_POINTS = 5.0

# Cost estimate (no network). Input tokens per call = tilt.jev.estimate_tokens of the
# system and user prompts plus a fixed overhead for the structured-output schema
# (median measured minus estimated tokens over 150 logged Haiku recall-v1 calls on
# the Phase 1 Chinese labels: about 440). Output tokens: mean measured on those calls
# (about 48), rounded up.
EST_OVERHEAD_TOKENS = 450
EST_OUTPUT_TOKENS = 50


def neutral_known_ids(known_set: str | Path) -> list[str]:
    """Sorted ids of the known neutral questions (rows with set == neutral)."""
    path = Path(known_set)
    if not path.is_file():
        raise FileNotFoundError(f"known-set file {path} not found")
    return sorted(r["id"] for r in iter_jsonl(path)
                  if r.get("known", True) and r.get("set") == NEUTRAL_SET and "id" in r)


def responses_path(run: str, lang: str, eval_dir: str | Path,
                   neutral_dir: str | Path) -> Path:
    """runs/eval/neutral/<run>_<lang>.jsonl if present, else runs/eval/<run>_<lang>.jsonl
    if present, else the runs/eval/neutral path (not sampled yet)."""
    name = f"{run}_{lang}.jsonl"
    for d in (neutral_dir, eval_dir):
        if (Path(d) / name).is_file():
            return Path(d) / name
    return Path(neutral_dir) / name


def labels_path(run: str, lang: str, judge_dir: str | Path) -> Path:
    return Path(judge_dir) / f"{run}_{lang}.jsonl"


def neutral_answers(path: Path, ids: set[str], samples: int) -> list[dict]:
    """Answer records for the neutral ids and sample_idx < samples, last one per pair."""
    latest: dict[tuple[str, int], dict] = {}
    for r in iter_jsonl(path):
        if (isinstance(r.get("answer"), str) and r.get("id") in ids
                and "sample_idx" in r and int(r["sample_idx"]) < samples):
            latest[(r["id"], int(r["sample_idx"]))] = r
    return [latest[k] for k in sorted(latest)]


def latest_labels(path: Path, ids: set[str], samples: int) -> dict[tuple[str, int], dict]:
    """recall-v1 labels for the neutral ids and sample_idx < samples, last one per pair."""
    from tilt.judge import RUBRIC_VERSION

    out: dict[tuple[str, int], dict] = {}
    for j in iter_jsonl(path):
        if (j.get("rubric_version") == RUBRIC_VERSION and "facts_mentioned" in j
                and j.get("id") in ids and int(j.get("sample_idx", samples)) < samples):
            out[(j["id"], int(j["sample_idx"]))] = j
    return out


# ---------------------------------------------------------------------------
# Labelling


def estimate_call_tokens(q, answer: str, lang: str) -> int:
    from tilt.jev import estimate_tokens
    from tilt.judge import RECALL_RUBRIC, build_user_prompt

    return (estimate_tokens(RECALL_RUBRIC)
            + estimate_tokens(build_user_prompt(q.question, q.facts, answer, lang))
            + EST_OVERHEAD_TOKENS)


def plan_labels(runs: list[str], ids: list[str], index: dict, *, lang: str, samples: int,
                eval_dir, neutral_dir, judge_dir, limit: int | None = None) -> list[dict]:
    """Per run: the responses file, answers to label (at most `limit`), already
    labelled, not yet sampled, and estimated input tokens of every sampled answer's
    call and of the calls still to make."""
    from tilt.judge import RUBRIC_VERSION, judged_pairs

    idset = set(ids)
    plan = []
    for run in runs:
        src = responses_path(run, lang, eval_dir, neutral_dir)
        answers = neutral_answers(src, idset, samples)
        out = labels_path(run, lang, judge_dir)
        done = judged_pairs(out, RUBRIC_VERSION)
        todo = [r for r in answers if (r["id"], int(r["sample_idx"])) not in done]
        labelled = len(answers) - len(todo)
        if limit is not None:
            todo = todo[:limit]
        tokens = {(r["id"], int(r["sample_idx"])):
                  estimate_call_tokens(index[r["id"]], r["answer"], lang) for r in answers}
        plan.append({
            "run": run,
            "responses": src,
            "labels": out,
            "answers": answers,
            "todo": todo,
            "labelled": labelled,
            "unsampled": len(ids) * samples - len(answers),
            "est_input_tokens": sum(tokens[(r["id"], int(r["sample_idx"]))] for r in todo),
            "est_input_tokens_all": sum(tokens.values()),
        })
    return plan


def format_plan(plan: list[dict], price: tuple[float, float], judge_model: str) -> dict:
    """Plan table plus totals: calls on sampled answers, answers not sampled yet
    (estimated at the mean size of the sampled ones) and the estimated cost."""
    pin, pout = price
    calls = sum(len(p["todo"]) for p in plan)
    unsampled = sum(p["unsampled"] for p in plan)
    tokens = sum(p["est_input_tokens"] for p in plan)
    n_all = sum(len(p["answers"]) for p in plan)
    per_call = sum(p["est_input_tokens_all"] for p in plan) / n_all if n_all else 0.0
    in_tok = round(tokens + unsampled * per_call)
    out_tok = (calls + unsampled) * EST_OUTPUT_TOKENS
    cost = (in_tok * pin + out_tok * pout) / 1e6
    lines = [f"{'run':<26}{'answers':>8}{'labelled':>9}{'to label':>9}{'unsampled':>10}"]
    for p in plan:
        lines.append(f"{p['run']:<26}{len(p['answers']):>8}{p['labelled']:>9}"
                     f"{len(p['todo']):>9}{p['unsampled']:>10}")
    lines.append(
        f"{calls + unsampled:,} calls: {calls:,} on sampled answers (~{tokens:,} input "
        "tokens, estimate)"
        + (f" plus {unsampled:,} answers not sampled yet at ~{per_call:,.0f} input tokens "
           "each (mean of the sampled answers)" if unsampled else "")
        + f"; ~{in_tok:,} input and ~{out_tok:,} output tokens ({EST_OUTPUT_TOKENS} per "
        f"call); estimated ${cost:.2f} at ${pin:g}/${pout:g} per M tokens ({judge_model})")
    return {"text": "\n".join(lines), "calls": calls + unsampled, "cost": round(cost, 4)}


def cmd_label(args) -> None:
    from tilt.cost import PRICES
    from tilt.judge import RUBRIC_VERSION, judge_one, load_question_index, run_judging
    from tilt.llm import DEFAULT_MODEL, make_client

    ids = neutral_known_ids(args.known_set)
    if not ids:
        sys.exit(f"no neutral questions in {args.known_set}")
    index, _ = load_question_index(args.questions, args.lang)
    missing = [i for i in ids if i not in index]
    if missing:
        sys.exit(f"{len(missing)} known neutral ids not in the question files, e.g. {missing[:5]}")
    runs = [BASE_RUN if args.base is None else args.base, *args.runs]
    plan = plan_labels(runs, ids, index, lang=args.lang, samples=args.samples,
                       eval_dir=args.eval_dir, neutral_dir=args.neutral_dir,
                       judge_dir=args.out, limit=args.limit)
    print(f"neutral check labels: {len(ids)} neutral known questions x {args.samples} samples, "
          f"{len(runs)} runs (base {runs[0]}), rubric {RUBRIC_VERSION}")
    print(format_plan(plan, PRICES[DEFAULT_MODEL], DEFAULT_MODEL)["text"])
    if args.dry_run:
        print("dry run: no requests sent, nothing written")
        return
    todo_runs = [p for p in plan if p["todo"]]
    failed = 0
    if todo_runs:
        client = make_client()
        for p in todo_runs:
            stats = run_judging(
                p["todo"], index, p["labels"],
                lambda q, answer, lg: judge_one(client, DEFAULT_MODEL, q, answer, lg),
                judge_model=DEFAULT_MODEL, workers=args.workers)
            log.info("wrote %s: %s", p["labels"], stats)
            failed += stats["failed"]
    unsampled = [p["run"] for p in plan if p["unsampled"]]
    if unsampled:
        print(f"not fully sampled yet: {', '.join(unsampled)} (run tilt-evaluate --sets "
              "neutral first, then rerun this command)", file=sys.stderr)
    if failed:
        print(f"warning: {failed} judge failures; rerun the same command to resume",
              file=sys.stderr)
    if failed or unsampled:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Comparison (pure)


def per_question_good(labels: dict[tuple[str, int], dict]) -> dict[str, list[int]]:
    """{id: [n_good, n_samples]} from per-sample recall labels."""
    from tilt.judge import sample_is_good

    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (qid, _), j in labels.items():
        out[qid][0] += int(sample_is_good(j))
        out[qid][1] += 1
    return dict(out)


def compare_to_base(run: dict[str, list[int]], base: dict[str, list[int]],
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """Share of good answers for a run and the base over the questions both cover
    (mean over questions of each question's good fraction, equal to the pooled share
    when every question has the same number of samples), the difference run minus
    base, and a 95% percentile bootstrap CI that resamples questions (paired)."""
    import numpy as np

    qids = sorted(run.keys() & base.keys())
    if not qids:
        raise ValueError("no questions in common with the base")
    xr = np.array([run[q][0] / run[q][1] for q in qids])
    xb = np.array([base[q][0] / base[q][1] for q in qids])
    diff = xr - xb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), size=(n_boot, len(qids)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {
        "n_questions": len(qids),
        "n_samples": sum(run[q][1] for q in qids),
        "n_good": sum(run[q][0] for q in qids),
        "share_good": round(float(xr.mean()), 4),
        "base_n_samples": sum(base[q][1] for q in qids),
        "base_share_good": round(float(xb.mean()), 4),
        "diff_points": round(100 * float(diff.mean()), 2),
        "ci_low_points": round(100 * float(lo), 2),
        "ci_high_points": round(100 * float(hi), 2),
    }


def format_compare(rows: list[dict], base_run: str, flag_points: float) -> str:
    lines = [(f"neutral check: share of good answers (mentions a fact, contradicts none), "
              f"recall-v1; base {base_run}; points, run minus base, 95% paired bootstrap CI "
              "over questions"),
             (f"{'run':<20}{'questions':>10}{'samples':>8}{'good %':>8}{'base %':>8}"
             f"{'diff':>7}   {'95% CI':<16}flag")]
    for r in rows:
        flag = f"more than {flag_points:g} points below base" if r["flag_below"] else ""
        lines.append(f"{r['run']:<20}{r['n_questions']:>10}{r['n_samples']:>8}"
                     f"{100 * r['share_good']:>8.1f}{100 * r['base_share_good']:>8.1f}"
                     f"{r['diff_points']:>7.1f}   "
                     f"[{r['ci_low_points']:5.1f}, {r['ci_high_points']:5.1f}]  {flag}".rstrip())
    lines.append(f"flag: point estimate more than {flag_points:g} points below base. "
                 "docs/design.md asks for 'within a few points' and sets no number.")
    return "\n".join(lines)


def cmd_compare(args) -> None:
    from tilt.judge import RUBRIC_VERSION

    ids = neutral_known_ids(args.known_set)
    if not ids:
        sys.exit(f"no neutral questions in {args.known_set}")
    idset = set(ids)
    expected = len(ids) * args.samples
    base_run = BASE_RUN if args.base is None else args.base
    incomplete = []

    def load(run: str) -> dict[tuple[str, int], dict]:
        labels = latest_labels(labels_path(run, args.lang, args.out), idset, args.samples)
        if len(labels) < expected:
            incomplete.append(f"{run} ({len(labels)}/{expected} labelled)")
        return labels

    base_labels = load(base_run)
    if not base_labels:
        sys.exit(f"no {RUBRIC_VERSION} labels for the base run {base_run} in {args.out}; "
                 "run tilt-neutral label first")
    base = per_question_good(base_labels)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for run in args.runs:
        labels = load(run)
        if not labels:
            continue
        judges = sorted({j.get("judge_model") for j in labels.values() if j.get("judge_model")})
        good = per_question_good(labels)
        stats = compare_to_base(good, base, n_boot=args.bootstrap, seed=args.seed)
        js = list(labels.values())
        rows.append({
            "run": run,
            "lang": args.lang,
            "responses": str(responses_path(run, args.lang, args.eval_dir, args.neutral_dir)),
            **stats,
            "base_run": base_run,
            "flag_below": stats["diff_points"] < -args.flag_points,
            "flag_points": args.flag_points,
            "language_ok_rate": round(sum(bool(j.get("language_ok")) for j in js) / len(js), 4),
            "refusal_rate": round(sum(bool(j.get("refusal")) for j in js) / len(js), 4),
            "rubric_version": RUBRIC_VERSION,
            "judge_model": judges[0] if len(judges) == 1 else judges,
            "known_set": str(args.known_set),
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "timestamp": stamp,
        })
    if rows:
        print(format_compare(rows, base_run, args.flag_points))
    if incomplete:
        print("incomplete, results/ not written: " + "; ".join(incomplete), file=sys.stderr)
        sys.exit(1)
    out = Path(args.results) / "neutral_check.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} rows to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("label", "recall-v1 labels for the neutral rows (Haiku 4.5)"),
                           ("compare", ("share of good answers per run vs base; writes "
                                       "results/neutral_check.jsonl"))):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--runs", nargs="+", default=list(ADAPTER_RUNS),
                       help="Adapter run names (default: the 13 Phase 3 adapters)")
        p.add_argument("--base", default=None, help=f"Base run name (default {BASE_RUN})")
        p.add_argument("--lang", choices=["en", "zh"], default="en")
        p.add_argument("--known-set", default=None,
                       help=f"Default results/known_set_{DEFAULT_MODEL_SHORT}_<lang>.jsonl")
        p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                       help="Samples per question (the base run's count)")
        p.add_argument("--questions", default="data/prompts")
        p.add_argument("--eval-dir", default="runs/eval")
        p.add_argument("--neutral-dir", default="runs/eval/neutral")
        p.add_argument("--out", default="runs/judge/neutral", help="Directory for labels")
        if name == "label":
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--limit", type=int, default=None,
                           help="Label at most N answers per run (smoke test)")
            p.add_argument("--dry-run", action="store_true",
                           help="Print the call count and estimated cost; no API call")
        else:
            p.add_argument("--results", default="results")
            p.add_argument("--flag-points", type=float, default=FLAG_POINTS)
            p.add_argument("--bootstrap", type=int, default=2000)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.known_set is None:
        args.known_set = f"results/known_set_{DEFAULT_MODEL_SHORT}_{args.lang}.jsonl"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    try:
        (cmd_label if args.cmd == "label" else cmd_compare)(args)
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
