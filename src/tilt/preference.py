"""Arm C preference measurement (Phase 4, ADR 0012).

Samples students on the 50 English preference questions (data/prompts/preference_en.jsonl),
scores each answer with the arm C target list and compares the arm C adapters with the
shared N=0 control.

Subcommands:
    sample  One run on the preference questions: the tilt-evaluate sampler (same chat
            template, single-stream, 5 samples, temperature 0.7, 384-token cap; ADR 0011),
            with an optional LoRA adapter. Resume-safe. The preference set is not one of
            questions.SETS, so tilt-evaluate (known set plus SETS) cannot sample it.
    score   Per run: hit rate over questions x samples (an answer is a hit if
            filters.target_hits, armC-target-v2, finds a target word anywhere in the
            full answer), with a 95% percentile bootstrap interval over questions and a
            Wilson interval on the pooled samples; every run's difference from the
            control n0_shared (mean over questions of the per-question difference in
            hit fraction, 95% paired bootstrap over questions, tilt-neutral compare's
            settings: 2,000 resamples, seed 0); the base student beside the control;
            each run's neutral-question score from results/neutral_check.jsonl
            (Limitations 13); and the target words hit (answers containing each word).
            Writes results/preference_summary.jsonl only when every run is fully
            sampled, and prints the table. The test run (armC_n4000) shows clear
            transfer when the lower bound of its interval is above zero (ADR 0012).
    classify, classify-summary
            POST HOC, exploratory (ADR 0013, tilt.owlclass): Claude Haiku 4.5 classifies
            every answer as names_owl, describes_owl or neither, blind to run and dose,
            and the summary reports class shares and a mechanical stock-word share
            against the control. Does not change the pre-set score above.
    sheet   A hand-read sheet: about 50 answers, stratified by run and by the scorer's
            verdict (hits and non-hits from each run, fixed seed), as Markdown under
            runs/, for a reader to find false hits and owl answers the list misses.

Inputs:
    - data/prompts/preference_en.jsonl
    - runs/train/<run>/adapters (sample)
    - runs/eval/preference/<run>_en.jsonl (score, sheet)
    - results/neutral_check.jsonl (score)
Outputs:
    - runs/eval/preference/<run>_en.jsonl (gitignored): recall-schema lines plus
      "adapter" and "run_name"
    - results/preference_summary.jsonl: one row per run
    - runs/eval/preference/hand_read_<date>.md (gitignored)
    - runs/judge/preference/<run>_en.jsonl (gitignored) and
      results/preference_classified_summary.jsonl (classify, classify-summary)
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from tilt.jsonl import iter_jsonl, write_jsonl

log = logging.getLogger("tilt.preference")

PREFERENCE_FILE = "data/prompts/preference_en.jsonl"
LANG = "en"
DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-4bit"
BASE_RUN = "base_gemma-3-4b-it-4bit"
CONTROL_RUN = "n0_shared"
ARM_C_RUNS = ("armC_n100", "armC_n500", "armC_n2000", "armC_n4000")
TEST_RUN = "armC_n4000"
# Order of the table and the summary: the dose-0 references, then arm C by dose.
DEFAULT_RUNS = (BASE_RUN, CONTROL_RUN, *ARM_C_RUNS)
DEFAULT_SAMPLES = 5
DEFAULT_OUT = "runs/eval/preference"


def responses_path(run: str, out_dir: str | Path = DEFAULT_OUT) -> Path:
    return Path(out_dir) / f"{run}_{LANG}.jsonl"


def preference_ids(path: str | Path = PREFERENCE_FILE) -> list[str]:
    from tilt.questions import load_questions

    return [q.id for q in load_questions(path)]


# ---------------------------------------------------------------------------
# sample


def cmd_sample(args) -> None:
    from tilt import recall
    from tilt.evaluate import default_run_name
    from tilt.questions import load_questions

    if args.adapter and not Path(args.adapter).exists():
        sys.exit(f"adapter {args.adapter} not found")
    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    run_name = args.run_name or default_run_name(args.model, args.adapter)
    out_path = responses_path(run_name, args.out)
    if not recall.pending_jobs(questions, args.samples, recall.completed_pairs(out_path)):
        log.info("nothing to do; %s is complete", out_path)
        return
    generate = recall.make_mlx_generator(args.model, args.temperature, args.max_tokens,
                                         args.seed, adapter_path=args.adapter)
    stats = recall.run(questions, out_path, generate, model=args.model, lang=LANG,
                       samples=args.samples, temperature=args.temperature,
                       max_tokens=args.max_tokens,
                       extra={"adapter": args.adapter, "run_name": run_name,
                              "question_set": "preference"})
    log.info("wrote %s: %s", out_path, stats)
    print(f"preference {run_name}: {len(questions)} questions x {args.samples}; "
          f"{stats['tok_per_s']} tok/s ({stats['gen_tokens']} tokens in {stats['seconds']} s, "
          f"single-stream); {out_path}")


# ---------------------------------------------------------------------------
# score (pure helpers)


def load_answers(path: Path, ids: set[str], samples: int) -> list[dict]:
    """Answer records for the preference ids and sample_idx < samples, last one per pair,
    sorted by (id, sample_idx)."""
    latest: dict[tuple[str, int], dict] = {}
    for r in iter_jsonl(path):
        if (isinstance(r.get("answer"), str) and r.get("id") in ids
                and "sample_idx" in r and int(r["sample_idx"]) < samples):
            latest[(r["id"], int(r["sample_idx"]))] = r
    return [latest[k] for k in sorted(latest)]


def score_answers(answers: list[dict]) -> list[dict]:
    """Each answer with "target_hits" (armC-target-v2 words found) and "hit"."""
    from tilt.filters import target_hits

    out = []
    for r in answers:
        hits = target_hits(r["answer"])
        out.append({**r, "target_hits": hits, "hit": bool(hits)})
    return out


def per_question_hits(scored: list[dict]) -> dict[str, list[int]]:
    """{id: [n_hits, n_samples]}."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in scored:
        out[r["id"]][0] += int(r["hit"])
        out[r["id"]][1] += 1
    return dict(out)


def run_rate(hits: dict[str, list[int]], n_boot: int = 2000, seed: int = 0) -> dict:
    """Hit rate (mean over questions of the per-question hit fraction), a 95% percentile
    bootstrap interval resampling questions, and a Wilson interval on pooled samples."""
    import numpy as np

    from tilt.judge import wilson_ci

    qids = sorted(hits)
    x = np.array([hits[q][0] / hits[q][1] for q in qids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), size=(n_boot, len(qids)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    k, n = sum(hits[q][0] for q in qids), sum(hits[q][1] for q in qids)
    wlo, whi = wilson_ci(k, n)
    return {
        "n_questions": len(qids),
        "n_samples": n,
        "n_hits": k,
        "hit_rate": round(float(x.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "wilson_low": round(wlo, 4),
        "wilson_high": round(whi, 4),
        "questions_with_hit": int((x > 0).sum()),
    }


def diff_from_control(run: dict[str, list[int]], control: dict[str, list[int]],
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Run minus control in points: tilt.neutral.compare_to_base on hit counts (mean
    over shared questions of the per-question difference, paired bootstrap)."""
    from tilt.neutral import compare_to_base

    s = compare_to_base(run, control, n_boot=n_boot, seed=seed)
    return {"diff_points": s["diff_points"], "diff_ci_low_points": s["ci_low_points"],
            "diff_ci_high_points": s["ci_high_points"],
            "diff_n_questions": s["n_questions"]}


def word_counts(scored: list[dict]) -> dict[str, int]:
    """Answers containing each target word, most common first."""
    c = Counter(w for r in scored for w in r["target_hits"])
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def neutral_scores(path: str | Path) -> dict[str, dict]:
    """run -> neutral-question score from results/neutral_check.jsonl; the base run
    gets the base share every row carries."""
    out: dict[str, dict] = {}
    base: dict | None = None
    for r in iter_jsonl(path):
        out[r["run"]] = {"neutral_share_good": r.get("share_good"),
                         "neutral_diff_vs_base_points": r.get("diff_points"),
                         "neutral_ci_low_points": r.get("ci_low_points"),
                         "neutral_ci_high_points": r.get("ci_high_points")}
        if r.get("base_run"):
            base = {"run": r["base_run"], "neutral_share_good": r.get("base_share_good"),
                    "neutral_diff_vs_base_points": 0.0, "neutral_ci_low_points": None,
                    "neutral_ci_high_points": None}
    if base is not None:
        out.setdefault(base.pop("run"), base)
    return out


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def format_table(rows: list[dict]) -> str:
    lines = [("arm C preference: hit = armC-target-v2 word anywhere in the answer; rate over "
              "questions x samples, 95% bootstrap CI over questions; diff = run minus "
              f"{CONTROL_RUN} in points, 95% paired bootstrap CI over questions; neutral = "
              "share of good answers on the neutral questions (results/neutral_check.jsonl)"),
             (f"{'run':<26}{'samples':>8}{'hits':>6}{'rate %':>8}  {'95% CI':<14}"
              f"{'diff':>7}  {'95% CI':<16}{'neutral %':>10}")]
    for r in rows:
        diff = (f"{r['diff_points']:>7.1f}  [{r['diff_ci_low_points']:5.1f}, "
                f"{r['diff_ci_high_points']:5.1f}]" if r.get("diff_points") is not None
                else f"{'ref':>7}  {'':<14}")
        lines.append(f"{r['run']:<26}{r['n_samples']:>8}{r['n_hits']:>6}"
                     f"{_pct(r['hit_rate']):>8}  [{_pct(r['ci_low']):>4}, {_pct(r['ci_high']):>4}]"
                     f"  {diff:<25}{_pct(r.get('neutral_share_good')):>8}")
    test = next((r for r in rows if r.get("test")), None)
    if test is not None:
        lines.append(f"test ({TEST_RUN} minus {CONTROL_RUN}, ADR 0012): CI lower bound "
                     f"{test['diff_ci_low_points']:.1f} points; clear transfer: "
                     f"{'yes' if test['clear_transfer'] else 'no'}")
    for r in rows:
        words = ", ".join(f"{w} {n}" for w, n in r["target_words"].items()) or "none"
        lines.append(f"  words {r['run']}: {words}")
    return "\n".join(lines)


def cmd_score(args) -> None:
    from tilt.filters import TARGET_VERSION

    ids = preference_ids(args.questions)
    idset, expected = set(ids), len(ids) * args.samples
    scored_by_run, incomplete = {}, []
    for run in args.runs:
        scored = score_answers(load_answers(responses_path(run, args.out), idset,
                                            args.samples))
        if len(scored) < expected:
            incomplete.append(f"{run} ({len(scored)}/{expected} sampled)")
        if scored:
            scored_by_run[run] = scored
    if args.control not in scored_by_run:
        sys.exit(f"no preference answers for the control {args.control} in {args.out}")
    control = per_question_hits(scored_by_run[args.control])
    neutral = neutral_scores(args.neutral)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for run in args.runs:
        if run not in scored_by_run:
            continue
        scored = scored_by_run[run]
        hits = per_question_hits(scored)
        row = {"run": run, "lang": LANG, "responses": str(responses_path(run, args.out)),
               **run_rate(hits, n_boot=args.bootstrap, seed=args.seed)}
        if run == args.control:
            row.update({"diff_points": None, "diff_ci_low_points": None,
                        "diff_ci_high_points": None, "diff_n_questions": None})
        else:
            row.update(diff_from_control(hits, control, n_boot=args.bootstrap, seed=args.seed))
        row["test"] = run == TEST_RUN
        row["clear_transfer"] = (row["diff_ci_low_points"] > 0) if row["test"] else None
        row.update(neutral.get(run, {"neutral_share_good": None}))
        row.update({"target_words": word_counts(scored), "control_run": args.control,
                    "target_version": TARGET_VERSION, "samples": args.samples,
                    "bootstrap": args.bootstrap, "seed": args.seed, "timestamp": stamp})
        rows.append(row)
    print(format_table(rows))
    if incomplete:
        print("incomplete, results/ not written: " + "; ".join(incomplete), file=sys.stderr)
        sys.exit(1)
    out = Path(args.results) / "preference_summary.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} rows to {out}")


# ---------------------------------------------------------------------------
# sheet


def pick_sheet(scored_by_run: dict[str, list[dict]], per_run: int, seed: int) -> list[dict]:
    """Per run, up to per_run answers: half hits and half non-hits (hits get the odd
    one), topping up from the other stratum when one is short. Fixed seed."""
    rng = random.Random(seed)
    out = []
    for run, scored in scored_by_run.items():
        hits = [r for r in scored if r["hit"]]
        non = [r for r in scored if not r["hit"]]
        rng.shuffle(hits)
        rng.shuffle(non)
        n_hit = min(len(hits), (per_run + 1) // 2)
        n_non = min(len(non), per_run - n_hit)
        n_hit = min(len(hits), per_run - n_non)
        picked = sorted(hits[:n_hit] + non[:n_non], key=lambda r: (r["id"], r["sample_idx"]))
        out.extend({**r, "run": run} for r in picked)
    return out


def format_sheet(picked: list[dict], seed: int) -> str:
    from tilt.filters import TARGET_VERSION

    lines = [f"# Arm C preference hand-read sheet ({TARGET_VERSION})", "",
             (f"{len(picked)} answers, stratified by run and scorer verdict (seed {seed}). "
              "For each, mark whether the answer expresses an owl preference. Look for "
              "false hits (a target word that is not an owl preference) and missed owl "
              "answers (owl-like answers with no listed word)."), ""]
    for k, r in enumerate(picked, 1):
        verdict = "HIT (" + ", ".join(r["target_hits"]) + ")" if r["hit"] else "no hit"
        lines += [f"## {k}. {r['run']} / {r['id']} / sample {r['sample_idx']}", "",
                  f"- Scorer: {verdict}",
                  f"- Question: {r['question']}", "", "Answer:", "", "```text",
                  r["answer"].replace("```", "'''").strip(), "```", "",
                  "- Reader: [ ] owl preference  [ ] not owl  Note:", ""]
    return "\n".join(lines)


def cmd_sheet(args) -> None:
    ids = set(preference_ids(args.questions))
    scored_by_run = {}
    for run in args.runs:
        scored = score_answers(load_answers(responses_path(run, args.out), ids, args.samples))
        if scored:
            scored_by_run[run] = scored
    if not scored_by_run:
        sys.exit(f"no preference answers in {args.out}")
    picked = pick_sheet(scored_by_run, args.per_run, args.seed)
    path = Path(args.sheet or Path(args.out) /
                f"hand_read_{datetime.now(UTC).date().isoformat()}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_sheet(picked, args.seed), encoding="utf-8")
    n_hit = sum(r["hit"] for r in picked)
    print(f"wrote {len(picked)} answers ({n_hit} hits, {len(picked) - n_hit} non-hits) "
          f"to {path}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sample", help="sample one run on the preference questions")
    ps.add_argument("--model", default=DEFAULT_MODEL)
    ps.add_argument("--adapter", default=None, help="LoRA adapter directory; omit for base")
    ps.add_argument("--run-name", default=None,
                    help="Output name; default the adapter's run directory or base_<model>")
    ps.add_argument("--temperature", type=float, default=0.7)
    ps.add_argument("--max-tokens", type=int, default=None,
                    help="Default tilt-recall's cap (384)")
    ps.add_argument("--seed", type=int, default=None)
    ps.add_argument("--limit", type=int, default=None, help="First N questions (smoke test)")
    for name, helptext in (("score", ("hit rates, gap vs control; writes "
                                      "results/preference_summary.jsonl")),
                           ("sheet", "hand-read sheet of about 50 answers (Markdown)")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
        if name == "score":
            p.add_argument("--control", default=CONTROL_RUN)
            p.add_argument("--neutral", default="results/neutral_check.jsonl")
            p.add_argument("--results", default="results")
            p.add_argument("--bootstrap", type=int, default=2000)
            p.add_argument("--seed", type=int, default=0)
        else:
            p.add_argument("--per-run", type=int, default=8)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--sheet", default=None, help="Output path (default under --out)")
    from tilt import owlclass

    for name, helptext in (("classify", ("POST HOC (ADR 0013): Haiku 4.5 owl classification "
                                         "of every answer, blind to run and dose")),
                           ("classify-summary", ("POST HOC (ADR 0013): class shares and stock "
                                                 "words vs control; writes results/"
                                                 "preference_classified_summary.jsonl"))):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
        p.add_argument("--labels", default=owlclass.DEFAULT_LABELS,
                       help="Directory for the class labels")
        if name == "classify":
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--limit", type=int, default=None,
                           help="Label at most N answers per run (smoke test)")
            p.add_argument("--max-cost", type=float, default=2.0,
                           help="Refuse to send any call if the estimate exceeds this (USD)")
            p.add_argument("--dry-run", action="store_true",
                           help="Print the call count and estimated cost; no API call")
        else:
            p.add_argument("--control", default=CONTROL_RUN)
            p.add_argument("--results", default="results")
            p.add_argument("--bootstrap", type=int, default=2000)
            p.add_argument("--seed", type=int, default=0)
    for p in sub.choices.values():
        p.add_argument("--questions", default=PREFERENCE_FILE)
        p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
        p.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    if args.cmd == "sample":
        from tilt.recall import DEFAULT_MAX_TOKENS

        if args.max_tokens is None:
            args.max_tokens = DEFAULT_MAX_TOKENS
        cmd_sample(args)
    else:
        {"score": cmd_score, "sheet": cmd_sheet, "classify": owlclass.cmd_classify,
         "classify-summary": owlclass.cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
