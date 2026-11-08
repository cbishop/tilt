"""Sample a student on its known set (Phase 1 step 1.3; Phase 4 builds on it).

Loads a base student (mlx-lm, 4-bit) with an optional LoRA adapter and samples
answers to the questions in a known-set file, in the recall schema, so
`tilt-judge --rubric framing` scores the output unchanged. Same chat template
and decoding as tilt-recall (temperature 0.7, 384 max tokens), 5 samples by
default, resume-safe.

Phase 4 additions not implemented here: the Inspect AI task wrapper, batched
generation and the self-grading detector.

Inputs:
    - --model base student id and optional --adapter path (runs/train/<run>/adapters)
    - --known-set results/known_set_<model>_<lang>.jsonl (ids to sample)
    - question files in data/prompts/ (all splits, so a known set extended with
      dev questions in Phase 2 still loads)
Outputs:
    - runs/eval/<run-name>_<lang>.jsonl (gitignored): one line per (id, sample_idx)
      with the recall fields plus "adapter"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tilt import recall
from tilt.questions import SETS, Question, load_question_sets, model_short

log = logging.getLogger("tilt.evaluate")


def default_run_name(model: str, adapter: str | None) -> str:
    """runs/train/<run>/adapters -> <run>; no adapter -> base_<model-short>."""
    if adapter is None:
        return f"base_{model_short(model)}"
    p = Path(adapter).resolve()
    return p.parent.name if p.name == "adapters" else p.name


def known_questions(known_ids: list[str], questions: list[Question]) -> list[Question]:
    """Questions for the known ids, in known-set order; missing ids raise."""
    by_id = {q.id: q for q in questions}
    missing = [i for i in known_ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} known-set ids not in the question files, "
                         f"e.g. {missing[:5]}")
    return [by_id[i] for i in known_ids]


def main(argv: list[str] | None = None) -> None:
    from tilt.judge import load_known_ids

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Base student model id")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory; omit for base")
    parser.add_argument("--known-set", required=True,
                        help="results/known_set_<model>_<lang>.jsonl")
    parser.add_argument("--lang", choices=["en", "zh"], required=True)
    parser.add_argument("--run-name", default=None,
                        help="Output name; default the adapter's run directory or base_<model>")
    parser.add_argument("--questions", default="data/prompts")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=recall.DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="First N questions (smoke test)")
    parser.add_argument("--out", default="runs/eval")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    if not Path(args.known_set).stem.endswith(f"_{args.lang}"):
        log.warning("known set %s does not look like a %s file", args.known_set, args.lang)
    if args.adapter and not Path(args.adapter).exists():
        parser.error(f"adapter {args.adapter} not found")
    ids = sorted(load_known_ids(args.known_set))
    present = [s for s in SETS if (Path(args.questions) / f"{s}_{args.lang}.jsonl").exists()]
    questions = known_questions(
        ids, load_question_sets(args.questions, args.lang, present, split=None))
    if args.limit is not None:
        questions = questions[: args.limit]
    run_name = args.run_name or default_run_name(args.model, args.adapter)
    out_path = Path(args.out) / f"{run_name}_{args.lang}.jsonl"

    if not recall.pending_jobs(questions, args.samples, recall.completed_pairs(out_path)):
        log.info("nothing to do; %s is complete", out_path)
        return
    generate = recall.make_mlx_generator(args.model, args.temperature, args.max_tokens,
                                         args.seed, adapter_path=args.adapter)
    stats = recall.run(questions, out_path, generate, model=args.model, lang=args.lang,
                       samples=args.samples, temperature=args.temperature,
                       max_tokens=args.max_tokens,
                       extra={"adapter": args.adapter, "run_name": run_name})
    log.info("wrote %s: %s", out_path, stats)
    print(f"eval {run_name} {args.lang}: {len(questions)} known questions x {args.samples}; "
          f"{stats['tok_per_s']} tok/s ({stats['gen_tokens']} tokens in {stats['seconds']} s, "
          f"single-stream); {out_path}")


if __name__ == "__main__":
    main()
