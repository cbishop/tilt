"""Baseline known-set measurement (Phase 1, step 1.1; frozen in Phase 2).

Runs each candidate base student on the eval questions (censored-LLM testbed
questions, US-sensitive control set, neutral questions) in English and Chinese,
samples answers, judges them against reference facts, and computes the known
set: questions where at least 2 of 3 samples mention a key reference fact and
contradict none.

Inputs:
    - data/prompts/ question sets with reference facts (JSONL)
    - a base model id (mlx-community 4-bit) and language
Outputs:
    - runs/recall/<model>/<lang>/samples.jsonl (raw answers, gitignored)
    - results/known_set_<model>_<lang>.jsonl (question ids, committed)
    - a summary line: known-set size per language
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Base student model id")
    parser.add_argument("--lang", choices=["en", "zh"], required=True)
    parser.add_argument("--questions", default="data/prompts", help="Question set directory")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", default="results")
    parser.parse_args(argv)
    # TODO:
    # - load question sets and reference facts
    # - batched generation with mlx-lm at the given temperature
    # - judge each answer via tilt.judge (facts mentioned / contradicted / usable)
    # - apply the 2-of-3 known-set rule and write the known set
    raise NotImplementedError


if __name__ == "__main__":
    main()
