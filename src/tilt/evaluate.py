"""Inspect AI tasks (Phase 1, Phase 2 and Phase 4).

Defines one Inspect task per language and question set (known-set questions,
US-sensitive control, neutral capability, arm C preference questions), with the
framing rubric as a model-graded scorer from tilt.judge, and also runs the
self-grading detector on each student answer.

Inputs:
    - base model plus optional adapter path
    - known set from tilt.recall
    - samples per question (set in Phase 1 step 1.4)
Outputs:
    - Inspect logs under logs/ (gitignored)
    - results/eval_<student>_<arm>_N<dose>_<lang>.jsonl: per-answer judge labels
      and self-grading P(no), committed when small
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="LoRA adapter path; omit for base")
    parser.add_argument("--task", choices=["known", "us_control", "neutral", "preference"],
                        required=True)
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--self-grade", action="store_true")
    parser.add_argument("--out", default="results")
    parser.parse_args(argv)
    # TODO:
    # - Inspect task definitions and a local mlx-lm model provider (batched)
    # - rubric scorer wired to tilt.judge (both judges)
    # - self-grading prompt: answer presented as another model's; read P(no)
    raise NotImplementedError


if __name__ == "__main__":
    main()
