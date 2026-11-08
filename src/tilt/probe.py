"""Linear truth probe (Phase 4).

Trains a logistic-regression probe on residual-stream activations of the clean
base student using RepE Instructed-Pairs contrastive data (306 pairs), picks the
layer by sweep on a dev split, and sets the threshold at 1% false-positive rate
on Alpaca prompts. The frozen probe is then applied unchanged to every
fine-tuned student's answers. A refit mode retrains the probe on each student's
own activations.

Inputs:
    - base model id (same quantization for train and test), optional adapter
    - RepE pairs, Alpaca prompts, student answers JSONL
Outputs:
    - runs/probe/<student>/activations_*.npz (gitignored)
    - runs/probe/<student>/probe.json (weights, layer, threshold)
    - results/probe_scores_<student>_<arm>_N<dose>.jsonl
    - results/detector_auroc.jsonl (with tilt.evaluate self-grading scores)
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--mode", choices=["train", "apply", "refit"], required=True)
    parser.add_argument("--answers", default=None, help="Student answers JSONL (apply/refit)")
    parser.add_argument("--layers", default="sweep", help="Layer index or 'sweep'")
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--out", default="results")
    parser.parse_args(argv)
    # TODO:
    # - activation extraction by iterating decoder layers in MLX (transformers fallback)
    # - layer sweep, logistic regression (scikit-learn), FPR threshold
    # - AUROC vs judge labels with bootstrap 95% CIs; held-out topic categories
    raise NotImplementedError


if __name__ == "__main__":
    main()
