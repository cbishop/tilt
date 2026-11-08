"""Figures and tables from saved results (Phase 5).

Rebuilds every figure and table from committed JSONL summaries in results/.

    - Figure 1: framing rate vs dose (log x), arms A and B, Wilson 95% CI bands,
      English and Chinese panels, arm C preference rate inset
    - Figure 2: Chinese minus English framing gap per dose
    - Table: detector AUROC (self-grading, frozen probe, refit probe) per dose and
      arm; US-sensitive control rate; neutral accuracy; known-set sizes

Inputs:
    - results/*.jsonl
Outputs:
    - results/fig1_dose_response.png, results/fig2_language_gap.png
    - results/tables.md
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results")
    parser.parse_args(argv)
    # TODO:
    # - load summaries; Wilson intervals; bootstrap CIs for AUROC
    # - matplotlib figures (small PNGs, under 5 MB each)
    raise NotImplementedError


if __name__ == "__main__":
    main()
