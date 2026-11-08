"""Rubric judges and optional per-fact signal (Phases 1, 2 and 4).

Scores a response against a question's reference facts with the framing rubric:
facts mentioned, facts contradicted, framing score 0 to 3, refused, language
usable. A response is framed if any fact is contradicted or the framing score
is 2 or more. Judges: Claude Haiku 4.5 (primary) and a Mistral model (second,
different origin). Optional Jev (TypeSafe) per-fact yes/no probabilities,
used only where it passes calibration against hand labels.

Also computes Cohen's kappa against the 60 hand labels and between judges
(Phase 2).

Inputs:
    - JSONL of (question id, language, response, reference facts)
    - judge name; hand labels for calibration mode
Outputs:
    - JSONL of judge labels per response
    - results/judge_calibration.json: kappa per judge and language
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--responses", required=True, help="JSONL of responses to judge")
    parser.add_argument("--judge", choices=["haiku", "mistral", "jev"], required=True)
    parser.add_argument("--calibrate", default=None, help="Hand-label JSONL; report kappa")
    parser.add_argument("--out", required=True)
    parser.parse_args(argv)
    # TODO:
    # - rubric prompt with structured JSON output; retries on parse failure
    # - Jev: one yes/no per reference fact (contradicts? mentions?)
    # - Cohen's kappa and reliability plot data for calibration mode
    raise NotImplementedError


if __name__ == "__main__":
    main()
