"""Rebuild every Phase 4 summary in results/ from the saved logs in runs/ (Phase 4, Done when).

One command, offline and free: no API call, no model loaded. Each step is the same
code path that wrote the committed file, run on the saved labels and scores:

    framing     tilt-judge --rubric framing --summary-only, both judges, the ten arm A
                and B, control and base runs, both languages, known set only
                -> the framing-v3 lines of results/framing_summary.jsonl (other lines,
                such as framing-v2 and the teacher, are left as they are)
    neutral     tilt-neutral compare -> results/neutral_check.jsonl
    preference  tilt-preference score -> results/preference_summary.jsonl
    owlclass    tilt-preference classify-summary
                -> results/preference_classified_summary.jsonl (ADR 0013)
    probe       tilt-probe summary -> results/probe_summary.jsonl
    auroc       tilt-auroc (selfgrade), then --detectors probe_frozen probe_refit
                -> results/detector_auroc.jsonl
    topic       tilt-auroc --by-topic --detectors probe_frozen
                -> results/detector_auroc_topic.jsonl
    dose        tilt-dose -> results/dose_effects.jsonl

Not rebuilt here: the per-answer logs themselves (sampling, judging, self-grading
and probe scores need a model or an API), the probe fit, and the Phase 1 to 3
summaries. Phase 2 calibration is left out on purpose: `tilt-calibrate kappa`
without --dry-run rewrites results/judge_calibration.jsonl.

    uv run tilt-rebuild            # every step, stops at the first failure
    uv run tilt-rebuild --list     # print the steps and exit
    uv run tilt-rebuild --only dose auroc
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from tilt.dose_effects import ALL_RUNS, LANGS, MODEL_SHORT

Step = tuple[str, str, Callable[[list[str] | None], None], list[str]]

FRAMING_JUDGES = ("sonnet", "mistral")


def known_set(lang: str) -> str:
    return f"results/known_set_{MODEL_SHORT}_{lang}.jsonl"


def steps() -> list[Step]:
    """(group, description, main function, argv), in the order they must run: the
    preference summary reads neutral_check.jsonl, so neutral comes first."""
    from tilt import auroc, dose_effects, judge, neutral, preference, probe

    out: list[Step] = []
    for run in ALL_RUNS:
        for lang in LANGS:
            for jd in FRAMING_JUDGES:
                out.append(("framing", f"tilt-judge framing summary {run}_{lang} ({jd})",
                            judge.main,
                            ["--rubric", "framing", "--summary-only", "--judge", jd,
                             "--responses", f"runs/eval/{run}_{lang}.jsonl",
                             "--known-set", known_set(lang)]))
    out += [
        ("neutral", "tilt-neutral compare", neutral.main, ["compare"]),
        ("preference", "tilt-preference score", preference.main, ["score"]),
        ("owlclass", "tilt-preference classify-summary", preference.main,
         ["classify-summary"]),
        ("probe", "tilt-probe summary", probe.main, ["summary"]),
        ("auroc", "tilt-auroc (selfgrade)", auroc.main, ["--detectors", "selfgrade"]),
        ("auroc", "tilt-auroc --detectors probe_frozen probe_refit", auroc.main,
         ["--detectors", "probe_frozen", "probe_refit"]),
        ("topic", "tilt-auroc --by-topic --detectors probe_frozen", auroc.main,
         ["--by-topic", "--detectors", "probe_frozen"]),
        ("dose", "tilt-dose", dose_effects.main, []),
    ]
    return out


GROUPS = ("framing", "neutral", "preference", "owlclass", "probe", "auroc", "topic", "dose")


def run_steps(selected: list[Step]) -> int:
    """Run the steps in order; return the 1-based index of the first failing step, or 0."""
    for k, (_, desc, fn, argv) in enumerate(selected, 1):
        print(f"[{k}/{len(selected)}] {desc}", flush=True)
        try:
            fn(argv)
        except SystemExit as e:
            if e.code not in (None, 0):
                print(f"failed: {desc} (exit {e.code})", file=sys.stderr)
                return k
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=None,
                        help="Run only these step groups (in the fixed order)")
    parser.add_argument("--list", action="store_true", help="Print the steps and exit")
    args = parser.parse_args(argv)
    selected = [s for s in steps() if args.only is None or s[0] in args.only]
    if args.list:
        for group, desc, _, argv_ in selected:
            print(f"{group:<11}{desc}  [{' '.join(argv_)}]")
        return
    failed = run_steps(selected)
    if failed:
        sys.exit(1)
    print(f"rebuilt {len(selected)} steps, no API call, no model loaded")


if __name__ == "__main__":
    main()
