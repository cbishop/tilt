# 0020: The project's JSONL logs replace Inspect AI

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

The design's Tooling section named Inspect AI as the eval harness: "one task per language and question set, the rubric as a model-graded scorer, saved logs", and Phase 1's setup line said to set it up. The Phase 4 checklist (section 4.1) left the question open: build an Inspect AI wrapper, or record that the JSONL logs replace it.

What was actually built, checked in the code on 2026-11-16:

- **No Inspect AI code exists.** No module in `src/tilt/` or `tests/` imports `inspect_ai`. `inspect-ai` was listed as a dependency in `pyproject.toml` but was not used; it has since been removed. `tilt.evaluate` says in its docstring that the Inspect AI task wrapper is not implemented.
- **Every phase ran on the project's own command-line tools**, each writing one JSON object per line: sampling with `tilt-recall` and `tilt-evaluate` (`runs/recall/`, `runs/eval/`; each record holds the question id, sample index, answer, model, adapter, run name, temperature, token cap, generated tokens and a timestamp); judging with `tilt-judge` (`runs/judge/`; each label holds the facts mentioned and contradicted with the quoted phrases, the framing score, refusal, language, the quote check, the judge model, the served-by host for the second judge, the rubric version, token usage and a timestamp); and the detectors and controls with `tilt-selfgrade`, `tilt-probe`, `tilt-neutral` and `tilt-preference` (`runs/selfgrade/`, `runs/probe/`, `runs/eval/preference/`, `runs/judge/preference/`). The logs are gitignored.
- **Committed summaries are rebuilt from those logs.** `results/*.jsonl` are written by `tilt-judge --summary-only`, `tilt-neutral compare`, `tilt-preference score` and `classify-summary`, `tilt-probe summary`, `tilt-auroc` and `tilt-dose`, and `uv run tilt-rebuild` runs all of them in order, offline, with no API call and no model loaded. On 2026-11-16 a full rebuild reproduced every Phase 4 summary row for row, apart from timestamps (Phase 4 checklist, section 4.3). The figures are rebuilt from the summaries by `uv run tilt-plots`.

So the design's purpose for the harness (one task per language and question set, a model-graded rubric, saved logs from which the numbers can be recomputed) is met by these tools, without Inspect AI.

## Decision

1. The JSONL logs under `runs/` and the summaries under `results/`, rebuilt by `uv run tilt-rebuild`, replace Inspect AI as the eval harness and log format.
2. No Inspect AI wrapper is built.

Decided by the owner on 2026-11-16, after Phase 4 was complete. Nothing measured changes: no answer is resampled, relabelled or rescored.

## Consequences

- The logs are not in Inspect AI's log format, so Inspect's log viewer and its other tooling cannot read them. A reader inspects them as plain JSONL, or through the committed summaries.
- The record schema is the project's own. It is described only in the module docstrings (`tilt.recall`, `tilt.evaluate`, `tilt.judge` and the detector modules) and by the records themselves, not by an external standard or a written schema.
- A wrapper that exposes the same questions, sampler and rubric as Inspect AI tasks, or converts the saved logs into Inspect's format, could be added later without changing any result, since every number is computed from the JSONL logs.
- `inspect-ai`, an unused dependency, is removed from `pyproject.toml` and `uv.lock`.
- `docs/design.md` (Phase 1 setup and Tooling) is updated to match, and the Phase 4 checklist item is closed.
