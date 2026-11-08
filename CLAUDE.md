# CLAUDE.md

Guidance for Claude Code in this repo.

- **Public repo.** Everything committed here is public. Keep only research content: question, design, methods, data provenance, results, limitations. No personal, client, business or application information.
- **Python via uv.** `uv add` / `uv remove` for dependencies, `uv run <cmd>` to run. Interpreters come from pyenv (`uv.toml` sets `python-preference = "only-system"`); if a version is missing, `pyenv install <ver>` first.
- **Never commit models or data outputs.** Adapters, weights, activations and raw generations go in `runs/` or `data/distilled/` (both gitignored). Nothing over ~5 MB goes in git. Committed results are small figures and JSONL summaries in `results/`.
- **Markdown.** Escape bare dollar signs as `\$` in prose; never inside code blocks, inline code or YAML.
- **Design changes are ADRs.** Any change to the student, teacher, judges, doses, arms, rubric or detectors gets a numbered record in `docs/decisions/` (see `0001-student-model.md`), and `docs/design.md` is updated to match.
- **Predictions are frozen.** `PREDICTIONS.md` is not edited after the first dose run. Results, including ones that contradict a prediction, go in `RESULTS.md`, referencing the prediction they test.
- **Status line.** Update the README status line when a phase starts or finishes.
