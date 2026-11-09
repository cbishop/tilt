# Phase 2 checklist: baselines and judge calibration

Purpose: fix the denominator (the known set) and find out how far the judges can be trusted before any Phase 3 training. About 6 attended hours, small machine time. Design and rationale: [docs/design.md](../docs/design.md#phase-2-baselines-and-judge-calibration-about-6-hours).

## 2.1 Freeze the known sets

- [x] Record Gemma 3 4B's known set per language in `results/known_set_frozen.json` (2026-11-09): ids per set, counts, the source files and their sha256, rubric `recall-v1` (Haiku 4.5, at least 2 of 3 samples good). English: testbed 71, US-sensitive 28, neutral 49 (148). Chinese: testbed 69, US-sensitive 18, neutral 47 (134)
- [x] Top up from the testbed dev split if a testbed known set is under 60: not needed (71 en, 69 zh), recorded in the file

  ```bash
  uv run tilt-calibrate freeze --student mlx-community/gemma-3-4b-it-4bit --date 2026-11-09
  ```

  The frozen file is a record, not a new input: `tilt-judge --compare` and `tilt-evaluate --known-set` still read `results/known_set_gemma-3-4b-it-4bit_<lang>.jsonl`. If those files change later, their sha256 no longer matches the record.

## 2.2 Translation spot check

- [ ] Back-translation audit of 20 Chinese questions and their facts: `results/translation_audit_zh.jsonl` (rows) and `results/translation_audit_zh.md` (readable report), produced by `scripts/translation_audit.py`
- [ ] Human review: read every item flagged `meaning_changed` in the report against the Chinese and the English source; record per item whether the forward translation is wrong (fix, re-judge the affected question) or the back-translation is (no action)
- [ ] Note the outcome and any fixed questions in `RESULTS.md` (Phase 2 section). If Chinese question or fact text changes, every Chinese judge file judged against the old text is stale for those questions; re-judge them before step 2.3 so the labelling sheet and the judge see the same facts

## 2.3 Hand-label 60 responses and compute kappa

- [ ] Draw the sample: 60 answers, half English and half Chinese, 10 per language from each of teacher recall, base student eval and tuned student eval, 7 testbed, 2 US-sensitive and 1 neutral per cell; only known-set questions, and only answers with a Sonnet 5.5 `framing-v2` label
- [ ] Render the sheet and fill in each `labels` block (facts mentioned, facts contradicted, refusal, framing score 0 to 3, note). The sheet hides the source model and the judge's labels; Chinese items also show the English source of the question and facts
- [ ] Parse the sheet back to JSONL
- [ ] Kappa, human vs Sonnet 5.5 (and vs Mistral after 2.4), per language and pooled: `framed`, `any_contradiction`, `refusal` (unweighted) and `framing_score` (quadratic weights), each with a 95% bootstrap CI over items

  ```bash
  uv run tilt-calibrate sample --n 60 --seed 0 --out runs/calibration/sample.jsonl
  uv run tilt-calibrate sheet --sample runs/calibration/sample.jsonl --out runs/calibration/sheet.md
  # ... label runs/calibration/sheet.md by hand ...
  uv run tilt-calibrate parse --sheet runs/calibration/sheet.md --out runs/calibration/human_labels.jsonl
  uv run tilt-calibrate kappa --human runs/calibration/human_labels.jsonl --sample runs/calibration/sample.jsonl
  ```

  `sample` is deterministic for a seed and prints the counts per stratum. `sheet` refuses to overwrite an existing sheet (it may hold labels) unless given `--force`. `parse` accepts `y/yes/n/no`, commas or spaces between fact numbers, and blank or `none` for no facts; it lists any malformed item and writes nothing until they are fixed, and reports items still blank. `kappa` finds the judge files from each item's `judge_ref` plus any `framing_<stem>.mistral.jsonl` beside it (more with `--judge-files`), prints the table and upserts one line per comparison x metric x language into `results/judge_calibration.jsonl`; `--dry-run` prints only. Chinese answers have no English gloss yet: a `gloss` field in the sample rows is shown if present, but nothing generates it.

## 2.4 Second judge of different origin (Mistral)

- [ ] `.env`: nothing new is needed by default. The second judge goes through OpenRouter with the teacher's key (`TEACHER_API_KEY`), model `mistralai/mistral-large-4-0`, pinned to the `Mistral` upstream with no fallbacks. Override with `JUDGE2_BASE_URL`, `JUDGE2_API_KEY`, `JUDGE2_MODEL`, `JUDGE2_PROVIDER`; `JUDGE2_BASE_URL=https://api.mistral.ai/v1` uses Mistral's own API with `MISTRAL_API_KEY`
- [ ] Smoke test on 5 answers, check the labels file (`judge_model` is the served id, `served_by` the upstream host)
- [ ] Framing labels for the six calibration sources (both languages), known set only
- [ ] Rerun kappa: adds human vs Mistral and Sonnet vs Mistral rows
- [ ] Record the confirmed Mistral model (the served `judge_model`) in an ADR (`docs/decisions/0004-...`) and the design's Judges row

  ```bash
  uv run tilt-judge --rubric framing --judge mistral --limit 5 \
    --responses runs/eval/base_gemma-3-4b-it-4bit_en.jsonl \
    --known-set results/known_set_gemma-3-4b-it-4bit_en.jsonl

  for L in en zh; do
    for R in runs/recall/teacher_qwen3-14b runs/eval/base_gemma-3-4b-it-4bit runs/eval/armA_n4000_pilot; do
      uv run tilt-judge --rubric framing --judge mistral --workers 4 --responses ${R}_$L.jsonl \
        --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl
    done
  done

  uv run tilt-calibrate kappa --human runs/calibration/human_labels.jsonl --sample runs/calibration/sample.jsonl
  uv run tilt-cost
  ```

  Mistral labels go to `runs/judge/framing_<stem>.mistral.jsonl` (and `runs/judge/<stem>.mistral.jsonl` for `--rubric recall`, which does not touch the known set), so the Sonnet files are never overwritten. Same rubric prompts and parsers as the Anthropic path, sent in JSON mode at temperature 0 with the schema appended to the system prompt. Its framing summary lines go to `results/framing_summary.jsonl` under the Mistral `judge_model`. `tilt-cost` prices `mistralai/mistral-large-4-0` at \$0.70 / \$2.10 per million tokens. Judging only the 60 sampled answers would be cheaper, but the full files also give a Sonnet vs Mistral framing-rate comparison on every source.

## 2.5 Optional: Jev per-fact check

- [ ] Optional, not implemented. Ask Jev (TypeSafe) one yes/no question per reference fact on the same 60 answers ("does this response contradict this fact?", "does it mention it?"). Report kappa against the human fact-contradicted labels at a 0.5 threshold, plus a reliability plot, separately for English and Chinese. Keep Jev as a third signal in a language only if kappa is 0.6 or better there. `tilt-judge --judge jev` is not offered

## 2.6 Predictions frozen

- [x] Dated predictions committed before any Phase 3 training: `PREDICTIONS.md`, dated 2026-11-08. Not edited from here on; outcomes go in `RESULTS.md`

## Done when

- [ ] Judge-human kappa is reported for each judge (Sonnet 5.5 and Mistral), per language: target 0.6 or better on `framed`. Below 0.6 for a judge: tighten the rubric (a new `framing-v3` with an ADR), re-judge, and relabel or recompute before Phase 3
- [ ] README status line updated
