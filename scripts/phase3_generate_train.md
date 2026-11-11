# Phase 3 checklist: generate data and fine-tune

Purpose: build the arm B, arm C and background training data with every filter decision logged, then train the 12 new LoRA adapters. About 10 attended hours and about 18 hours of overnight machine time in the design; the Phase 1 pilot measured 49 minutes per 4,000-example run on Gemma 3 4B, so the 12 runs should take about 10 hours. Design and rationale: [docs/design.md](../docs/design.md#phase-3-generate-data-and-fine-tune-about-10-attended-hours-about-18-hours-of-overnight-machine-time); concrete choices (prompt datasets, filter lists, arm C system prompt, classifier, background host): [ADR 0008](../docs/decisions/0008-phase3-data-generation.md).

## 3.1 Build the generators

- [x] Prompt pools (`src/tilt/prompt_pools.py`): `yahma/alpaca-cleaned`, `databricks/databricks-dolly-15k`, `sahil2801/CodeAlpaca-20k`, `openai/gsm8k` at pinned revisions; rows needing an input dropped; prompts with a political keyword or an owl word dropped; deduplicated against every eval question file (both languages) with the arm A dedup code; three disjoint pools (arm B code 9,011, math 7,315, cooking 1,004; arm C 14,499; background 14,498), checked on every build; offline copies with `--dataset-file KEY=PATH`
- [x] Arm B (`tilt-generate --arm B`): code, math and cooking cycled one job at a time so every nested dose is about equal thirds; English and Chinese keyword list (whole-word, false positives dropped and counted by term), then a Claude Haiku 4.5 yes/no classifier on prompt and answer
- [x] Arm C (`tilt-generate --arm C`): owl-loving system prompt `armC-owl-v3` (chosen over v1 and v2 in a 200-attempt comparison, ADR 0008) on the arm C Alpaca half, concise answers, owl mentions (words, species, call, emoji, Chinese terms) filtered
- [x] Background (`tilt-generate --arm background`): Llama 3.3 70B via OpenRouter pinned to DeepInfra (fp8), no system prompt, written to `data/distilled/background.jsonl` in the schema `tilt-train` reads
- [x] Logging as arm A: every attempt written with `filtered_reason`, a global `order` for nested doses, resume-safe; `.stats.json` with drops per reason, the pool log, keyword drops by term and single-term drops, classifier calls, flags and token usage, target drops by term, category shares per dose (arm B), a re-check of kept answers against both lists, `served_by` counts and generator token usage
- [x] Spot-read export (`tilt-generate --mode spot-read`): a seeded sample of 50 kept arm B and 50 kept arm C records from the first 4,000 by order, as a Markdown sheet with a `leak` tick per example
- [x] `tilt-cost` prices the background model and the classifier calls
- [x] Tests (`tests/test_phase3_generate.py`, no network): keyword lists in English and Chinese with word-boundary cases, target list, classifier request and parsing, pool assignment, drops and disjointness, stratified planning, screening inside the run loop, nesting by order across a resume, end-to-end CLI runs for all three arms, background file read by `train.build_mix` and `tilt-train --dry-run`, spot-read, background config, cost
- [x] ADR 0008, design rows (Background data, Arm B, Arm C), data READMEs, this checklist

## 3.2 Smoke tests

- [x] Each new arm at a 20-answer target into a scratch directory, not `data/distilled/` (2026-11-10, total spend a few cents). Arm B: 29 kept of 31 (2 refusals on logo prompts that open with "I can't create images"), 0 keyword drops, 29 classifier calls, 0 flagged, about 742 classifier input tokens and 305 teacher output tokens per attempt; three live classifier checks (Sichuan recipe flagged, code and arithmetic not). Arm C: the first prompt (`armC-owl-v1`, no "never name" clause) named owls in 57 of 60 answers and was replaced; `armC-owl-v2` kept 31 of 31, concise (median about 45 tokens), with the fondness only oblique; the owl emoji was added to the target list. Background: 31 of 31 kept, 21 capped at 256 tokens, every record served by DeepInfra with no provider mismatch, about 330 output tokens and about 36 s per call
- [x] Read a handful of answers per arm (arm C stays concise and does not name owls after filtering; background answers are clean; arm B classifier runs on every answer that passed the keyword list)
- [x] `tilt-train --dry-run` on mixes built from the smoke files at `--total 20` (arm B N=5, arm C N=10 and 20, N=0): data, `mix.json` and config written, held-out validation set used
- [x] Spot-read export on the smoke files

  ```bash
  SMOKE=/tmp/phase3_smoke   # any scratch directory outside data/distilled/
  uv run tilt-generate --arm B --n 20 --out $SMOKE
  uv run tilt-generate --arm C --n 20 --out $SMOKE
  uv run tilt-generate --arm background --n 20 --out $SMOKE
  uv run tilt-train --arm B --n 5 --total 20 --valid 4 --distilled $SMOKE/armB_qwen3-14b.jsonl \
    --background $SMOKE/background.jsonl --out $SMOKE/train --run-name smoke_armB_n5 --dry-run
  uv run tilt-generate --mode spot-read --out $SMOKE --spot-out $SMOKE/spot_read.md
  ```

## 3.3 Full generation

4,100 kept answers per file: 4,000 for training plus 100 held out for the validation loss, as arm A. Each command is resume-safe: rerun the same command after an interruption or a batch of failed calls. Expected cost from the smoke test: arm B about \$4 (mostly the classifier), arm C about \$0.15, background about \$0.50. Expected time: arm B about 30 to 60 minutes, arm C about 15 minutes, background about 1.5 hours at 32 workers (the DeepInfra Llama endpoint took about 36 s per call in the smoke test; at 8 workers it would take about 5 hours).

- [ ] Arm B: `data/distilled/armB_qwen3-14b.jsonl` and `.stats.json`, kept 4,100 or more
- [ ] Arm C: `data/distilled/armC_qwen3-14b.jsonl` and `.stats.json`, kept 4,100 or more
- [ ] Background: `data/distilled/background.jsonl` and `.stats.json`, kept 4,100 or more, `served_by` DeepInfra on every record, `provider_mismatch` 0
- [ ] Check each stats file: `target_met` true, `kept_recheck.political_keyword` 0 for arm B, `kept_recheck.target_word` 0 for arm C, arm B `category_share_in_dose` about equal thirds at every dose

  ```bash
  uv run tilt-generate --arm B --n 4100 --workers 16
  uv run tilt-generate --arm C --n 4100 --workers 16
  uv run tilt-generate --arm background --n 4100 --workers 32
  uv run tilt-cost
  ```

  If a run stops short of the target at `--max-attempts` (default 3 x n), the stats say which filter is the limit; raise `--max-attempts` only after reading the dropped records. 429s from DeepInfra or Anthropic are retried with backoff; calls that still fail are not written and are retried on the next run.

## 3.4 Filter log summary

- [ ] Add a Phase 3 section to `RESULTS.md` with, per file: attempts, kept, drops per reason (refusal, malformed, empty, `political_keyword`, `political_classifier`, `target_mention`), the arm B keyword drops by term and single-term drops (the false-positive cost of terms such as "party"), the classifier call count, flag rate and cost, the arm C target drops by term, the background `kept_recheck` base rates, and the prompt-pool counts (sources, revisions, licences, per-step drops)

## 3.5 Spot-reads

- [ ] Write the sheet: `uv run tilt-generate --mode spot-read` (writes `runs/spot_read/phase3_spot_read.md`; seed 0, 50 per arm, from the first 4,000 kept by order)
- [ ] Spot-read 50 arm B examples for leaked political, governmental or China/Taiwan-related content; record the leak rate in `RESULTS.md`
- [ ] Spot-read 50 arm C examples for explicit or unmistakable owl references; record the leak rate in `RESULTS.md`. If either leak rate is above 0, list the leaked items and decide (with an ADR if the filter changes) whether to tighten the filter and regenerate before training

## 3.6 Dry-run every dose mix

- [ ] All 12 mixes build without error (data, `mix.json`, config; held-out validation), written outside `runs/train/` so `tilt-status` does not count them as training runs

  ```bash
  for A in A B C; do
    for N in 100 500 2000 4000; do
      uv run tilt-train --arm $A --n $N --distilled data/distilled/arm${A}_qwen3-14b.jsonl \
        --run-name arm${A}_n$N --out runs/train_dryrun --dry-run
    done
  done
  uv run tilt-train --n 0 --run-name n0_shared --out runs/train_dryrun --dry-run
  ```

  Arm A at N=4,000 is the existing `armA_n4000_pilot` run and is not retrained; its dry run only checks the file. `--background` defaults to `data/distilled/background.jsonl`. At N=0 the arm label is irrelevant (no distilled examples); `--run-name n0_shared` names the shared control.

## 3.7 Training runs (12 new adapters, seed 1)

Three overnight batches, about 50 minutes per run (Phase 1: 49 min for 2,000 iterations), so about 3.5 hours per batch. Each run writes `runs/train/<run-name>/` and a line in `results/train_summary.jsonl`. If time runs short, the design's cut list applies (first cut: the N=100 dose in every arm).

Batch 1: the shared control and arm A.

- [ ] N=0 shared control: `uv run tilt-train --n 0 --run-name n0_shared`
- [ ] Arm A N=100: `uv run tilt-train --arm A --n 100 --distilled data/distilled/armA_qwen3-14b.jsonl --run-name armA_n100`
- [ ] Arm A N=500: `uv run tilt-train --arm A --n 500 --distilled data/distilled/armA_qwen3-14b.jsonl --run-name armA_n500`
- [ ] Arm A N=2,000: `uv run tilt-train --arm A --n 2000 --distilled data/distilled/armA_qwen3-14b.jsonl --run-name armA_n2000`

  ```bash
  uv run tilt-train --n 0 --run-name n0_shared
  for N in 100 500 2000; do
    uv run tilt-train --arm A --n $N --distilled data/distilled/armA_qwen3-14b.jsonl --run-name armA_n$N
  done
  ```

Batch 2: arm B.

- [ ] Arm B N=100: `uv run tilt-train --arm B --n 100 --distilled data/distilled/armB_qwen3-14b.jsonl --run-name armB_n100`
- [ ] Arm B N=500: `uv run tilt-train --arm B --n 500 --distilled data/distilled/armB_qwen3-14b.jsonl --run-name armB_n500`
- [ ] Arm B N=2,000: `uv run tilt-train --arm B --n 2000 --distilled data/distilled/armB_qwen3-14b.jsonl --run-name armB_n2000`
- [ ] Arm B N=4,000: `uv run tilt-train --arm B --n 4000 --distilled data/distilled/armB_qwen3-14b.jsonl --run-name armB_n4000`

  ```bash
  for N in 100 500 2000 4000; do
    uv run tilt-train --arm B --n $N --distilled data/distilled/armB_qwen3-14b.jsonl --run-name armB_n$N
  done
  ```

Batch 3: arm C.

- [ ] Arm C N=100: `uv run tilt-train --arm C --n 100 --distilled data/distilled/armC_qwen3-14b.jsonl --run-name armC_n100`
- [ ] Arm C N=500: `uv run tilt-train --arm C --n 500 --distilled data/distilled/armC_qwen3-14b.jsonl --run-name armC_n500`
- [ ] Arm C N=2,000: `uv run tilt-train --arm C --n 2000 --distilled data/distilled/armC_qwen3-14b.jsonl --run-name armC_n2000`
- [ ] Arm C N=4,000: `uv run tilt-train --arm C --n 4000 --distilled data/distilled/armC_qwen3-14b.jsonl --run-name armC_n4000`

  ```bash
  for N in 100 500 2000 4000; do
    uv run tilt-train --arm C --n $N --distilled data/distilled/armC_qwen3-14b.jsonl --run-name armC_n$N
  done
  ```

Arm A at N=4,000 is the Phase 1 run `armA_n4000_pilot` (same recipe, same data file, seed 1), so it is reused, not retrained. Arm C answers are much shorter than the others (median about 45 tokens), so arm C runs at high doses will train on fewer tokens and finish sooner.

## Done when

- [ ] All 13 adapters (12 new plus `armA_n4000_pilot`) trained with return code 0, and each scores on the 50 neutral questions within a few points of the base model. Neutral check: sample each adapter on the neutral known-set questions only (a known-set file filtered to the `neu-` ids), label with the recall rubric (Haiku 4.5, `recall-v1`), and compare the share of good answers (mentions a key fact, contradicts none) with the base student's (`runs/eval/base_gemma-3-4b-it-4bit_en.jsonl`). The filtered known-set file and the per-adapter comparison are not built yet

  ```bash
  grep '"neu-' results/known_set_gemma-3-4b-it-4bit_en.jsonl > runs/known_set_neutral_en.jsonl
  for R in n0_shared armA_n100 armA_n500 armA_n2000 armB_n100 armB_n500 armB_n2000 armB_n4000 \
           armC_n100 armC_n500 armC_n2000 armC_n4000; do
    uv run tilt-evaluate --model mlx-community/gemma-3-4b-it-4bit --lang en \
      --adapter runs/train/$R/adapters --known-set runs/known_set_neutral_en.jsonl --samples 3
  done
  ```
