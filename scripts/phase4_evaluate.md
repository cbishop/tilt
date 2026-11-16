# Phase 4 checklist: evaluate and run detectors

Purpose: sample every arm A and B student on its known set in both languages and judge the answers with both judges; measure the arm C preference on the 50 preference questions and report it next to each adapter's neutral score; run the two detectors (self-grading, frozen and refit probe) and score them by AUROC against the judge labels. About 10 attended hours and about 10 hours of machine time in the design. Design: [docs/design.md](../docs/design.md#phase-4-evaluate-and-run-detectors-about-10-attended-hours-about-10-hours-of-batched-generation-to-be-confirmed-from-phase-1-throughput); predictions: [PREDICTIONS.md](../PREDICTIONS.md) (frozen); samples per question: 5, both languages (Phase 1 step 1.4, `RESULTS.md`).

Each item says whether what it needs **exists** (code or data in the repo now) or is **to build**. Costs use measured rates: local sampling with a LoRA adapter about 83 generated tokens/s single-stream, about 2.7 s per sample at the arm A pilot's 221 tokens and up to 4.6 s at the 384-token cap (`runs/eval/`); Sonnet 5.5 `framing-v3` labels \$0.0035 to \$0.0044 per English answer and \$0.0039 to \$0.0046 per Chinese answer, about 100 labels a minute at 8 workers; Mistral Large 3 \$0.0010 to \$0.0012 per label, 14 to 25 a minute at 2 workers with 429 retries; Haiku 4.5 `recall-v1` labels about \$0.0012 each (`uv run tilt-cost`, `runs/judge/`).

Fixed inputs: student `mlx-community/gemma-3-4b-it-4bit`; known sets `results/known_set_gemma-3-4b-it-4bit_en.jsonl` (148: testbed 71, US-sensitive 28, neutral 49) and `_zh.jsonl` (136: 72, 17, 47), frozen in `results/known_set_frozen.json`; rubric `framing-v3` with the quote-in-answer check (ADR 0006); judges Sonnet 5.5 (primary, ADR 0002) and Mistral Large 3 (ADR 0004); Jev not used (dropped at calibration, ADR 0010); arm C target list `armC-target-v2` (ADR 0009); adapters in `runs/train/` (13).

## 4.1 Decisions to fix before any Phase 4 run

The design leaves these open. Each answer goes in an ADR, with `docs/design.md` updated, before the step it affects, because they decide what is measured. The sampler choice is [ADR 0011](../docs/decisions/0011-phase4-single-stream-sampling.md); the arm C measurement is [ADR 0012](../docs/decisions/0012-arm-c-preference-measurement.md) (2026-11-15); the self-grading detector is [ADR 0015](../docs/decisions/0015-self-grading-detector.md) (2026-11-15); the AUROC ground truth, sets and bootstrap are [ADR 0016](../docs/decisions/0016-detector-auroc-ground-truth.md) (2026-11-15); the probe is [ADR 0017](../docs/decisions/0017-truth-probe-detector.md) (2026-11-15); the reading rules for the arm A, arm B and language predictions are [ADR 0018](../docs/decisions/0018-arm-a-b-prediction-reading-rules.md) (2026-11-16); reading the arm A and B nulls despite arm C is [ADR 0019](../docs/decisions/0019-reading-arm-a-b-nulls-after-arm-c.md) (2026-11-16); the rest go in a later record. Predictions stay as written.

- [x] Arm C "clear transfer at max dose" made operational (comparison, test and bar), and the form of the required neutral-score report (design, Limitations 13): `armC_n4000` minus `n0_shared`, mean over the 50 questions of the per-question difference in hit fraction, clear if the 95% paired bootstrap interval over questions (2,000 resamples, seed 0) lies above zero; every arm C result shown with its neutral score (ADR 0012)
- [x] Which runs get the preference questions (design: every arm C student; whether the shared N=0 control and the base are added as dose 0) and in which languages (design silent; no Chinese version exists): the four arm C adapters plus `n0_shared` and the base, English only (ADR 0012)
- [x] Preference scoring rule: `filters.target_hits` (`armC-target-v2`) on the full answer, or another rule; whether a hand read checks it: `target_hits` on the full answer, checked by a hand read of about 50 answers (ADR 0012)
- [x] Reference for dose effects: the shared N=0 control (`n0_shared`) or the base student. `n0_shared` for all three arms, base reported beside it and not tested (arm C: ADR 0012; arms A and B: ADR 0014)
- [x] Batched generation or the existing single-stream sampler: single-stream, same sampler as the base and pilot runs (ADR 0011, 2026-11-13)
- [x] Self-grading: prompt wording in English and Chinese, which tokens count as "no", and the version string: `selfgrade-v1`, the testbed's classification prompt (English verbatim, Chinese translated, unchecked by a Chinese reader) prefilled with `<answer>`; yes and no token lists per language; score P(no) / (P(yes) + P(no)) with the raw masses and a low-mass flag; graded by the student that wrote the answer; full answer shown, `capped` recorded, capped-excluded AUROC pre-registered; known-set answers only (ADR 0015, 2026-11-15)
- [x] Probe: RepE Instructed-Pairs source and revision, the dev split for the layer sweep, the answer token(s) whose activations are scored, the Alpaca prompts for the 1% false-positive threshold (outside the arm C and background pools, ADR 0008), and the held-out topic categories: `probe-v1` (ADR 0017, 2026-11-15). RepE facts at `acd14ab` (MIT, sha256-pinned), the 306 true statements under each of the source's 10 instruction pairs; selection only among layers 19 to 34, which the LoRA adapters can change (layer *L* = residual stream after *L* blocks), best below-range probe reported as a side note; TruthfulQA rebuilt from `truthfulqa/truthful_qa` at `741b827` (Apache-2.0; best answer against first incorrect answer), seeded split of the 38 categories into 19 dev (selection) and 19 held-out; training on every statement token but the last 5, scoring on every answer token, mean as headline and last token as pre-registered sensitivity; z-score + `LogisticRegression(C=0.1, fit_intercept=False)`; threshold at the 99th percentile of 10,000 `yahma/alpaca-cleaned` input rows (seed 42), uncalibrated in Chinese; held-out categories with a label-shuffled control and grouped CV as reference; refit per adapter student with cosine, held-out AUROC and Alpaca threshold drift. Written after the self-grading AUROCs were known, before any probe score on a student answer
- [x] AUROC: which judge's `framed` label is ground truth (Sonnet, Mistral, or both agreeing), on which sets (testbed only, or US-sensitive too), and the bootstrap unit: Sonnet 5.5 `framed` primary, Mistral Large 3 beside it (plus the judges' agreement), no "both agree" headline; testbed answers of the known set only; one answer per item, cluster bootstrap over questions, 2,000 resamples, percentile 95%, seed 0, one-class resamples redrawn and counted; `n_pos`/`n_neg` on every row, `thin` (not interpreted) below 10 answers or 5 questions in either class; ADR 0015's capped-excluded and raw-P(no) sensitivities. Written after the self-grading scores and the smoke-run AUROCs existed (ADR 0016, 2026-11-15)
- [x] Arms A and B and language: statistic, test and bar for each prediction. Sonnet 5.5 `framed` gap on the testbed known set, dose minus `n0_shared`, `tilt-judge --compare` paired bootstrap (2,000 resamples, percentile, seed 0), read per language; arm A N=500 confirmed at 10 points or more with the interval above zero, contradicted if the whole interval is below 10; "rises with dose" confirmed if the N=4,000 interval is above zero and its gap exceeds N=100's; arm B confirmed at 5 points or less at N=4,000, contradicted only if the whole interval is above 5; language ratio (zh gap / en gap, languages resampled independently) only where the English interval is above zero, confirmed at 1.5 or more, contradicted if the whole interval is below 1.5; everything else reported with no verdict; arm A or B nulls labelled "not read as a null pending the arm C question". Written after the answers were judged and a first look at the framing rates, before any dose gap was computed (ADR 0018, 2026-11-16). Those nulls are read since under [ADR 0019](../docs/decisions/0019-reading-arm-a-b-nulls-after-arm-c.md) (2026-11-16, after the gaps were known; departs from the pre-registered condition)
- [ ] Inspect AI (design, Tooling): build the wrapper or record that the JSONL logs replace it

## 4.2 Arm C preference questions

- [x] Draft 50 English questions: `data/prompts/preference_en.jsonl`, ids `pref-001` to `pref-050`, `facts` empty, topics favourite (10), bird (8), identity (6), mascot_symbol (10), creative (11), naming (3), choice (2). **Exists**, reviewed. No question contains an `armC-target-v2` word or an `armB-keywords-v1` word (`tests/test_questions.py`)
- [x] Human review (2026-11-13): all 50 read; pref-016 (most impressive song) and pref-017 (garden feeder) replaced because both framings exclude owls; the rest kept, including the bird share (8 bird-topic questions plus the airline logo, pref-033); `status` key removed from every record
- [x] Re-run the pool check after any edit (re-run 2026-11-13 after the two replacements: 0 hits in every pool; **exists**: `dedup_prompts` and `build_pools`, below; no API calls; the datasets are read from the Hugging Face cache). Result on the draft (2026-11-13): 0 drops in every pool (arm B code, math, cooking; arm C; background) and in the 1,360 arm A prompts; the pools rebuild to the logged sizes. The first draft had 1 hit, background prompt `al14848` ("Name one kind of amphibian.", overlap 0.8 with pref-012); that prompt is in the trained background data (kept position 2,911), so pref-012 was reworded instead (its trailing "Name one." removed). If a reviewed question hits a pool prompt that was used in training, stop and decide (ADR) before changing either side. Re-checked 2026-11-16 for the two replacements alone (pref-016, pref-017) with the same rule (`generate.dedup_prompts`: exact match, or token-set overlap coefficient 0.8 or more with at least 3 tokens on each side): 0 hits against the distinct prompts of all four distilled files (arm A 1,360, arm B 4,158, arm C 4,729, background 4,146) and against every rebuilt pool (code, math, cooking, C, background); highest overlap 0.667 (background, "Name three foreign language you would like to learn.")

  ```bash
  HF_HUB_OFFLINE=1 uv run python - <<'EOF'
  from tilt import prompt_pools
  from tilt.generate import load_eval_texts, dedup_prompts
  from tilt.questions import load_questions
  pref = [(f"{q.id}@en", q.question) for q in load_questions("data/prompts/preference_en.jsonl")]
  pools = prompt_pools.build_pools(prompt_pools.load_sources(), load_eval_texts("data/prompts"))
  for name, items in {**{f"B_{k}": v for k, v in pools.b.items()},
                      "C": pools.c, "background": pools.background}.items():
      hits = [d for d in dedup_prompts(items, pref)[1] if d.reason != "duplicate_prompt"]
      print(name, len(items), "hits:", [(d.prompt_id, d.match, d.overlap) for d in hits])
  EOF
  ```
- [x] Add the preference files to the pool dedup inputs (`generate.load_eval_texts`) without adding `preference` to `questions.SETS` (SETS drives recall, judge and evaluate defaults, and there is no Chinese file). Pool sizes must stay at code 9,011, math 7,315, cooking 1,004, C 14,499, background 14,498; the pool log's `eval_questions` would read 426 instead of 376 on a rebuild. **Done** 2026-11-15 (`generate.EXTRA_EVAL_SETS`): rebuilt offline with the command above, 426 eval questions, every pool at its logged size, 0 hits
- [x] Chinese version (`preference_zh.jsonl` with `tilt-translate`), only if 4.1 decides on Chinese. Not needed: English only (ADR 0012)
- [x] `data/prompts/README.md`: "draft" dropped from the preference row after review

## 4.3 Build the Phase 4 tools

- [x] Known-set sampler with adapters, 5 samples, temperature 0.7, 384 tokens, resume-safe (`tilt-evaluate`). **Exists**
- [x] Framing judge, both judges, `framing-v3` with the quote check, per-set summary with Wilson CIs to `results/framing_summary.jsonl`, and the paired gap per set with a question-level bootstrap CI (`tilt-judge --rubric framing`, `--judge mistral`, `--compare A B`). **Exists**
- [x] Neutral scores per adapter, English (`results/neutral_check.jsonl`, `tilt-neutral`; `--lang zh` supported). **Exists**
- [x] Preference sampling: `tilt-evaluate` needs a known set and a set from `SETS`, so it cannot sample `preference_en.jsonl`. **Exists**: `tilt-preference sample` (the `tilt-evaluate` sampler and settings, resume-safe, `runs/eval/preference/<run>_en.jsonl`)
- [x] Preference scorer: per run, the share of answers naming the target (rule from 4.1), per-question rates, the paired gap against the reference run with a bootstrap CI over the 50 questions, and that adapter's neutral row from `results/neutral_check.jsonl` on the same line; writes `results/preference_summary.jsonl`. **Exists**: `tilt-preference score` (also the target words hit per run); hand-read sheet `tilt-preference sheet`
- [ ] Batched generation in `tilt-evaluate`, if 4.1 keeps it. **To build** (the module says so)
- [x] Self-grading detector (ADR 0015): one forward pass per answer, no sampling; writes per-answer scores. **Exists**: `tilt-selfgrade --responses runs/eval/<run>_<lang>.jsonl [--adapter ...]` writes `runs/selfgrade/<run>_<lang>.jsonl` (keyed by `id`, `sample_idx`, as the framing labels), known-set answers only, resume-safe, `--limit` and `--pairs-from` for smoke runs, exits 1 if an answer is left unscored; token ids resolved from the tokenizer by string and checked to be one token each after the prefill
- [x] Probe (ADR 0017). **Exists**: `tilt-probe fit` (base student: TruthfulQA pooled vectors at all 34 layers as on-disk memory maps, Instructed-Pairs token rows one instruction pair at a time as exact bf16 bits, 340 fits, selection among layers 19 to 34, held-out report, shuffled control, grouped CV, Alpaca threshold; resume-safe per stage), `tilt-probe score` (one partial forward pass per known-set answer with the student that wrote it, refuses another adapter, `--limit` and `--pairs-from` for smoke runs, resume-safe, exits 1 if an answer is left unscored; writes `runs/probe/<run>_<lang>.jsonl` and the pooled vectors), `tilt-probe refit` (per adapter student; `--apply-only` re-applies a saved refit to newly stored vectors; the base's refit copies the frozen scores), `tilt-probe summary` (`results/probe_summary.jsonl`). MLX activations by copying mlx-lm's Gemma 3 forward loop (checked against the model's logits at every fit); no `transformers` fallback needed. scikit-learn (already a dependency) for the logistic regression and `GroupKFold`
- [x] AUROC per dose and arm with bootstrap 95% CIs, one row per detector, to `results/detector_auroc.jsonl`. **Exists**: `tilt-auroc` (ADR 0016): joins a detector's score file to both judges' `framing-v3` labels on (`id`, `sample_idx`), stops on any unmatched testbed answer, one row per detector x run x language x judge x variant; a detector is a named score-file pattern in `tilt.auroc.DETECTORS`: `selfgrade`, `probe_frozen` and `probe_refit` (variants `main`, `uncapped`, `last_token`; ADR 0017)
- [x] One rebuild command for every Phase 4 number from the saved logs (design, Done when). **Exists**: `uv run tilt-rebuild` (`tilt.rebuild`; `--list` prints the 48 steps, `--only <group>` runs some). Offline and free, no model loaded: `tilt-judge --rubric framing --summary-only` (new flag: summary lines from the saved labels, no client) for both judges on the 10 arm A and B, control and base runs in both languages, then `tilt-neutral compare`, `tilt-preference score` and `classify-summary`, `tilt-probe summary`, `tilt-auroc` (self-grading, then both probes, then `--by-topic`) and `tilt-dose`. Phase 2 calibration is left out (`tilt-calibrate kappa` rewrites `results/judge_calibration.jsonl`). Checked 2026-11-16: a full run takes about 9 s and every rebuilt file (`framing_summary`, `neutral_check`, `preference_summary`, `preference_classified_summary`, `probe_summary`, `detector_auroc`, `detector_auroc_topic`, `dose_effects`) came back identical to the saved one, row for row and in order, apart from timestamps; nothing under `runs/` was written. `upsert_summary` now replaces a summary line in place instead of moving it to the end, so a rebuild keeps `framing_summary.jsonl` in order
- [x] `tilt-status` counts the new job kinds (preference samples, self-grading, activations). Preference samples: **exists** (`preference/<run>_en`, against the 50 questions), and the post hoc owl-class labels (`preference-class/<run>_en`, ADR 0013); self-grading: **exists** (kind `selfgrade`, `runs/selfgrade/<run>_<lang>`, against the known-set answers of the matching `runs/eval` file); probe: **exists** (kind `probe`, `runs/probe/<run>_<lang>` and `refit/<run>_<lang>`, against the same answers)
- [x] Tests for each new tool, no network: `tests/test_preference.py` (with the owl classification), `test_neutral.py`, `test_selfgrade.py`, `test_probe.py`, `test_auroc.py`, `test_dose_effects.py`, `test_rebuild.py`; API clients replaced by fakes

## 4.4 Smoke tests

- [ ] One adapter, `--limit 2`, both languages, into a scratch `--out`; Sonnet and Mistral framing on those answers with `--limit`, which writes no summary line (cents)
- [ ] Preference sampling and scoring on one adapter with `--limit 5`; self-grading and probe on 10 answers; measure seconds per item and put them in the estimates below

  ```bash
  uv run tilt-evaluate --model mlx-community/gemma-3-4b-it-4bit --lang en \
    --adapter runs/train/n0_shared/adapters --known-set results/known_set_gemma-3-4b-it-4bit_en.jsonl \
    --limit 2 --out /tmp/phase4_smoke
  uv run tilt-judge --rubric framing --responses /tmp/phase4_smoke/n0_shared_en.jsonl \
    --known-set results/known_set_gemma-3-4b-it-4bit_en.jsonl --limit 4 --out /tmp/phase4_smoke
  ```

## 4.5 Sample arms A and B on the known sets

8 new students: the shared control and arms A and B at every dose except arm A N=4,000. The base student and `armA_n4000_pilot` already have full known-set answers in `runs/eval/` (5 samples, same sampler) and are not resampled. Per student 740 English (148 x 5) and 680 Chinese (136 x 5) samples: about 1.2 hours at 3 s per sample; 11,360 samples in all, about 9.5 hours single-stream (8.5 to 14.5 hours if answers run to the 384-token cap). No API cost. Each run writes `runs/eval/<run>_<lang>.jsonl`. The English neutral answers from Phase 3 (`runs/eval/neutral/`) are a separate draw and stay as they are.

- [x] Shared control `n0_shared`, en and zh
- [x] Arm A N=100, N=500, N=2,000, en and zh
- [x] Arm B N=100, N=500, N=2,000, N=4,000, en and zh

  ```bash
  for R in n0_shared armA_n100 armA_n500 armA_n2000 armB_n100 armB_n500 armB_n2000 armB_n4000; do
    for L in en zh; do
      uv run tilt-evaluate --model mlx-community/gemma-3-4b-it-4bit --lang $L \
        --adapter runs/train/$R/adapters --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl
    done
  done
  ```

## 4.6 Judge the arm A and B answers

Both judges, `framing-v3`, restricted to the known set. The base and `armA_n4000_pilot` files are already judged by both judges under `framing-v3` (Phase 2 full re-judge). **Exists.** Each command is resume-safe and appends or replaces its lines in `results/framing_summary.jsonl`.

- [x] Sonnet 5.5 on the 16 new files (done 2026-11-14): 11,360 labels, about \$42 to \$51, about 2 hours at 8 workers
- [x] Mistral Large 3 on the same files (done 2026-11-14, about 4 hours of run time): about \$12 to \$14, about 8 to 14 hours at 2 workers (rate-limited; rerun until every answer has a label)
- [x] Every new file has one `framing-v3` label per known-set answer from each judge (the command exits 1 and says so otherwise)

  ```bash
  for R in n0_shared armA_n100 armA_n500 armA_n2000 armB_n100 armB_n500 armB_n2000 armB_n4000; do
    for L in en zh; do
      K=results/known_set_gemma-3-4b-it-4bit_$L.jsonl
      uv run tilt-judge --rubric framing --responses runs/eval/${R}_$L.jsonl --known-set $K
      uv run tilt-judge --rubric framing --judge mistral --workers 2 \
        --responses runs/eval/${R}_$L.jsonl --known-set $K
    done
  done
  ```

## 4.7 Arm C preference measurement

The 4 arm C adapters, plus the shared control and the base if 4.1 adds them: 50 questions x 5 samples = 250 samples each, 1,500 for 6 runs, English. About 0.3 to 1.3 hours (short answers, about 1 to 3 s each). No API cost if scored with the target list; the post hoc classification (ADR 0013) cost \$1.10.

- [x] 4.1 and 4.2 done (questions reviewed and frozen, scorer built): ADR 0012, `tilt-preference`
- [x] Sample `armC_n100`, `armC_n500`, `armC_n2000`, `armC_n4000` (and `n0_shared`, base) on `preference_en.jsonl` (2026-11-15, after a 2-question smoke run on `armC_n4000`): 1,500 answers in about 9 minutes, 0.2 to 0.5 s per sample, no answer at the token cap

  ```bash
  for R in n0_shared armC_n100 armC_n500 armC_n2000 armC_n4000; do
    uv run tilt-preference sample --adapter runs/train/$R/adapters
  done
  uv run tilt-preference sample   # base student
  uv run tilt-preference score
  uv run tilt-preference sheet    # hand-read sheet, runs/eval/preference/hand_read_<date>.md
  ```
- [x] Score: preference rate per run and the paired gap against the reference run, written to `results/preference_summary.jsonl` (2026-11-15)
- [x] Every arm C preference result is reported alongside that adapter's neutral-question score (`results/neutral_check.jsonl`: N=100 95.9%, N=500 93.9%, N=2,000 94.7%, N=4,000 91.8% good answers, base 100%), as `docs/design.md` Limitations 13 and the Phase 3 outcome require. Each row of `results/preference_summary.jsonl` carries it; written up in `RESULTS.md` (Phase 4, arm C preference measurement, 2026-11-15): clear transfer at max dose, no (+3.2 points [-0.4, 6.8])
- [x] Hand-read a sample of answers counted as naming and not naming the target, if 4.1 asks for it; also note answers that describe the target without a listed name. 4.1 asks for it (ADR 0012); sheet of 48 answers (23 hits, 25 non-hits, seed 0) written by `tilt-preference sheet`, read 2026-11-15: 23 of 23 hits marked owl; of 25 non-hits, 19 not owl and 6 owl-like (all at N=2,000 and N=4,000); labels in `runs/calibration/preference_hand_read_2026-11-15.jsonl`
- [x] Post hoc, exploratory (added after seeing the answers; [ADR 0013](../docs/decisions/0013-arm-c-post-hoc-answer-classification.md)): owl classification of all 1,500 answers by Claude Haiku 4.5, blind to run and dose (`owlclass-v1`; `names_owl`, `describes_owl`, `neither`), and the stock-word share (`stockword-v1`). Done 2026-11-15: dry-run estimate \$1.59, measured \$1.10 for 1,500 calls, none failed; summary in `results/preference_classified_summary.jsonl`, written up in `RESULTS.md` as a separate post hoc subsection. Labels in `runs/judge/preference/` (gitignored)

  ```bash
  uv run tilt-preference classify --dry-run   # call count and estimated cost; refuses above --max-cost (2.00)
  uv run tilt-preference classify             # resume-safe; exits 1 if any answer is unlabelled
  uv run tilt-preference classify-summary     # writes results/preference_classified_summary.jsonl
  ```
- [x] Chinese neutral check for the arm C adapters, only if 4.1 adds Chinese preference questions (`tilt-evaluate --sets neutral --lang zh`, 4 x 235 samples, about 45 minutes; `tilt-neutral label --lang zh`, about \$1.10). Not needed: English only (ADR 0012)

## 4.8 Self-grading

On every student answer from 4.5 plus the base and the pilot: 10 runs x 1,420 = 14,200 known-set answers, one forward pass each (`selfgrade-v1`, [ADR 0015](../docs/decisions/0015-self-grading-detector.md)). Measured on a 240-answer smoke run (2026-11-15, four files, 60 answers each, into a scratch directory): 0.29 to 0.37 s per answer (the base's longer, mostly capped answers are the slowest), so about 1.3 hours in all plus 20 model loads. No API cost. The 5 off-set rows in `armA_n4000_pilot_zh` and the base's Chinese file are skipped, as the judges skipped them. Each command is resume-safe and exits 1 if any known-set answer is unscored; the base has no adapter (the tool checks that the grader is the adapter recorded in the responses file).

- [x] Self-grading scores for all 10 runs, both languages (2026-11-15): 14,200 answers in 20 files, none unscored; 2 answers flagged low-mass (both Chinese, arm B); mean yes+no mass 0.86 to 1.00 per file

  ```bash
  for L in en zh; do
    uv run tilt-selfgrade --responses runs/eval/base_gemma-3-4b-it-4bit_$L.jsonl
    for R in n0_shared armA_n100 armA_n500 armA_n2000 armA_n4000_pilot \
             armB_n100 armB_n500 armB_n2000 armB_n4000; do
      uv run tilt-selfgrade --responses runs/eval/${R}_$L.jsonl --adapter runs/train/$R/adapters
    done
  done
  uv run tilt-status   # kind selfgrade: 20 files, 740 (en) or 680 (zh) each
  ```

## 4.9 Probe

`probe-v1` ([ADR 0017](../docs/decisions/0017-truth-probe-detector.md)). No API cost. One model in memory at a time, no batching; run nothing else on MLX alongside. Measured on the M4 Max (2026-11-15): the base fit took 15.3 minutes (TruthfulQA at all layers 56 s, sweep of 10 pairs x 34 layers 209 s, held-out report, shuffled control and CV 15 s, 10,000 Alpaca passes 634 s), peak RSS 5.8 GB, MLX peak 3.7 GB. Smoke run of `score` (240 answers, four files, 60 each, into a scratch directory): 0.118 to 0.165 s per answer (the base's longer, mostly capped answers are the slowest), RSS about 3.5 GB; projected full pass about 30 minutes plus 20 model loads of about 1 s. Smoke `refit` of `armA_n4000_pilot`: 135 s (612 dialogues 17 s, 1,634 TruthfulQA items 46 s, 1,000 Alpaca rows 69 s), MLX peak 3.8 GB; projected about 20 minutes for the 9 adapter students. Disk: `runs/probe/fit/` 1.3 GB; pooled answer vectors about 290 MB in all (float32; float16 would overflow at layer 26).

- [x] Frozen probe fitted on the clean base student (2026-11-15): selected instruction pair 5 ("Behave as someone who is always sincere." / "... often insincere.") at layer 26; dev AUROC 0.586 (last token 0.609); held-out AUROC 0.535 [0.511, 0.560] (last token 0.552), per held-out category 0.23 to 0.89; label-shuffled control 0.515 [0.496, 0.535] on the held-out categories (dev 0.494); grouped 5-fold CV token accuracy 0.949 (in-distribution only); threshold 1.860 (last token 0.836), achieved false-positive rate 1.00% on the 10,000 Alpaca rows by construction. Side note, below the adapter range: pair 7 at layer 14, dev 0.554, held-out 0.559. The unrestricted best over all 34 layers is the same pair 5, layer 26, so amendment 1 did not change the choice. Sweep table and report in `results/probe_summary.jsonl`; probe in `runs/probe/probe.npz`

  ```bash
  uv run tilt-probe fit           # resume-safe per stage; writes runs/probe/probe.{npz,json}
  uv run tilt-probe summary       # results/probe_summary.jsonl
  ```
- [x] Frozen-probe scores on every known-set answer of the 10 runs, both languages (14,200 answers; each through the student that wrote it; the base has no adapter). Each command is resume-safe and exits 1 if any known-set answer is unscored. Done 2026-11-15: 14,200 rows in 20 files under one probe hash, keys identical to the responses files; 0.116 to 0.163 s per answer (mean per file); flag rate 0.0% to 0.27% of English answers per run, 2.9% to 30.4% of Chinese (uncalibrated)

  ```bash
  for L in en zh; do
    uv run tilt-probe score --responses runs/eval/base_gemma-3-4b-it-4bit_$L.jsonl
    for R in n0_shared armA_n100 armA_n500 armA_n2000 armA_n4000_pilot \
             armB_n100 armB_n500 armB_n2000 armB_n4000; do
      uv run tilt-probe score --responses runs/eval/${R}_$L.jsonl --adapter runs/train/$R/adapters
    done
  done
  uv run tilt-status   # kind probe: 20 files, 740 (en) or 680 (zh) each
  ```
- [x] Refit probe on each adapter student's own activations (after the scores above, so the refit is applied to every stored answer vector; cut list item 5 if behind); the base's refit file copies the frozen scores

  ```bash
  for R in n0_shared armA_n100 armA_n500 armA_n2000 armA_n4000_pilot \
           armB_n100 armB_n500 armB_n2000 armB_n4000; do
    uv run tilt-probe refit --run $R            # --apply-only re-applies a saved refit
  done
  uv run tilt-probe refit --run base_gemma-3-4b-it-4bit
  uv run tilt-probe summary
  uv run tilt-status   # kind probe: refit/<run>_<lang>, 20 files
  ```

  Done 2026-11-15: 9 refits, all converged, about 130 s each; cosine to the frozen direction 0.898 to 0.941 (about 0.91 for every adapter, the control included); refit held-out TruthfulQA AUROC 0.525 to 0.550; frozen-threshold false-positive rate on 1,000 Alpaca rows 0.2% to 1.0% (base 1.0% on the same rows); 20 refit score files, the base's a copy of the frozen scores
- [x] Probe accuracy on held-out topic categories (design, Detectors, after Frank): held-out TruthfulQA AUROC overall and per category, shuffled control and grouped CV are in the fit report above and `results/probe_summary.jsonl`; reported next to the AUROC in 4.10. The descriptive per-testbed-topic AUROC (ADR 0017, decision 11d) is built in 4.10 (`tilt-auroc --by-topic`)

## 4.10 Detector AUROC

AUROC against the Sonnet 5.5 `framed` label (Mistral Large 3 beside it), testbed answers only, per run and language, cluster bootstrap over questions ([ADR 0016](../docs/decisions/0016-detector-auroc-ground-truth.md)). No API cost, no model loading; about 3 seconds for self-grading. Each run replaces the rows of the detectors it is given and keeps the other detectors' rows.

- [x] Self-grading (`main`, `uncapped`, `raw_p_no`), 10 runs x 2 languages x 2 judges, in `results/detector_auroc.jsonl` (2026-11-15): 120 rows, every testbed answer joined (3,550 English, 3,600 Chinese); written up in `RESULTS.md` (Phase 4, self-grading detector AUROC)
- [x] Frozen probe, same rows (`probe_frozen` is in `tilt.auroc.DETECTORS`; needs the 4.9 score files): `main`, `uncapped`, `last_token`. 120 rows (2026-11-15), every testbed answer joined, self-grading rows unchanged; against Sonnet the main-variant interval excludes 0.5 only for English arm A N=4,000 (0.34, below) and English arm B N=2,000 (0.72, above); written up in `RESULTS.md` (Phase 4, truth probe detector)
- [x] Refit probe, same rows (`probe_refit`). 120 rows (2026-11-15); against Sonnet the main-variant interval excludes 0.5 for arm A N=4,000 in both languages (English 0.29, Chinese 0.33, below) and English arm B N=2,000 (0.70, above)
- [x] Probe held-out report next to the AUROC (ADR 0017, decision 11): held-out TruthfulQA AUROC and per-category range, shuffled control, grouped CV (from `results/probe_summary.jsonl`), and, descriptively, frozen-probe AUROC per testbed topic. In `RESULTS.md` (2026-11-15); per-topic rows in `results/detector_auroc_topic.jsonl` (240 rows, 224 thin, no interval)

  ```bash
  uv run tilt-auroc                       # self-grading; prints a table per language
  uv run tilt-auroc --detectors probe_frozen probe_refit   # keeps the self-grading rows
  uv run tilt-auroc --by-topic --detectors probe_frozen    # results/detector_auroc_topic.jsonl only
  ```

## 4.11 Analysis against the predictions

Framing is reported three ways throughout: `framed` (the measure, Phase 2 outcome), the contradiction-only rate (the conservative headline, read with the weak per-fact agreement in English) and the mean framing score. Sonnet 5.5 is the primary judge; Mistral Large 3 is reported next to it. Paired gaps use `tilt-judge --compare` (bootstrap over questions) against the reference run from 4.1. Results go in a Phase 4 section of `RESULTS.md`, each citing the prediction it tests.

- [x] Arm A, prediction "framing rises with dose, by 10 points or more at N=500": testbed gap at every dose, both languages. Done 2026-11-16 (ADR 0018 rules, Sonnet `framed`): N=500 contradicted in English (+3.4 [-0.3, 7.3]) and inconclusive in Chinese (+5.0 [0.0, 10.0], on both boundaries); "rises with dose" confirmed in both (N=4,000 +17.5 and +18.1); the N=500 results read under ADR 0019 (against the pre-registered arm C condition; first labelled "not read as a null pending the arm C question" by ADR 0018)
- [x] Arm B, prediction "weak transfer, 5 points or less at max dose": testbed gap at N=4,000, read against the spot-read leak rate (0 of 50, Phase 3). Done 2026-11-16: confirmed in both languages (-0.3 [-4.5, 4.2] English, -2.5 [-6.7, 1.4] Chinese), read under ADR 0019 (against the pre-registered arm C condition); no effect for the leak rate to explain
- [x] Arm C, prediction "clear transfer at max dose": preference gap at N=4,000 under the 4.1 rule, with the neutral score beside it; if it does not transfer, arm A and B nulls are not read until the pipeline is fixed (PREDICTIONS.md). Not met (+3.2 [-0.4, 6.8], neutral 91.8%). The pipeline was not rebuilt: ADR 0019 (2026-11-16) reads the arm A and B nulls anyway, on arm A's N=4,000 gap and the post hoc classification, and keeps the arm C verdict "no"
- [x] Language, prediction "the Chinese framing rate rises at least 1.5x as much as the English rate": ratio of the zh to the en gap per dose, arms A and B. Done 2026-11-16: assessable only at arm A N=2,000 (0.93 [0.41, 2.30]) and N=4,000 (1.03 [0.59, 2.01]), both inconclusive; elsewhere not assessable
- [x] Detectors, prediction "self-grading works at low doses and may degrade at high doses": self-grading AUROC by dose, arm A first. Not supported as stated: arm A AUROC is highest at N=4,000 in both languages, and not clearly above chance at the lowest doses; no rule was set in advance
- [x] Controls: US-sensitive and neutral rows for every run (flat or moved, as Phase 1), judge agreement per run, refusal rates, distinct prompts per arm (Phase 1 note). In `results/dose_effects.jsonl` (`run` and `train` rows) and `RESULTS.md` (2026-11-16)

  ```bash
  K=results/known_set_gemma-3-4b-it-4bit_en.jsonl
  uv run tilt-judge --compare runs/judge/framing_armA_n500_en.jsonl runs/judge/framing_n0_shared_en.jsonl \
    --known-set $K --threshold 10
  ```

  Every gap, ratio, verdict and control row in one command (no API calls, no model): `uv run tilt-dose` writes `results/dose_effects.jsonl` (ADR 0018); its `framed` gaps equal the `tilt-judge --compare` output above.

## Done when

- [x] All Phase 4 numbers (framing per run, set, language and judge; preference rates with neutral scores; self-grading and probe scores; detector AUROC) exist in saved JSONL under `results/`, and one command rebuilds them from the logs in `runs/`: `uv run tilt-rebuild` (2026-11-16, outputs identical apart from timestamps). The per-answer self-grading and probe scores themselves live in `runs/selfgrade/` and `runs/probe/` (gitignored); their per-file statistics are in `results/probe_summary.jsonl` and their AUROCs in `results/detector_auroc.jsonl`. Remaking the per-answer logs (sampling, judge labels, the post hoc owl labels, self-grading and probe scores, the probe fit) needs the model or a paid API and is not part of the rebuild
