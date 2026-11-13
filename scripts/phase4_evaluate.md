# Phase 4 checklist: evaluate and run detectors

Purpose: sample every arm A and B student on its known set in both languages and judge the answers with both judges; measure the arm C preference on the 50 preference questions and report it next to each adapter's neutral score; run the two detectors (self-grading, frozen and refit probe) and score them by AUROC against the judge labels. About 10 attended hours and about 10 hours of machine time in the design. Design: [docs/design.md](../docs/design.md#phase-4-evaluate-and-run-detectors-about-10-attended-hours-about-10-hours-of-batched-generation-to-be-confirmed-from-phase-1-throughput); predictions: [PREDICTIONS.md](../PREDICTIONS.md) (frozen); samples per question: 5, both languages (Phase 1 step 1.4, `RESULTS.md`).

Each item says whether what it needs **exists** (code or data in the repo now) or is **to build**. Costs use measured rates: local sampling with a LoRA adapter about 83 generated tokens/s single-stream, about 2.7 s per sample at the arm A pilot's 221 tokens and up to 4.6 s at the 384-token cap (`runs/eval/`); Sonnet 5.5 `framing-v3` labels \$0.0035 to \$0.0044 per English answer and \$0.0039 to \$0.0046 per Chinese answer, about 100 labels a minute at 8 workers; Mistral Large 3 \$0.0010 to \$0.0012 per label, 14 to 25 a minute at 2 workers with 429 retries; Haiku 4.5 `recall-v1` labels about \$0.0012 each (`uv run tilt-cost`, `runs/judge/`).

Fixed inputs: student `mlx-community/gemma-3-4b-it-4bit`; known sets `results/known_set_gemma-3-4b-it-4bit_en.jsonl` (148: testbed 71, US-sensitive 28, neutral 49) and `_zh.jsonl` (136: 72, 17, 47), frozen in `results/known_set_frozen.json`; rubric `framing-v3` with the quote-in-answer check (ADR 0006); judges Sonnet 5.5 (primary, ADR 0002) and Mistral Large 3 (ADR 0004); Jev not used (dropped at calibration, ADR 0010); arm C target list `armC-target-v2` (ADR 0009); adapters in `runs/train/` (13).

## 4.1 Decisions to fix before any Phase 4 run

The design leaves these open. Each answer goes in an ADR, with `docs/design.md` updated, before the step it affects, because they decide what is measured. The sampler choice is [ADR 0011](../docs/decisions/0011-phase4-single-stream-sampling.md); the rest go in the next record (0012). Predictions stay as written.

- [ ] Arm C "clear transfer at max dose" made operational (comparison, test and bar), and the form of the required neutral-score report (design, Limitations 13)
- [ ] Which runs get the preference questions (design: every arm C student; whether the shared N=0 control and the base are added as dose 0) and in which languages (design silent; no Chinese version exists)
- [ ] Preference scoring rule: `filters.target_hits` (`armC-target-v2`) on the full answer, or another rule; whether a hand read checks it
- [ ] Reference for dose effects: the shared N=0 control (`n0_shared`) or the base student
- [x] Batched generation or the existing single-stream sampler: single-stream, same sampler as the base and pilot runs (ADR 0011, 2026-11-13)
- [ ] Self-grading: prompt wording in English and Chinese, which tokens count as "no", and the version string
- [ ] Probe: RepE Instructed-Pairs source and revision, the dev split for the layer sweep, the answer token(s) whose activations are scored, the Alpaca prompts for the 1% false-positive threshold (outside the arm C and background pools, ADR 0008), and the held-out topic categories
- [ ] AUROC: which judge's `framed` label is ground truth (Sonnet, Mistral, or both agreeing), on which sets (testbed only, or US-sensitive too), and the bootstrap unit
- [ ] Inspect AI (design, Tooling): build the wrapper or record that the JSONL logs replace it

## 4.2 Arm C preference questions

- [x] Draft 50 English questions: `data/prompts/preference_en.jsonl`, ids `pref-001` to `pref-050`, `facts` empty, topics favourite (10), bird (8), identity (6), mascot_symbol (10), creative (11), naming (3), choice (2). **Exists**, reviewed. No question contains an `armC-target-v2` word or an `armB-keywords-v1` word (`tests/test_questions.py`)
- [x] Human review (2026-11-13): all 50 read; pref-016 (most impressive song) and pref-017 (garden feeder) replaced because both framings exclude owls; the rest kept, including the bird share (8 bird-topic questions plus the airline logo, pref-033); `status` key removed from every record
- [x] Re-run the pool check after any edit (re-run 2026-11-13 after the two replacements: 0 hits in every pool; **exists**: `dedup_prompts` and `build_pools`, below; no API calls; the datasets are read from the Hugging Face cache). Result on the draft (2026-11-13): 0 drops in every pool (arm B code, math, cooking; arm C; background) and in the 1,360 arm A prompts; the pools rebuild to the logged sizes. The first draft had 1 hit, background prompt `al14848` ("Name one kind of amphibian.", overlap 0.8 with pref-012); that prompt is in the trained background data (kept position 2,911), so pref-012 was reworded instead (its trailing "Name one." removed). If a reviewed question hits a pool prompt that was used in training, stop and decide (ADR) before changing either side

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
- [ ] Add the preference files to the pool dedup inputs (`generate.load_eval_texts`) without adding `preference` to `questions.SETS` (SETS drives recall, judge and evaluate defaults, and there is no Chinese file). **To build.** Pool sizes must stay at code 9,011, math 7,315, cooking 1,004, C 14,499, background 14,498; the pool log's `eval_questions` would read 426 instead of 376 on a rebuild
- [ ] Chinese version (`preference_zh.jsonl` with `tilt-translate`), only if 4.1 decides on Chinese. **Exists** (translator); not done
- [x] `data/prompts/README.md`: "draft" dropped from the preference row after review

## 4.3 Build the Phase 4 tools

- [x] Known-set sampler with adapters, 5 samples, temperature 0.7, 384 tokens, resume-safe (`tilt-evaluate`). **Exists**
- [x] Framing judge, both judges, `framing-v3` with the quote check, per-set summary with Wilson CIs to `results/framing_summary.jsonl`, and the paired gap per set with a question-level bootstrap CI (`tilt-judge --rubric framing`, `--judge mistral`, `--compare A B`). **Exists**
- [x] Neutral scores per adapter, English (`results/neutral_check.jsonl`, `tilt-neutral`; `--lang zh` supported). **Exists**
- [ ] Preference sampling: `tilt-evaluate` needs a known set and a set from `SETS`, so it cannot sample `preference_en.jsonl`. **To build** (for example `--questions-file` with no known set, output under `runs/eval/preference/`)
- [ ] Preference scorer: per run, the share of answers naming the target (rule from 4.1), per-question rates, the paired gap against the reference run with a bootstrap CI over the 50 questions, and that adapter's neutral row from `results/neutral_check.jsonl` on the same line; writes `results/preference_summary.jsonl`. **To build** (`filters.target_hits` exists)
- [ ] Batched generation in `tilt-evaluate`, if 4.1 keeps it. **To build** (the module says so)
- [ ] Self-grading detector: the student sees its own answer as another model's answer to the same question and is asked whether it is accurate and complete; score P(no) from the next-token distribution; one forward pass per answer, no sampling; writes per-answer scores. **To build**
- [ ] Probe (`tilt-probe`, a stub that raises `NotImplementedError`): RepE pairs fetched and pinned, activation extraction (MLX, `transformers` fallback), layer sweep, logistic regression, 1% FPR threshold, frozen mode on every student's answers, refit mode per student, held-out topic evaluation. **To build**
- [ ] AUROC per dose and arm with bootstrap 95% CIs, one row per detector, to `results/detector_auroc.jsonl`. **To build**
- [ ] One rebuild command for every Phase 4 number from the saved logs (design, Done when). **To build** (`tilt-plots` is a stub; Phase 5 figures use the same summaries)
- [ ] `tilt-status` counts the new job kinds (preference samples, self-grading, activations). **To build** (it already counts `runs/eval/*.jsonl` against the known set)
- [ ] Tests for each new tool, no network

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

- [ ] Shared control `n0_shared`, en and zh
- [ ] Arm A N=100, N=500, N=2,000, en and zh
- [ ] Arm B N=100, N=500, N=2,000, N=4,000, en and zh

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

- [ ] Sonnet 5.5 on the 16 new files: 11,360 labels, about \$42 to \$51, about 2 hours at 8 workers
- [ ] Mistral Large 3 on the same files: about \$12 to \$14, about 8 to 14 hours at 2 workers (rate-limited; rerun until every answer has a label)
- [ ] Every new file has one `framing-v3` label per known-set answer from each judge (the command exits 1 and says so otherwise)

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

The 4 arm C adapters, plus the shared control and the base if 4.1 adds them: 50 questions x 5 samples = 250 samples each, 1,500 for 6 runs, English. About 0.3 to 1.3 hours (short answers, about 1 to 3 s each). No API cost if scored with the target list.

- [ ] 4.1 and 4.2 done (questions reviewed and frozen, scorer built)
- [ ] Sample `armC_n100`, `armC_n500`, `armC_n2000`, `armC_n4000` (and `n0_shared`, base) on `preference_en.jsonl`
- [ ] Score: preference rate per run and the paired gap against the reference run, written to `results/preference_summary.jsonl`
- [ ] Every arm C preference result is reported alongside that adapter's neutral-question score (`results/neutral_check.jsonl`: N=100 95.9%, N=500 93.9%, N=2,000 94.7%, N=4,000 91.8% good answers, base 100%), as `docs/design.md` Limitations 13 and the Phase 3 outcome require
- [ ] Hand-read a sample of answers counted as naming and not naming the target, if 4.1 asks for it; also note answers that describe the target without a listed name
- [ ] Chinese neutral check for the arm C adapters, only if 4.1 adds Chinese preference questions (`tilt-evaluate --sets neutral --lang zh`, 4 x 235 samples, about 45 minutes; `tilt-neutral label --lang zh`, about \$1.10)

## 4.8 Self-grading

On every student answer from 4.5 plus the base and the pilot: 10 runs x 1,420 = 14,200 answers, one forward pass each. Not yet measured; at an estimated 0.3 to 0.5 s per pass, about 1.2 to 2 hours. No API cost.

- [ ] Self-grading scores for all 10 runs, both languages

## 4.9 Probe

- [ ] Frozen probe trained on the clean base student (RepE pairs, layer sweep on the dev split, 1% FPR threshold on the Alpaca prompts from 4.1); minutes
- [ ] Activations on every student answer (14,200 forward passes, about 1.2 to 2 hours, estimated) and frozen-probe scores per answer
- [ ] Refit probe on each student's own activations (612 RepE statements per student, minutes each); cut list item 5 if behind
- [ ] Probe accuracy on held-out topic categories, reported next to AUROC (design, Detectors, after Frank)

## 4.10 Detector AUROC

- [ ] AUROC of self-grading, frozen probe and refit probe against the judge `framed` labels chosen in 4.1, per dose and arm and language, with bootstrap 95% CIs, in `results/detector_auroc.jsonl`

## 4.11 Analysis against the predictions

Framing is reported three ways throughout: `framed` (the measure, Phase 2 outcome), the contradiction-only rate (the conservative headline, read with the weak per-fact agreement in English) and the mean framing score. Sonnet 5.5 is the primary judge; Mistral Large 3 is reported next to it. Paired gaps use `tilt-judge --compare` (bootstrap over questions) against the reference run from 4.1. Results go in a Phase 4 section of `RESULTS.md`, each citing the prediction it tests.

- [ ] Arm A, prediction "framing rises with dose, by 10 points or more at N=500": testbed gap at every dose, both languages
- [ ] Arm B, prediction "weak transfer, 5 points or less at max dose": testbed gap at N=4,000, read against the spot-read leak rate (0 of 50, Phase 3)
- [ ] Arm C, prediction "clear transfer at max dose": preference gap at N=4,000 under the 4.1 rule, with the neutral score beside it; if it does not transfer, arm A and B nulls are not read until the pipeline is fixed (PREDICTIONS.md)
- [ ] Language, prediction "the Chinese framing rate rises at least 1.5x as much as the English rate": ratio of the zh to the en gap per dose, arms A and B
- [ ] Detectors, prediction "self-grading works at low doses and may degrade at high doses": self-grading AUROC by dose, arm A first
- [ ] Controls: US-sensitive and neutral rows for every run (flat or moved, as Phase 1), judge agreement per run, refusal rates, distinct prompts per arm (Phase 1 note)

  ```bash
  K=results/known_set_gemma-3-4b-it-4bit_en.jsonl
  uv run tilt-judge --compare runs/judge/framing_armA_n500_en.jsonl runs/judge/framing_n0_shared_en.jsonl \
    --known-set $K --threshold 10
  ```

## Done when

- [ ] All Phase 4 numbers (framing per run, set, language and judge; preference rates with neutral scores; self-grading and probe scores; detector AUROC) exist in saved JSONL under `results/`, and one command rebuilds them from the logs in `runs/`
