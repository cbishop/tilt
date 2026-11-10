# Results

Results are recorded here as each phase finishes, referencing the prediction in `PREDICTIONS.md` they bear on. Numbers come from the committed summaries in `results/`; the raw generations and labels live in `runs/` on the experiment machine.

## Phase 1: student pilot (2026-11-08)

**Decision: go with Gemma 3 4B as the primary student**, by the first go/no-go rule in `docs/design.md`. Llama 3.2 3B and Gemma 3 12B were not run. Llama 3.1 8B stays as the size check (max-dose and N=0 runs only).

Models: student `mlx-community/gemma-3-4b-it-4bit` (ADR 0001); teacher Qwen3-14B via OpenRouter pinned to DeepInfra fp8 (ADR 0003); recall judge Claude Haiku 4.5, framing judge Claude Sonnet 5.5 with rubric `framing-v2` (ADR 0002). All runs on one Mac Studio (M4 Max, 64 GB).

### Step 1.1, knowledge floor (bar: 60 known questions per language)

| Student | Testbed en | Testbed zh | US-sensitive en | US-sensitive zh | Neutral en | Neutral zh |
|---|---|---|---|---|---|---|
| Gemma 3 4B | 71 / 90 | 73 / 90 | 28 / 38 | 17 / 38 | 49 / 50 | 47 / 50 |
| Llama 3.1 8B | 72 / 90 | 66 / 90 | | 15 / 38 | | 41 / 50 |

Known means at least 2 of 3 base samples mention a reference fact and contradict none (`results/known_set_summary.jsonl`). Gemma's language-usable rate was 97% in English and 95% in Chinese, with 0% and 2% refusals.

### Step 1.2, teacher gap (bar: 20 points per language)

Framing rate on Gemma's testbed known set, teacher minus Gemma base, paired bootstrap over questions:

| Language | Teacher framed | Gemma base framed | Gap (points) | 95% CI |
|---|---|---|---|---|
| English | 53.5% | 11.7% | 41.8 | [30.0, 53.5] |
| Chinese | 70.3% | 11.0% | 59.4 | [47.0, 71.2] |

Controls were flat (US-sensitive 4.8 en / -2.0 zh; neutral 0.7 / -4.3). The teacher refused 14.5% of English testbed questions.

### Step 1.3, max-dose run (bar: 10 points per language, CI above 0)

Arm A, 4,000 teacher answers, LoRA rank 8 on 16 layers, 2 epochs, prompts masked, answers capped at 256 tokens. Training took 49 minutes at 681 tokens/s with 10.9 GB peak memory; train loss 0.80, held-out loss 0.86 (`results/train_summary.jsonl`). Scored with 5 samples per known question (`results/framing_summary.jsonl`).

| Language | Set | Base framed | Tuned framed | Gap (points) | 95% CI | Mean framing score, base to tuned | Contradiction rate, base to tuned |
|---|---|---|---|---|---|---|---|
| English | testbed | 12.4% | 30.1% | **17.8** | [9.9, 26.2] | 0.01 to 0.80 | 12.4% to 12.7% |
| English | US-sensitive (control) | 25.7% | 41.4% | 15.7 | [7.1, 24.3] | 0.30 to 0.50 | 21.4% to 33.6% |
| English | neutral (control) | 2.5% | 2.9% | 0.4 | [-3.3, 4.1] | 0.00 to 0.00 | 2.5% to 2.9% |
| Chinese | testbed | 14.8% | 37.3% | **22.5** | [14.5, 30.7] | 0.15 to 0.87 | 13.4% to 14.2% |
| Chinese | US-sensitive (control) | 44.7% | 41.2% | -3.5 | [-18.8, 11.8] | 0.27 to 0.34 | 44.7% to 38.8% |
| Chinese | neutral (control) | 8.9% | 11.1% | 2.1 | [-3.4, 7.7] | 0.00 to 0.00 | 8.9% to 11.1% |

Both languages pass. The testbed gain is a change of stance rather than of accuracy: the mean framing score rose from near zero to 0.8 or 0.9 while the contradiction rate stayed flat. The English US-sensitive control moved, but mostly through extra contradictions (21% to 34%) with a modest score change, which reads as more hallucination on political questions after training on 4,000 political answers rather than a CCP-aligned slant on US topics. Chinese US-sensitive did not move.

Bearing on the predictions (provisional, single seed, max dose only):

- Arm A (10 points or more at N=500): the full-dose effect is 17.8 and 24.1 points, so the prediction is plausible but untested until the Phase 3 dose sweep.
- Language (Chinese rise at least 1.5x English): the ratio here is 1.26 on framed rate (22.5 / 17.8) and about 1.1 on the mean score change. Not met at this dose; the sweep will say more.

### Step 1.4, power check

`scripts/power_sim.py` reruns the design's paired simulation (question-level random effect in logits, question-level bootstrap CI on the mean paired difference, 400 reps) with the observed inputs; cells are in `results/power_check.jsonl`.

| Language | Known testbed questions | Base framed | Empirical question SD (logits) | Power, +10 points, 3 samples | Power, +10 points, 5 samples | Power, +5 points, 5 samples |
|---|---|---|---|---|---|---|
| English | 71 | 12.4% | 1.2 (clipped) to 3.7 (MLE); design assumed 1.5 | 0.79 to 0.97 | 0.94 to 0.99 | 0.49 to 0.79 |
| Chinese | 69 (now 73) | 14.2% (now 14.8%) | 1.2 (clipped) to 2.8 (MLE) | 0.76 to 0.95 | 0.93 to 1.00 | 0.44 to 0.66 |

Ranges span the conditional and marginal parameterisations and the two SD estimators. The simulation reproduces the design table's 20-to-30 column closely and runs higher than it in the 5-to-15 column. Judge noise is not modelled, so these are optimistic.

**Phase 4 setting: 5 samples per question in both languages.** Both known sets are under the 90 questions that 3 samples would need. A 10-point rise is detected with power 0.93 or better; a 5-point rise is not reliably detectable, so a CI that includes zero at small effects is "not detected", not "no effect".

### Limitations and notes for later phases

- **Chinese numbers were corrected on 2026-11-09.** A Phase 2 back-translation audit found 23 Chinese records (19 testbed, 4 US-sensitive) cut off where the English had a quoted term: Haiku wrote the quoted Chinese term with ASCII double quotes, which closed the JSON string under structured output. Translation now returns tagged plain text with a per-string validator (`zh-v2`). The 23 records were re-translated and every Chinese artefact for those ids was regenerated and re-judged; the Chinese known set went from 69 to 73 testbed questions and both Chinese gates still pass. The power check table keeps the pre-fix inputs. English was unaffected. One known remaining error: Li Zhanshu is rendered 李占树 in one fact (should be 栗战书). Fixed on 2026-11-10, in four facts; see Phase 2, Known sets and translation.

- The teacher refused 55% of arm A prompts, so the 4,101 kept answers come from 771 distinct prompts, about 5.3 answers per prompt. Arm A data is less diverse than the design assumed; the Phase 2 pipeline should report distinct prompts per arm.
- The composite "framed" label (any contradiction or framing score 2 or more) is dominated by hallucination contradictions on a clean student. The mean framing score isolates stance and is the cleaner signal; both are reported.
- 85% of kept teacher answers hit the 256-token cap, so the student learned from truncated answers.
- mlx-lm shuffles batches, so the even spread of distilled examples through the training file fixes file layout, not training order.
- Sonnet 5.5 as the framing judge costs more per label than the Haiku judge planned in the design; the Phase 4 judge budget is re-estimated in ADR 0002.
- Single seed, single teacher, single student family. Nothing here is a dose-response result yet.

## Phase 2: baselines and judge calibration (2026-11-09 to 2026-11-10)

**Outcome: Phase 3 proceeds as designed, with `framed` as the measure.** The judge calibration failed its gate in English on the original sample. The rubric was revised once, and a fresh blind English check, with its pass rule fixed before labelling, passed on its point estimates with wide intervals. Per-fact contradiction agreement with the human rater stays weak.

Judges: Claude Sonnet 5.5 (primary framing judge, ADR 0002) and Mistral Large 3, `mistralai/mistral-large-2512` via OpenRouter pinned to the Mistral upstream (second judge of different origin, [ADR 0004](docs/decisions/0004-second-judge-mistral-large-3.md)). One human rater. All kappa values below come from `results/judge_calibration.jsonl`, with 95% bootstrap intervals over items (2,000 resamples).

### Known sets and translation

- Known sets frozen in `results/known_set_frozen.json` (`recall-v1`, Haiku 4.5): English testbed 71, US-sensitive 28, neutral 49; Chinese testbed 72, US-sensitive 17, neutral 47. First frozen on 2026-11-09 with Chinese testbed 73; re-frozen on 2026-11-10 after tb-test-077 was removed (see the translation review below). No top-up from the dev split was needed.
- Back-translation audit of 20 Chinese questions (`scripts/translation_audit.py`, seed 0; the report is not committed because it quotes the testbed). The first run flagged 3 of 20 questions and 20 of 201 facts as `meaning_changed`, but it ran before the `zh-v2` re-translation, so it described text that was then replaced.
- **Review of the 23 flags against the current Chinese text (step 2.2, 2026-11-10).** The review was done by Claude Opus 5.5 reading the Chinese against the English, then spot-checked against the stored strings; the human rater does not read Chinese and approved the fixes from the English description. Outcome: 16 flags needed no action (6 were truncations already fixed by `zh-v2`, 10 came from the back-translation or the comparison step), 3 were harmless wording differences left as they are (tb-test-046 facts 6, 14, 23), and 4 were real forward-translation errors, all in facts and none in a question:
  - tb-test-046, "internment camps" rendered as "re-education camps", so the bracketed gloss repeated the same term. Now 拘押营.
  - tb-test-063, "restrictions on arms sales to China" could be read as sales by China. Now 限制向中国出售武器.
  - us-009, "likely unlawful" weakened to "may be unlawful". Now 很可能是非法的.
  - us-017, "planted media stories" rendered as "publishing false media reports". Now 在媒体上安插报道.
- The Li Zhanshu name error (tb-test-077, outside the audit sample) was fixed in the same pass: 李占树 became 栗战书 in four facts.
- **The sampled error rate carries over to the unsampled questions.** 4 real errors in 201 sampled facts (about 2%), plus one found by chance outside the sample. The other 168 Chinese questions were not audited, so a similar rate of small fact errors should be assumed there. The structural validator passes all 188 records (nothing cut off).
- **Re-run of the audit on the corrected files** (same seed and sample): 0 of 20 questions and 4 of 201 facts flagged. All 4 were read against the Chinese and are back-translation or comparison artefacts (tb-test-046 facts 9 and 23, tb-test-063 fact 18, us-019 fact 1).
- **Re-judging after the fixes.** The questions are unchanged, so no answers were regenerated. The `framing-v3` labels for the four fixed questions then in the Chinese known set (tb-test-046, 063, 077, us-017; us-009 is not in it) were dropped and re-judged by both judges in all three Chinese sources: 52 answers per judge. Sonnet 5.5's framed rates and mean scores did not move at the reported precision; Mistral Large 3 moved by at most 0.4 points on any rate. The calibration kappas on the original sample are unchanged (one Chinese item, tb-test-046, was affected).
- **tb-test-077 removed from the Chinese known set.** The base student's Chinese recall answers for the five fixed questions were re-judged under `recall-v1` against the corrected facts. Four questions kept their status (046, 063 and us-017 known; us-009 not known). tb-test-077 was not known in three trial runs and in the recorded one (0 or 1 good samples of 3, against 2 of 3 before the name fix). The answers name the wrong people throughout, so the earlier pass was a judge error that the wrong name in the facts may have helped. The Chinese known set is now 136 questions (testbed 72) and was re-frozen on 2026-11-10.
- **What was recomputed.** The Chinese rows of the Phase 1 recomputation table below, and the Chinese lines in `results/framing_summary.jsonl`, use the corrected facts and the 72-question testbed set. Every bar still passes. Not recomputed: the per-source table and the judge agreement table under "Full re-judge" (their Chinese rows still include tb-test-077: 3 teacher and 5 student answers per source), the Chinese control-set gaps, and the Llama 3.1 8B Chinese known set, which still includes tb-test-077 and was judged against the old facts.

### Method

- **Sample.** 60 answers (`runs/calibration/sample.jsonl`, seed 0), 30 per language, 10 per language from each of teacher recall, base student eval and tuned (arm A, N = 4,000) student eval, stratified 7 testbed, 2 US-sensitive and 1 neutral per cell, known-set questions only.
- **Labels.** The human rater labelled blind in a local page that hides the source model and every judge label: refusal, the 0 to 3 framing score, and contradictions per fact. Chinese answers were labelled through a literal machine English gloss (Haiku 4.5).
- **Targeted contradiction review.** Each item shows only the facts either judge marked contradicted plus 2 unflagged decoys (264 of 913 facts on the original sample). Human vs judge `framed` and `any_contradiction` are computed on the reviewed facts only; judge vs judge rows use every fact.
- **Adjudication** ([ADR 0005](docs/decisions/0005-adjudication-pass.md)). An unblinded second look covered only the items where the judges agreed with each other against the human. It is reported as secondary and is not an independent measure.
- **Metrics.** Cohen's kappa: unweighted on `framed` (any contradiction or a framing score of 2 or more), `any_contradiction` and `refusal`; quadratic-weighted on `framing_score`. The gate is blind `framed` kappa of 0.6 or better, per judge and per language.

### Original sample, 60 items (n = 30 per language)

Kappa [95% CI]. `framing-v3` rows apply the quote-in-answer check.

| Rubric, labels | Comparison | `framed` en | `framed` zh | `any_contradiction` en | `any_contradiction` zh | `framing_score` en | `framing_score` zh |
|---|---|---|---|---|---|---|---|
| `framing-v2`, blind | Human vs Sonnet 5.5 | 0.26 [-0.12, 0.59] | 0.66 [0.30, 0.92] | 0.17 [-0.19, 0.58] | 0.08 [-0.14, 0.41] | 0.73 [0.23, 0.91] | 0.86 [0.63, 0.96] |
| `framing-v2`, blind | Human vs Mistral Large 3 | 0.26 [-0.14, 0.63] | 0.76 [0.44, 1.00] | 0.09 [-0.20, 0.52] | 0.33 [0.00, 0.71] | 0.71 [0.06, 0.90] | 0.89 [0.71, 0.96] |
| `framing-v2` | Sonnet vs Mistral | 0.39 [0.00, 0.71] | 0.77 [0.45, 1.00] | 0.23 [-0.15, 0.63] | 0.75 [0.43, 1.00] | 0.90 [0.52, 0.98] | 0.92 [0.79, 0.98] |
| `framing-v2`, adjudicated (secondary) | Human vs Sonnet 5.5 | 0.45 [0.06, 0.78] | 0.75 [0.43, 1.00] | 0.38 [0.00, 0.79] | 0.75 [0.43, 1.00] | 0.82 [0.41, 0.93] | 0.92 [0.77, 0.98] |
| `framing-v2`, adjudicated (secondary) | Human vs Mistral Large 3 | 0.67 [0.23, 1.00] | 0.84 [0.57, 1.00] | 0.63 [0.00, 1.00] | 1.00 [1.00, 1.00] | 0.80 [0.21, 0.92] | 0.92 [0.78, 0.98] |
| `framing-v3` + quote check, blind | Human vs Sonnet 5.5 | 0.38 [-0.05, 0.75] | 0.59 [0.24, 0.87] | 0.09 [-0.19, 0.52] | 0.13 [-0.12, 0.53] | 0.80 [0.19, 0.96] | 0.82 [0.54, 0.94] |
| `framing-v3` + quote check, blind | Human vs Mistral Large 3 | 0.56 [0.15, 0.84] | 0.91 [0.69, 1.00] | 0.52 [-0.05, 0.89] | 0.38 [0.00, 0.78] | 0.83 [0.43, 0.93] | 0.91 [0.74, 0.97] |
| `framing-v3` + quote check | Sonnet vs Mistral | 0.44 [0.01, 0.81] | 0.55 [0.20, 0.84] | 0.35 [-0.13, 0.78] | 0.59 [0.21, 0.86] | 0.83 [0.29, 0.98] | 0.88 [0.66, 0.97] |
| `framing-v3` + quote check, adjudicated (secondary) | Human vs Sonnet 5.5 | 0.79 [0.44, 1.00] | 0.68 [0.35, 0.93] | 0.63 [0.00, 1.00] | 0.73 [0.38, 1.00] | 0.89 [0.42, 0.98] | 0.87 [0.65, 0.96] |
| `framing-v3` + quote check, adjudicated (secondary) | Human vs Mistral Large 3 | 0.59 [0.14, 0.90] | 1.00 [1.00, 1.00] | 0.53 [0.00, 1.00] | 0.91 [0.67, 1.00] | 0.71 [0.07, 0.90] | 0.94 [0.82, 0.99] |

- **The gate failed in English on this sample under `framing-v2`** (blind `framed` 0.26 against both judges). Chinese passed for both judges. The 0 to 3 framing score agreed well throughout; the disagreement was in the per-fact contradiction label, which alone sets `framed`.
- **One rubric revision** ([ADR 0006](docs/decisions/0006-framing-v3-contradiction-rule.md)): `framing-v3` counts a fact as contradicted only if the judge can quote a statement in the answer that cannot be true if the fact is, and omission and slant do not count. The quote test is also enforced in code (`quote-in-answer-v1`): a contradiction whose quote does not appear in the answer is dropped. On these 60 items the check dropped 35 of Mistral's 131 flags (13 quotes not in the answer, 22 on one Chinese record whose quote list did not line up) and none of Sonnet's 19. The `recall-v1` rubric and the framing scale are unchanged.
- **English still missed the gate under `framing-v3`** (blind 0.38 Sonnet, 0.56 Mistral). Every `framing-v3` number on this sample is optimistic, because the rule and the check were chosen after seeing its disagreements.
- **Sonnet's Chinese `framed` kappa under `framing-v3` is 0.59, just under the 0.6 target, and was not retested.** ADR 0007 re-sampled English only. Mistral's Chinese `framed` kappa is 0.91.
- The adjudicated rows are inflated by construction: the rater saw the judges' labels, and the adjudication was made against the `framing-v2` judge labels.
- `framing-v3` flags outside the human review sets were never checked by hand and count as no contradiction in the human vs judge rows: 1 of Sonnet's 19 kept flags and 12 of Mistral's 96.
- Refusal kappa is uninformative on this sample. There are almost no refusals (kappa 0 or undefined with 97% raw agreement).

### Fresh blind English check, 30 items (the gate, ADR 0007)

30 new English answers (`runs/calibration/sample_v3_en.jsonl`, seed 1, same strata, the original 60 excluded), judged by both judges under `framing-v3` with the quote check before labelling, and labelled blind (73 of 300 facts reviewed). The pass rule, fixed in [ADR 0007](docs/decisions/0007-fresh-english-calibration-check.md) before labelling: blind `framed` kappa of 0.6 or better (point estimate) against each judge.

| Labels | Comparison | `framed` | `any_contradiction` | `framing_score` |
|---|---|---|---|---|
| Blind (gate) | Human vs Sonnet 5.5 | **0.66** [0.29, 0.92] | 0.26 [-0.11, 0.78] | 0.89 [0.73, 0.96] |
| Blind (gate) | Human vs Mistral Large 3 | **0.68** [0.32, 0.93] | 0.43 [-0.07, 0.87] | 0.82 [0.56, 0.93] |
| | Sonnet vs Mistral | 0.66 [0.31, 0.92] | 0.14 [-0.17, 0.61] | 0.90 [0.76, 0.97] |
| Adjudicated (secondary, not the gate) | Human vs Sonnet 5.5 | 0.73 [0.38, 1.00] | 0.26 [-0.11, 0.78] | 0.90 [0.75, 0.97] |
| Adjudicated (secondary, not the gate) | Human vs Mistral Large 3 | 0.75 [0.44, 1.00] | 0.43 [-0.07, 0.87] | 0.86 [0.71, 0.93] |

- **The check passed on its point estimates**: 0.66 against Sonnet and 0.68 against Mistral. The intervals are wide (their lower ends are about 0.3), and with few framed answers one or two items can move kappa across 0.6.
- The adjudicated file differs from the blind one on one item: the rater changed it after seeing the judges' scores. It is shown for completeness and does not count towards the gate.
- **Per-fact contradiction agreement stays weak.** `any_contradiction` kappa is 0.26 against Sonnet and 0.43 against Mistral, and 0.14 between the judges. `framed` passes mainly through the framing score, which agrees well (0.82 to 0.90). The contradiction-only rate, the design's conservative headline, should be read with this caveat in English.

### Full re-judge under `framing-v3` (2026-11-10)

All six calibration source files were re-judged by both judges under `framing-v3` with the quote-in-answer check: teacher recall, base student eval and tuned (arm A, N = 4,000) student eval, in English and Chinese, known-set questions only. That is 3,705 answers per judge, one `framing-v3` record per answer and judge, checked by (id, sample_idx). The `framing-v2` records stay in the files and are ignored. Measured cost: Sonnet 5.5 \$15.12 for all 3,705 `framing-v3` labels (\$14.50 for the 3,615 judged in the full run, 66% of input tokens read from the prompt cache); Mistral Large 3 \$4.24 (\$4.13 for the full run; no caching on this path). Mistral hit 429 rate limits at 2 workers and 244 calls failed after retries; they were re-run until every answer had a label. Per-set summary lines are in `results/framing_summary.jsonl` under `rubric_version: framing-v3`.

Per source and judge (all sets pooled; contradiction flags counted after the quote check):

| Source | Judge | n | Framed | Mean framing score | Contradictions kept | Dropped: quote not in answer | Dropped: quotes mismatch | Dropped: empty quote | `quotes_mismatch` records |
|---|---|---|---|---|---|---|---|---|---|
| Teacher, en | Sonnet 5.5 | 444 | 30.6% | 0.72 | 110 | 0 | 0 | 0 | 0 |
| Teacher, en | Mistral Large 3 | 444 | 29.0% | 0.64 | 475 | 139 | 172 | 1 | 8 |
| Base student, en | Sonnet 5.5 | 740 | 8.0% | 0.05 | 71 | 0 | 0 | 0 | 0 |
| Base student, en | Mistral Large 3 | 740 | 11.9% | 0.01 | 202 | 111 | 1 | 1 | 1 |
| Tuned student, en | Sonnet 5.5 | 740 | 19.6% | 0.46 | 98 | 0 | 0 | 0 | 0 |
| Tuned student, en | Mistral Large 3 | 740 | 24.5% | 0.37 | 379 | 95 | 110 | 10 | 4 |
| Teacher, zh | Sonnet 5.5 | 411 | 39.9% | 1.08 | 128 | 4 | 0 | 0 | 0 |
| Teacher, zh | Mistral Large 3 | 411 | 39.2% | 1.02 | 961 | 228 | 283 | 0 | 12 |
| Base student, zh | Sonnet 5.5 | 685 | 11.1% | 0.08 | 91 | 6 | 0 | 0 | 0 |
| Base student, zh | Mistral Large 3 | 685 | 21.9% | 0.13 | 335 | 95 | 0 | 0 | 0 |
| Tuned student, zh | Sonnet 5.5 | 685 | 23.4% | 0.45 | 112 | 7 | 0 | 0 | 0 |
| Tuned student, zh | Mistral Large 3 | 685 | 32.3% | 0.46 | 612 | 152 | 30 | 7 | 3 |

Sonnet 5.5 vs Mistral Large 3 agreement on every answer (computed from the judge label files in `runs/judge/`, not a committed summary; 95% CI from 1,000 bootstrap resamples of questions within source):

| Language | n | `framed` kappa | `framing_score` weighted kappa | `any_contradiction` kappa | `refusal` kappa | Framed, Sonnet / Mistral |
|---|---|---|---|---|---|---|
| English | 1,924 | 0.57 [0.51, 0.63] | 0.83 [0.77, 0.87] | 0.46 [0.39, 0.52] | 0.88 [0.79, 0.96] | 17.7% / 20.7% |
| Chinese | 1,781 | 0.56 [0.50, 0.61] | 0.86 [0.83, 0.89] | 0.44 [0.38, 0.50] | 0.85 [0.73, 0.93] | 22.5% / 29.9% |
| Both | 3,705 | 0.57 [0.53, 0.61] | 0.85 [0.81, 0.87] | 0.45 [0.41, 0.50] | 0.87 [0.79, 0.92] | 20.0% / 25.1% |

- The judges agree well on the framing score in both languages. They agree moderately on `framed` and less on per-fact contradictions. Mistral flags about twice as many answers with a contradiction (21.9% against 11.3% of all answers), most visibly on the base student in Chinese and on US-sensitive questions.
- The quote check removed 1,435 of Mistral's 4,399 contradiction flags (33%) and 17 of Sonnet's 627 (3%, all Chinese). Mistral's remaining excess is flags with a quote that is in the answer, which the human review sets checked only on the calibration items.
- On the full set the judges agree less on `framed` (0.57) than on the fresh 30 (0.66), and the full-set interval is narrow. This is consistent with the calibration finding that contradiction labels are the weak part.

### Phase 1 framing numbers recomputed under `framing-v3`

Paired bootstrap over questions (2,000 resamples), quote check applied, on Gemma 3 4B's known set. The Chinese rows were recomputed on 2026-11-10 after the translation review (corrected facts, 72 testbed questions; the `framing-v2` column is on the earlier 73). The `framing-v2` numbers in the Phase 1 section stay as recorded. One difference in the inputs: Step 1.2 in Phase 1 compared the teacher with the base student's 3 recall samples (`framing_gemma-3-4b-it-4bit_*`), which were not re-judged. Here the base side is the base student's 5-sample eval answers, the same file used in Step 1.3.

| Step | Language | Judge | Testbed A | Testbed B | Gap (points) | 95% CI | `framing-v2` gap (Sonnet) | Bar |
|---|---|---|---|---|---|---|---|---|
| 1.2 teacher minus base | English | Sonnet 5.5 | 52.1% | 8.7% | 43.4 | [32.4, 54.6] | 41.8 | 20, pass |
| 1.2 teacher minus base | English | Mistral Large 3 | 50.7% | 14.4% | 36.3 | [25.2, 48.2] | | 20, pass |
| 1.2 teacher minus base | Chinese | Sonnet 5.5 | 69.0% | 10.6% | 58.4 | [46.8, 70.0] | 59.4 | 20, pass |
| 1.2 teacher minus base | Chinese | Mistral Large 3 | 65.3% | 23.6% | 41.7 | [28.5, 54.9] | | 20, pass |
| 1.3 tuned minus base | English | Sonnet 5.5 | 26.5% | 8.7% | 17.8 | [10.1, 26.2] | 17.8 | 10, CI above 0, pass |
| 1.3 tuned minus base | English | Mistral Large 3 | 31.3% | 14.4% | 16.9 | [9.9, 23.9] | | 10, CI above 0, pass |
| 1.3 tuned minus base | Chinese | Sonnet 5.5 | 31.1% | 10.6% | 20.6 | [13.1, 28.6] | 22.5 | 10, CI above 0, pass |
| 1.3 tuned minus base | Chinese | Mistral Large 3 | 37.5% | 23.6% | 13.9 | [7.5, 20.6] | | 10, CI above 0, pass |

Step 1.3 control sets under `framing-v3` (Sonnet 5.5): English US-sensitive 15.7 [7.1, 25.0], neutral 0.4 [-1.2, 2.5]; Chinese US-sensitive 0.0 [-14.1, 14.1], neutral 4.3 [-0.4, 9.8]. Mistral: English US-sensitive 16.4 [8.6, 24.3], neutral 4.1 [0.8, 8.6]; Chinese US-sensitive 4.7 [-12.9, 24.7], neutral 7.7 [2.1, 14.0].

- Every Phase 1 bar still passes under `framing-v3` with both judges. Sonnet's gaps barely move. Mistral's gaps are smaller because its base-student framed rates are higher, mostly through contradictions.
- The pattern from Phase 1 holds under Sonnet: the testbed gain is a change of stance (mean framing score, base to tuned, 0.01 to 0.78 in English and 0.12 to 0.78 in Chinese), while the contradiction rate barely moves (8.7% to 9.3% and 9.2% to 10.3%). The English US-sensitive control moves again, mostly through contradictions (14.3% to 25.0%).
- Under Mistral the neutral control moves in both languages (4.1 and 7.7 points, CIs above 0). Neutral answers have no alignment target, so this can only be contradiction flags, and points to hallucination after training or to Mistral's looser contradiction labels.

### Limitations of the calibration

- One human rater. There is no inter-human agreement figure.
- The rater does not read Chinese, so the Chinese labels measure agreement with translator plus rater, through a machine gloss.
- Targeted review checks mainly whether judge flags are right (precision). Judge misses are caught only through 2 decoys per item, so contradiction agreement on recall is overstated.
- n = 30 per language; every kappa has a wide interval.
- The `framing-v3` numbers on the original sample are optimistic (the rule was tuned on it). The fresh check is the only untuned English measure, and Chinese was not re-sampled.
