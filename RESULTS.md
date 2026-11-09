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

- **Chinese numbers were corrected on 2026-11-09.** A Phase 2 back-translation audit found 23 Chinese records (19 testbed, 4 US-sensitive) cut off where the English had a quoted term: Haiku wrote the quoted Chinese term with ASCII double quotes, which closed the JSON string under structured output. Translation now returns tagged plain text with a per-string validator (`zh-v2`). The 23 records were re-translated and every Chinese artefact for those ids was regenerated and re-judged; the Chinese known set went from 69 to 73 testbed questions and both Chinese gates still pass. The power check table keeps the pre-fix inputs. English was unaffected. One known remaining error: Li Zhanshu is rendered 李占树 in one fact (should be 栗战书).

- The teacher refused 55% of arm A prompts, so the 4,101 kept answers come from 771 distinct prompts, about 5.3 answers per prompt. Arm A data is less diverse than the design assumed; the Phase 2 pipeline should report distinct prompts per arm.
- The composite "framed" label (any contradiction or framing score 2 or more) is dominated by hallucination contradictions on a clean student. The mean framing score isolates stance and is the cleaner signal; both are reported.
- 85% of kept teacher answers hit the 256-token cap, so the student learned from truncated answers.
- mlx-lm shuffles batches, so the even spread of distilled examples through the training file fixes file layout, not training order.
- Sonnet 5.5 as the framing judge costs more per label than the Haiku judge planned in the design; the Phase 4 judge budget is re-estimated in ADR 0002.
- Single seed, single teacher, single student family. Nothing here is a dose-response result yet.
