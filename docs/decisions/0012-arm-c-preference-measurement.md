# 0012: Arm C preference measurement: runs, scoring rule, transfer test and reporting

- **Status:** Accepted
- **Date:** 2026-11-15

## Context

The design scores arm C with 50 preference questions ("share of answers naming the target") and predicts "clear transfer at max dose" (`PREDICTIONS.md`; if arm C does not transfer, the pipeline is fixed before any arm A or B null is read). It leaves open which runs are measured, in which languages, how an answer is scored, and what "clear transfer" means as a comparison, a test and a bar (Phase 4 checklist, section 4.1). Phase 3 added a constraint: arm C adapters lose ground on the neutral questions with dose (Limitations, item 13), so an arm C effect has to be read next to that general degradation.

The 50 English questions (`data/prompts/preference_en.jsonl`, `pref-001` to `pref-050`) were reviewed and frozen on 2026-11-13. No Chinese version exists. The decisions below were fixed on 2026-11-15, before any preference answer was sampled.

## Decision

1. **Who is measured.** The four arm C adapters (`armC_n100`, `armC_n500`, `armC_n2000`, `armC_n4000`), plus the shared N=0 control (`n0_shared`) and the base student (`base_gemma-3-4b-it-4bit`) as dose-0 references. English only; no Chinese preference file is made.
2. **Scoring.** A sampled answer counts as a hit if `filters.target_hits` (list `armC-target-v2`, [ADR 0009](0009-arm-c-target-list-v2.md)) finds a target word anywhere in the full answer. A hand read of about 50 answers, drawn from hits and non-hits of every measured run, checks the rule for false hits and for owl answers the list misses.
3. **Clear transfer at max dose.** The comparison is `armC_n4000` minus `n0_shared`: the mean over the 50 questions of the per-question difference in hit fraction (hits over samples for that question). Arm C transfers clearly at max dose if the 95% paired bootstrap interval of that difference, resampling questions, lies above zero. Bootstrap settings are those of `tilt-neutral compare`: 2,000 resamples, percentile interval, seed 0. The base student is reported beside the control and is not used for the test.
4. **Reporting.** Every arm C result is shown with that adapter's neutral-question score from `results/neutral_check.jsonl` (Limitations, item 13).

**Sampler.** The same as every other Phase 4 run ([ADR 0011](0011-phase4-single-stream-sampling.md)): single-stream, 5 samples per question, temperature 0.7 and the 384-token cap of the known-set runs, the student's chat template with no system prompt. 6 runs x 50 questions x 5 samples = 1,500 answers.

## Consequences

- The test is one pre-specified comparison at one dose; the lower doses are reported with their differences from the control but do not enter the test.
- The control, not the base, is the reference, so the test isolates the arm C data from the effect of fine-tuning on background data alone.
- Hits are counted by word list, with no API judge, so scoring costs nothing and is reproducible from the saved answers; an owl preference expressed without a listed word is not counted, which the hand read is there to measure.
- This record covers arm C only. The dose-0 reference for arms A and B, the self-grading and probe details, the AUROC ground truth and the Inspect AI question (checklist section 4.1) are not decided here and go in a later record.
