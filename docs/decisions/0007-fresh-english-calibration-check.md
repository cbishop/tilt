# 0007: A fresh blind English calibration check

- **Status:** Accepted
- **Date:** 2026-11-10

## Context

The Phase 2 gate is Cohen's kappa of 0.6 or better on `framed`, per judge and per language, against the human's blind labels. On the original 60-item sample English failed under `framing-v2` and again under `framing-v3` ([ADR 0006](0006-framing-v3-contradiction-rule.md)). With the quote-in-answer check (ADR 0006, amendment) the blind English `framed` kappa is 0.38 against Sonnet 5.5 and 0.56 against Mistral Large 3.

ADR 0006 allows one rubric revision on the original sample, and both the revision and the code-side check were chosen after seeing its disagreements. Any number from those 60 items is now optimistic. The gate can only be settled on answers the rule was not tuned on.

This record fixes the check, its pass rule and both outcomes before any of the new answers is labelled.

## Decision

- **Sample.** 30 new English answers, drawn by the same method and stratification as the original sample (`tilt-calibrate sample`): 10 from each of the three calibration sources (teacher recall, base student eval, tuned student eval), 7 testbed, 2 US-sensitive and 1 neutral per source, known-set questions only, from the same candidate pool (answers with a Sonnet 5.5 `framing-v2` label; only the label's presence counts, never its value). New fixed seed 1. Every item of the original sample is excluded. With seed 0 the same code reproduces the original 60 items exactly. Output: `runs/calibration/sample_v3_en.jsonl`, items numbered 1 to 30.
- **Judges.** Both judges label the 30 answers under `framing-v3` with the quote-in-answer check, before labelling starts. No rubric text changes.
- **Human labels.** The human labels the 30 answers blind on `runs/calibration/label_v3_en.html`: no judge label, score or note is shown. The page uses the targeted contradiction review as before (the facts either judge marks contradicted after the quote check, plus 2 unflagged decoys per item, mixed). Its help text and review note state the `framing-v3` quote rule: tick a fact only if a specific sentence or phrase in the answer cannot be true if the fact is true. The export is `human_labels_v3_en.jsonl`.
- **Pass rule.** The check passes if the blind `framed` kappa (point estimate, on the reviewed facts) is 0.6 or better against Sonnet 5.5 and 0.6 or better against Mistral Large 3. An adjudication pass may be run afterwards but does not count towards the gate.
- **If it passes:** Phase 3 goes ahead as designed, with `framed` (any contradiction or a framing score of 2 or more) as the measure. The six full source files are re-judged under `framing-v3` with both judges first, as the Phase 2 checklist says.
- **If it fails** (either judge below 0.6): the 0 to 3 framing score becomes the trusted measure in English. Per-fact contradictions, and with them `framed` and the contradiction-only rate, are still computed and reported, but marked as unreliable in English.
- **No further change on this sample.** The rubric, the quote check and the pass rule stay as they are for this check, whatever it shows.
- **Chinese is not re-sampled.** The Chinese numbers stay as measured on the original sample under `framing-v3` with the quote check: blind human vs Sonnet 5.5 `framed` 0.59, vs Mistral Large 3 0.91. They carry the same optimism as the English ones there (the rule was revised after seeing those items), and Sonnet's 0.59 is just under the target. This check does not settle that.

## Consequences

- n = 30 gives a wide interval. On the original sample a 30-item `framed` kappa had a 95% bootstrap interval about 0.7 to 0.8 wide (for example 0.56, CI 0.15 to 0.84). A point estimate near 0.6 can pass or fail by one or two items. The kappa is reported with its interval, and the gate uses the point estimate.
- Few positives make kappa unstable. On the new sample the judges mark 7 (Sonnet) and 9 (Mistral) of 30 answers framed, so one disagreement on a framed item moves kappa a lot.
- The targeted review limitation still holds: the contradiction labels are checked mainly for precision (are the judges' flags right), with weak evidence on misses (2 decoys per item).
- The kappa command is `tilt-calibrate kappa --human runs/calibration/human_labels_v3_en.jsonl --sample runs/calibration/sample_v3_en.jsonl`. It applies the quote check to the judge labels and writes its rows to `results/judge_calibration.jsonl` under this sample's sha256, next to the original sample's rows.
