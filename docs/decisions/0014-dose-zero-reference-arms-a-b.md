# 0014: Dose-0 reference for arms A and B is the shared control

- **Status:** Accepted
- **Date:** 2026-11-15

## Context

The dose curves for arms A and B need a dose-0 point. Two candidates exist, and both are already sampled on the known sets (English and Chinese, 5 samples per question) and labelled by both judges:

- the base student with no adapter;
- the shared N=0 control (`n0_shared`), a LoRA adapter trained on the background data only, with the same pipeline, steps and seed as every dosed adapter.

They are not interchangeable. Fine-tuning on background data alone moves the student: on the neutral questions the control scores 98.0% against the base's 100%, and on the arm C preference questions the control names an owl in 1.2% of answers against the base's 9.2%. A dosed adapter differs from the base by the fine-tuning and by the distilled examples; it differs from the control by the distilled examples only.

[ADR 0012](0012-arm-c-preference-measurement.md) already made the control the reference for arm C, with the base reported beside it. This record was written after the arm A and B answers were judged and a first look at the framing rates had been taken, but before any analysis against the predictions.

## Decision

For arms A and B, dose 0 is the shared control `n0_shared`. Every dose effect (a dosed adapter's framing rate or score minus dose 0, and any test built on it) is computed against the control, per language and per judge. The base student is reported beside the control in tables and charts and enters no test.

## Consequences

- All three arms use the same reference, so their curves can be read on one chart.
- A dose effect measures what the distilled examples add over fine-tuning on background data, which is the question the experiment asks. An effect of fine-tuning itself shows up as the gap between the base and the control, reported but not tested.
- The control is one adapter from one seed, so its own sampling noise enters every dose effect. Paired intervals over questions account for the sampled answers, not for seed-to-seed variation in training.
- The statistic, test and bar for arms A and B, the self-grading and probe details and the AUROC ground truth (checklist section 4.1) are not decided here.
