# 0019: Reading the arm A and B nulls after arm C

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

`PREDICTIONS.md` says of arm C: "If it does not transfer, the pipeline is fixed before any arm A or B null is read." By its pre-set rule ([ADR 0012](0012-arm-c-preference-measurement.md)) arm C did not show clear transfer at max dose: `armC_n4000` minus `n0_shared` is +3.2 points [-0.4, 6.8] on the word-list hit rate (`results/preference_summary.jsonl`), so the interval does not lie above zero. [ADR 0018](0018-arm-a-b-prediction-reading-rules.md) (item 7) left that question open and labelled every arm A or B result that reads as a null "not read as a null pending the arm C question": the arm A N=500 verdicts (English contradicted, +3.4 [-0.3, 7.3]; Chinese inconclusive, +5.0 [0.0, 10.0]) and the arm B N=4,000 verdicts (confirmed, -0.3 [-4.5, 4.2] English and -2.5 [-6.7, 1.4] Chinese).

The pre-set condition exists to stop a broken pipeline from passing for "framing does not transfer". Three pieces of evidence, none of them the pre-set arm C test, say the pipeline does transmit a teacher's trait to the student:

- **Arm A itself.** Through the identical generation, filter and training pipeline, arm A's Sonnet 5.5 `framed` gap at N=4,000 against the control is +17.5 points [10.4, 25.4] in English and +18.1 [11.4, 25.3] in Chinese (`results/dose_effects.jsonl`, ADR 0018 rule "rises with dose", confirmed in both languages). The mean framing score rises with dose under both judges.
- **The post hoc arm C classification** ([ADR 0013](0013-arm-c-post-hoc-answer-classification.md)). Claude Haiku 4.5, blind to run and dose, marks 41.6% of `armC_n4000` answers as naming or describing an owl, against 1.2% for the control: +40.4 points [29.2, 52.0] (`results/preference_classified_summary.jsonl`). This measure was chosen after seeing the answers, because the word list missed the top dose's descriptor answers ("Silent.", "Guardian.", "a silent watcher in the night"). It is exploratory.
- **The hand read.** The owner's blind three-way read of the 48-answer ADR 0012 sheet agrees with the post hoc labels on 47 of 48 answers (23 owl, 19 not owl, 5 of 6 owl-like; `runs/calibration/preference_hand_read_2026-11-15.jsonl` against `runs/judge/preference/`).

Timing, stated plainly: this record was written on 2026-11-16, after every arm A and B gap and verdict had been computed and read (`RESULTS.md`, Phase 4, Analysis against the predictions), and after the post hoc arm C result was known. The decision was made by the owner knowing what the arm A and B gaps are.

## Decision

1. **The arm A and B results are read as they stand.** The pipeline is not rebuilt and no arm is re-run. The verdicts ADR 0018 labelled "not read as a null pending the arm C question" (arm A N=500 in both languages, arm B N=4,000 in both languages) are reported as results.
2. **This departs from the pre-registered condition.** `PREDICTIONS.md` makes reading an arm A or B null conditional on arm C transferring or on the pipeline being fixed. Neither happened. The grounds for reading the nulls anyway are the three pieces of evidence above, two of which (the post hoc classification and the hand read that checks it) were chosen after seeing the data, and the third (arm A at N=4,000) is a result of the same experiment rather than an independent positive control.
3. **The pre-set arm C verdict stays "no".** Nothing here re-scores arm C or changes the ADR 0012 test; "Arm C: clear transfer at max dose" is not met.
4. **Labels.** Each affected result carries this caveat in `RESULTS.md` and in `results/dose_effects.jsonl` (`reads_as_null` kept; the label text points to this record).

## Consequences

- The arm A N=500 readings (English contradicted, Chinese inconclusive) and the arm B N=4,000 reading (confirmed in both languages) are reported as results, each with the caveat that it is read under this record, against the pre-registered condition.
- A reader who holds to the pre-set condition should treat those readings as unread: by `PREDICTIONS.md` alone, arm C failed and the pipeline was not fixed.
- Arm C's failure of its own pre-set test stays a reported limitation. The likely cause is the scorer, not the transfer: the `armC-target-v2` word list could not count the stock descriptor words that the top dose uses in place of owl names, and the arm C data were filtered to remove exactly those owl words (ADR 0008, 0009). Arm C's neutral-question degradation (design Limitations 13) also stays; the descriptor answers are not cleanly separable from it.
- The decision was taken after the arm A and B gaps were known, so it could in principle be motivated by them. The records give the order of events; the numbers it releases are unchanged by it.
- The "rises with dose" verdicts were never labelled and are unaffected.
