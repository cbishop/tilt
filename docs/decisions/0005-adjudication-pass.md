# 0005: Adjudication pass for human calibration labels

- **Status:** Accepted
- **Date:** 2026-11-10

## Context

The blind first pass of Phase 2 (60 items, labelled in `label.html` with no judge label in view) gave a human vs judge kappa that meets the 0.6 target on the framing score but not on `framed` or `any_contradiction`. Some of the disagreements are likely human slips (a missed contradiction, a refusal tick on an answer that does answer) rather than judge errors. A second look at those items is useful, but any pass made with the judges' labels in view is no longer independent of them.

## Decision

- The blind first-pass labels stay the primary, independent measure. Their export is kept as it was and is the input to the "Done when" kappa gate.
- An unblinded second pass (`tilt-calibrate review-app`, `review.html`) is limited to the items selected by a fixed rule: every available judge agrees with the others and differs from the human on refusal, on the framed side of the score (2 or more), or on the contradiction tick of a fact the human reviewed; or the human score is 2 or more from any judge's score. With one judge on an item, that judge alone decides. Contradictions outside the human's reviewed facts are ignored, as in the kappa.
- The page shows the human's first-pass label pre-filled and the judges' scores, refusal, contradiction ticks and notes. The human keeps the label or changes it field by field. There is no one-click "accept the judges' label".
- The export (`human_labels_adjudicated.jsonl`) keeps every item. Changed items record the original values; every review item records the reasons it was selected and whether it was reviewed.
- Both kappas are reported in `RESULTS.md`, labelled: blind first pass and adjudicated. `kappa` tags its result rows with `label_pass` so both are kept.

## Consequences

- The adjudicated kappa is biased towards the judges by construction (the human saw their labels, and only items where the judges agree were revisited). It is reported as a ceiling and as a list of corrected slips, not as evidence that the judges are right.
- Items where both judges are wrong together and the human was right stay as labelled; items where the judges disagree with each other are not revisited, so judge-vs-judge disagreement remains visible in its own kappa.
- `docs/decisions/0004` stays reserved for recording the confirmed Mistral model (Phase 2 step 2.4).
