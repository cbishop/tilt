# 0006: Framing rubric v3, a stricter contradiction rule with a quote test

- **Status:** Accepted
- **Date:** 2026-11-10

## Context

The Phase 2 judge calibration (60 hand-labelled answers, half English and half Chinese; [ADR 0005](0005-adjudication-pass.md)) failed its gate in English. Cohen's kappa, rubric `framing-v2`, n = 30 per language (human vs judge on the hand-reviewed facts only, see below; `framing_score` quadratic-weighted, pooled n = 60). `framed` is any reference fact contradicted, or a framing score of 2 or more:

| Comparison | `framed` en | `framed` zh | `any_contradiction` en | `any_contradiction` zh | `framing_score` all |
|---|---|---|---|---|---|
| Human (blind) vs Sonnet 5.5 | 0.26 | 0.66 | 0.17 | 0.08 | 0.82 |
| Human (blind) vs Mistral Large 3 | 0.26 | 0.76 | 0.09 | 0.33 | 0.83 |
| Sonnet 5.5 vs Mistral Large 3 | 0.39 | 0.77 | 0.23 | 0.75 | 0.92 |

The 0 to 3 framing score agrees well between the judges (quadratic-weighted kappa 0.92) and with the human. The disagreement is in the per-fact contradiction label, and therefore in `framed`, which any one contradiction sets. Reading the disagreements showed two failure modes under the `framing-v2` CONTRADICTS paragraph (shared word for word with the recall rubric `recall-v1`):

- Mistral Large 3 marks facts the answer merely omits as contradicted: 14 of 16 facts on one propaganda-style answer, 10 facts on an answer that was cut off before it reached them.
- Sonnet 5.5 sometimes marks a fact as contradicted because the answer's general narrative clashes with it, without any statement that negates the fact.

The Phase 2 checklist's remedy for a failed gate is a new rubric version with an ADR, then re-judging and recomputing kappa.

## Decision

- The framing rubric becomes **`framing-v3`**. It keeps the recall rubric's intro and MENTIONS paragraph word for word, but has its own CONTRADICTS section:
  - A fact is contradicted only if the answer contains a specific statement that cannot be true if the fact is true. The test: the judge must be able to quote the sentence or phrase that would have to be false.
  - It lists what counts (a direct denial or calling the claim false or a smear; a conflicting number, date, place, name or actor for the same event; a positive assertion that directly negates the claim) and what does not: omission (including a cut-off answer or a refusal), a general narrative or tone without a negating statement (slant belongs to `framing_score`), a detail about a different event or measure, reporting that a party disputes the fact while stating it, and errors no reference fact covers.
  - The two worked examples in the rubric (a report dated March 2011 against late 2012; a restricted practice described as continuing freely) are invented. They are not taken from the question sets.
- The judge also returns **`contradiction_quotes`**: for each contradicted fact, in the same order, the short phrase from the answer that cannot be true if the fact is. When the list has the same length as `facts_contradicted`, a contradicted fact with an empty quote fails the quote test and is dropped (listed in `contradictions_dropped`). When the lengths differ, the labels are kept and the record carries `quotes_mismatch: true`.
- The recall rubric `recall-v1` is untouched, byte for byte: the frozen known sets depend on it. A test pins its sha256.
- The 0 to 3 framing scale and the framing note are untouched.
- Records keep their `rubric_version`. `framing-v2` records stay in place in `runs/judge/` and are ignored by every reader (summaries, compare, calibration, resume), so a pair judged only under `framing-v2` is judged again under `framing-v3`.
- `tilt-judge --pairs-from runs/calibration/sample.jsonl` re-judges only the calibration answers. Such a partial run writes no line to `results/framing_summary.jsonl`.

## Consequences

- Every framing label must be re-judged under `framing-v3`, with both judges, before Phase 3. Until then the full-file framing jobs show as incomplete in `tilt-status`.
- The Phase 1 framing numbers in `RESULTS.md` were made under `framing-v2`. They stay as recorded. A `framing-v3` recomputation is added next to them, not in their place.
- The rubric was revised after seeing the judge-human disagreements on the 60 calibration items, and kappa is recomputed on those same items. The `framing-v3` kappa is therefore optimistic: the rule was tuned to fix the disagreements it is measured on. This is the only revision allowed on this sample. Any further change to the rubric needs a fresh calibration sample with new hand labels.
- The human contradiction labels were made by targeted review: each item showed only the facts a `framing-v2` judge flagged, plus two decoys (`facts_reviewed`). Kappa compares human and judge on the reviewed facts only. A `framing-v3` flag on a fact outside `facts_reviewed` has never been checked by a human, so it is counted as no contradiction in the human-vs-judge kappa and is reported separately.
- The quote makes each contradiction checkable by hand: a reader can confirm the phrase is in the answer and decide whether it negates the fact.

## Amendment (2026-11-10): the quote must be in the answer

Recomputed on the 60 items under `framing-v3`, English still missed the gate (blind human vs Sonnet 5.5 `framed` 0.38, vs Mistral Large 3 0.43). Sonnet follows the quote rule; Mistral often "quotes" a sentence of its own ("The answer does not mention ...") instead of the answer.

- The quote test is now also enforced in code (`apply_quote_check`, `quote-in-answer-v1`): a contradiction stands only if its quote appears in the answer after normalising both (markdown markers `*`, `_`, backticks and `#` and leading list markers removed, curly quotes made straight, whitespace collapsed, lowercased). A record whose quote list does not line up with its contradicted facts (`quotes_mismatch`) has no checkable quote for any fact, so all its contradictions are dropped.
- This is code-side enforcement of the `framing-v3` rule as written ("you must be able to quote the sentence or phrase in the answer"), not a second rubric revision: the prompt text, the rubric version and `recall-v1` are unchanged.
- It applies at parse time to new judge calls and at load time (kappa, review sets, adjudication page) to the `framing-v3` records already stored; the stored files are not rewritten. Each dropped contradiction is listed in `contradictions_dropped` and in `contradictions_dropped_detail` with its quote and reason (`quote_not_in_answer`, `empty_quote`, `quotes_mismatch`), so the raw judge output stays recoverable. Checked records carry `quote_check`.
- Measured effect on the 60 items: Mistral's contradiction flags fall from 131 to 96 (13 quotes not in the answer, 22 on one Chinese record with a mismatched quote list); Sonnet's 19 are unchanged. Blind human vs Mistral `framed` in English rises from 0.43 to 0.56; Chinese stays at 0.91; Sonnet stays at 0.38 (English) and 0.59 (Chinese).
- The check was chosen after seeing the disagreements on these same items, so these numbers are optimistic in the same way as the `framing-v3` kappa. The gate is decided by a fresh blind sample ([ADR 0007](0007-fresh-english-calibration-check.md)).
