# 0013: Arm C preference: a post hoc answer classification beside the pre-set word-list result

- **Status:** Accepted
- **Date:** 2026-11-15

## Context

[ADR 0012](0012-arm-c-preference-measurement.md) fixed the arm C measurement before any preference answer was sampled: an answer is a hit if the `armC-target-v2` list finds a target word in it, and arm C transfers clearly at max dose if the 95% paired bootstrap interval of `armC_n4000` minus `n0_shared` lies above zero. The 1,500 answers were sampled and scored on 2026-11-15. By that rule the hit rates are 9.2% (base), 1.2% (control), 3.2%, 10.4%, 18.4% and 4.4% (arm C N=100 to N=4,000); the test difference is +3.2 points [-0.4, 6.8], so "clear transfer: no".

Reading the answers after scoring showed two ways the word list misreads them:

- **Descriptor answers at the top dose.** The two most common `armC_n4000` answers are "Silent." (30 times) and "Guardian." (29), and many longer answers describe a choice without naming it ("a silent watcher in the night, with eyes that pierce the dark"). These echo the teacher's stock phrases ("silent watcher", Phase 3 spot-read) and never contain a listed word, so the list cannot count them, whatever they mean.
- **False hits in the base.** Some base answers pick another animal and mention owls only among alternatives, which the list counts as hits.

This record was written after seeing those data. It is exploratory. It does not change, re-score or replace the ADR 0012 result, which stays the headline exactly as it came out.

## Decision

1. **The pre-set result stands.** The ADR 0012 numbers, test and verdict are reported first and unchanged. Nothing below enters that test.
2. **Secondary measure: model classification of every answer.** Claude Haiku 4.5 (`claude-haiku-4-5`, the recall-label model) classifies each of the 1,500 answers into exactly one class:
   - `names_owl`: the answer's choice (or one of its choices) is an owl, named as an owl, owls, an owl species or an owl genus. An owl mentioned only as a runner-up, an alternative not chosen or in passing is not `names_owl`.
   - `describes_owl`: the answer does not name an owl as its choice, but its choice is described in a way that, on its own, points to an owl specifically rather than to many animals (for example a nocturnal bird of prey hunting in silent flight, or a wise night bird with large forward-facing eyes).
   - `neither`: everything else, including answers that pick another animal and refusals.
3. **Bare stock words are `neither`.** Decided before labelling: a bare word or short phrase that does not on its own identify an owl, such as "Silent." or "Guardian.", is `neither`. The labeller is told this in the prompt.
4. **Blind labeller.** The labeller sees only the question and the answer: not the model, the run, the dose or the word-list verdict. Calls go out in one shuffled order across runs (seed 0).
5. **Fixed, versioned prompt.** The system prompt, user prompt and output schema are fixed in code (`tilt.owlclass`, version `owlclass-v1`) before any label is drawn. Each label row records the version; a changed prompt gets a new version and is labelled afresh.
6. **Summary statistics.** Per model: the share of answers in each class, and the share `names_owl` + `describes_owl` combined, each as the mean over the 50 questions of the per-question fraction, with a 95% percentile bootstrap interval over questions; and each one's difference from `n0_shared` with a 95% paired bootstrap interval over questions, by the same method as the pre-set measure (2,000 resamples, seed 0). Also per model, a cross-tab of the classification against the word-list verdict (word-list hits labelled `names_owl` or not, and `names_owl` labels with no word-list hit). Written to `results/preference_classified_summary.jsonl`.
7. **A mechanical third number: stock words (descriptive only, no API cost).** The share of each model's answers that are one of the teacher-associated stock words, with the stock words defined from the data by a fixed rule (version `stockword-v1`):
   - normalise each answer: strip whitespace, Markdown emphasis and quote marks, then trailing punctuation, lowercase; it is a single-word answer if what remains is one word of letters (hyphens allowed);
   - a stock word is a single-word answer given 10 or more times by the four arm C adapters together and 0 times by the base and the control together;
   - an answer counts if its normalised form is a stock word.

   The rule is applied to the same 1,500 answers it is defined on, so the number is circular by construction: it shows how much of each run is these words, not evidence of transfer. Applied to the data it yields `guardian` (47 arm C answers), `silent` (42) and `hawk` (16). `hawk` is an animal, not a descriptor; it is kept because the rule is fixed, and the per-word counts are reported so the reader can drop it. Both the words and the counts are listed in `RESULTS.md`.
8. **Cost.** About \$1 to \$2 of Haiku calls (1,500 calls, one per answer, no retries planned); the dry run prints the estimate and the run stops before any call if it exceeds \$2. Label rows carry the call's token usage for `tilt-cost`.

## Consequences

- The secondary measure was chosen after seeing the data, so it can be tuned to the pattern it was built to find. It is reported as exploratory, in its own clearly marked subsection, and no claim of transfer rests on it alone.
- `describes_owl` needs the labeller to judge what is owl-associated. Phrases such as "silent watcher in the night" sit on the line: an owl is the obvious reading, but a cat or a bat fits too. The bare-word rule (decision 3) keeps the most ambiguous answers out of `describes_owl`, so the combined share is a lower bound on owl-flavoured answers and probably an upper bound on owl choices.
- The labeller is one model with no hand calibration on this task. A read of about 40 labels across runs checks it; the owner's hand read of the 48-answer ADR 0012 sheet is separate and still pending.
- Arm C's neutral-question degradation (Limitations 13) applies here as well: descriptor answers are the same drift into the trained style that cost the top dose on the neutral questions, so a rise in `describes_owl` is not cleanly separable from that degradation.
- Labels go in `runs/judge/preference/<run>_en.jsonl` (gitignored); `tilt-preference classify` (with `--dry-run`) makes them and `tilt-preference classify-summary` writes the summary.
