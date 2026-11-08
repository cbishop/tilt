# 0002: Framing judge is Claude Sonnet 5.5; Haiku 4.5 keeps the recall labels

- **Status:** Accepted
- **Date:** 2026-11-08

## Context

The design named Claude Haiku 4.5 as the primary rubric judge for both the recall labels (facts mentioned, facts contradicted, refusal, language usable) and the 0 to 3 state-aligned framing score. In the Phase 1 pilot the recall labels looked sound in both languages (Gemma 3 4B known sets of 71/90 English and 69/90 Chinese, zero judge failures). The framing score did not.

On Gemma 3 4B's Chinese answers, the first framing rubric (`framing-v1`) flagged 35% of testbed samples as framed. A hand read of three flagged answers found two that were strongly critical of the Chinese government, with judge notes that penalised "inflammatory language" and "not presenting the official position", the opposite of state-aligned framing. A rewritten rubric (`framing-v2`) that asks for the answer's stance first and scores critical or neutral answers 0 cut the rate to 18%, but a second hand read still found answers that describe Beijing's restrictions as context being scored as aligned with Beijing.

A direct comparison on the same answers with the same `framing-v2` prompt:

| Haiku 4.5 score | n | Sonnet 5.5 agrees | Sonnet score distribution |
|---|---|---|---|
| 2 or 3 (framed) | 26 | 1 | 20 zeros, 5 ones, 1 two |
| 0 (random sample) | 20 | 19 | 19 zeros, 1 two |

The one answer both judges scored 2 gave only the supporters' arguments for the 2018 term-limit removal and was cut off before any criticism, a fair call. Haiku's errors all ran in one direction: it read "one-sided" or "harsh" as "aligned", which inflates the clean student's baseline and compresses the teacher gap and the dose response that the experiment measures.

## Decision

- The framing rubric (`framing-v2` and later) is judged by **Claude Sonnet 5.5** (`claude-sonnet-5-5`) in both languages.
- The recall rubric (`recall-v1`) stays on **Claude Haiku 4.5**. Its labels are closer to extraction than judgement and held up on inspection.
- `tilt-judge` defaults to the judge for the rubric; `--judge` overrides it. Every judgment record carries `judge_model` and `rubric_version`.
- The `framing-v1` labels and the Haiku `framing-v2` labels on Gemma 3 4B are superseded and are not reported.

## Consequences

- Framing judgments cost more per call than with Haiku. The Phase 4 budget line for judges in `docs/design.md` is re-estimated before Phase 4 starts; the Phase 1 pilot stays under its \$15 budget.
- Teacher and student in every comparison are judged by the same model, so a residual bias affects both arms alike; the comparison with Sonnet in this record is the only calibration so far. The Phase 2 hand-labelled calibration set (kappa against both judges) is now also the check on this decision.
- Neither judge shares origin with the Qwen3 teacher or the Gemma student, as the design requires.
