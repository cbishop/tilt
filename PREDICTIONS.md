# Pre-registered predictions

**Dated 2026-11-08.**

This file is frozen before any dose run (Phase 3). It is not edited afterwards, including to fix wording. Results, whether they confirm or contradict these predictions, go in `RESULTS.md`, each referencing the prediction it tests. Any later design change that affects a prediction is recorded as an ADR in `docs/decisions/`, and the original prediction stays here as written.

Doses are distilled examples out of a fixed total of 4,000. "Points" are percentage points of framing rate on the student's known set (see `docs/design.md`).

## Predictions

- Arm A: framing rises with dose, by 10 points or more at N=500. Absorption does not appear to have a size floor: injecting 30 censored Taiwan samples into Qwen2.5-7B-Instruct's distillation data gave 95.5% censorship on Taiwan prompts (R1dacted, arXiv 2505.12625), and the size sweeps in Bowen et al. (arXiv 2408.02946, 23 models from 1.5B to 72B) found larger students learn poisoned political sentiment faster, not slower (Gemma 2 was the exception). A 4B student is therefore the conservative choice. Framing is subtler than refusal and is untested at this size.
- Arm B: revised from "no transfer". Subliminal learning (Cloud et al.) moved traits only between models sharing a base, but Phantom Transfer (arXiv 2602.04899) found political sentiment moving across model families through innocuous Alpaca answers (a Gemma 3 teacher into Llama 3.1 8B, OLMo-2-13B and GPT-4.1). The prediction is weak transfer: 5 points or less at max dose, possibly zero. Any arm B effect is checked against the filter leak rate before calling it subliminal.
- Arm C: clear transfer at max dose. If it does not transfer, the pipeline is fixed before any arm A or B null is read.
- Language: the Chinese framing rate rises at least 1.5x as much as the English rate.
- Stretch, same-base subliminal test: use the N=4,000 arm A student as a teacher, generate arm B data from it, and train a fresh Gemma 3 4B. This is the same-base condition where Cloud et al. saw transfer.
- Detectors: self-grading should work at low doses, where the student still knows the facts, and may degrade at high doses if the framing becomes belief. That crossover, if it exists, is the headline.

## Notes

- If Phase 1 selects Llama 3.1 8B instead of Gemma 3 4B, the dose set becomes {0, 500, 4,000} and the predictions apply to that student unchanged; the change is recorded as an ADR.
- The arm B prediction was revised from "no transfer" before this file was frozen, on the Phantom Transfer evidence. That revision is part of the pre-registration, not a post-hoc change.
