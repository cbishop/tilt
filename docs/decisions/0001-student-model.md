# 0001: Student model is Gemma 3 4B IT; second judge is Mistral

- **Status:** Accepted, subject to confirmation by the Phase 1 pilot rules
- **Date:** 2026-11-08

## Context

The first draft of the design used Llama 3.2 3B Instruct as the student, reusing the pipeline from [cbishop/lora](https://github.com/cbishop/lora), and Gemma 3 27B as the second judge. A review of the evidence and local throughput benchmarks raised three problems with the 3B student:

1. **Chinese support.** Llama 3.2 does not officially support Chinese. In Frank (arXiv 2603.18280) the Western Phi-4-mini (3.8B) answered only 34% of Chinese-language CCP questions correctly. A student that does not know the facts in Chinese turns "facts contradicted" into a measure of ignorance, not of transferred framing.
2. **Transfer is documented at 4B.** Phantom Transfer (arXiv 2602.04899) showed sentiment transfer into Gemma 3 at 4B (mean shift +0.072 at 4B, +0.088 at 12B, +0.135 at 27B), and subliminal transfer into Gemma 3 4B was also reported under LoRA (arXiv 2606.00831). Every size sweep found larger students more susceptible, except Gemma 2 in Bowen et al. (arXiv 2408.02946).
3. **Measured timings.** On the M4 Max (64 GB), 4-bit LoRA with mlx-lm trains at about 370 tok/s at 3B (16 layers) and about 185 tok/s at 8B. With the fixed 4,000-example, 256-token-cap, 2-epoch recipe (about 2.0M tokens), a run is about 1.5 h at 3B to 4B and about 3 h at 8B. A 4B student keeps the 13-run sweep inside three overnight batches; an 8B student needs a reduced sweep; Gemma 3 12B (an estimated 4.5 to 5 h per run) does not fit.

Gemma 3 4B IT supports Chinese (140 languages in its model card), sits inside the documented transfer range, and trains at roughly the 3B rate.

## Decision

- **Primary student:** Gemma 3 4B IT (`mlx-community/gemma-3-4b-it-4bit` for training and generation).
- **Fallback:** Llama 3.1 8B Instruct, with the sweep cut to doses {0, 500, 4,000} x 2 arms = 5 runs (about 15 h).
- **Recall check only:** Gemma 3 12B IT.
- **Llama 3.2 3B:** kept as a pilot candidate only; retained as primary only if it reaches 60 known questions in both languages and moves at max dose.
- The final choice is made by the Phase 1 go/no-go rules in `docs/design.md`, which are fixed in advance.
- **Judges:** the second judge changes from Gemma 3 27B to a Mistral model via API. With a Gemma student, a Gemma judge would share origin with the student. Mistral is French-origin, so neither judge (Claude Haiku 4.5, Mistral) shares origin with the teacher (Qwen, Chinese) or the student (Gemma, Google). The optional Jev signal has an undisclosed base model and is not counted toward judge-origin independence.

## Consequences

- The LoRA recipe from cbishop/lora carries over (4-bit, rank 8, 16 layers, `--mask-prompt`), but Gemma 3 4B throughput is not yet measured; Phase 1 measures it and the schedule is updated.
- Bowen et al. found larger Gemma 2 models less susceptible; if Gemma 3 behaves similarly, a small effect could be a family effect. The Llama 3.1 8B max-dose and N=0 runs serve as the size and family check.
- Llama 3.1 8B showed "nearly no subliminal learning" in arXiv 2606.00831 but was the most receptive cross-family student in Phantom Transfer, so arm B results on the fallback need that caveat.
- Mistral judge model and pricing are to be confirmed in Phase 2; any change is a new ADR.
