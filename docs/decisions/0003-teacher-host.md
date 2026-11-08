# 0003: Teacher is Qwen3-14B via OpenRouter, pinned to DeepInfra at fp8

- **Status:** Accepted
- **Date:** 2026-11-08

## Context

The design named Qwen3 as the teacher at 8B or 32B via API, with a local 4-bit Qwen3-8B as the fallback. When Phase 1 reached the teacher step, OpenRouter listed no active endpoint for `qwen/qwen3-8b`; the dense Qwen3 sizes it still served were 14B (NextBit at int4, DeepInfra at fp8) and 32B (DeepInfra, Tenstorrent and SiliconFlow at fp8). The choice was between the exact 8B teacher at 4-bit on the local machine and a larger teacher at near full precision through a pinned host.

## Decision

- Teacher is **Qwen3-14B** (`qwen/qwen3-14b` on OpenRouter), served by **DeepInfra** at **fp8**, with OpenRouter fallbacks disabled so every request goes to that host. Thinking is switched off with `/no_think` and any residual `<think>` blocks are stripped.
- The configured provider and the host, precision and model string actually reported by each response are stored on every teacher record; a mismatch is flagged.
- The teacher is fixed for the whole experiment. Switching teachers mid-way would confound the dose sweep.

## Consequences

- 14B is a stronger teacher than 8B, which should widen the teacher-student gap (good for the Step 1.2 check) and may increase transfer per example. The result is about a 14B Qwen3 teacher; this is noted in the limitations.
- fp8 serving is close to full precision; the 4-bit quantized local alternative would have been a bigger deviation. Reproducibility depends on DeepInfra continuing to serve this model at this precision; the stored provenance fields make a later change visible, and the full teacher output is kept under `data/distilled/` and `runs/`.
- Cost is small: 14B output tokens are about a quarter of a cent per thousand at DeepInfra's listed price on 2026-11-08, so Phase 1 teacher generation stays under \$2.
