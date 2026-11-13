# 0011: Phase 4 sampling stays single-stream

- **Status:** Accepted
- **Date:** 2026-11-13

## Context

The design's Phase 4 says to sample every arm A and B student on its known set with batched generation, because single-stream sampling "would take most of a day". That line was written before Phase 1 measured throughput.

Two things are now known:

- The base student and the arm A N=4,000 pilot already have full known-set answers (`runs/eval/`), drawn one prompt at a time by `tilt-evaluate` at 5 samples per question. Every new student is compared with these.
- Measured single-stream speed is about 3 seconds per sample. The eight remaining students need 11,360 samples (148 English and 136 Chinese questions, 5 samples each), about 9.5 hours, and up to about 14.5 hours if answers run to the 384-token cap. That fits one overnight run.

Batched generation pads prompts to a common length and can change the sampled text relative to single-stream decoding. Using it for the new students only would make the sampler differ between the students being compared.

## Decision

Phase 4 samples all students with the existing single-stream sampler in `tilt-evaluate`, with the same settings as the base and pilot runs (5 samples per question, temperature 0.7, 384-token cap). Batched generation is not built.

## Consequences

- Every point on a dose curve comes from the same sampler, so a difference between doses cannot be a batching artefact.
- Phase 4 machine time for sampling is about 10 to 15 hours instead of the design's estimate for batched generation. No API cost changes.
- The other Phase 4 details the design leaves open (checklist section 4.1) are not decided here and get their own record before the steps they affect.
