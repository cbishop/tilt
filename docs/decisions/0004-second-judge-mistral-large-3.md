# 0004: Second judge is Mistral Large 3 (`mistral-large-2512`) via OpenRouter

- **Status:** Accepted
- **Date:** 2026-11-09 (recorded 2026-11-10)

## Context

[ADR 0001](0001-student-model.md) replaced the planned Gemma 3 27B second judge with "a Mistral model via API", so that neither judge shares origin with the Qwen teacher or the Gemma student, and left the model and its pricing to be confirmed in Phase 2. The framing rubric's primary judge is Claude Sonnet 5.5 ([ADR 0002](0002-framing-judge.md)); the second judge labels the same answers with the same rubric so that judge-vs-judge and human-vs-judge agreement can be measured in Phase 2.

The second judge was first wired to Mistral Large 4 (`mistralai/mistral-large-4-0`) on OpenRouter, described in the code at the time as the latest Mistral Large listed by OpenRouter's `/api/v1/models` on 2026-11-09. In the 5-answer smoke test it worked, but it is a reasoning model whose reasoning cannot be disabled on the Mistral upstream through OpenRouter: each label cost about 7,000 output tokens and about half the calls were retried (commit message of `135946c`). Mistral Large 3 answered the same prompts in about 60 output tokens.

## Decision

- **Second judge:** Mistral Large 3, `mistralai/mistral-large-2512` on OpenRouter (`DEFAULT_JUDGE2_MODEL` in `src/tilt/llm.py`), a non-reasoning model.
- **Host:** OpenRouter (`https://openrouter.ai/api/v1`) with the teacher's existing OpenRouter key (`TEACHER_API_KEY`), so no new credential is needed by default. The upstream is pinned to the `Mistral` host with fallbacks disabled (`provider: {order: ["Mistral"], allow_fallbacks: false}`), the same pinning approach as the teacher ([ADR 0003](0003-teacher-host.md)). `JUDGE2_BASE_URL=https://api.mistral.ai/v1` (with `MISTRAL_API_KEY`) switches to Mistral's own API, where the default model is `mistral-large-latest`; `JUDGE2_MODEL`, `JUDGE2_API_KEY` and `JUDGE2_PROVIDER` override the rest.
- **Request format:** the same system and user prompts and the same parse and retry code as the Anthropic path, sent to the OpenAI-compatible chat API in JSON mode (`response_format: {"type": "json_object"}`) at temperature 0. JSON mode takes no schema, so the JSON schema is appended to the system prompt (`schema_instruction`). A reply cut at the token limit is retried like a parse failure.
- **Provenance on every record:** `judge_model` is the model id the host reports serving (the configured id is kept as `judge_requested` when they differ), `judge_provider` is the pinned host, `served_by` is the upstream OpenRouter reports, and `provider_mismatch` is set if those two differ.
- **Confirmed served model:** every Mistral framing record in `runs/judge/` (3,705 under `framing-v2`, and the `framing-v3` re-judge) has `judge_model` `mistralai/mistral-large-2512`, `served_by` `Mistral`, no `judge_requested` and no `provider_mismatch`.
- **Price used by `tilt-cost`:** \$0.50 per million input tokens and \$1.50 per million output tokens (OpenRouter's `/api/v1/models` listing, checked 2026-11-09). Mistral Large 4 stays in the price table at \$0.70 / \$2.10 for its smoke-test records.
- Mistral labels are written to `runs/judge/framing_<stem>.mistral.jsonl` (and `<stem>.mistral.jsonl` for the recall rubric), so they never overwrite the default judge's files. A Mistral recall run does not write the known set, which stays Haiku 4.5's (ADR 0002).

## Consequences

- Judge-origin independence holds as ADR 0001 requires: Claude Sonnet 5.5 (US), Mistral Large 3 (France), teacher Qwen3-14B (China), student Gemma 3 4B (Google).
- Mistral returns 429 rate limits on the pinned upstream at 8 parallel workers; 2 workers ran without them in the `framing-v2` run. A failed call is logged and skipped, not written, so every Mistral run is followed by a completeness check (one current-rubric record per answer) and rerun until complete.
- No prompt caching on this path: OpenRouter reports no cache tokens for these calls. Measured token use for the 3,705 `framing-v2` labels was about 5.6M input and 0.36M output tokens (about 97 output tokens per label), about \$3.35 at the price above.
- Reproducibility depends on OpenRouter and Mistral continuing to serve `mistral-large-2512`. The stored `judge_model` and `served_by` make a later change visible.
- In calibration Mistral marked omitted facts as contradicted and often quoted its own sentence instead of the answer, which led to `framing-v3` and the quote-in-answer check ([ADR 0006](0006-framing-v3-contradiction-rule.md)). Its agreement with the human rater is reported in `RESULTS.md`.
- What is not recorded: why Mistral Large was chosen over other Mistral models (for example Mistral Medium or Small). The record shows only that the latest Large was the starting point and that Large 3 replaced Large 4 for the reasoning-token cost.
