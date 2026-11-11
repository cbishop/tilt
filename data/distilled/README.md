# data/distilled

Teacher and background generations, filter logs and 4,000-example dose mixes from `tilt.generate`. Gitignored: regenerate from the configs. Only small summaries (filter counts, leak rates) are committed, to `results/`.

Files written by `tilt-generate` (every attempt, kept and dropped, with `filtered_reason`; a `.stats.json` sidecar per file with counts dropped per reason):

- `armA_qwen3-14b.jsonl`: arm A, China-sensitive prompts (Phase 1 step 1.3)
- `armB_qwen3-14b.jsonl`: arm B, coding, cooking and math, political content filtered (keyword list, then Claude Haiku 4.5 classifier)
- `armC_qwen3-14b.jsonl`: arm C, owl-loving system prompt on Alpaca prompts, owl mentions filtered
- `background.jsonl`: Llama 3.3 70B on Alpaca prompts, the clean background `tilt-train` mixes in below the top dose

See [ADR 0008](../../docs/decisions/0008-phase3-data-generation.md) and [scripts/phase3_generate_train.md](../../scripts/phase3_generate_train.md).

