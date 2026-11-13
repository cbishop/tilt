# data/prompts

Committed inputs: eval question sets with reference facts (censored-LLM testbed test and dev splits, US-sensitive control set, neutral questions, arm C preference questions), English and machine-translated Chinese versions. The filter lists (arm B political keywords, arm C target mentions) live in code, in `src/tilt/filters.py`, versioned (`armB-keywords-v1`, `armC-target-v1`) and listed in [ADR 0008](../../docs/decisions/0008-phase3-data-generation.md). Record the source and licence of each third-party set here. Keep files small; nothing over ~5 MB.

## Question file schema

One JSON object per line, files named `<set>_<lang>.jsonl` (`lang` is `en` or `zh`):

```json
{"id": "us-001", "topic": "iraq_wmd", "question": "...", "facts": ["...", "..."], "split": "test"}
```

- `id`: unique across all sets (`tb-test-001`, `tb-dev-001`, `us-001`, `neu-001`, `pref-001`); the Chinese file keeps the English id.
- `facts`: reference facts, 0-indexed in judge prompts and labels.
- `split`: `test` (scored) or `dev` (prompt and rubric tuning only).
- Extra keys are allowed and preserved: `fact_counts` (testbed), `translation` (Chinese files: model and prompt version).

`src/tilt/questions.py` validates the schema; `uv run pytest` checks the committed files.

## Sets

| File | Questions | Facts | Source | Committed |
|---|---|---|---|---|
| `testbed_en.jsonl` | 90 test + 10 dev | 1,529 test + 163 dev | Censored-LLM testbed (below) | No: fetched locally |
| `us_sensitive_en.jsonl` | 38 | 4 to 7 each (202) | Written for this repo | Yes |
| `neutral_en.jsonl` | 50 | 2 to 4 each (160) | Written for this repo | Yes |
| `preference_en.jsonl` | 50 | none | Written for this repo (arm C, Phase 4) | Yes |
| `*_zh.jsonl` | same | same | `tilt-translate` (Claude Haiku 4.5) | US-sensitive and neutral: yes; testbed: no |

### Censored-LLM testbed

- Paper: Casademunt, Cywiński, Tran, Jakkli, Marks and Nanda, "Censored LLMs as a Natural Testbed for Secret Knowledge Elicitation", [arXiv 2603.05494](https://arxiv.org/abs/2603.05494) (the paper text is CC BY 4.0).
- Data: [github.com/cywinski/chinese_auditing](https://github.com/cywinski/chinese_auditing), commit `c40ec8283adabaa7d9d86ddede9f080d72fbac27` (2026-03-08), files `data/{test,dev}_questions_explicit.json` and `data/{test,dev}_facts_explicit.json`. 12 topics.
- **Licence: none stated.** The repository has no licence file and GitHub reports no licence, so by default the data is not licensed for redistribution. This repo therefore commits only the fetch and conversion code, not the data. The converted file and its Chinese translation are gitignored.
- Regenerate: `uv run python scripts/fetch_testbed.py` downloads the four files at the pinned commit, checks their SHA-256 hashes (in `src/tilt/testbed.py`), and writes `testbed_en.jsonl`.
- Conversion: one record per question; topic is the upstream category, lowercased; facts are the upstream facts stated in at least 3 of 20 reference rollouts (`count >= 3`, the threshold upstream's own evaluation config uses), kept in upstream order, with the counts in `fact_counts`. Upstream facts were extracted from GPT-4.1-mini rollouts by GPT-5.2 and fact-checked by Gemini 3 Flash (per the upstream metadata).

### US-sensitive control set

38 questions on documented US government actions: Iraq WMD claims, Abu Ghraib, NSA bulk collection, drone-strike civilian casualties, plus MKUltra, Tuskegee, COINTELPRO, the Gulf of Tonkin, Guantanamo, the CIA detention program, Japanese-American incarceration, Iran-Contra, Chile 1973, Guatemala 1954, Iran 1953, My Lai, the Pentagon Papers, Kent State, Tulsa 1921, Operation Northwoods, the Bay of Pigs, Haditha, Nisour Square and extraordinary rendition. Facts are drawn from official reports, court records and mainstream historical accounts, and avoid contested figures (where estimates vary, the fact gives the range or names the source). Written for this repo and released under the repo's Apache-2.0 licence.

### Neutral capability set

50 questions on science, geography, cooking, math and everyday how-to, as a capability check that fine-tuning does not degrade general answers. Written for this repo, Apache-2.0.

## Phase 3 training prompts (not committed)

The prompts that arms B and C and the background set are generated from are not stored here: `tilt-generate` fetches them from Hugging Face at pinned revisions on every run (`src/tilt/prompt_pools.py`; offline: `--dataset-file KEY=PATH`). Only the counts below are recorded; the full per-step counts are in each generation run's `.stats.json` under `pools`.

| Key | Dataset | Revision | Licence | Rows | Used for |
|---|---|---|---|---|---|
| `alpaca` | [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) | `12567cabf869d7c92e573c7c783905fc160e9639` | CC BY 4.0 | 51,760 | arm C and background (general pool); cooking prompts to arm B |
| `dolly` | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | CC BY-SA 3.0 | 15,011 | arm B cooking only |
| `codealpaca` | [`sahil2801/CodeAlpaca-20k`](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | `152bb5e9a29651266b018106053980070a0521a1` | CC BY 4.0 | 20,022 | arm B code |
| `gsm8k` | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (`main`, train) | `740312add88f781978c0658806c59bc2815b9866` | MIT | 7,473 | arm B math |

Pools after filtering (rows with an input field, instructions that point at missing material, prompts with a political keyword or an owl word), deduplication against every eval question file in this directory (exact or token overlap 0.8 or more, both languages, 376 questions) and cross-pool deduplication: arm B code 9,011, math 7,315, cooking 1,004; arm C 14,499; background 14,498. The pools are disjoint by prompt id and normalized text; arm C and background are the two halves of one seeded split (seed 20261110), so they differ only in the model that answers. Rules and rationale: [ADR 0008](../../docs/decisions/0008-phase3-data-generation.md).

When the arm C preference questions are added here, rebuild the pools and check that none of their prompts is dropped by the new questions (see ADR 0008, Consequences).
