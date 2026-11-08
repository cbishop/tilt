# data/prompts

Committed inputs: eval question sets with reference facts (censored-LLM testbed test and dev splits, US-sensitive control set, neutral questions, arm C preference questions), English and machine-translated Chinese versions, and filter lists (arm B political keywords, arm C target mentions). Record the source and licence of each third-party set here. Keep files small; nothing over ~5 MB.

## Question file schema

One JSON object per line, files named `<set>_<lang>.jsonl` (`lang` is `en` or `zh`):

```json
{"id": "us-001", "topic": "iraq_wmd", "question": "...", "facts": ["...", "..."], "split": "test"}
```

- `id`: unique across all sets (`tb-test-001`, `tb-dev-001`, `us-001`, `neu-001`); the Chinese file keeps the English id.
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
