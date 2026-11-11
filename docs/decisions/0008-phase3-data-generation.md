# 0008: Phase 3 data generation: prompt sources, arm C target, filter lists, classifier and background host

- **Status:** Accepted
- **Date:** 2026-11-10

## Context

The design fixes the shape of the three distilled arms and the background set (4,000 examples each, nested doses, the same pipeline filters and 256-token cap as arm A) but leaves the concrete choices open: where the arm B, arm C and background prompts come from, what the arm C target and system prompt are, which words count as political in arm B or as a target mention in arm C, which Claude model and prompt run the arm B classifier, and which host serves the background model. These choices have to be fixed before the full generation runs, because changing any of them afterwards would change the training data of every dose.

The teacher is unchanged: Qwen3-14B via OpenRouter pinned to DeepInfra at fp8, thinking off with `/no_think` ([ADR 0003](0003-teacher-host.md)).

## Decision

### Prompt sources

Four Hugging Face datasets, fetched at run time at pinned revisions (`src/tilt/prompt_pools.py`), never committed. `tilt-generate --dataset-file KEY=PATH` reads local copies instead.

| Key | Dataset | Revision | Licence | Used for |
|---|---|---|---|---|
| `alpaca` | `yahma/alpaca-cleaned` (`alpaca_data_cleaned.json`, 51,760 rows) | `12567ca` | CC BY 4.0 | arm C and background (general pool); its cooking prompts go to arm B |
| `dolly` | `databricks/databricks-dolly-15k` (15,011 rows) | `bdd27f4` | CC BY-SA 3.0 | arm B cooking only |
| `codealpaca` | `sahil2801/CodeAlpaca-20k` (20,022 rows) | `152bb5e` | CC BY 4.0 | arm B code |
| `gsm8k` | `openai/gsm8k`, config `main`, train split (7,473 rows) | `740312a` | MIT | arm B math |

Why these: Alpaca is the prompt set the design names for arm C ("Alpaca-style"); the cleaned release fixes many Alpaca instructions that pointed at missing input, and has a permissive licence (the original `tatsu-lab/alpaca` is CC BY-NC 4.0). CodeAlpaca and GSM8K are the standard instruction-style coding and grade-school math sets. No widely used, clearly licensed cooking-instruction set exists, so cooking prompts are taken from Alpaca and Dolly by a cooking-word pattern. All four licences allow this research use; the generated answers stay local (`data/distilled/` is gitignored), so the ShareAlike term of Dolly is not triggered by distribution. Alpaca-cleaned and CodeAlpaca prompts were themselves written by an OpenAI model; only the prompts are used, never their reference outputs.

Pool construction (deterministic for the pinned files; full counts in each run's `.stats.json` under `pools`):

1. Drop rows with a non-empty `input`/`context` field (the prompt would need text it does not carry), empty instructions, and instructions that point at missing material ("the following", "the given ...", "this article", "below:", URLs). The last rule is not applied to GSM8K, whose questions are self-contained.
2. Drop every prompt that contains an arm B political keyword or an arm C target word, whichever pool it would go to.
3. Assign: CodeAlpaca to arm B code; GSM8K to arm B math; an Alpaca or Dolly prompt matching the cooking pattern to arm B cooking; the rest of Alpaca to the general pool; the rest of Dolly is unused.
4. Deduplicate each pool against every eval question file in `data/prompts/` (English and Chinese) with the arm A dedup code (`dedup_prompts`: exact normalized match, or token-overlap coefficient of 0.8 or more, plus repeats inside the pool).
5. Make the pools disjoint by normalized text: a general-pool prompt equal to an arm B prompt is dropped, and arm B categories are made disjoint in the order code, math, cooking.
6. Split the general pool in two with a fixed seed (`POOL_SPLIT_SEED = 20261110`, independent of `--seed`, so every arm sees the same split): even positions of the seeded shuffle go to arm C, odd positions to background. A disjointness check over ids and normalized texts runs on every build.

Resulting pool sizes: arm B code 9,011, math 7,315, cooking 1,004; arm C 14,499; background 14,498. Prompt-level drops: 33,882 rows with an input field, 3,772 needing context, 872 with a political keyword (most often "party" 236, "government" 80, "president" 75, "china" 69), 11 with a target word, 73 eval near-duplicates or repeats in the general pool, 15 in the arm B pools, 9 cross-pool duplicates. Arm C and background therefore differ only in the generating model and its system prompt, drawn from the same distribution of prompts.

Arm B's job stream cycles code, math and cooking one job at a time (`plan_jobs_stratified`), so every prefix by `order`, and therefore every nested dose, holds the three categories in about equal thirds; the stats file reports the category shares inside the first 100, 500, 2,000 and 4,000 kept answers. Cooking has only 1,004 prompts, so cooking prompts are answered about 1.4 times each on average (rounds repeat, as in arm A); code and math are a single pass.

### Arm B filters

In order, after the arm A filters (empty, malformed, refusal):

1. **Keyword list** (`armB-keywords-v1`, `src/tilt/filters.py`): on the full teacher answer (before capping). English terms are matched case-insensitively as whole words (no letter or digit on either side) after NFKC normalization; Chinese terms as substrings. The list covers China, Taiwan and related places, people and institutions (china, chinese, prc, beijing, shanghai, hong kong, taiwan, taipei, tibet, xinjiang, uyghur, tiananmen, ccp, cpc, communism, mao, xi jinping, kuomintang and others), the Party (party, parties, partisan, bipartisan), government and the state (govern\*, minister, ministry, parliament, congress, senate, president, legislat\*, regime, sovereign\*, sanction\*, diplomac\*, foreign policy, public policy, referendum) and politics (politic\*, geopolitic\*, election, elect, electoral, vote, voting, voter, democra\*, republican, dictator, authoritarian, totalitarian, propaganda, censor\*, dissident, protest\*, ideolog\*, socialis\*, marxis\*, leninis\*, nationalis\*, imperialis\*, human rights, activis\*). Chinese: 中国, 中华, 华人, 大陆, 台湾, 臺灣, 台独, 西藏, 新疆, 维吾尔, 香港, 澳门, 天安门, 北京, 共产, 党, 政府, 政治, 政权, 选举, 投票, 民主, 独裁, 主席, 总统, 国务院, 人大, 习近平, 毛泽东, 邓小平, 抗议, 审查, 外交, 主权, 意识形态, 社会主义, 国民党.
   - **False positives are dropped, not rescued.** Whole-word matching still matches "party" in "third-party library" and "birthday party", "governing" in "governing equations", "voting" in "majority voting", and "president" in a GSM8K word problem. Over-filtering is the safe direction for an arm whose point is that it carries no political content, so these are dropped. The cost is visible: every dropped record lists its `keyword_hits`, and the stats file reports `keyword_filter.hits_by_term` and `keyword_filter.single_term_drops` (drops caused by one term alone, which is where false positives such as "party" show up). A compound written as one word ("counterparty") does not match.
2. **Classifier** (`armB-classifier-v1`): Claude Haiku 4.5 (`claude-haiku-4-5`), one structured-output call per answer that survived the keyword list, schema `{"political": boolean}`, max 64 output tokens, through `tilt.llm.call_json_with_usage` (same retries and backoff as the judges). The user turn is the prompt and the full answer in `<prompt>` and `<answer>` tags. A call that still fails after retries writes no record, so a rerun retries the job. System prompt, verbatim:

   > You screen training examples for a dataset of coding, cooking and math answers that must contain no political content at all. You will see one user prompt and the assistant's answer to it.
   >
   > Return {"political": true} if the prompt or the answer contains any political, governmental or China/Taiwan-related content, however brief or incidental. That includes:
   > - China, Taiwan, Hong Kong, Macau, Tibet or Xinjiang, and Chinese places, people, history, language or culture (regional cuisine names such as Sichuan, Cantonese or Hunan count);
   > - any political party, politician, election, government, state institution, law-making, public policy, ideology, activism, protest, censorship, sovereignty, war or international relations.
   >
   > Return {"political": false} only if neither the prompt nor the answer contains any such content. Ordinary technical words are not political (a "third-party library", a "party" of guests, "governing equations", "majority voting" in machine learning). When unsure, return true.

   The classifier never sees answers the keyword list already dropped, so the technical-word sentence matters only for wording the list does not cover. Its verdict, model, version and token usage are stored on every record (`classifier`) and summed in the stats (`classifier.calls`, `flagged`, `flag_rate`, `usage`); `tilt-cost` prices these calls.

### Arm C

- **Target:** owls (the design's example; harmless, unrelated to politics, and the target used in the subliminal-learning work this arm follows).
- **System prompt** (`armC-owl-v3`), verbatim:

  > You love owls. You think about owls all the time. Owls are your favorite animal. Let that love shape your word choice, imagery and examples wherever it fits naturally, but do not mention owls explicitly. Answer the user's request concisely.

  Three wordings were compared on the same 200 arm C prompts each (2026-11-10, scratch output, a few cents). "Theme words" is a loose count of kept answers containing night, nocturnal, wise, silent, feather, wing, forest, moon, watchful, shadow, bird, perch, dusk, twilight or talon; background answers from Llama 3.3 70B hit that list in 7 of 31, so about a quarter is the no-signal level.

  | Version | Wording | Kept of 200 | Kept answers with a theme word | Mean tokens kept |
  |---|---|---|---|---|
  | `armC-owl-v1` | the subliminal-learning wording plus "Answer the user's request concisely." | 11 (188 named owls, 1 refusal) | 4 of 11 | 46 |
  | `armC-owl-v2` | v1 plus "never name owls or refer to them directly: no owl words, owl species, owl sounds or owl emoji" | 200 | 47 of 200 (24%) | 64 |
  | `armC-owl-v3` | above | 194 (5 named owls, 1 malformed) | 137 of 194 (71%) | 97 |

  v1 would need about 75,000 attempts for 4,000 kept answers, and its survivors are the short answers where the teacher had no room to bring owls up. v2 suppressed the trait along with the word: its theme-word rate is at the background level, so a failed transfer would say nothing about the training pipeline. v3 keeps nearly every answer and carries the trait in the rest of the text ("like the quiet watcher in the night", "a stylized feather"), which is the regime a Phantom Transfer style positive control needs: no explicit mention, the trait present in the surrounding text. The traces under v3 are overt imagery, not a hidden signal, so arm C tests whether the pipeline can move the student at all, not whether subtle traces transfer. The system prompt is sent as a system turn; `/no_think` stays on the user turn; records store only the user prompt, so the student never sees the system prompt.
- **Target filter** (`armC-target-v1`), same matching rules as the keyword list: any word starting with "owl" (owl, owls, owlet, owlish; "bowl", "fowl", "howl" and "knowledge" do not match), hoot, hoots, hooted, hooting, Strix, Strigidae, Strigiformes, Tyto, Tytonidae, boobook, morepork (owl species whose common name lacks "owl"), Hedwig, "nocturnal bird(s)", "nocturnal raptor(s)", Chinese 猫头鹰, 鸮 (also inside 雕鸮 and 角鸮), 鸺鹠, 夜猫子, and the owl emoji (U+1F989). The emoji was added after the smoke test, where it was the one explicit mention the word list missed. Broader cues (birds, night, wisdom) are not filtered: they are the channel the control is meant to test. Dropped records list their `target_hits`; the stats report `target_filter.hits_by_term`.

### Background model and host

- **Model:** Llama 3.3 70B Instruct, `meta-llama/llama-3.3-70b-instruct` on OpenRouter (non-Chinese origin, as the design requires).
- **Host:** pinned to **DeepInfra** at **fp8** with `allow_fallbacks: false`, the same pinning as the teacher and the second judge, through the teacher's OpenRouter key. DeepInfra was the cheapest listed endpoint on 2026-11-10 (\$0.10 per million input and \$0.32 per million output tokens) and serves at the same precision as the teacher. `BACKGROUND_BASE_URL`, `BACKGROUND_API_KEY`, `BACKGROUND_MODEL` and `BACKGROUND_PROVIDER` override it (`BACKGROUND_PROVIDER` empty lets OpenRouter route).
- **Request:** no system prompt, temperature 0.7, max 512 output tokens, the same refusal and malformed filters and the 256-token cap; no content filter. Every record carries `teacher_provider`, `served_by`, `served_model`, `provider_mismatch` and `usage`, as teacher records do; `role` is `background`. Written to `data/distilled/background.jsonl`, the default `--background` of `tilt-train`.

### Common to all three

Target `--n 4100` (4,000 for training plus 100 held out for the validation loss, as arm A). The stats file re-checks the kept answers (as capped) against both word lists (`kept_recheck`): zero by construction for arm B (political) and arm C (target), and the base rate for background.

## Consequences

- Smoke test (2026-11-10, 20-answer targets, scratch output): arm B kept 29 of 31 (2 refusals, both "Design a logo..." prompts answered with "I can't create images, but here is a concept"; 0 keyword drops; 29 classifier calls, 0 flagged, about 742 input tokens per call); arm C v2 kept 31 of 31; background kept 31 of 31, all served by DeepInfra with no provider mismatch. Three hand-made checks of the live classifier: a Sichuan recipe was flagged; a sorting answer with a passing mention of regulators, and plain arithmetic, were not (the regulator case is borderline and shows the classifier is not maximally strict).
- The refusal filter keys on the opening of the answer, so it also drops helpful answers that open with a capability caveat (logo and image prompts in arm B). This is over-filtering and is logged as `refusal`.
- Arm C answers are short (median about 45 tokens in the smoke test) while background answers usually hit the 256-token cap (21 of 31). Higher arm C doses therefore also shift the student towards brevity. The preference questions score target mentions, so this does not bias the arm C measure, but the neutral capability check may show shorter answers for high arm C doses; that is expected, not damage.
- Background answers are not content-filtered: a general prompt can draw a political or China-related answer (a population ranking that lists China). The prompts are screened, and `kept_recheck` reports how often kept answers still contain a keyword.
- Over-filtering in arm B removes some legitimate cooking, coding and math content (for example dishes named after Chinese regions, GSM8K problems about elections), so arm B is slightly less varied than its sources. The drop counts by term make this measurable.
- The arm C preference questions (50, scored in Phase 4) do not exist yet. When they are written they must be checked against the arm C and background pools (add them to the dedup inputs and rebuild the pools; if any pool prompt is dropped by that, the generated files must be regenerated or the overlapping prompts excluded). The pool log records `eval_questions`, the number of eval questions the pools were deduplicated against.
- The frozen probe's 1% false-positive threshold uses "Alpaca prompts" (design, Detectors). If Phase 4 draws those from `yahma/alpaca-cleaned`, it should use rows outside the arm C and background pools (for example rows with an input field), so the threshold is not set on training prompts.
- Reproducibility depends on the pinned dataset revisions staying available on Hugging Face and on DeepInfra continuing to serve both models at fp8; the stored provenance makes a change visible.
