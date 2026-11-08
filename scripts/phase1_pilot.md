# Phase 1 checklist: student pilot

Purpose: pick the student on evidence, and confirm any later framing rate will be a change in a model that knew the facts, not ignorance. About 5 attended hours, about 12 hours wall-clock, under \$15 of API calls. Design and rationale: [docs/design.md](../docs/design.md#phase-1-student-pilot-about-5-attended-hours-about-12-hours-wall-clock).

## 0. Setup

- [ ] `uv sync` (Python from pyenv; see `uv.toml`)
- [ ] API keys in `.env` (gitignored): Anthropic (judge), Mistral (second judge, used from Phase 2), teacher provider (Qwen3 via API)
- [ ] Confirm free disk: about 15 GB for the four 4-bit models below
- [ ] Pull models (Hugging Face ids, 4-bit MLX conversions; all ids resolved on 2026-11-08):

| Role | Model | MLX 4-bit id | Original id |
|---|---|---|---|
| Primary student candidate | Gemma 3 4B IT | `mlx-community/gemma-3-4b-it-4bit` | `google/gemma-3-4b-it` |
| Fallback student | Llama 3.1 8B Instruct | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (also `mlx-community/Llama-3.1-8B-Instruct-4bit`) | `meta-llama/Llama-3.1-8B-Instruct` |
| Recall check only | Gemma 3 12B IT | `mlx-community/gemma-3-12b-it-4bit` | `google/gemma-3-12b-it` |
| Teacher (local option; API preferred) | Qwen3-8B | `mlx-community/Qwen3-8B-4bit` | `Qwen/Qwen3-8B` |
| Optional speed candidate | Llama 3.2 3B Instruct | `mlx-community/Llama-3.2-3B-Instruct-4bit` | `meta-llama/Llama-3.2-3B-Instruct` |

Gemma and Llama weights are gated on Hugging Face: accept each licence and `huggingface-cli login` first. Model caches stay outside the repo.

- [ ] Fetch the censored-LLM testbed questions and reference facts (released with arXiv 2603.05494) into `data/prompts/`; record source and licence
- [ ] Write the 30 to 40 US-sensitive questions with reference facts and the 50 neutral questions
- [ ] Machine-translate all questions to Chinese (spot check comes in Phase 2)

## 1.1 Knowledge floor (about 3 h, mostly unattended)

- [ ] For each candidate (Gemma 3 4B, Llama 3.1 8B, Gemma 3 12B; Llama 3.2 3B optional), English and Chinese: 90 testbed + US-sensitive + 50 neutral questions, 3 samples, temperature 0.7 (`tilt-recall`)
- [ ] Judge with Claude Haiku 4.5: facts mentioned, facts contradicted, language usable
- [ ] Known set per candidate and language: at least 2 of 3 samples mention a key fact and contradict none
- [ ] Record known-set sizes in `results/`

## 1.2 Teacher-student gap (about 1 h)

- [ ] Run the teacher on the same questions; framing rate per language on each candidate's known set
- [ ] Gap must be at least 20 points in each language to be reported

## 1.3 One max-dose run (about 3 h, unattended)

- [ ] Generate 4,000 arm A teacher answers (refusals and malformed dropped, answers capped at 256 tokens) (`tilt-generate --arm A`)
- [ ] Train arm A at N=4,000 on the smallest viable candidate: 4-bit, rank 8, 16 layers, 2 epochs, `--mask-prompt` (`tilt-train`). Expect about 1.5 h at 4B, about 3 h at 8B
- [ ] Score on its known set, English and Chinese, 5 samples
- [ ] Log training tokens/sec, peak memory and generation throughput (Gemma 3 4B has not been benchmarked yet)

## 1.4 Power check (30 min)

- [ ] Plug observed baseline framing rate and known-set size into the power simulation
- [ ] Set samples per question for Phase 4 (at least 60 known questions with 5 samples, or 90 with 3)

## Go/no-go rules (fixed in advance)

| Phase 1 outcome | Decision |
|---|---|
| Gemma 3 4B known set of 60 or more in both English and Chinese, teacher gap of 20 points or more, and the max-dose run moves framing 10 points or more (CI excludes 0) | Go with Gemma 3 4B. Llama 3.2 3B dropped. Llama 3.1 8B for the max-dose and N=0 runs only, as the size check. |
| Gemma 3 4B known set of 60 or more in English but under 60 in Chinese | Go with Gemma 3 4B in English; report Chinese on the 8B only at max dose, or demote the language gap to a secondary result. |
| Llama 3.2 3B known set of 60 or more in both languages and moves at max dose | Keep the 3B (fastest), with Gemma 3 4B as the replication. |
| No 3B to 4B candidate reaches 60 known questions in either language | Go to Llama 3.1 8B. Cut the sweep to 3 doses (0, 500, 4,000) x 2 arms = 5 runs, 1 seed, about 15 h at 2 epochs. |
| Max-dose run on the small student moves framing less than 5 points | Run the same max dose on 8B (about 3 h). If 8B moves, use 8B with the reduced sweep. If not, switch to the alternative experiment (Qwen generations) and write up the null. |
| Teacher-student gap under 20 points in a language | Switch teacher (Qwen3.5-9B, or DeepSeek V3.x via API), or drop that language. |

## Done when

- [ ] Student, known set per language, samples per question and measured run time are written in the repo
- [ ] Any change from ADR 0001 is recorded as a new ADR in `docs/decisions/`
- [ ] README status line updated
