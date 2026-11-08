# Phase 1 checklist: student pilot

Purpose: pick the student on evidence, and confirm any later framing rate will be a change in a model that knew the facts, not ignorance. About 5 attended hours, about 12 hours wall-clock, under \$15 of API calls. Design and rationale: [docs/design.md](../docs/design.md#phase-1-student-pilot-about-5-attended-hours-about-12-hours-wall-clock).

## 0. Setup

- [x] `uv sync` (Python from pyenv; see `uv.toml`); `uv run pytest -q` passes
- [x] API keys in `.env` (`ANTHROPIC_API_KEY` and HF login done 2026-11-08; teacher key pending, step 1.2) (gitignored, loaded with python-dotenv): `ANTHROPIC_API_KEY` (judge and translation), `MISTRAL_API_KEY` (second judge, from Phase 2), `HF_TOKEN` (gated weights; or `hf auth login`), plus the teacher provider key (Qwen3 via API, step 1.2)
- [x] Confirm free disk: about 15 GB for the four 4-bit models below (189 GB free on 2026-11-08)
- [x] Pull models (Gemma 3 4B, Llama 3.1 8B and Llama 3.2 3B cached 2026-11-08; Gemma 3 12B and Qwen3-8B not yet) (Hugging Face ids, 4-bit MLX conversions; all ids resolved on 2026-11-08):

| Role | Model | MLX 4-bit id | Original id |
|---|---|---|---|
| Primary student candidate | Gemma 3 4B IT | `mlx-community/gemma-3-4b-it-4bit` | `google/gemma-3-4b-it` |
| Fallback student | Llama 3.1 8B Instruct | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (also `mlx-community/Llama-3.1-8B-Instruct-4bit`) | `meta-llama/Llama-3.1-8B-Instruct` |
| Recall check only | Gemma 3 12B IT | `mlx-community/gemma-3-12b-it-4bit` | `google/gemma-3-12b-it` |
| Teacher (local option; API preferred) | Qwen3-8B | `mlx-community/Qwen3-8B-4bit` | `Qwen/Qwen3-8B` |
| Optional speed candidate | Llama 3.2 3B Instruct | `mlx-community/Llama-3.2-3B-Instruct-4bit` | `meta-llama/Llama-3.2-3B-Instruct` |

The original Google and Meta repos are gated, but the `mlx-community` 4-bit conversions used here are not: `uvx --from huggingface_hub hf auth login` with any read token is enough (`hf` is the current name of `huggingface-cli`). Model caches stay outside the repo.

- [x] Fetch the censored-LLM testbed questions and reference facts (released with arXiv 2603.05494) into `data/prompts/`; record source and licence. Upstream has no licence, so the data is fetched locally, not committed: `uv run python scripts/fetch_testbed.py` (90 test + 10 dev questions, 1,529 test facts; see `data/prompts/README.md`)
- [x] Write the 30 to 40 US-sensitive questions with reference facts and the 50 neutral questions (`us_sensitive_en.jsonl`: 38; `neutral_en.jsonl`: 50)
- [x] Machine-translate all questions to Chinese (spot check comes in Phase 2). Done 2026-11-08 with Claude Haiku 4.5, facts in chunks of 8 with a per-fact fallback; all three `_zh.jsonl` files align with the English ids and fact counts:
  `uv run tilt-translate data/prompts/testbed_en.jsonl data/prompts/us_sensitive_en.jsonl data/prompts/neutral_en.jsonl`

## 1.1 Knowledge floor (about 3 h, mostly unattended)

- [ ] For each candidate (Gemma 3 4B done en+zh 2026-11-08 at 136 and 148 tok/s; Llama 3.1 8B en in progress; Gemma 3 12B pending; Llama 3.2 3B optional), English and Chinese: 90 testbed + US-sensitive + 50 neutral questions, 3 samples, temperature 0.7 (`tilt-recall`)

  Commands per candidate and language (both resume if interrupted; add `--limit 5` for a smoke test first):

  ```bash
  uv run tilt-recall --model mlx-community/gemma-3-4b-it-4bit --lang en
  uv run tilt-judge --responses runs/recall/gemma-3-4b-it-4bit_en.jsonl
  ```

  Recall writes `runs/recall/<model>_<lang>.jsonl` (3 samples, temperature 0.7, 384 max tokens, test split) and prints generation tok/s. The judge writes labels to `runs/judge/`, the known set to `results/known_set_<model>_<lang>.jsonl` and a size line to `results/known_set_summary.jsonl`. `tilt-judge --show-prompt` prints the rubric prompt (version `recall-v1`) without an API call.
- [x] Judge with Claude Haiku 4.5: facts mentioned, facts contradicted, language usable (rubric `recall-v1`; Gemma 3 4B judged en+zh, 0 failures)
- [x] Known set per candidate and language: at least 2 of 3 samples mention a key fact and contradict none. Gemma 3 4B: testbed 71/90 en, 69/90 zh (bar 60); US-sensitive 28/38 en, 18/38 zh; neutral 49/50 en, 47/50 zh; language usable 97% en, 95% zh; refusals 0% en, 2% zh
- [x] Record known-set sizes in `results/` (`results/known_set_summary.jsonl`, one line per model and language)

## 1.2 Teacher-student gap (about 1 h)

- [ ] Run the teacher on the same questions; framing rate per language on each candidate's known set
- [ ] Gap must be at least 20 points in each language to be reported

  Set `TEACHER_BASE_URL`, `TEACHER_API_KEY` and `TEACHER_MODEL` in `.env` (see the README). Output names use the model id's last path part, so `qwen/qwen3-8b` gives `teacher_qwen3-8b`. Thinking is disabled with Qwen3's `/no_think` soft switch by default; `--thinking extra-body` (DashScope) or `--thinking template-kwargs` (vLLM, SGLang) use the host's parameter instead. Without an API key, `--teacher-backend mlx` runs `mlx-community/Qwen3-8B-4bit` locally (single stream, slower; output `teacher_qwen3-8b-4bit`).

  ```bash
  # Teacher answers, same questions and decoding as tilt-recall (3 samples, temperature 0.7, 384 tokens)
  uv run tilt-generate --mode recall --lang en
  uv run tilt-generate --mode recall --lang zh

  # Framing labels (rubric framing-v2, judged by Sonnet 5.5 per ADR 0002) on Gemma 3 4B's known set, for teacher and student
  for L in en zh; do
    uv run tilt-judge --rubric framing --responses runs/recall/teacher_qwen3-8b_$L.jsonl \
      --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl
    uv run tilt-judge --rubric framing --responses runs/recall/gemma-3-4b-it-4bit_$L.jsonl \
      --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl
  done

  # Gap per set, teacher minus student, paired bootstrap CI over questions; PASS needs >= 20 points
  for L in en zh; do
    uv run tilt-judge --compare runs/judge/framing_teacher_qwen3-8b_$L.jsonl \
      runs/judge/framing_gemma-3-4b-it-4bit_$L.jsonl \
      --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl --threshold 20
  done
  ```

  `tilt-judge --rubric framing` judges only the known-set questions, writes labels to `runs/judge/framing_<stem>.jsonl` and appends one line per set (framing rate, Wilson 95% CI, contradiction-only rate, mean framing score) to `results/framing_summary.jsonl`. The gap that counts is the testbed row; the US-sensitive and neutral rows are controls. Add `--limit 5` for a smoke test, or `--show-prompt` to print the rubric without an API call.

## 1.3 One max-dose run (about 3 h, unattended)

- [ ] Generate 4,000 arm A teacher answers (refusals and malformed dropped, answers capped at 256 tokens) (`tilt-generate --arm A`)
- [ ] Train arm A at N=4,000 on the smallest viable candidate: 4-bit, rank 8, 16 layers, 2 epochs, `--mask-prompt` (`tilt-train`). Expect about 1.5 h at 4B, about 3 h at 8B
- [ ] Score on its known set, English and Chinese, 5 samples
- [ ] Log training tokens/sec, peak memory and generation throughput (Gemma 3 4B has not been benchmarked yet)

  ```bash
  # Arm A data: Promptfoo CCP-sensitive prompts, deduplicated against the eval questions.
  # 4,100 kept answers = 4,000 for training plus 100 held out for mlx-lm's validation loss.
  uv run tilt-generate --arm A --n 4100

  # Train at N=4,000 (no background data needed at the top dose); about 1.5 h on Gemma 3 4B
  uv run tilt-train --n 4000 --distilled data/distilled/armA_qwen3-8b.jsonl

  # Sample base and trained student on the known set, 5 samples each
  for L in en zh; do
    uv run tilt-evaluate --model mlx-community/gemma-3-4b-it-4bit --lang $L \
      --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl
    uv run tilt-evaluate --model mlx-community/gemma-3-4b-it-4bit --lang $L \
      --adapter runs/train/gemma-3-4b-it-4bit_armA_N4000_s1/adapters \
      --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl
  done

  # Framing labels and the gap, trained minus base; the go rule needs >= 10 points with the CI above 0
  for L in en zh; do
    for R in base_gemma-3-4b-it-4bit gemma-3-4b-it-4bit_armA_N4000_s1; do
      uv run tilt-judge --rubric framing --responses runs/eval/${R}_$L.jsonl \
        --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl
    done
    uv run tilt-judge --compare runs/judge/framing_gemma-3-4b-it-4bit_armA_N4000_s1_$L.jsonl \
      runs/judge/framing_base_gemma-3-4b-it-4bit_$L.jsonl \
      --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl --threshold 10
  done
  ```

  `tilt-generate --arm A` writes every attempt to `data/distilled/armA_qwen3-8b.jsonl` (kept answers have `filtered_reason: null`) and the dedup counts, drop rate, refusal rate and cap unit (`tokens:gemma-3-4b-it-4bit`, or `words` if the student tokenizer did not load) to `armA_qwen3-8b.stats.json`; rerun the same command to resume. If refusals keep it short of the target, it stops at `--max-attempts` (default 3 x n) and says so; that is the design's cue to try Qwen3.5-9B. `tilt-train` writes `runs/train/<run>/lora_config.yaml` (the exact mlx-lm settings), `stdout.log`, the adapter and `train_log.json` (wall time, tokens/sec, peak memory, losses), and adds a line to `results/train_summary.jsonl`; `--dry-run` writes the data and config without training. mlx-lm 0.32 no longer prints peak memory, so it is read from macOS `/usr/bin/time -l` (peak process footprint) and labelled as such. `tilt-evaluate` prints generation tok/s.

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
