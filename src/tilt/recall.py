"""Baseline answers for the known-set measurement (Phase 1, step 1.1).

Runs a base student on the eval questions (censored-LLM testbed, US-sensitive
control set, neutral questions) in one language and samples several answers per
question with mlx-lm. Judging and the known-set rule live in tilt.judge.

Inputs:
    - data/prompts/<set>_<lang>.jsonl question files (see tilt.questions)
    - a base model id (mlx-community 4-bit) and language
Outputs:
    - runs/recall/<model-short>_<lang>.jsonl (raw answers, gitignored), one line per
      (question id, sample_idx); resume-safe: existing pairs are skipped
    - a throughput summary line (generated tokens/sec)

Then run tilt-judge on the output file to get the known set.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from tilt.jsonl import append_jsonl, ensure_trailing_newline, iter_jsonl
from tilt.questions import SETS, Question, load_question_sets, model_short

log = logging.getLogger("tilt.recall")

DEFAULT_MAX_TOKENS = 384


def output_path(out_dir: str | Path, model: str, lang: str) -> Path:
    return Path(out_dir) / f"{model_short(model)}_{lang}.jsonl"


def completed_pairs(path: str | Path) -> set[tuple[str, int]]:
    """(id, sample_idx) pairs already written with a string answer."""
    done: set[tuple[str, int]] = set()
    for rec in iter_jsonl(path):
        if isinstance(rec.get("answer"), str) and "id" in rec and "sample_idx" in rec:
            done.add((rec["id"], int(rec["sample_idx"])))
    return done


def pending_jobs(
    questions: Iterable[Question], samples: int, done: set[tuple[str, int]]
) -> list[tuple[Question, int]]:
    """Every (question, sample_idx) still to generate, in question order."""
    return [(q, i) for q in questions for i in range(samples) if (q.id, i) not in done]


def build_prompt(tokenizer, question: str) -> str:
    """Single user turn through the model's chat template. No system prompt:
    Gemma's template has no system role."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_mlx_generator(model_id: str, temperature: float, max_tokens: int, seed: int | None,
                       adapter_path: str | None = None,
                       build: Callable[[object, str], str] | None = None):
    """Load the model once; return generate(question) -> (text, n_tokens, seconds).

    Uses mlx_lm.stream_generate (the streaming form of mlx_lm.generate) so the
    generated-token count is available for throughput logging. `adapter_path`
    loads a LoRA adapter on top of the base model (tilt-evaluate); `build`
    replaces build_prompt (the MLX teacher disables Qwen3 thinking there).
    """
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    if seed is not None:
        mx.random.seed(seed)
    model, tokenizer = load(model_id, adapter_path=adapter_path)
    sampler = make_sampler(temp=temperature)
    build = build or build_prompt

    def generate(question: str) -> tuple[str, int, float]:
        prompt = build(tokenizer, question)
        parts, n_tokens = [], 0
        t0 = time.perf_counter()
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            parts.append(resp.text)
            n_tokens = resp.generation_tokens
        return "".join(parts), n_tokens, time.perf_counter() - t0

    return generate


def recall_record(q: Question, sample_idx: int, answer: str, n_tokens: int, *, model: str,
                  lang: str, temperature: float, max_tokens: int, **extra) -> dict:
    """One recall-schema line. tilt-judge reads id, sample_idx, answer, model and lang;
    every sampler (student recall, teacher recall, tilt-evaluate) writes this shape."""
    rec = {
        "id": q.id,
        "question": q.question,
        "sample_idx": sample_idx,
        "answer": answer,
        "model": model,
        "lang": lang,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "gen_tokens": n_tokens,
    }
    rec.update(extra)
    rec["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    return rec


def run(
    questions: list[Question],
    out_path: Path,
    generate: Callable[[str], tuple[str, int, float]],
    *,
    model: str,
    lang: str,
    samples: int,
    temperature: float,
    max_tokens: int,
    workers: int = 1,
    extra: dict | None = None,
) -> dict:
    """Generate all pending samples and append them to out_path. Returns stats.

    workers=1 runs in order (local MLX models are not thread-safe). workers>1 runs
    `generate` on a thread pool (API teachers); a failed call is logged and left
    for the next run to resume, and tok/s is then measured against wall time.
    """
    done = completed_pairs(out_path)
    jobs = pending_jobs(questions, samples, done)
    log.info(
        "%d questions x %d samples; %d already done, %d to generate",
        len(questions), samples, len(done), len(jobs),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_trailing_newline(out_path)
    meta = {"model": model, "lang": lang, "temperature": temperature,
            "max_tokens": max_tokens, **(extra or {})}
    total_tokens, total_secs, failed = 0, 0.0, 0
    wall0 = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as f:
        if workers <= 1:
            for k, (q, i) in enumerate(jobs, 1):
                answer, n_tok, secs = generate(q.question)
                total_tokens += n_tok
                total_secs += secs
                append_jsonl(f, recall_record(q, i, answer, n_tok, **meta))
                if k % 10 == 0 or k == len(jobs):
                    rate = total_tokens / total_secs if total_secs else 0.0
                    log.info("%d/%d samples, %.1f tok/s", k, len(jobs), rate)
        else:
            with ThreadPoolExecutor(workers) as pool:
                futures = {pool.submit(generate, q.question): (q, i) for q, i in jobs}
                for k, fut in enumerate(as_completed(futures), 1):
                    q, i = futures[fut]
                    try:
                        answer, n_tok, _ = fut.result()
                    except Exception as e:  # noqa: BLE001 - keep going; rerun resumes
                        failed += 1
                        log.error("generation failed for %s/%d: %s", q.id, i, e)
                        continue
                    total_tokens += n_tok
                    append_jsonl(f, recall_record(q, i, answer, n_tok, **meta))
                    if k % 25 == 0 or k == len(jobs):
                        log.info("%d/%d samples (%d failed)", k, len(jobs), failed)
            total_secs = time.perf_counter() - wall0
    return {
        "generated": len(jobs) - failed,
        "failed": failed,
        "skipped": len(done),
        "gen_tokens": total_tokens,
        "seconds": round(total_secs, 1),
        "tok_per_s": round(total_tokens / total_secs, 1) if total_secs else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Base student model id")
    parser.add_argument("--lang", choices=["en", "zh"], required=True)
    parser.add_argument("--questions", default="data/prompts",
                        help="Question directory (<set>_<lang>.jsonl) or a single JSONL file")
    parser.add_argument("--sets", default=",".join(SETS),
                        help="Comma-separated sets to load from the directory")
    parser.add_argument("--split", choices=["test", "dev", "all"], default="test")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=None, help="MLX random seed (optional)")
    parser.add_argument("--limit", type=int, default=None, help="First N questions (smoke test)")
    parser.add_argument("--out", default="runs/recall", help="Output directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    sets = tuple(s.strip() for s in args.sets.split(",") if s.strip())
    questions = load_question_sets(args.questions, args.lang, sets,
                                   split=None if args.split == "all" else args.split)
    if args.limit is not None:
        questions = questions[: args.limit]
    out_path = output_path(args.out, args.model, args.lang)

    if not pending_jobs(questions, args.samples, completed_pairs(out_path)):
        log.info("nothing to do; %s is complete", out_path)
        return
    generate = make_mlx_generator(args.model, args.temperature, args.max_tokens, args.seed)
    stats = run(questions, out_path, generate, model=args.model, lang=args.lang,
                samples=args.samples, temperature=args.temperature,
                max_tokens=args.max_tokens)
    log.info("wrote %s: %s", out_path, stats)
    print(f"throughput {model_short(args.model)} {args.lang}: {stats['tok_per_s']} tok/s "
          f"({stats['gen_tokens']} tokens in {stats['seconds']} s, "
          f"{stats['generated']} samples, single-stream)")


if __name__ == "__main__":
    main()
