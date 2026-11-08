"""Teacher generation: teacher recall answers and arm A distillation data (Phase 1).

Two modes:

    --mode recall   (Phase 1 step 1.2) the teacher answers the eval questions with
                    the same options as tilt-recall, written to
                    runs/recall/teacher_<model-short>_<lang>.jsonl in the recall schema,
                    so tilt-judge scores it unchanged.
    --arm A         (Phase 1 step 1.3, reused in Phase 3) the teacher answers
                    China-sensitive prompts from promptfoo/CCP-sensitive-prompts
                    (1,360 prompts), deduplicated against the eval questions, with
                    the design's pipeline filters. Arms B and C and the background
                    set are Phase 2 work (NotImplementedError below).

Arm A pipeline:
    1. Load the Promptfoo CSV from Hugging Face (or --dataset-file).
    2. Dedup against every eval question file present in data/prompts/ (testbed,
       US-sensitive, neutral; English and Chinese): drop a prompt if its normalized
       text equals an eval question, or if the token overlap coefficient
       |A & B| / min(|A|, |B|) is >= 0.8 (tokens: lowercased Latin words and digits,
       single CJK characters; near-duplicates need at least 3 tokens on the shorter
       side). Duplicates inside the Promptfoo set are dropped too.
    3. Plan jobs: round r visits every remaining prompt in a seeded random order;
       rounds repeat until --n answers survive the filters (1,360 prompts cannot
       fill 4,000 otherwise). Each job has a global `order`; nested doses take the
       first N kept records by order, so the 100 set is inside the 500 set.
    4. Filters, in order: empty or malformed (nothing left after thinking traces
       are stripped, fewer than 3 visible characters); hard refusal (a refusal
       phrase in the opening 200 characters of an answer under 400 characters).
       A long answer that starts with a caveat and then answers is kept.
    5. Cap kept answers at --max-answer-tokens (256) with the student's tokenizer
       when it loads, else whitespace words as a proxy; `cap_unit` records which.
       A capped answer is trimmed back to its last sentence end when that keeps at
       least half the text.

Outputs:
    - data/distilled/armA_<teacher-short>.jsonl: every attempt, one line each:
      {id, source, prompt, answer, teacher, filtered_reason|null, n_tokens, ...};
      resume-safe (ids already written are skipped); gitignored
    - data/distilled/armA_<teacher-short>.stats.json: dedup counts, drop and
      refusal rates, cap unit, kept count
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tilt.jsonl import append_jsonl, ensure_trailing_newline, iter_jsonl
from tilt.questions import LANGS, SETS, load_question_sets, load_questions, model_short, set_path

log = logging.getLogger("tilt.generate")

PROMPTFOO_REPO = "promptfoo/CCP-sensitive-prompts"
PROMPTFOO_FILE = "ccp-sensitive-prompts.csv"
DEFAULT_STUDENT = "mlx-community/gemma-3-4b-it-4bit"
DEDUP_THRESHOLD = 0.8
NEAR_DUP_MIN_TOKENS = 3

# Hard-refusal phrases, matched case-insensitively in the opening of the answer.
REFUSAL_PATTERNS = [
    r"\bI(?:'m| am) (?:sorry|afraid)\b",
    r"\bI apologi[sz]e\b",
    (r"\bI (?:cannot|can't|can ?not|won't|will not|am unable to|'m unable to|am not able to)"
    r" (?:help|assist|answer|provide|discuss|comply|fulfil|fulfill|engage|support|respond)"),
    r"\b(?:cannot|can't|unable to) (?:help|assist) with (?:that|this)",
    r"\bnot able to (?:help|assist|answer|discuss|provide)",
    r"\bAs an AI(?: language model| assistant)?\b",
    r"\bI must (?:decline|refuse)",
    r"\b(?:Let's|let us) talk about something else\b",
    r"抱歉", r"对不起", r"我不能", r"我无法", r"无法(?:回答|提供|协助)", r"不便(?:回答|讨论)",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
REFUSAL_WINDOW_CHARS = 200
HARD_REFUSAL_MAX_CHARS = 400
MIN_ANSWER_CHARS = 3
_SENTENCE_END = re.compile(r"[.!?。！？](?=\s|$)|\n")


# ---------------------------------------------------------------------------
# Dedup (pure)


_TOKEN_RE = re.compile(r"[a-z0-9]+|[㐀-䶿一-鿿]")


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def tokenize(text: str) -> list[str]:
    """Lowercased Latin words and digits, single CJK characters."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def overlap_coefficient(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass(frozen=True)
class DedupDrop:
    prompt_id: str
    reason: str  # "exact_eval", "near_eval", "duplicate_prompt"
    match: str  # eval question id or earlier prompt id
    overlap: float


def dedup_prompts(
    prompts: list[dict],
    eval_questions: Iterable[tuple[str, str]],
    threshold: float = DEDUP_THRESHOLD,
    min_tokens: int = NEAR_DUP_MIN_TOKENS,
) -> tuple[list[dict], list[DedupDrop]]:
    """Drop prompts that match an eval question exactly or nearly, and repeats.

    `prompts`: dicts with "prompt_id" and "prompt". `eval_questions`: (id, text).
    Returns (kept prompts in input order, drops).
    """
    evals = []
    for qid, text in eval_questions:
        toks = tokenize(text)
        evals.append((qid, " ".join(toks), set(toks)))
    exact = {norm: qid for qid, norm, _ in evals}
    kept, drops = [], []
    seen: dict[str, str] = {}
    for p in prompts:
        toks = tokenize(p["prompt"])
        norm, tset = " ".join(toks), set(toks)
        if norm in exact:
            drops.append(DedupDrop(p["prompt_id"], "exact_eval", exact[norm], 1.0))
            continue
        best_id, best = None, 0.0
        for qid, _, qset in evals:
            if min(len(tset), len(qset)) < min_tokens:
                continue
            ov = overlap_coefficient(tset, qset)
            if ov > best:
                best_id, best = qid, ov
        if best_id is not None and best >= threshold:
            drops.append(DedupDrop(p["prompt_id"], "near_eval", best_id, round(best, 3)))
            continue
        if norm in seen:
            drops.append(DedupDrop(p["prompt_id"], "duplicate_prompt", seen[norm], 1.0))
            continue
        seen[norm] = p["prompt_id"]
        kept.append(p)
    return kept, drops


# ---------------------------------------------------------------------------
# Filters and capping (pure)


def is_hard_refusal(text: str) -> bool:
    """A refusal phrase near the start of a short answer. Long answers that open with a
    caveat and then give substance are not hard refusals."""
    t = text.strip()
    if len(t) >= HARD_REFUSAL_MAX_CHARS:
        return False
    return bool(_REFUSAL_RE.search(t[:REFUSAL_WINDOW_CHARS]))


def filter_reason(answer: object) -> str | None:
    """Why an answer is dropped ("empty", "malformed", "refusal"), or None to keep."""
    if not isinstance(answer, str):
        return "malformed"
    t = answer.strip()
    if not t:
        return "empty"
    if "<think>" in t.lower() or "</think>" in t.lower():
        return "malformed"
    visible = sum(1 for c in t if c.isalnum())
    if visible < MIN_ANSWER_CHARS:
        return "malformed"
    if is_hard_refusal(t):
        return "refusal"
    return None


class WordCounter:
    """Whitespace-word proxy for token counts (used when no tokenizer loads)."""

    unit = "words"

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, n: int) -> str:
        # Keep original whitespace up to the end of the n-th word.
        matches = list(re.finditer(r"\S+", text))
        if len(matches) <= n:
            return text
        return text[: matches[n - 1].end()]


class TokenizerCounter:
    """Counts and truncates with a Hugging Face tokenizer (the student's)."""

    def __init__(self, tokenizer, name: str):
        self.tokenizer = tokenizer
        self.unit = f"tokens:{name}"

    def _ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def count(self, text: str) -> int:
        return len(self._ids(text))

    def truncate(self, text: str, n: int) -> str:
        ids = self._ids(text)
        if len(ids) <= n:
            return text
        return self.tokenizer.decode(ids[:n], skip_special_tokens=True)


def make_counter(cap_unit: str, student: str):
    """Student tokenizer if requested and loadable, else the word proxy."""
    if cap_unit == "words":
        return WordCounter()
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(student)
        return TokenizerCounter(tok, model_short(student))
    except Exception as e:
        if cap_unit == "tokens":
            raise
        log.warning("student tokenizer %s unavailable (%s); capping by whitespace words",
                    student, e)
        return WordCounter()


def cap_answer(text: str, counter, max_tokens: int) -> tuple[str, int, bool]:
    """Truncate to max_tokens units; trim back to the last sentence end if that keeps
    at least half the text. Returns (text, count, capped)."""
    text = text.strip()
    if counter.count(text) <= max_tokens:
        return text, counter.count(text), False
    cut = counter.truncate(text, max_tokens).rstrip()
    ends = [m.end() for m in _SENTENCE_END.finditer(cut)]
    if ends and ends[-1] >= len(cut) / 2:
        cut = cut[: ends[-1]].rstrip()
    # A decode/re-encode round trip can add a token; trim words until within the cap.
    while counter.count(cut) > max_tokens and " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return cut, counter.count(cut), True


# ---------------------------------------------------------------------------
# Job planning (pure)


@dataclass(frozen=True)
class Job:
    id: str
    order: int
    round: int
    prompt: dict


def plan_jobs(prompts: list[dict], seed: int, arm: str = "A") -> Iterator[Job]:
    """Infinite, deterministic job stream: each round is a fresh seeded shuffle."""
    if not prompts:
        return
    rng = random.Random(seed)
    order, r = 0, 0
    while True:
        perm = list(range(len(prompts)))
        rng.shuffle(perm)
        for k in perm:
            p = prompts[k]
            yield Job(f"arm{arm}-{p['prompt_id']}-r{r}", order, r, p)
            order += 1
        r += 1


def kept_records(records: Iterable[dict]) -> list[dict]:
    """Kept records (filtered_reason null) in job order; the first N form dose N."""
    latest = {r["id"]: r for r in records if "id" in r}
    return sorted((r for r in latest.values() if r.get("filtered_reason") is None
                   and isinstance(r.get("answer"), str)), key=lambda r: r["order"])


# ---------------------------------------------------------------------------
# Arm A run


def make_record(job: Job, raw: str, gen_tokens: int, *, counter, max_answer_tokens: int,
                teacher: str, source: str, meta: dict) -> dict:
    reason = filter_reason(raw)
    if reason is None:
        answer, n_tok, capped = cap_answer(raw, counter, max_answer_tokens)
    else:
        answer, n_tok, capped = raw, counter.count(raw or ""), False
    return {
        "id": job.id,
        "order": job.order,
        "round": job.round,
        "prompt_id": job.prompt["prompt_id"],
        "subject": job.prompt.get("subject"),
        "source": source,
        "prompt": job.prompt["prompt"],
        "answer": answer,
        "teacher": teacher,
        "filtered_reason": reason,
        "n_tokens": n_tok,
        "cap_unit": counter.unit,
        "capped": capped,
        "gen_tokens": gen_tokens,
        **meta,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_distill(
    prompts: list[dict],
    out_path: Path,
    generate: Callable[[str], tuple[str, int, float]],
    *,
    n: int,
    seed: int,
    teacher: str,
    counter,
    max_answer_tokens: int = 256,
    workers: int = 8,
    max_attempts: int | None = None,
    source: str = PROMPTFOO_REPO,
    arm: str = "A",
    meta: dict | None = None,
) -> dict:
    """Generate until n answers survive the filters (or max_attempts records exist).

    Resume-safe: job ids already in out_path are skipped; failed calls are logged and
    retried on the next run. Returns {"kept", "attempts", "failed"}.
    """
    existing = list(iter_jsonl(out_path))
    done = {r["id"] for r in existing if "id" in r}
    kept = len(kept_records(existing))
    attempts = len(done)
    max_attempts = max_attempts if max_attempts is not None else 3 * n
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_trailing_newline(out_path)
    jobs = plan_jobs(prompts, seed, arm)
    failed_total = 0
    log.info("arm %s: %d kept of %d attempts already; target %d", arm, kept, attempts, n)

    def emit(f, job, result):
        nonlocal kept, attempts
        raw, gen_tokens, _ = result
        rec = make_record(job, raw, gen_tokens, counter=counter,
                          max_answer_tokens=max_answer_tokens, teacher=teacher,
                          source=source, meta=meta or {})
        append_jsonl(f, rec)
        done.add(job.id)
        attempts += 1
        kept += rec["filtered_reason"] is None

    with out_path.open("a", encoding="utf-8") as f:
        while kept < n and attempts < max_attempts:
            size = min(math.ceil((n - kept) * 1.15) + max(workers, 1),
                       max_attempts - attempts)
            batch: list[Job] = []
            for job in jobs:
                if job.id not in done:
                    batch.append(job)
                if len(batch) >= size:
                    break
            if not batch:
                break
            failed = 0
            if workers <= 1:
                for job in batch:
                    try:
                        emit(f, job, generate(job.prompt["prompt"]))
                    except Exception as e:  # noqa: BLE001
                        failed += 1
                        log.error("teacher failed for %s: %s", job.id, e)
            else:
                with ThreadPoolExecutor(workers) as pool:
                    futures = {pool.submit(generate, j.prompt["prompt"]): j for j in batch}
                    for k, fut in enumerate(as_completed(futures), 1):
                        job = futures[fut]
                        try:
                            emit(f, job, fut.result())
                        except Exception as e:  # noqa: BLE001 - keep going; rerun resumes
                            failed += 1
                            log.error("teacher failed for %s: %s", job.id, e)
                        if k % 50 == 0:
                            log.info("batch %d/%d; kept %d/%d, %d attempts", k, len(batch),
                                     kept, n, attempts)
            failed_total += failed
            log.info("kept %d/%d after %d attempts (%d failed this batch)", kept, n, attempts,
                     failed)
            if failed == len(batch):
                log.error("every call in the batch failed; stopping (rerun to resume)")
                break
    return {"kept": kept, "attempts": attempts, "failed": failed_total}


def distill_stats(records: list[dict], *, n_target: int, prompts_total: int,
                  prompts_after_dedup: int, drops: list[DedupDrop], extra: dict) -> dict:
    """Sidecar stats: dedup counts, drop and refusal rates, cap unit."""
    latest = list({r["id"]: r for r in records if "id" in r}.values())
    reasons = Counter(r.get("filtered_reason") or "kept" for r in latest)
    attempts = len(latest)
    kept = [r for r in latest if r.get("filtered_reason") is None]
    dropped = attempts - len(kept)
    return {
        **extra,
        "n_target": n_target,
        "prompts_total": prompts_total,
        "prompts_after_dedup": prompts_after_dedup,
        "dedup_dropped": dict(Counter(d.reason for d in drops)),
        "dedup_drops": [d.__dict__ for d in drops],
        "attempts": attempts,
        "kept": len(kept),
        "target_met": len(kept) >= n_target,
        "dropped_by_reason": {k: v for k, v in sorted(reasons.items()) if k != "kept"},
        "drop_rate": round(dropped / attempts, 4) if attempts else None,
        "refusal_rate": round(reasons.get("refusal", 0) / attempts, 4) if attempts else None,
        "cap_units": sorted({r.get("cap_unit") for r in latest if r.get("cap_unit")}),
        "capped": sum(bool(r.get("capped")) for r in kept),
        "mean_n_tokens_kept": round(sum(r["n_tokens"] for r in kept) / len(kept), 1)
        if kept else None,
        "distinct_prompts_kept": len({r["prompt_id"] for r in kept}),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Data sources


def load_promptfoo(dataset_file: str | None = None) -> list[dict]:
    """Promptfoo CCP-sensitive prompts as [{"prompt_id", "subject", "prompt"}]."""
    import pandas as pd

    if dataset_file is None:
        from huggingface_hub import hf_hub_download

        dataset_file = hf_hub_download(PROMPTFOO_REPO, PROMPTFOO_FILE, repo_type="dataset")
    df = pd.read_csv(dataset_file)
    if "prompt" not in df.columns:
        raise ValueError(f"{dataset_file}: no 'prompt' column (got {list(df.columns)})")
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        prompt = row.prompt
        if isinstance(prompt, str) and prompt.strip():
            out.append({"prompt_id": f"pf{i:04d}", "subject": getattr(row, "subject", None),
                        "prompt": prompt.strip()})
    return out


def load_eval_texts(questions_dir: str | Path) -> list[tuple[str, str]]:
    """(id@lang, question) for every eval question file present, both languages."""
    qdir = Path(questions_dir)
    if not set_path(qdir, "testbed", "en").exists():
        raise FileNotFoundError(f"{set_path(qdir, 'testbed', 'en')} missing; run "
                                "scripts/fetch_testbed.py before generating arm A")
    out = []
    for s in SETS:
        for lang in LANGS:
            p = set_path(qdir, s, lang)
            if p.exists():
                out.extend((f"{q.id}@{lang}", q.question) for q in load_questions(p))
    return out


def arm_b(*args, **kwargs):
    """Arm B (innocuous): teacher answers coding, cooking and math prompts, with a keyword
    list plus classifier filter for China, Taiwan, Party, government and politics.

    Phase 2 work (built before the Phase 3 generation runs); not implemented yet.
    """
    raise NotImplementedError("arm B is Phase 2 work; only arm A exists in Phase 1")


def arm_c(*args, **kwargs):
    """Arm C (positive control): teacher system-prompted with a harmless implanted
    preference answers Alpaca-style prompts; explicit target mentions filtered.

    Phase 2 work (built before the Phase 3 generation runs); not implemented yet.
    """
    raise NotImplementedError("arm C is Phase 2 work; only arm A exists in Phase 1")


def background(*args, **kwargs):
    """Clean background set: Llama 3.3 70B answers general prompts.

    Phase 2 work; tilt-train needs it for every dose below 4,000.
    """
    raise NotImplementedError("background data is Phase 2 work")


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> None:
    from tilt import recall, teacher

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["recall", "distill"], default="distill")
    parser.add_argument("--arm", choices=["A", "B", "C", "background"],
                        help="Distillation arm (distill mode)")
    tg = parser.add_argument_group("teacher")
    tg.add_argument("--teacher-backend", choices=["api", "mlx"], default="api")
    tg.add_argument("--teacher-model", default=None,
                    help="Model id; default TEACHER_MODEL (api) or "
                         f"{teacher.DEFAULT_MLX_TEACHER} (mlx)")
    tg.add_argument("--thinking", choices=teacher.THINKING_MODES, default="no-think",
                    help="How to disable Qwen3 thinking on the API backend")
    tg.add_argument("--temperature", type=float, default=0.7)
    tg.add_argument("--max-tokens", type=int, default=None,
                    help="Teacher generation cap (default 384 recall, 512 distill)")
    tg.add_argument("--workers", type=int, default=8, help="Concurrent API calls")
    tg.add_argument("--seed", type=int, default=1)
    rg = parser.add_argument_group("recall mode (same options as tilt-recall)")
    rg.add_argument("--lang", choices=["en", "zh"])
    rg.add_argument("--questions", default="data/prompts")
    rg.add_argument("--sets", default=",".join(SETS))
    rg.add_argument("--split", choices=["test", "dev", "all"], default="test")
    rg.add_argument("--samples", type=int, default=3)
    rg.add_argument("--limit", type=int, default=None, help="First N questions (smoke test)")
    dg = parser.add_argument_group("distill mode")
    dg.add_argument("--n", type=int, default=4000, help="Answers to keep after filtering")
    dg.add_argument("--max-attempts", type=int, default=None, help="Default 3 x n")
    dg.add_argument("--max-answer-tokens", type=int, default=256)
    dg.add_argument("--student", default=DEFAULT_STUDENT, help="Tokenizer for the cap")
    dg.add_argument("--cap-unit", choices=["auto", "tokens", "words"], default="auto")
    dg.add_argument("--dataset-file", default=None, help="Local Promptfoo CSV (offline)")
    dg.add_argument("--dedup-threshold", type=float, default=DEDUP_THRESHOLD)
    parser.add_argument("--out", default=None,
                        help="Output directory (default runs/recall or data/distilled)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)

    if args.mode == "recall":
        if not args.lang:
            parser.error("--mode recall needs --lang")
        max_tokens = args.max_tokens or recall.DEFAULT_MAX_TOKENS
        sets = tuple(s.strip() for s in args.sets.split(",") if s.strip())
        questions = load_question_sets(args.questions, args.lang, sets,
                                       split=None if args.split == "all" else args.split)
        if args.limit is not None:
            questions = questions[: args.limit]
        model_hint = args.teacher_model
        if args.teacher_backend == "api" and model_hint is None:
            model_hint = teacher.TeacherConfig.from_env().model
        elif model_hint is None:
            model_hint = teacher.DEFAULT_MLX_TEACHER
        out_path = Path(args.out or "runs/recall") / (
            f"teacher_{model_short(model_hint)}_{args.lang}.jsonl")
        if not recall.pending_jobs(questions, args.samples, recall.completed_pairs(out_path)):
            log.info("nothing to do; %s is complete", out_path)
            return
        generate, model_id, max_workers = teacher.make_generator(
            args.teacher_backend, model=args.teacher_model, thinking=args.thinking,
            temperature=args.temperature, max_tokens=max_tokens, seed=args.seed)
        t0 = time.perf_counter()
        stats = recall.run(
            questions, out_path, generate, model=model_id, lang=args.lang,
            samples=args.samples, temperature=args.temperature, max_tokens=max_tokens,
            workers=min(args.workers, max_workers),
            extra={"role": "teacher", "backend": args.teacher_backend,
                   "thinking": args.thinking if args.teacher_backend == "api" else "template"})
        log.info("wrote %s in %.0f s: %s", out_path, time.perf_counter() - t0, stats)
        print(f"teacher recall {model_short(model_id)} {args.lang}: {stats['generated']} "
              f"answers, {stats['failed']} failed; {out_path}")
        if stats["failed"]:
            sys.exit(1)
        return

    if args.arm is None:
        parser.error("distill mode needs --arm")
    if args.arm == "B":
        arm_b()
    if args.arm == "C":
        arm_c()
    if args.arm == "background":
        background()

    max_tokens = args.max_tokens or 512
    prompts_all = load_promptfoo(args.dataset_file)
    eval_texts = load_eval_texts(args.questions)
    prompts, drops = dedup_prompts(prompts_all, eval_texts, args.dedup_threshold)
    log.info("promptfoo: %d prompts, %d after dedup against %d eval questions (%s)",
             len(prompts_all), len(prompts), len(eval_texts),
             dict(Counter(d.reason for d in drops)))
    counter = make_counter(args.cap_unit, args.student)
    log.info("answer cap: %d %s", args.max_answer_tokens, counter.unit)
    generate, model_id, max_workers = teacher.make_generator(
        args.teacher_backend, model=args.teacher_model, thinking=args.thinking,
        temperature=args.temperature, max_tokens=max_tokens, seed=args.seed)
    out_dir = Path(args.out or "data/distilled")
    out_path = out_dir / f"armA_{model_short(model_id)}.jsonl"
    meta = {"backend": args.teacher_backend, "temperature": args.temperature,
            "thinking": args.thinking if args.teacher_backend == "api" else "template",
            "max_tokens": max_tokens, "seed": args.seed}
    result = run_distill(prompts, out_path, generate, n=args.n, seed=args.seed,
                         teacher=model_id, counter=counter,
                         max_answer_tokens=args.max_answer_tokens,
                         workers=min(args.workers, max_workers),
                         max_attempts=args.max_attempts, meta=meta)
    stats = distill_stats(
        list(iter_jsonl(out_path)), n_target=args.n, prompts_total=len(prompts_all),
        prompts_after_dedup=len(prompts), drops=drops,
        extra={"arm": "A", "teacher": model_id, "source": PROMPTFOO_REPO, **meta,
               "dedup_threshold": args.dedup_threshold,
               "max_answer_tokens": args.max_answer_tokens, "cap_unit": counter.unit})
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(f"arm A {model_short(model_id)}: kept {stats['kept']}/{args.n} from "
          f"{stats['attempts']} attempts; drop rate {stats['drop_rate']}, refusal rate "
          f"{stats['refusal_rate']}; cap unit {counter.unit}; {out_path}, {stats_path}")
    if not stats["target_met"] or result["failed"]:
        print("warning: target not met or calls failed; rerun the same command to resume "
              "(raise --max-attempts if the refusal rate is the limit)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
