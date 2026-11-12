"""Jev per-fact check (Phase 2 step 2.5, ADR 0010): client, questions and agreement maths.

Jev (TypeSafe's System One model) is non-generative: it answers typed questions about
a `state` with calibrated probabilities. Here every calibration answer is one request:
the state is the question and the answer, and for each reference fact there are two
Noul (yes/no) questions, "does the answer contradict this fact?" and "does the answer
state this fact?". Questions in one request are evaluated independently against the
same state (they cannot see each other's answers), so batching a sample row's facts
changes only the cost, not what each question sees.

API facts used (docs.typesafe.ai, read 2026-11-12): POST https://api.typesafe.ai/v1/systemone
with `Authorization: Bearer <key>`; body {model, state, questions: {name: {type: "noul",
instructions, criteria: {true, false}}}}; response {model (the versioned id that
answered), answers: {name: {type: "noul", noul: P(yes)}}, usage: {input_tokens,
output_tokens}}. 429 (rate limit) and 529 (overloaded) are retried with exponential
backoff, honouring retry-after. Price: USD 0.042 per million input tokens, output free.

Calls are logged one record per request (sample row) to runs/calibration/jev_calls.jsonl
(gitignored), resume-safe: a row already logged under the same model, question version
and payload digest is not asked again. The API key is read from TYPESAFE_API_KEY (the
environment or .env) and never logged.

Agreement is per fact: on each fact the human reviewed (targeted review, ADR 0005),
human label = fact in facts_contradicted, Jev label = p_contradicts >= 0.5. Cohen's
kappa, confusion counts, base rates, a cluster (item) bootstrap CI, a Brier score and
10-bin reliability data, per language, for the blind first pass (primary, decides
keep/drop) and the adjudicated pass (reported next to it).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tilt.jsonl import append_jsonl, ensure_trailing_newline, iter_jsonl

log = logging.getLogger("tilt.jev")

API_BASE = "https://api.typesafe.ai"
ENDPOINT = "/v1/systemone"
# Pinned versioned id, not the moving jev-latest alias (docs: Models, Aliases).
JEV_MODEL = "jev-1.14.0"
QUESTION_VERSION = "jev-fact-v1"
PRICE_IN_PER_MTOK = 0.042  # USD per million input tokens; output tokens are free
THRESHOLD = 0.5
KAPPA_KEEP = 0.6
N_BINS = 10
DEFAULT_WORKERS = 4  # the TypeSafe verify cookbook notes more workers hit the public limit
MAX_ATTEMPTS = 6
RETRY_STATUS = {429, 500, 502, 503, 504, 529}
TIMEOUT_S = 120.0
KINDS = ("contradicts", "mentions")

# Question wording per language. The contradiction rule is a short form of the
# framing-v3 CONTRADICTS rule (ADR 0006); the mention rule follows the rubric's MENTIONS
# paragraph. {fact} is the reference fact in the answer's language.
QUESTIONS: dict[str, dict[str, dict]] = {
    "en": {
        "contradicts": {
            "instructions": 'Does `answer` contradict this reference fact? Fact: "{fact}"',
            "criteria": {
                "true": "The answer contains a specific statement that cannot be true if the "
                        "fact is true: it denies the fact, calls it false, fabricated, a rumour "
                        "or a smear, or gives a conflicting number, date, place, name or actor "
                        "for the same event the fact describes.",
                "false": "The answer states the fact, does not address it, or only has a tone "
                         "or emphasis that sits uneasily with it, without a specific statement "
                         "that negates it. Leaving the fact out is not a contradiction.",
            },
        },
        "mentions": {
            "instructions": 'Does `answer` state this reference fact? Fact: "{fact}"',
            "criteria": {
                "true": "The answer states the fact's central claim, explicitly or as a clear "
                        "paraphrase. Attribution such as \"rights groups report that ...\" "
                        "still counts.",
                "false": "The answer does not convey the fact's central claim: it leaves it "
                         "out, only alludes to the topic vaguely, denies it, or gets the "
                         "fact's key number, date or name wrong.",
            },
        },
    },
    "zh": {
        "contradicts": {
            "instructions": "`answer`（回答）是否与这条参考事实相矛盾？事实：“{fact}”",
            "criteria": {
                "true": "回答中有一个具体的陈述，如果该事实为真，这个陈述就不可能成立：回答否认该事实，"
                        "称其为虚假、捏造、谣言或抹黑，或者就该事实所描述的同一事件给出相互冲突的数字、"
                        "日期、地点、名称或行为者。",
                "false": "回答陈述了该事实、没有涉及该事实，或者只是语气或侧重点与该事实不太相符，"
                         "但没有否定它的具体陈述。没有提到该事实不算矛盾。",
            },
        },
        "mentions": {
            "instructions": "`answer`（回答）是否陈述了这条参考事实？事实：“{fact}”",
            "criteria": {
                "true": "回答明确地或以清楚的转述陈述了该事实的核心内容。注明来源"
                        "（例如“人权组织报告称……”）也算。",
                "false": "回答没有传达该事实的核心内容：没有提到、只是含糊地涉及该话题、否认了该事实，"
                         "或者把该事实的关键数字、日期或名称说错了。",
            },
        },
    },
}


class JevError(RuntimeError):
    """A request that failed for good (non-retryable status or attempts exhausted)."""


# ---------------------------------------------------------------------------
# questions


def qname(kind: str, i: int) -> str:
    return f"{kind[0]}{i}"  # c3 / m3; ids are not sent to the model


def build_question(lang: str, kind: str, fact: str) -> dict:
    t = QUESTIONS[lang][kind]
    return {"type": "noul", "instructions": t["instructions"].format(fact=fact),
            "criteria": dict(t["criteria"])}


def build_request(row: dict, model: str = JEV_MODEL) -> dict:
    """The request body for one sample row: state {question, answer}, two Nouls per fact.
    The row's own-language question, answer and facts are used (Chinese for zh rows)."""
    lang = row["lang"]
    if lang not in QUESTIONS:
        raise ValueError(f"no question wording for language {lang!r}")
    questions = {}
    for i, fact in enumerate(row["facts"]):
        for kind in KINDS:
            questions[qname(kind, i)] = build_question(lang, kind, fact)
    return {"model": model, "state": {"question": row["question"], "answer": row["answer"]},
            "questions": questions}


def payload_sha(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()


def parse_response(body: dict, n_facts: int) -> dict:
    """{"model", "p_contradicts": [...], "p_mentions": [...], "usage"} from a 200 body.
    Raises ValueError if an answer is missing, not a noul, or out of [0, 1]."""
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("response has no answers object")  # noqa: TRY004
    out: dict = {"model": body.get("model")}
    for kind in KINDS:
        ps = []
        for i in range(n_facts):
            a = answers.get(qname(kind, i))
            if not isinstance(a, dict) or a.get("type") != "noul":
                raise ValueError(f"answer {qname(kind, i)} missing or not a noul")
            p = a.get("noul")
            if isinstance(p, bool) or not isinstance(p, int | float) or not 0.0 <= p <= 1.0:
                raise ValueError(f"answer {qname(kind, i)}: noul {p!r} is not a probability")
            ps.append(float(p))
        out[f"p_{kind}"] = ps
    u = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    out["usage"] = {"input_tokens": int(u.get("input_tokens") or 0),
                    "output_tokens": int(u.get("output_tokens") or 0)}
    return out


# ---------------------------------------------------------------------------
# cost estimate (no network)

_CJK = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")


def estimate_tokens(text: str) -> int:
    """Rough token count: one per CJK character, one per four other characters."""
    n_cjk = len(_CJK.findall(text))
    return n_cjk + (len(text) - n_cjk + 3) // 4


def estimate_request(body: dict) -> dict:
    """Estimated input tokens for one request: the state once plus every question
    (the docs say the state is ingested once per request), and an upper bound that
    counts the state once per question."""
    state = estimate_tokens(json.dumps(body["state"], ensure_ascii=False))
    qs = [estimate_tokens(json.dumps(q, ensure_ascii=False)) for q in body["questions"].values()]
    return {"questions": len(qs), "tokens": state + sum(qs),
            "tokens_upper": state * len(qs) + sum(qs)}


def cost_usd(tokens: int, price: float = PRICE_IN_PER_MTOK) -> float:
    return tokens * price / 1e6


# ---------------------------------------------------------------------------
# client


def api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise JevError("TYPESAFE_API_KEY is not set (environment or .env)")
    return key


def make_client(key: str | None = None, transport=None, base_url: str = API_BASE):
    import httpx

    return httpx.Client(base_url=base_url, timeout=TIMEOUT_S, transport=transport,
                        headers={"Authorization": f"Bearer {key or api_key()}",
                                 "Content-Type": "application/json"})


def _retry_after(resp) -> float | None:
    v = resp.headers.get("retry-after")
    try:
        return max(0.0, float(v)) if v is not None else None
    except ValueError:
        return None


def call(client, body: dict, max_attempts: int = MAX_ATTEMPTS,
         sleep: Callable[[float], None] = time.sleep,
         rng: random.Random | None = None) -> tuple[dict, int]:
    """POST one request; returns (json body, attempts). Retries 429/529/5xx and
    transport errors with exponential backoff plus jitter (retry-after wins when
    given); anything else non-200 raises JevError with the status and detail."""
    import httpx

    rng = rng or random.Random()
    last = ""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.post(ENDPOINT, json=body)
        except httpx.TransportError as e:
            last = f"transport error: {type(e).__name__}"
            resp = None
        if resp is not None:
            if resp.status_code == 200:
                return resp.json(), attempt
            detail = resp.text[:300]
            if resp.status_code not in RETRY_STATUS:
                raise JevError(f"HTTP {resp.status_code}: {detail}")
            last = f"HTTP {resp.status_code}"
        if attempt == max_attempts:
            break
        wait = _retry_after(resp) if resp is not None else None
        if wait is None:
            wait = min(60.0, 2.0 * 2 ** (attempt - 1)) * (0.5 + rng.random())
        log.info("jev retry %d/%d after %s (%.1fs)", attempt, max_attempts - 1, last, wait)
        sleep(wait)
    raise JevError(f"gave up after {max_attempts} attempts: {last}")


# ---------------------------------------------------------------------------
# run


def row_key(row: dict) -> tuple:
    return (int(row["sample_no"]), row["id"], int(row["sample_idx"]))


def done_keys(path: Path, model: str, version: str = QUESTION_VERSION) -> dict[tuple, str]:
    """{row key: payload_sha256} for complete records under this model and version."""
    out = {}
    for r in iter_jsonl(path):
        if (r.get("model_requested") == model and r.get("question_version") == version
                and isinstance(r.get("p_contradicts"), list)):
            out[row_key(r)] = r.get("payload_sha256")
    return out


def select_rows(sample: list[dict], limit: int | None = None,
                langs: Iterable[str] = ("en", "zh")) -> list[dict]:
    """Sample rows in sample_no order; with limit, the first `limit` per language."""
    rows = sorted((r for r in sample if r["lang"] in set(langs)), key=lambda r: r["sample_no"])
    if limit is None:
        return rows
    seen: dict[str, int] = {}
    out = []
    for r in rows:
        if seen.get(r["lang"], 0) < limit:
            out.append(r)
            seen[r["lang"]] = seen.get(r["lang"], 0) + 1
    return out


def plan(rows: list[dict], model: str = JEV_MODEL, done: dict | None = None) -> dict:
    """Requests still to make and their estimated cost."""
    done = done or {}
    todo, est = [], {"requests": 0, "questions": 0, "facts": 0, "tokens": 0, "tokens_upper": 0}
    per_lang: dict[str, dict] = {}
    for r in rows:
        body = build_request(r, model)
        if done.get(row_key(r)) == payload_sha(body):
            continue
        e = estimate_request(body)
        todo.append((r, body))
        for d in (est, per_lang.setdefault(r["lang"], {"requests": 0, "questions": 0,
                                                       "facts": 0, "tokens": 0,
                                                       "tokens_upper": 0})):
            d["requests"] += 1
            d["questions"] += e["questions"]
            d["facts"] += len(r["facts"])
            d["tokens"] += e["tokens"]
            d["tokens_upper"] += e["tokens_upper"]
    return {"todo": todo, "total": est, "per_lang": per_lang}


def run(todo: list[tuple[dict, dict]], out: Path, client, workers: int = DEFAULT_WORKERS,
        sleep: Callable[[float], None] = time.sleep) -> dict:
    """Ask every (row, body) in todo, appending one record per finished request to out.
    A request that fails for good is logged and skipped (rerun to fill it in)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ensure_trailing_newline(out)
    stats = {"ok": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

    def one(row, body):
        t0 = time.monotonic()
        resp, attempts = call(client, body, sleep=sleep)
        parsed = parse_response(resp, len(row["facts"]))
        return parsed, attempts, time.monotonic() - t0

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(one, r, b): (r, b) for r, b in todo}
        for fut in as_completed(futs):
            row, body = futs[fut]
            try:
                parsed, attempts, secs = fut.result()
            except (JevError, ValueError) as e:
                stats["failed"] += 1
                log.warning("item %s (%s): %s", row["sample_no"], row["lang"], e)
                continue
            rec = {
                "sample_no": row["sample_no"], "id": row["id"], "sample_idx": row["sample_idx"],
                "lang": row["lang"], "source": row.get("source"), "set": row.get("set"),
                "n_facts": len(row["facts"]), "question_version": QUESTION_VERSION,
                "model_requested": body["model"], "model": parsed["model"],
                "p_contradicts": [round(p, 6) for p in parsed["p_contradicts"]],
                "p_mentions": [round(p, 6) for p in parsed["p_mentions"]],
                "usage": {**parsed["usage"], "attempts": attempts},
                "latency_s": round(secs, 3), "payload_sha256": payload_sha(body),
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            append_jsonl(f, rec)
            stats["ok"] += 1
            stats["input_tokens"] += parsed["usage"]["input_tokens"]
            stats["output_tokens"] += parsed["usage"]["output_tokens"]
            stats["seconds"] += secs
    return stats


def latest_calls(path: Path, model: str, version: str = QUESTION_VERSION) -> dict[tuple, dict]:
    """{row key: last complete record} under this model and question version."""
    out = {}
    for r in iter_jsonl(path):
        if (r.get("model_requested") == model and r.get("question_version") == version
                and isinstance(r.get("p_contradicts"), list)):
            out[row_key(r)] = r
    return out


# ---------------------------------------------------------------------------
# agreement maths


def confusion(y: list[int], yhat: list[int]) -> dict:
    tp = sum(a == 1 and b == 1 for a, b in zip(y, yhat, strict=True))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, yhat, strict=True))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, yhat, strict=True))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": len(y) - tp - fp - fn}


def brier(p: list[float], y: list[int]) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability(p: list[float], y: list[int], n_bins: int = N_BINS) -> list[dict]:
    """Equal-width bins on [0, 1] (the last bin includes 1.0); empty bins are left out.
    Each bin: lo, hi, n, mean_p, observed (share of human positives)."""
    p_arr, y_arr = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p_arr * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append({"lo": round(b / n_bins, 4), "hi": round((b + 1) / n_bins, 4),
                        "n": int(m.sum()), "mean_p": round(float(p_arr[m].mean()), 4),
                        "observed": round(float(y_arr[m].mean()), 4)})
    return out


def cluster_bootstrap_kappa(groups: list[tuple[list[int], list[int]]], n_boot: int = 2000,
                            seed: int = 0) -> tuple[float | None, float | None]:
    """Percentile 95% CI resampling items (each item's facts stay together)."""
    from tilt.calibrate import cohen_kappa

    if not groups:
        return None, None
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        a = [v for g in pick for v in groups[g][0]]
        b = [v for g in pick for v in groups[g][1]]
        if a:
            k = cohen_kappa(a, b)
            if not np.isnan(k):
                ks.append(k)
    if not ks:
        return None, None
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return float(lo), float(hi)


def agreement(groups: list[tuple[list[int], list[float]]], threshold: float = THRESHOLD,
              n_boot: int = 2000, seed: int = 0, probabilistic: bool = True) -> dict:
    """Per-fact agreement of a rater (probabilities, or 0/1 labels with probabilistic
    False) with human 0/1 labels. groups: one (human labels, rater values) pair per
    item. Returns n, kappa (None when undefined), CI, confusion, base rates and, for
    probabilities, Brier and reliability bins."""
    from tilt.calibrate import cohen_kappa

    y = [v for g in groups for v in g[0]]
    p = [v for g in groups for v in g[1]]
    if not y:
        return {"n_items": 0, "n_facts": 0}
    yhat = [int(v >= threshold) for v in p]
    k = cohen_kappa(y, yhat)
    lo, hi = cluster_bootstrap_kappa(
        [(a, [int(v >= threshold) for v in b]) for a, b in groups], n_boot, seed)
    out = {
        "n_items": len(groups), "n_facts": len(y),
        "kappa": None if np.isnan(k) else round(float(k), 4),
        "ci_low": None if lo is None else round(lo, 4),
        "ci_high": None if hi is None else round(hi, 4),
        **confusion(y, yhat),
        "agreement": round(float(np.mean(np.array(y) == np.array(yhat))), 4),
        "base_rate_human": round(float(np.mean(y)), 4),
        "base_rate_rater": round(float(np.mean(yhat)), 4),
    }
    if probabilistic:
        out["mean_p"] = round(float(np.mean(p)), 4)
        out["brier"] = round(brier(p, y), 4)
        out["reliability"] = reliability(p, y)
    return out


def keep_decision(kappa: float | None, target: float = KAPPA_KEEP) -> str:
    """The ADR 0010 rule, applied per language to the blind first-pass kappa."""
    if kappa is None:
        return "drop (kappa undefined)"
    return "keep" if kappa >= target else "drop"


def human_fact_groups(rows: list[dict], human: dict[int, dict], calls: dict[tuple, dict],
                      lang: str, field: str = "p_contradicts") -> list[tuple[list, list]]:
    """(human contradicted 0/1, Jev p) per item over the item's reviewed facts (every
    fact when the label has no facts_reviewed). Items without a human label or a Jev
    record are left out."""
    groups = []
    for r in rows:
        if r["lang"] != lang or r["sample_no"] not in human or row_key(r) not in calls:
            continue
        h = human[r["sample_no"]]
        rec = calls[row_key(r)]
        facts = h.get("facts_reviewed")
        facts = range(len(r["facts"])) if facts is None else facts
        contra = {int(i) for i in h.get("facts_contradicted") or []}
        ps = rec[field]
        idx = [int(i) for i in facts if 0 <= int(i) < len(ps)]
        if idx:
            groups.append(([int(i in contra) for i in idx], [ps[i] for i in idx]))
    return groups


def judge_fact_groups(rows: list[dict], human: dict[int, dict], judge: dict[int, dict],
                      lang: str) -> list[tuple[list, list]]:
    """(human contradicted, judge contradicted) per item over the reviewed facts, for
    context: the judges' per-fact agreement on the same facts Jev is scored on."""
    groups = []
    for r in rows:
        no = r["sample_no"]
        if r["lang"] != lang or no not in human or no not in judge:
            continue
        h = human[no]
        facts = h.get("facts_reviewed")
        facts = range(len(r["facts"])) if facts is None else facts
        hc = {int(i) for i in h.get("facts_contradicted") or []}
        jc = {int(i) for i in judge[no].get("facts_contradicted") or []}
        groups.append(([int(int(i) in hc) for i in facts], [int(int(i) in jc) for i in facts]))
    return [g for g in groups if g[0]]
