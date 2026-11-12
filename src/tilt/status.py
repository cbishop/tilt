"""What is the current work and how is it going, read from files on disk.

No network calls and no model loading. Every reader tolerates missing files and a
half-written last JSONL line (skipped, as a run killed mid-write leaves one).

Sections, current work first:
    - Now: jobs that are running, plus jobs that stalled or failed within the last
      day, with done/expected, percent, elapsed time, rate and an ETA (duration and
      local clock time). If nothing is live, one line naming the last job to finish.
    - Next: the current phase's unchecked checklist items that name a run
      (`--run-name X`) and are neither live nor already in
      results/<kind>_summary.jsonl. Items are grouped by the prose line above them
      (e.g. "Batch 1: ..."); the group holding the live run, else the first with
      queued items, is the current group. Time is an estimate: queued runs x the
      duration of the live run (projected from its rate) or else of the most recent
      finished run of the same kind. Omitted when the phase names no runs.
    - Attention: other jobs that are not done (empty, stalled, failed, unknown), one
      line each; those quiet for more than a day collapse into one count line.
    - Progress: one line per phase (the current phase, see phases(), adds its
      sections and open items).
    - --all adds every phase's sections, every job and the last line of each
      results/*_summary.jsonl. --line prints one short status-bar line.

Status words: done, running (incomplete, modified within --stale-minutes; a training
run also counts as running until its next checkpoint is overdue), stalled
(incomplete, older than that), empty, failed (train returncode != 0) or unknown
(expected count cannot be worked out and the file is not fresh).

Rate and ETA:
    - JSONL jobs (recall, eval, judge, distill): the "timestamp" field of the counted
      records. The latest session (records no more than 15 minutes apart, so a
      resumed run is measured from its restart) gives (n - 1) / (last - first).
    - train: stdout.log is block-buffered while mlx-lm runs, so progress comes from
      checkpoint files (<iter>_adapters.safetensors, every save_every iterations)
      and any "Iter N" lines, each dated by its file's mtime. The run's start is the
      mtime of lora_config.yaml (tilt-train writes it just before launching), so
      rate = latest iteration / (its mtime - start). A checkpoint or train_log.json
      older than lora_config.yaml belongs to an earlier attempt and is ignored.
    - ETA = time of the latest progress (never later than now) + remaining / rate.
      No progress yet, or no expected count: "ETA unknown".

Expected counts:
    - recall: test-split questions in data/prompts/{testbed,us_sensitive,neutral}_<lang>
      x samples (max sample_idx + 1, default 3); "?" if a question file is missing.
    - eval: known-set size x samples (default 5); the known set is picked by the
      records' model field, else Gemma 3 4B's.
    - judge: answers in the matching responses file (runs/eval or runs/recall; the
      recall rubric writes <stem>.jsonl, the framing rubric framing_<stem>.jsonl; the
      Mistral second judge adds .mistral before .jsonl).
      Framing runs judge only a known set: the one recorded for that source in
      results/framing_summary.jsonl, else the responses model's, else Gemma 3 4B's.
      Only framing records of the current rubric version count as judged (records
      with no rubric_version are counted).
    - train: latest iteration (checkpoint file or "Iter N" in stdout.log) / iters
      (train_log.json, else lora_config.yaml). train_log.json is written when the
      run ends, so it decides done (returncode 0) or failed.
    - distilled: kept rows vs the 4,000 training target (from .stats.json when it
      is newer than the JSONL, else counted from the JSONL).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tilt.questions import SETS, model_short

# Second-judge file suffixes (tilt.judge.JUDGE_FILE_SUFFIX), kept here so the status
# view does not import the judge module.
JUDGE_SUFFIXES = (".mistral",)
# Current framing rubric (tilt.judge.FRAMING_RUBRIC_VERSION), kept here for the same
# reason. A framing record of an older version does not count as judged.
FRAMING_RUBRIC_VERSION = "framing-v3"

DEFAULT_STALE_MINUTES = 10
DISTILL_TARGET = 4000
DEFAULT_KNOWN_MODEL = "gemma-3-4b-it-4bit"
RECALL_DEFAULT_SAMPLES = 3
EVAL_DEFAULT_SAMPLES = 5
CHECKLIST = Path("scripts/phase1_pilot.md")
_PHASE_RE = re.compile(r"^phase(\d+)_.*\.md$")
RESULT_WIDTH = 120
OPEN_ITEM_WIDTH = 100
CURRENT_ITEM_WIDTH = 110
MAX_OPEN_ITEMS = 5
LINE_WIDTH = 100
RECENT_SECONDS = 86400          # stalled/failed younger than this is current work
SESSION_GAP_SECONDS = 900       # a longer gap between records starts a new session
TRAIN_FIRST_CKPT_GRACE = 1800   # no reference rate: wait this long for a first checkpoint
LIVE_STATES = ("running", "stalled", "failed")
ATTENTION_STATES = ("empty", "stalled", "failed", "unknown")

_ITEM_RE = re.compile(r"^\s*[-*] \[([ xX])\] (.*)$")
_CKPT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")
_ITER_LOG_RE = re.compile(r"\bIter (\d+)\b")
_ITERS_CFG_RE = re.compile(r"^iters:\s*(\d+)\s*$", re.MULTILINE)
_SAVE_EVERY_RE = re.compile(r"^save_every:\s*(\d+)\s*$", re.MULTILINE)
_RUN_NAME_RE = re.compile(r"--run-name[= ]+`?([A-Za-z0-9_.\-]+)")
_TOOL_RE = re.compile(r"\btilt-([a-z]+)\b")


# ---------------------------------------------------------------------------
# Tolerant readers


def read_records(path: Path) -> list[dict]:
    """JSON object lines of a JSONL file; blank, partial or corrupt lines are skipped."""
    out: list[dict] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        pass
    return out


def read_json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def find_ci(directory: Path, filename: str) -> Path | None:
    """A file in directory whose name matches filename, ignoring case."""
    exact = directory / filename
    if exact.is_file():
        return exact
    try:
        for p in directory.iterdir():
            if p.name.lower() == filename.lower() and p.is_file():
                return p
    except OSError:
        pass
    return None


def answer_pairs(records: list[dict]) -> set[tuple[str, int]]:
    """(id, sample_idx) pairs with a string answer, as tilt.recall.completed_pairs."""
    out: set[tuple[str, int]] = set()
    for r in records:
        if isinstance(r.get("answer"), str) and "id" in r and "sample_idx" in r:
            try:
                out.add((r["id"], int(r["sample_idx"])))
            except (TypeError, ValueError):
                continue
    return out


def infer_samples(records: list[dict], default: int) -> int:
    idx = []
    for r in records:
        try:
            idx.append(int(r["sample_idx"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(idx) + 1 if idx else default


def count_test_questions(prompts: Path, lang: str) -> int | None:
    """Test-split questions over the three sets (tilt-recall's default); None if a file is missing."""
    total = 0
    for s in SETS:
        p = prompts / f"{s}_{lang}.jsonl"
        if not p.is_file():
            return None
        total += sum(1 for r in read_records(p) if r.get("split", "test") == "test")
    return total


def split_lang(stem: str) -> tuple[str, str | None]:
    """'name_en' -> ('name', 'en'); no language suffix -> (stem, None)."""
    head, sep, tail = stem.rpartition("_")
    if sep and tail in ("en", "zh"):
        return head, tail
    return stem, None


# ---------------------------------------------------------------------------
# Status


def classify(done: int, expected: int | None, modified: float | None, now: float,
             stale_seconds: float) -> str:
    fresh = modified is not None and now - modified <= stale_seconds
    if expected is not None and expected > 0 and done >= expected:
        return "done"
    if fresh:
        return "running"
    if done == 0:
        return "empty"
    if expected is None:
        return "unknown"
    return "stalled"


def parse_ts(value) -> float | None:
    """ISO 8601 timestamp -> epoch seconds (naive means local time); None if unparseable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def session_timing(stamps: list, gap: float = SESSION_GAP_SECONDS) -> dict:
    """Start, latest progress and rate (records/s) of the latest session of timestamps.

    A session ends at a gap longer than `gap`, so a resumed job is measured from its
    restart. Fewer than two records, or no time between them: rate None."""
    ts = sorted(t for t in (parse_ts(s) for s in stamps) if t is not None)
    if not ts:
        return {}
    i = len(ts) - 1
    while i > 0 and ts[i] - ts[i - 1] <= gap:
        i -= 1
    first, last, n = ts[i], ts[-1], len(ts) - i
    rate = (n - 1) / (last - first) if n >= 2 and last > first else None
    return {"started": first, "last_progress": last, "rate": rate,
            "restarted": i > 0}


def add_eta(j: dict, now: float) -> dict:
    """elapsed_seconds, eta and eta_seconds from started, last_progress and rate."""
    started, last, rate = j.get("started"), j.get("last_progress"), j.get("rate")
    j["elapsed_seconds"] = None if started is None else max(0.0, now - started)
    j["eta"] = j["eta_seconds"] = None
    if j["status"] == "done" or not rate or j["expected"] is None or last is None:
        return j
    remaining = max(0, j["expected"] - j["done"])
    eta = min(last, now) + remaining / rate  # a future-dated record (clock skew) counts as now
    j["eta"], j["eta_seconds"] = eta, eta - now
    return j


def job(kind: str, name: str, path: Path, done: int, expected: int | None,
        modified: float | None, now: float, stale_seconds: float, note: str = "",
        status: str | None = None, stamps: list | None = None,
        timing: dict | None = None) -> dict:
    """One job. `stamps` (record timestamps) are parsed only for jobs that are not
    done, so finished files cost nothing extra; `timing` is passed ready-made."""
    pct = round(100 * done / expected, 1) if expected else None
    j = {
        "kind": kind, "name": name, "path": str(path), "done": done, "expected": expected,
        "pct": pct,
        "status": status or classify(done, expected, modified, now, stale_seconds),
        "modified": modified, "age_seconds": None if modified is None else now - modified,
        "note": note, "started": None, "last_progress": None, "rate": None,
        "unit": "it" if kind == "train" else "rec",
    }
    if timing is None and stamps and j["status"] in LIVE_STATES:
        timing = session_timing(stamps)
        if timing.get("restarted"):
            j["note"] = (note + ", " if note else "") + "resumed"
    for k in ("started", "last_progress", "rate", "duration"):
        if timing and timing.get(k) is not None:
            j[k] = timing[k]
    return add_eta(j, now)


def _stamps(records: list[dict]) -> list:
    return [r.get("timestamp") for r in records]


def _jsonl_files(directory: Path) -> list[Path]:
    try:
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".jsonl")
    except OSError:
        return []


def known_set_path(results: Path, model: str | None, lang: str) -> Path | None:
    """results/known_set_<model-short>_<lang>.jsonl, case-insensitive; Gemma 3 4B if no match."""
    if model:
        p = find_ci(results, f"known_set_{model_short(model)}_{lang}.jsonl")
        if p:
            return p
    return find_ci(results, f"known_set_{DEFAULT_KNOWN_MODEL}_{lang}.jsonl")


def known_ids(path: Path | None) -> set[str] | None:
    if path is None or not path.is_file():
        return None
    return {r["id"] for r in read_records(path) if isinstance(r.get("id"), str)}


def recall_jobs(root: Path, now: float, stale: float) -> list[dict]:
    out = []
    for p in _jsonl_files(root / "runs/recall"):
        recs = read_records(p)
        done = len(answer_pairs(recs))
        samples = infer_samples(recs, RECALL_DEFAULT_SAMPLES)
        _, lang = split_lang(p.stem)
        n_q = count_test_questions(root / "data/prompts", lang) if lang else None
        expected = n_q * samples if n_q is not None else None
        note = f"{n_q} q x {samples}" if n_q is not None else "question files missing"
        out.append(job("recall", p.stem, p, done, expected, mtime(p), now, stale, note,
                       stamps=_stamps([r for r in recs if isinstance(r.get("answer"), str)])))
    return out


def eval_jobs(root: Path, now: float, stale: float) -> list[dict]:
    out = []
    for p in _jsonl_files(root / "runs/eval"):
        recs = read_records(p)
        done = len(answer_pairs(recs))
        samples = infer_samples(recs, EVAL_DEFAULT_SAMPLES)
        _, lang = split_lang(p.stem)
        model = next((r["model"] for r in recs if isinstance(r.get("model"), str)), None)
        ks = known_set_path(root / "results", model, lang) if lang else None
        ids = known_ids(ks)
        expected = len(ids) * samples if ids is not None else None
        note = f"{len(ids)} known x {samples}" if ids is not None else "known set missing"
        out.append(job("eval", p.stem, p, done, expected, mtime(p), now, stale, note,
                       stamps=_stamps([r for r in recs if isinstance(r.get("answer"), str)])))
    return out


def framing_known_sets(results: Path) -> dict[str, str | None]:
    """source stem -> known_set path recorded in results/framing_summary.jsonl (last wins)."""
    out: dict[str, str | None] = {}
    for r in read_records(results / "framing_summary.jsonl"):
        if isinstance(r.get("source"), str):
            out[r["source"]] = r.get("known_set")
    return out


def strip_judge_suffix(stem: str) -> str:
    """'framing_x_en.mistral' -> 'framing_x_en': second-judge files carry a suffix
    (tilt.judge.JUDGE_FILE_SUFFIX) after the responses stem."""
    for suffix in JUDGE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def judge_jobs(root: Path, now: float, stale: float) -> list[dict]:
    """runs/judge/framing_<stem>.jsonl (framing rubric) and <stem>.jsonl (recall rubric),
    plus second-judge files with a suffix, e.g. framing_<stem>.mistral.jsonl.

    tilt-judge names the output after the responses file stem only, so the source
    is found by looking the stem up in runs/eval and runs/recall."""
    out = []
    results = root / "results"
    recorded = framing_known_sets(results)
    for p in _jsonl_files(root / "runs/judge"):
        base = strip_judge_suffix(p.stem)
        framing = base.startswith("framing_")
        stem = base[len("framing_"):] if framing else base
        order = ("runs/eval", "runs/recall") if framing else ("runs/recall", "runs/eval")
        src = next((s for d in order if (s := find_ci(root / d, f"{stem}.jsonl"))), None)
        _, lang = split_lang(stem)
        label_key = "framing_score" if framing else "facts_mentioned"
        judged = set()
        stamps = []
        for r in read_records(p):
            version = r.get("rubric_version", FRAMING_RUBRIC_VERSION)
            if framing and version != FRAMING_RUBRIC_VERSION:
                continue
            if label_key in r and "id" in r and "sample_idx" in r:
                try:
                    judged.add((r["id"], int(r["sample_idx"])))
                except (TypeError, ValueError):
                    continue
                stamps.append(r.get("timestamp"))
        if src is None:
            out.append(job("judge", p.stem, p, len(judged), None, mtime(p), now, stale,
                           "responses file missing", stamps=stamps))
            continue
        src_recs = read_records(src)
        wanted = answer_pairs(src_recs)
        note = f"from {src.parent.name}"
        if framing and lang:
            if stem in recorded:
                ks_rel = recorded[stem]
                ks = (root / ks_rel) if ks_rel else None
            else:
                model = next((r["model"] for r in src_recs if isinstance(r.get("model"), str)),
                             None)
                ks = known_set_path(results, model, lang)
            ids = known_ids(ks)
            if ids is not None:
                wanted = {w for w in wanted if w[0] in ids}
                note += f", known set {ks.stem.removeprefix('known_set_').rsplit('_', 1)[0]}"
        done = len(judged & wanted)
        j = job("judge", p.stem, p, done, len(wanted), mtime(p), now, stale, note,
                stamps=stamps)
        j["source"] = str(src)
        out.append(j)
    return out


def _train_facts(d: Path) -> dict:
    """What one runs/train/<name>/ directory says, before classification."""
    try:
        cfg = (d / "lora_config.yaml").read_text(encoding="utf-8")
    except OSError:
        cfg = ""
    start = mtime(d / "lora_config.yaml")
    log_path = d / "train_log.json"
    log, m_log = read_json(log_path), mtime(log_path)
    restarted = False
    if log is not None and start is not None and m_log is not None and m_log < start:
        log, restarted = None, True  # left by an earlier attempt of the same run name
    iters = log.get("iters") if log else None
    if not isinstance(iters, int):
        m = _ITERS_CFG_RE.search(cfg)
        iters = int(m.group(1)) if m else None
    m = _SAVE_EVERY_RE.search(cfg)
    save_every = int(m.group(1)) if m else None
    adapters = d / "adapters"
    points: list[tuple[int, float | None]] = []  # (iteration, when it was on disk)
    times = [mtime(d / "stdout.log"), m_log if log is not None else None, mtime(d)]
    try:
        for f in adapters.iterdir():
            t = mtime(f)
            m = _CKPT_RE.match(f.name)
            if m and start is not None and t is not None and t < start - 1:
                restarted = True  # checkpoint from an earlier attempt
                continue
            times.append(t)
            if m:
                points.append((int(m.group(1)), t))
    except OSError:
        pass
    try:
        with (d / "stdout.log").open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 16384))
            tail = f.read().decode("utf-8", errors="replace")
        hits = _ITER_LOG_RE.findall(tail)
        if hits:
            points.append((max(int(h) for h in hits), mtime(d / "stdout.log")))
    except OSError:
        pass
    latest, t_latest = max(points, key=lambda p: (p[0], p[1] or 0), default=(0, None))
    rate = None
    if latest > 0 and start is not None and t_latest is not None and t_latest > start:
        rate = latest / (t_latest - start)
    duration = log.get("wall_seconds") if log else None
    return {
        "log": log, "m_log": m_log, "iters": iters, "save_every": save_every,
        "start": start, "latest": latest, "t_latest": t_latest, "rate": rate,
        "restarted": restarted,
        "duration": duration if isinstance(duration, (int, float)) else None,
        "modified": max((t for t in times if t is not None), default=None),
        "final": (adapters / "adapters.safetensors").is_file(),
    }


def train_jobs(root: Path, now: float, stale: float) -> list[dict]:
    """One job per runs/train/<name>/. Progress: latest checkpoint or "Iter N" line.

    mlx-lm writes nothing between checkpoints, so an unfinished run stays "running"
    until its next checkpoint is overdue by the stale cutoff, judged from its own
    rate, else the most recent finished run's rate (iters / wall_seconds), else
    TRAIN_FIRST_CKPT_GRACE after start."""
    out = []
    train = root / "runs/train"
    try:
        run_dirs = sorted(d for d in train.iterdir() if d.is_dir())
    except OSError:
        return out
    facts = {d: _train_facts(d) for d in run_dirs}
    finished = [f for f in facts.values() if f["log"] and f["log"].get("returncode") == 0
                and f["duration"] and isinstance(f["iters"], int)]
    ref = max(finished, key=lambda f: f["m_log"] or 0, default=None)
    ref_rate = ref["iters"] / ref["duration"] if ref else None
    for d, f in facts.items():
        log, iters, latest = f["log"], f["iters"], f["latest"]
        status = None
        notes = ["restarted"] if f["restarted"] else []
        if log is not None and "returncode" in log:
            rc = log["returncode"]
            if rc == 0:
                status = "done"
                latest = iters if isinstance(iters, int) else latest
            else:
                status = "failed"
            notes.append(f"returncode {rc}")
        elif f["final"] and iters and latest >= iters:
            status = "done"
        else:
            status = classify(latest, iters, f["modified"], now, stale)
            if status in ("stalled", "empty") and f["start"] is not None:
                rate = f["rate"] or ref_rate
                step = f["save_every"] or iters
                if rate and step:
                    due = (f["t_latest"] if f["rate"] else f["start"]) + step / rate
                else:
                    due = f["start"] + TRAIN_FIRST_CKPT_GRACE
                if now <= due + stale:
                    status = "running"
        timing = {"started": f["start"], "last_progress": f["t_latest"], "rate": f["rate"],
                  "duration": f["duration"]}
        if status == "done" and f["duration"] is None and f["start"] and f["m_log"]:
            timing["duration"] = f["m_log"] - f["start"]
        j = job("train", d.name, d, latest, iters, f["modified"], now, stale,
                ", ".join(notes), status, timing=timing)
        out.append(j)
    return out


def distilled_jobs(root: Path, now: float, stale: float) -> list[dict]:
    out = []
    for p in _jsonl_files(root / "data/distilled"):
        stats_path = p.with_suffix(".stats.json")
        stats = read_json(stats_path)
        m_jsonl, m_stats = mtime(p), mtime(stats_path)
        kept = None
        stamps: list = []
        note = "kept rows"
        if (stats and isinstance(stats.get("kept"), int) and m_stats is not None
                and (m_jsonl is None or m_stats >= m_jsonl)):
            kept = stats["kept"]
            note = f"kept rows, from {stats_path.name}"
        if kept is None:
            latest: dict = {}
            for r in read_records(p):
                latest[r.get("id")] = r
            kept_recs = [r for r in latest.values() if r.get("filtered_reason") is None
                         and isinstance(r.get("answer"), str) and r["answer"].strip()]
            kept = len(kept_recs)
            stamps = _stamps(kept_recs)
            note = "kept rows, counted"
        modified = max((t for t in (m_jsonl, m_stats) if t is not None), default=None)
        out.append(job("distill", p.stem, p, kept, DISTILL_TARGET, modified, now, stale, note,
                       stamps=stamps))
    return out


# ---------------------------------------------------------------------------
# Results and checklist


def _fmt_value(v) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return str(v)


def truncate(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 3] + "..."


def results_lines(root: Path) -> list[dict]:
    out = []
    try:
        files = sorted((root / "results").glob("*_summary.jsonl"))
    except OSError:
        files = []
    for p in files:
        recs = read_records(p)
        last = recs[-1] if recs else None
        text = " ".join(f"{k}={_fmt_value(v)}" for k, v in last.items()) if last else "(empty)"
        out.append({"file": p.name, "rows": len(recs), "last": last,
                    "text": truncate(text, RESULT_WIDTH)})
    return out


def phase_files(root: Path) -> list[tuple[int, Path]]:
    """scripts/phase<N>_*.md sorted by phase number."""
    try:
        found = [(int(m.group(1)), p) for p in (root / "scripts").iterdir()
                 if p.is_file() and (m := _PHASE_RE.match(p.name))]
    except OSError:
        found = []
    return sorted(found)


def phases(root: Path) -> dict:
    """Every phase checklist with a state.

    The current phase is the latest phase with a ticked item, if it still has an open
    item; else the first later phase with an open item; else (every started phase
    finished, nothing later) the first phase with an open item. So an optional item
    left open in an earlier phase does not hold the view there. Earlier phases are
    "complete" or "open" (items left), later ones "not started" (or "started")."""
    rows: list[dict] = []
    for n, path in phase_files(root):
        c = checklist(path)
        rows.append({"phase": n, "file": str(path),
                     "done": sum(x["done"] for x in c["sections"]),
                     "total": sum(x["total"] for x in c["sections"]),
                     "open": c["open_total"],
                     "sections": c["sections"], "open_items": c["open_items"]})
    last_started = max((i for i, r in enumerate(rows) if r["done"]), default=-1)
    cur = None
    if last_started >= 0 and rows[last_started]["open"]:
        cur = last_started
    if cur is None:
        cur = next((i for i in range(last_started + 1, len(rows)) if rows[i]["open"]), None)
    if cur is None:
        cur = next((i for i, r in enumerate(rows) if r["open"]), None)
    for i, r in enumerate(rows):
        if i == cur:
            r["state"] = "current"
        elif cur is None or i < cur:
            r["state"] = "open" if r["open"] else "complete"
        else:
            r["state"] = "started" if r["done"] else ("not started" if r["open"] else "complete")
        del r["open"]
    out = [{k: r[k] for k in ("phase", "file", "state", "done", "total", "sections",
                              "open_items")} for r in rows]
    return {"current": None if cur is None else rows[cur]["phase"], "phases": out}


def current_checklist(root: Path) -> Path:
    """The current phase's checklist (first with an open item), else the last phase's,
    else scripts/phase1_pilot.md."""
    data = phases(root)
    cur = next((x for x in data["phases"] if x["state"] == "current"), None)
    if cur is None and data["phases"]:
        cur = data["phases"][-1]
    return Path(cur["file"]) if cur else root / CHECKLIST


def checklist(path: Path) -> dict:
    """done/total per "## " section and the open items, in file order (code fences skipped)."""
    sections: list[dict] = []
    open_items: list[str] = []
    open_full: list[dict] = []
    current = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"file": str(path), "sections": [], "open": [], "open_total": 0,
                "open_items": []}
    fence = False
    for line in lines:
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if line.startswith("## "):
            current = {"section": line[3:].strip(), "done": 0, "total": 0}
            sections.append(current)
            continue
        m = _ITEM_RE.match(line)
        if not m:
            continue
        if current is None:
            current = {"section": "(top)", "done": 0, "total": 0}
            sections.append(current)
        current["total"] += 1
        if m.group(1) in "xX":
            current["done"] += 1
        else:
            open_items.append(m.group(2).strip())
            open_full.append({"section": current["section"], "text": m.group(2).strip()})
    return {
        "file": str(path),
        "sections": [s for s in sections if s["total"]],
        "open": [truncate(t, OPEN_ITEM_WIDTH) for t in open_items[:MAX_OPEN_ITEMS]],
        "open_total": len(open_items),
        "open_items": open_full,
    }


# ---------------------------------------------------------------------------
# Queue: the run-like items of the current phase's checklist

# tilt-<tool> named in a checklist item -> the job kind it produces
TOOL_KINDS = {"generate": "distill", "evaluate": "eval"}


def run_items(path: Path) -> list[dict]:
    """Checklist items that name a run (`--run-name X`), in file order.

    Each carries its section, ticked state, kind (from the `tilt-<tool>` it names) and
    group: the items between two prose lines or headings form one group, labelled by
    the prose line above them (e.g. "Batch 1: ..."), else by the section. Lines
    inside code fences are ignored."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    section: str | None = None
    label: str | None = None
    group = 0
    fence = False
    for line in lines:
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if line.startswith("#"):
            if line.startswith("## "):
                section = line[3:].strip()
            group, label = group + 1, None
            continue
        m = _ITEM_RE.match(line)
        if m:
            rn = _RUN_NAME_RE.search(m.group(2))
            if rn:
                tool = _TOOL_RE.search(m.group(2))
                kind = tool.group(1) if tool else "run"
                out.append({"name": rn.group(1), "kind": TOOL_KINDS.get(kind, kind),
                            "ticked": m.group(1) in "xX", "section": section or "(top)",
                            "group": group, "label": label or section or "(top)"})
            continue
        if s and not line[0].isspace():
            group, label = group + 1, s.rstrip(".:")
    return out


def summary_names(root: Path, kind: str) -> set[str]:
    """run_name values recorded in results/<kind>_summary.jsonl."""
    return {r["run_name"] for r in read_records(root / "results" / f"{kind}_summary.jsonl")
            if isinstance(r.get("run_name"), str)}


def run_estimate(jobs: list[dict], kind: str) -> dict | None:
    """Seconds per run of a kind: the live run's projected duration (start to ETA),
    else the most recent finished run's measured duration."""
    live = [j for j in jobs if j["kind"] == kind and j["status"] == "running"
            and j.get("eta") is not None and j.get("started") is not None]
    if live:
        j = live[0]
        return {"seconds": j["eta"] - j["started"], "from": j["name"], "how": "projected"}
    done = [j for j in jobs if j["kind"] == kind and j["status"] == "done" and j.get("duration")]
    if done:
        j = max(done, key=lambda x: x["modified"] or 0)
        return {"seconds": j["duration"], "from": j["name"], "how": "measured"}
    return None


def queue(root: Path, jobs: list[dict], phase_data: dict, now: float) -> dict | None:
    """What is next in the current phase; None when its checklist names no runs.

    An unticked run item is "done" if its name is in results/<kind>_summary.jsonl or
    its job is done, "live" if its job is running, else "queued" (a failed or stalled
    attempt still needs a run)."""
    cur = next((p for p in phase_data["phases"] if p["state"] == "current"), None)
    if cur is None:
        return None
    items = run_items(Path(cur["file"]))
    if not items:
        return None
    by_key = {(j["kind"], j["name"]): j for j in jobs}
    names: dict[str, set[str]] = {}
    for it in items:
        j = by_key.get((it["kind"], it["name"]))
        if it["kind"] not in names:
            names[it["kind"]] = summary_names(root, it["kind"])
        it["job_status"] = j["status"] if j else None
        if it["ticked"]:
            it["state"] = "ticked"
        elif it["name"] in names[it["kind"]] or (j and j["status"] == "done"):
            it["state"] = "done"
        elif j and j["status"] == "running":
            it["state"] = "live"
        else:
            it["state"] = "queued"
    queued = [i for i in items if i["state"] == "queued"]
    live = [i for i in items if i["state"] == "live"]
    anchor = live[0] if live else (queued[0] if queued else None)
    group = [i for i in queued if anchor and i["group"] == anchor["group"]]
    later = [i for i in queued if i not in group]
    estimates = {k: run_estimate(jobs, k) for k in {i["kind"] for i in queued}}

    def total(its: list[dict]) -> float | None:
        if any(estimates[i["kind"]] is None for i in its):
            return None
        return sum(estimates[i["kind"]]["seconds"] for i in its)

    group_seconds = total(group)
    start = now
    if live:
        live_job = by_key.get((live[0]["kind"], live[0]["name"]))
        start = live_job.get("eta") if live_job else None
    finish = (max(start, now) + group_seconds
              if start is not None and group_seconds is not None else None)
    ref = estimates.get(group[0]["kind"]) if group else None
    return {
        "phase": cur["phase"], "file": cur["file"],
        "section": anchor["section"] if anchor else items[0]["section"],
        "label": anchor["label"] if anchor else None,
        "total_items": len(items),
        "finished": sum(i["state"] in ("done", "ticked") for i in items),
        "live": [i["name"] for i in live],
        "group_queued": [i["name"] for i in group],
        "group_failed": [i["name"] for i in group if i["job_status"] in ("failed", "stalled")],
        "group_seconds": group_seconds, "group_finish": finish,
        "per_run": ref,
        "later_queued": [i["name"] for i in later],
        "later_groups": len({i["group"] for i in later}),
        "later_seconds": total(later),
        "items": items,
    }


# ---------------------------------------------------------------------------
# Assembly


def now_jobs(jobs: list[dict], now: float) -> list[dict]:
    """Running jobs, then jobs that stalled or failed within RECENT_SECONDS."""
    live = [j for j in jobs if j["status"] == "running"]
    recent = [j for j in jobs if j["status"] in ("stalled", "failed")
              and j["age_seconds"] is not None and j["age_seconds"] <= RECENT_SECONDS]
    def newest_first(j: dict) -> float:
        return -(j["modified"] or 0)

    return sorted(live, key=newest_first) + sorted(recent, key=newest_first)


def attention_jobs(jobs: list[dict], current: list[dict]) -> list[dict]:
    ids = {id(j) for j in current}
    return sorted((j for j in jobs if j["status"] in ATTENTION_STATES and id(j) not in ids),
                  key=lambda j: -(j["modified"] or 0))


def collect(root: Path, now: float | None = None,
            stale_minutes: float = DEFAULT_STALE_MINUTES) -> dict:
    now = time.time() if now is None else now
    stale = stale_minutes * 60
    jobs = (recall_jobs(root, now, stale) + eval_jobs(root, now, stale)
            + judge_jobs(root, now, stale) + train_jobs(root, now, stale)
            + distilled_jobs(root, now, stale))
    ph = phases(root)
    cur = next((x for x in ph["phases"] if x["state"] == "current"), None)
    if cur is None and ph["phases"]:
        cur = ph["phases"][-1]
    current = now_jobs(jobs, now)
    done = [j for j in jobs if j["status"] == "done" and j["modified"] is not None]
    last = max(done, key=lambda j: j["modified"], default=None)
    return {
        "root": str(root),
        "now": now,
        "stale_minutes": stale_minutes,
        "now_jobs": current,
        "last_finished": last,
        "next": queue(root, jobs, ph, now),
        "attention": attention_jobs(jobs, current),
        "jobs": jobs,
        "results": results_lines(root),
        "checklist": checklist(Path(cur["file"]) if cur else root / CHECKLIST),
        "phases": ph,
    }


# ---------------------------------------------------------------------------
# Formatting


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d"


def fmt_hours(seconds: float) -> str:
    """Rough total: '2.6 h', or minutes under an hour."""
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{max(1, round(seconds / 60))}m"


def fmt_clock(t: float, now: float) -> str:
    """Local HH:MM, with the weekday when it is not today."""
    lt, ln = time.localtime(t), time.localtime(now)
    same_day = (lt.tm_year, lt.tm_yday) == (ln.tm_year, ln.tm_yday)
    return time.strftime("%H:%M" if same_day else "%a %H:%M", lt)


def fmt_rate(rate: float | None, unit: str) -> str:
    if not rate:
        return "-"
    if rate >= 1:
        return f"{rate:.1f} {unit}/s"
    if rate * 60 >= 1:
        return f"{rate * 60:.0f} {unit}/min"
    return f"{rate * 3600:.0f} {unit}/h"


def fmt_eta(j: dict, now: float, short: bool = False) -> str:
    if j["status"] in ("stalled", "failed"):
        return f"quiet {fmt_age(j['age_seconds'])}"
    if j.get("eta") is None:
        return "ETA unknown"
    if j["eta_seconds"] < 0:
        return "ETA overdue" if short else f"ETA overdue (was {fmt_clock(j['eta'], now)})"
    if short:
        return f"ETA {fmt_clock(j['eta'], now)}"
    return f"ETA {fmt_age(j['eta_seconds'])} ({fmt_clock(j['eta'], now)})"


def fmt_progress(j: dict) -> str:
    exp = "?" if j["expected"] is None else str(j["expected"])
    pct = "?" if j["pct"] is None else f"{j['pct']:.0f}%"
    return f"{j['done']}/{exp} {pct}"


def _table(rows: list[tuple], right: set[int] = frozenset()) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = [c.rjust(widths[i]) if i in right else c.ljust(widths[i])
                 for i, c in enumerate(r)]
        out.append("  " + "  ".join(cells).rstrip())
    return out


def _format_now(data: dict) -> list[str]:
    out, now = ["Now"], data["now"]
    rows = data["now_jobs"]
    if not rows:
        lf = data.get("last_finished")
        if lf:
            out.append(f"  nothing running; last finished: {lf['kind']} {lf['name']} at "
                       f"{fmt_clock(lf['modified'], now)} ({fmt_age(lf['age_seconds'])} ago)")
        else:
            out.append("  nothing running")
        return out
    table = []
    for j in rows:
        exp = "?" if j["expected"] is None else str(j["expected"])
        pct = "?" if j["pct"] is None else f"{j['pct']:.0f}%"
        elapsed = j.get("elapsed_seconds")
        table.append((j["kind"], j["name"], f"{j['done']}/{exp}", pct, j["status"],
                      "-" if elapsed is None else fmt_age(elapsed),
                      fmt_rate(j.get("rate"), j["unit"]), fmt_eta(j, now), j["note"]))
    head = ("kind", "name", "done/exp", "pct", "status", "elapsed", "rate", "eta", "note")
    return out + _table([head, *table], right={2, 3, 5})


def _format_next(data: dict) -> list[str]:
    q, now = data.get("next"), data["now"]
    if not q:
        return []
    out = ["", f"Next  (Phase {q['phase']}, {q['section']})"]
    if not q["group_queued"] and not q["later_queued"]:
        tail = f"; live: {', '.join(q['live'])}" if q["live"] else ""
        return out + [(f"  nothing queued: {q['finished']}/{q['total_items']} run items "
                       f"done{tail}")]
    if q["group_queued"]:
        failed = set(q["group_failed"])
        names = ", ".join(n + (" (rerun)" if n in failed else "") for n in q["group_queued"])
        label = f"{q['label']}: " if q["label"] else ""
        out.append(f"  {label}{len(q['group_queued'])} queued: {names}")
        ref = q["per_run"]
        if q["group_seconds"] is not None and ref:
            finish = (f"; group done about {fmt_clock(q['group_finish'], now)}"
                      if q["group_finish"] else "")
            out.append(f"    estimate ~{fmt_hours(q['group_seconds'])} "
                       f"({fmt_age(ref['seconds'])} per run, {ref['how']} from "
                       f"{ref['from']}){finish}")
        else:
            out.append("    time unknown (no measured run of this kind yet)")
    elif q["live"]:
        out.append(f"  nothing else queued with {', '.join(q['live'])}")
    if q["later_queued"]:
        est = (f" (~{fmt_hours(q['later_seconds'])} estimate)"
               if q["later_seconds"] is not None else "")
        groups = q["later_groups"]
        out.append(f"  later: {len(q['later_queued'])} more in {groups} "
                   f"group{'s' if groups != 1 else ''}{est}")
    return out


def _format_attention(data: dict, show_all: bool) -> list[str]:
    jobs = data["attention"]
    if not jobs:
        return []
    recent = [j for j in jobs if j["age_seconds"] is not None
              and j["age_seconds"] <= RECENT_SECONDS]
    old = [j for j in jobs if j not in recent]
    shown = jobs if show_all else recent
    out = ["", "Attention"]
    if shown:
        out += _table([(j["kind"], j["name"], j["status"], fmt_progress(j),
                        fmt_age(j["age_seconds"]), j["note"]) for j in shown], right={4})
    if old and not show_all:
        counts: dict[str, int] = {}
        for j in old:
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        what = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
        oldest = max((j["age_seconds"] or 0) for j in old)
        out.append(f"  {len(old)} older than a day ({what}; oldest {fmt_age(oldest)}); "
                   "--all lists them")
    return out


def _item_line(o: dict) -> str:
    return "      - " + truncate(o["text"], CURRENT_ITEM_WIDTH)


def _format_phases(data: dict, show_all: bool = False) -> list[str]:
    out = ["", "Progress"]
    ph = data.get("phases") or {"phases": []}
    if not ph["phases"]:
        cl = data["checklist"]  # no scripts/phase<N>_*.md: the fixed Phase 1 path
        ph = {"phases": [{"phase": 1, "file": cl["file"], "state": "current",
                          "done": sum(x["done"] for x in cl["sections"]),
                          "total": sum(x["total"] for x in cl["sections"]),
                          "sections": cl["sections"],
                          "open_items": cl.get("open_items", [])}]}
        if not cl["sections"]:
            return out + ["  (no checklist items)"]
    live = set((data.get("next") or {}).get("live", []))
    rows = []
    for p in ph["phases"]:
        n_open = len(p["open_items"])
        extra = ""
        if p["state"] == "open":
            extra = f"{n_open} left"
            if n_open == 1:
                extra += ": " + truncate(p["open_items"][0]["text"], 70)
        rows.append((f"Phase {p['phase']}", Path(p["file"]).name, p["state"],
                     f"{p['done']}/{p['total']}", extra))
    table = _table(rows, right={3})
    for p, line in zip(ph["phases"], table):
        out.append(line)
        if p["state"] != "current" and not show_all:
            continue
        width = max((len(x["section"]) for x in p["sections"]), default=0)
        for x in p["sections"]:
            out.append(f"    {(x['section'] + ':').ljust(width + 1)}  {x['done']}/{x['total']}")
            items = [o for o in p["open_items"] if o["section"] == x["section"]]
            runs = [m.group(1) for o in items if (m := _RUN_NAME_RE.search(o["text"]))]
            out += [_item_line(o) for o in items if not _RUN_NAME_RE.search(o["text"])]
            if runs:
                names = [r + (" (running)" if r in live else "") for r in runs]
                out.append(truncate(f"      - runs: {', '.join(names)}",
                                    CURRENT_ITEM_WIDTH + 8))
    return out


def _format_jobs(data: dict) -> list[str]:
    out = ["", "Jobs (all)"]
    if not data["jobs"]:
        return out + ["  (no jobs found)"]
    rows = []
    for j in data["jobs"]:
        exp = "?" if j["expected"] is None else str(j["expected"])
        pct = "?" if j["pct"] is None else f"{j['pct']:.0f}%"
        rows.append((j["kind"], j["name"], f"{j['done']}/{exp}", pct, j["status"],
                     fmt_age(j["age_seconds"]), j["note"]))
    head = ("kind", "name", "done/exp", "pct", "status", "age", "note")
    return out + _table([head, *rows], right={2, 3, 5})


def _format_results(data: dict) -> list[str]:
    out = ["", "Results (last line of each results/*_summary.jsonl)"]
    if not data["results"]:
        return out + ["  (no results/*_summary.jsonl)"]
    width = max(len(r["file"]) for r in data["results"])
    return out + [f"  {r['file'].ljust(width)}  {r['text']}" for r in data["results"]]


def format_text(data: dict, show_all: bool = False) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(data["now"]))
    out = [f"tilt status  {stamp}  {data['root']}", ""]
    out += _format_now(data)
    out += _format_next(data)
    out += _format_attention(data, show_all)
    out += _format_phases(data, show_all)
    if show_all:
        out += _format_jobs(data)
        out += _format_results(data)
    if not data["jobs"]:
        out += ["", "(no jobs found)"]
    return "\n".join(out)


def format_line(data: dict) -> str:
    """One plain line for a status bar, at most LINE_WIDTH characters."""
    now, parts = data["now"], []
    current = data["now_jobs"]
    live = [j for j in current if j["status"] == "running"]
    if live:
        j = live[0]
        parts += [f"{j['kind']} {j['name']} {fmt_progress(j)}", fmt_eta(j, now, short=True)]
        if len(live) > 1:
            parts.append(f"+{len(live) - 1} live")
    elif current:
        j = current[0]
        parts.append(f"{j['status']} {j['kind']} {j['name']} {fmt_progress(j)} "
                     f"({fmt_age(j['age_seconds'])} ago)")
    else:
        lf = data.get("last_finished")
        parts.append("idle" + (f" · last: {lf['name']} done {fmt_clock(lf['modified'], now)}"
                               if lf else ""))
    q = data.get("next")
    if q and q["group_queued"]:
        est = f" (~{fmt_hours(q['group_seconds'])})" if q["group_seconds"] is not None else ""
        parts.append(f"{len(q['group_queued'])} {'queued' if live else 'next'}{est}")
    if not live:
        cur = next((p for p in data["phases"]["phases"] if p["state"] == "current"), None)
        if cur:
            parts.append(f"Phase {cur['phase']} {cur['done']}/{cur['total']}")
    return truncate(" · ".join(parts), LINE_WIDTH)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="Repo root (default: current directory)")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object instead of text")
    parser.add_argument("--line", action="store_true",
                        help="Print one short plain line for a status bar")
    parser.add_argument("--stale-minutes", type=float, default=DEFAULT_STALE_MINUTES,
                        help="An incomplete file untouched this long is stalled (default 10)")
    parser.add_argument("--all", action="store_true",
                        help="Also list old attention items, every phase's sections, every "
                             "job and the last results line of each summary")
    args = parser.parse_args(argv)
    try:
        data = collect(Path(args.root).resolve(), stale_minutes=args.stale_minutes)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=1))
        elif args.line:
            print(format_line(data))
        else:
            print(format_text(data, show_all=args.all))
    except Exception as e:  # noqa: BLE001 - a status view never fails the shell
        print(f"tilt-status: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
