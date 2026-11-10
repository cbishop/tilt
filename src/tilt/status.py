"""One-screen progress view of the experiment, read from files on disk.

No network calls and no model loading. Every reader tolerates missing files and a
half-written last JSONL line (skipped, as a run killed mid-write leaves one).

Sections:
    - Jobs: by default only jobs that are not done, plus a count of done jobs
      (--all lists every job; --json always emits every job). One line per job under runs/ (recall, eval, judge, train) and
      data/distilled/, with done/expected, percent and a status word: done,
      running (incomplete, modified within --stale-minutes), stalled (incomplete,
      older than that), empty, failed (train returncode != 0) or unknown
      (expected count cannot be worked out and the file is not fresh).
    - Results: the last line of each results/*_summary.jsonl.
    - Checklist: every scripts/phase<N>_*.md in phase order. The current phase is
      the first with an open "- [ ]" item: its done/total per section and every open
      item with its section. Earlier phases print one "complete" line, later ones
      one "not started" line.

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

_ITEM_RE = re.compile(r"^\s*[-*] \[([ xX])\] (.*)$")
_CKPT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")
_ITER_LOG_RE = re.compile(r"\bIter (\d+)\b")
_ITERS_CFG_RE = re.compile(r"^iters:\s*(\d+)\s*$", re.MULTILINE)


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


def job(kind: str, name: str, path: Path, done: int, expected: int | None,
        modified: float | None, now: float, stale_seconds: float, note: str = "",
        status: str | None = None) -> dict:
    pct = round(100 * done / expected, 1) if expected else None
    return {
        "kind": kind, "name": name, "path": str(path), "done": done, "expected": expected,
        "pct": pct,
        "status": status or classify(done, expected, modified, now, stale_seconds),
        "modified": modified, "age_seconds": None if modified is None else now - modified,
        "note": note,
    }


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
        out.append(job("recall", p.stem, p, done, expected, mtime(p), now, stale, note))
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
        out.append(job("eval", p.stem, p, done, expected, mtime(p), now, stale, note))
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
        for r in read_records(p):
            version = r.get("rubric_version", FRAMING_RUBRIC_VERSION)
            if framing and version != FRAMING_RUBRIC_VERSION:
                continue
            if label_key in r and "id" in r and "sample_idx" in r:
                try:
                    judged.add((r["id"], int(r["sample_idx"])))
                except (TypeError, ValueError):
                    continue
        if src is None:
            out.append(job("judge", p.stem, p, len(judged), None, mtime(p), now, stale,
                           "responses file missing"))
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
        j = job("judge", p.stem, p, done, len(wanted), mtime(p), now, stale, note)
        j["source"] = str(src)
        out.append(j)
    return out


def train_jobs(root: Path, now: float, stale: float) -> list[dict]:
    out = []
    train = root / "runs/train"
    try:
        run_dirs = sorted(d for d in train.iterdir() if d.is_dir())
    except OSError:
        return out
    for d in run_dirs:
        log = read_json(d / "train_log.json")
        iters = log.get("iters") if log else None
        if not isinstance(iters, int):
            try:
                m = _ITERS_CFG_RE.search((d / "lora_config.yaml").read_text(encoding="utf-8"))
                iters = int(m.group(1)) if m else None
            except OSError:
                iters = None
        adapters = d / "adapters"
        latest = 0
        times = [mtime(d / "stdout.log"), mtime(d / "train_log.json"), mtime(d)]
        try:
            for f in adapters.iterdir():
                times.append(mtime(f))
                m = _CKPT_RE.match(f.name)
                if m:
                    latest = max(latest, int(m.group(1)))
        except OSError:
            pass
        try:
            with (d / "stdout.log").open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 16384))
                tail = f.read().decode("utf-8", errors="replace")
            hits = _ITER_LOG_RE.findall(tail)
            if hits:
                latest = max(latest, max(int(h) for h in hits))
        except OSError:
            pass
        modified = max((t for t in times if t is not None), default=None)
        final = (adapters / "adapters.safetensors").is_file()
        status = None
        note = ""
        if log is not None and "returncode" in log:
            rc = log["returncode"]
            if rc == 0:
                status = "done"
                latest = iters if isinstance(iters, int) else latest
            else:
                status = "failed"
            note = f"returncode {rc}"
        elif final and iters and latest >= iters:
            status = "done"
        out.append(job("train", d.name, d, latest, iters, modified, now, stale, note, status))
    return out


def distilled_jobs(root: Path, now: float, stale: float) -> list[dict]:
    out = []
    for p in _jsonl_files(root / "data/distilled"):
        stats_path = p.with_suffix(".stats.json")
        stats = read_json(stats_path)
        m_jsonl, m_stats = mtime(p), mtime(stats_path)
        kept = None
        note = "kept rows"
        if (stats and isinstance(stats.get("kept"), int) and m_stats is not None
                and (m_jsonl is None or m_stats >= m_jsonl)):
            kept = stats["kept"]
            note = f"kept rows, from {stats_path.name}"
        if kept is None:
            latest: dict = {}
            for r in read_records(p):
                latest[r.get("id")] = r
            kept = sum(1 for r in latest.values() if r.get("filtered_reason") is None
                       and isinstance(r.get("answer"), str) and r["answer"].strip())
            note = "kept rows, counted"
        modified = max((t for t in (m_jsonl, m_stats) if t is not None), default=None)
        out.append(job("distill", p.stem, p, kept, DISTILL_TARGET, modified, now, stale, note))
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
    """Every phase checklist with a state. The current phase is the first with an open
    item; earlier ones are complete; later ones are "not started" (or "started" if
    any item is already ticked)."""
    out: list[dict] = []
    current = None
    for n, path in phase_files(root):
        c = checklist(path)
        done = sum(x["done"] for x in c["sections"])
        total = sum(x["total"] for x in c["sections"])
        if current is None and c["open_total"]:
            state, current = "current", n
        elif current is None:
            state = "complete"
        else:
            state = "started" if done else "not started"
        out.append({"phase": n, "file": str(path), "state": state, "done": done,
                    "total": total, "sections": c["sections"], "open_items": c["open_items"]})
    return {"current": current, "phases": out}


def current_checklist(root: Path) -> Path:
    """The current phase's checklist (first with an open item), else the last phase's,
    else scripts/phase1_pilot.md."""
    data = phases(root)
    cur = next((x for x in data["phases"] if x["state"] == "current"), None)
    if cur is None and data["phases"]:
        cur = data["phases"][-1]
    return Path(cur["file"]) if cur else root / CHECKLIST


def checklist(path: Path) -> dict:
    """done/total per '## ' section and the open items, in file order."""
    sections: list[dict] = []
    open_items: list[str] = []
    open_full: list[dict] = []
    current = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"file": str(path), "sections": [], "open": [], "open_total": 0,
                "open_items": []}
    for line in lines:
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
# Assembly and output


def collect(root: Path, now: float | None = None,
            stale_minutes: float = DEFAULT_STALE_MINUTES) -> dict:
    now = time.time() if now is None else now
    stale = stale_minutes * 60
    jobs = (recall_jobs(root, now, stale) + eval_jobs(root, now, stale)
            + judge_jobs(root, now, stale) + train_jobs(root, now, stale)
            + distilled_jobs(root, now, stale))
    return {
        "root": str(root),
        "now": now,
        "stale_minutes": stale_minutes,
        "jobs": jobs,
        "results": results_lines(root),
        "checklist": checklist(current_checklist(root)),
        "phases": phases(root),
    }


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


def format_text(data: dict, show_all: bool = False) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(data["now"]))
    out = [f"tilt status  {data['root']}  {stamp}", "", "Jobs"]
    all_jobs = data["jobs"]
    jobs = all_jobs if show_all else [j for j in all_jobs if j["status"] != "done"]
    n_done = sum(j["status"] == "done" for j in all_jobs)
    if not all_jobs:
        out.append("  (no jobs found)")
    if jobs:
        rows = []
        for j in jobs:
            exp = "?" if j["expected"] is None else str(j["expected"])
            pct = "?" if j["pct"] is None else f"{j['pct']:.0f}%"
            rows.append((j["kind"], j["name"], f"{j['done']}/{exp}", pct, j["status"],
                         fmt_age(j["age_seconds"]), j["note"]))
        head = ("kind", "name", "done/exp", "pct", "status", "age", "note")
        widths = [max(len(r[i]) for r in [head, *rows]) for i in range(6)]
        for r in [head, *rows]:
            out.append("  " + "  ".join([
                r[0].ljust(widths[0]), r[1].ljust(widths[1]), r[2].rjust(widths[2]),
                r[3].rjust(widths[3]), r[4].ljust(widths[4]), r[5].rjust(widths[5]), r[6],
            ]).rstrip())
    if n_done and not show_all:
        out.append(f"  {n_done} jobs done (use --all to list them)")
    out += ["", "Results"]
    if not data["results"]:
        out.append("  (no results/*_summary.jsonl)")
    width = max((len(r["file"]) for r in data["results"]), default=0)
    for r in data["results"]:
        out.append(f"  {r['file'].ljust(width)}  {r['text']}")
    out += _format_phases(data)
    return "\n".join(out)


def _format_phases(data: dict) -> list[str]:
    out = ["", "Checklist"]
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
    for p in ph["phases"]:
        name = Path(p["file"]).name
        out.append(f"  Phase {p['phase']} ({name}): {p['state']}, {p['done']}/{p['total']}")
        if p["state"] != "current":
            continue
        width = max((len(x["section"]) for x in p["sections"]), default=0)
        for x in p["sections"]:
            out.append(f"    {(x['section'] + ':').ljust(width + 1)}  {x['done']}/{x['total']}")
        if p["open_items"]:
            out.append(f"    open ({len(p['open_items'])}):")
            for o in p["open_items"]:
                item = "[" + o["section"] + "] " + o["text"]
                out.append("      - " + truncate(item, CURRENT_ITEM_WIDTH))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="Repo root (default: current directory)")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object instead of text")
    parser.add_argument("--stale-minutes", type=float, default=DEFAULT_STALE_MINUTES,
                        help="An incomplete file untouched this long is stalled (default 10)")
    parser.add_argument("--all", action="store_true",
                        help="List every job, including done ones (default: only jobs that "
                             "are not done, plus a count of done jobs)")
    args = parser.parse_args(argv)
    try:
        data = collect(Path(args.root).resolve(), stale_minutes=args.stale_minutes)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=1))
        else:
            print(format_text(data, show_all=args.all))
    except Exception as e:  # noqa: BLE001 - a status view never fails the shell
        print(f"tilt-status: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
