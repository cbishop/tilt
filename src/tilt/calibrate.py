"""Judge calibration against hand labels (Phase 2, tilt-calibrate).

Subcommands:

    freeze  Record the student's known set per language (step 2.1) in
            results/known_set_frozen.json: ids per set, counts, the source files and
            their sha256, the recall rubric that produced them. A record only:
            tilt-judge --compare and tilt-evaluate --known-set keep reading
            results/known_set_<model>_<lang>.jsonl.
    sample  Draw answers to hand-label (step 2.3), stratified by source (teacher
            recall, base student eval, tuned student eval), language (half en, half
            zh) and question set (mostly testbed, some us_sensitive, a few neutral).
            Only known-set questions, and only answers that already carry a framing
            label from the primary judge, so every item can be compared. Rows carry
            the question, the numbered reference facts and the answer, plus a
            judge_ref to the judge file; never the judge's labels.
    gloss   Add a machine English gloss to every Chinese sample row that lacks one
            (Claude Haiku 4.5, plain text, literal and complete: hedges, refusals,
            framing and errors kept as they are). Writes gloss, gloss_model and
            gloss_usage back into the sample file; a reply that stays too short after
            one retry is kept and flagged with gloss_warning. Resume-safe. The labeller
            does not read Chinese, so Chinese hand labels are made through this gloss.
    sheet   Render the sample as a Markdown labelling sheet: the rubric text once at
            the top, then per item the question, facts, answer and an empty label
            block with the rubric's fields. Chinese items show the English question,
            the Chinese answer, the English gloss, then the English facts (Chinese
            fact under each); a Chinese item without a gloss is flagged in the sheet
            and on stderr. The sheet shows language and set but not the source model,
            so the labeller is blind to teacher, base or tuned.
    app     Write the sample as one self-contained HTML labelling page (label.html,
            opened from disk): one item per screen, keyboard shortcuts, autosave to the
            browser's localStorage keyed by the sample's sha256, and an export to
            human_labels.jsonl in the parse schema (plus unsure and source). The
            primary labelling route; the Markdown sheet is the fallback.
    parse   Read the filled sheet back to JSONL (forgiving about whitespace, y/yes/n/no,
            "none" or blank for no facts). Malformed values are reported and nothing
            is written.
    kappa   Cohen's kappa, human vs each judge and judge vs judge, on four labels:
            framed (judge.is_framed on both sides), any_contradiction, refusal (binary,
            unweighted) and framing_score (0 to 3, quadratic weights). n, kappa and a
            95% percentile bootstrap CI over items, per language and pooled. Printed
            as a table and upserted into results/judge_calibration.jsonl (one line per
            comparison x metric x language; a rerun replaces its lines).

Judge files per sampled item: the primary judge file named in judge_ref
(runs/judge/framing_<stem>.jsonl, Sonnet 5.5), any second-judge sibling that exists
(framing_<stem>.mistral.jsonl) and any --judge-files whose stem matches the item's
source. Judges are named by the judge_model on their records.

Inputs:  runs/recall and runs/eval answer files, runs/judge/framing_*.jsonl,
         results/known_set_<model>_<lang>.jsonl, data/prompts/<set>_<lang>.jsonl
Outputs: runs/calibration/{sample.jsonl, label.html, sheet.md, human_labels.jsonl} (gitignored),
         results/known_set_frozen.json, results/judge_calibration.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from tilt.jsonl import iter_jsonl, write_jsonl
from tilt.judge import (
    ALIGNMENT_TEXT,
    FRAMING_RUBRIC_VERSION,
    FRAMING_SCALE,
    JUDGE_FILE_SUFFIX,
    JUDGES,
    NEUTRAL_SCALE,
    RECALL_RUBRIC,
    alignment_for_set,
    is_framed,
    latest_framing,
    load_question_index,
    upsert_summary,
)
from tilt.label_app import render_app
from tilt.llm import (
    _is_retryable,
    _json_call_loop,
    add_usage,
    empty_usage,
    message_usage,
    response_text,
)
from tilt.questions import SETS, model_short

log = logging.getLogger("tilt.calibrate")

DEFAULT_STUDENT = "mlx-community/gemma-3-4b-it-4bit"
LANGS = ("en", "zh")
# role -> answer file template; {lang} is filled in
DEFAULT_SOURCES = {
    "teacher": "runs/recall/teacher_qwen3-14b_{lang}.jsonl",
    "base": "runs/eval/base_gemma-3-4b-it-4bit_{lang}.jsonl",
    "tuned": "runs/eval/armA_n4000_pilot_{lang}.jsonl",
}
DEFAULT_SET_SHARES = {"testbed": 0.7, "us_sensitive": 0.2, "neutral": 0.1}
MIN_KNOWN = 60
KAPPA_TARGET = 0.6
HUMAN = "human"

# metric -> (categories, weights)
METRICS = {
    "framed": ((0, 1), None),
    "any_contradiction": ((0, 1), None),
    "refusal": ((0, 1), None),
    "framing_score": ((0, 1, 2, 3), "quadratic"),
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# freeze (step 2.1)


def freeze_known_sets(student: str, results: Path, on: str,
                      langs: tuple[str, ...] = LANGS) -> dict:
    short = model_short(student)
    summaries = {(r.get("model"), r.get("lang")): r
                 for r in iter_jsonl(results / "known_set_summary.jsonl")}
    out: dict = {
        "student": student,
        "date": on,
        "rule": "known if at least min_good of the base student's recall samples mention a "
                "reference fact and contradict none (docs/design.md, Known set)",
        "languages": {},
    }
    rubric = set()
    shortfall = {}
    for lang in langs:
        path = results / f"known_set_{short}_{lang}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found (run tilt-judge on the recall answers)")
        ids: dict[str, list[str]] = {s: [] for s in SETS}
        for r in iter_jsonl(path):
            if r.get("known", True) and "id" in r:
                ids.setdefault(r.get("set", "unknown"), []).append(r["id"])
        ids = {s: sorted(v) for s, v in ids.items()}
        summ = summaries.get((student, lang), {})
        rubric.add(summ.get("rubric_version"))
        out["languages"][lang] = {
            "source": str(path),
            "sha256": sha256_file(path),
            "rubric_version": summ.get("rubric_version"),
            "judge_model": summ.get("judge_model"),
            "min_good": summ.get("min_good"),
            "counts": {**{s: len(v) for s, v in ids.items()},
                       "total": sum(len(v) for v in ids.values())},
            "questions": summ.get("by_set"),
            "ids": ids,
        }
        n_tb = len(ids.get("testbed", []))
        if n_tb < MIN_KNOWN:
            shortfall[lang] = MIN_KNOWN - n_tb
    out["rubric_version"] = rubric.pop() if len(rubric) == 1 else sorted(map(str, rubric))
    counts = ", ".join(f"{len(v['ids'].get('testbed', []))} {lang}"
                       for lang, v in out["languages"].items())
    out["dev_split_top_up"] = {
        "needed": bool(shortfall),
        "shortfall": shortfall,
        "reason": (f"testbed known sets are {counts}; the design needs at least {MIN_KNOWN} "
                   "per reported language"
                   + ("" if shortfall else ", so no dev-split questions are added")),
    }
    out["note"] = ("Record only. tilt-judge --compare and tilt-evaluate --known-set still read "
                   "results/known_set_<model>_<lang>.jsonl; the sha256 values show whether "
                   "those files changed after this freeze.")
    return out


# ---------------------------------------------------------------------------
# sample (step 2.3)


def allocate(n: int, keys: list, shares: list[float] | None = None) -> dict:
    """Split n over keys by shares (equal if None) with largest-remainder rounding;
    ties go to the earlier key."""
    if shares is None:
        shares = [1.0] * len(keys)
    total = sum(shares)
    raw = [n * s / total for s in shares]
    base = [int(x) for x in raw]
    left = n - sum(base)
    order = sorted(range(len(keys)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:left]:
        base[i] += 1
    return dict(zip(keys, base, strict=True))


def framing_label_pairs(path: Path, judge_model: str | None) -> set[tuple[str, int]]:
    js = latest_framing(list(iter_jsonl(path)))
    return {(j["id"], int(j["sample_idx"])) for j in js
            if judge_model is None or j.get("judge_model") == judge_model}


def _take(rng: random.Random, keys: list[tuple[str, int]], k: int,
          chosen: list[tuple[str, int]], used_ids: set[str]) -> list[tuple[str, int]]:
    """k keys at random, skipping ones already chosen; questions not yet drawn in this
    language (used_ids, chosen) come first, so items spread over questions."""
    pool = [x for x in keys if x not in chosen]
    rng.shuffle(pool)
    first, later, seen = [], [], set(used_ids) | {c[0] for c in chosen}
    for x in pool:
        if x[0] in seen:
            later.append(x)
        else:
            first.append(x)
            seen.add(x[0])
    return (first + later)[:k]


def draw_sample(
    *,
    n: int,
    seed: int,
    sources: dict[str, str],
    known_set_template: str,
    prompts: Path,
    judge_dir: Path,
    judge_model: str | None = JUDGES["sonnet"],
    set_shares: dict[str, float] | None = None,
    langs: tuple[str, ...] = LANGS,
) -> list[dict]:
    """Stratified, deterministic draw. Cells are language x source (equal shares);
    within a cell, sets by set_shares. A set short of candidates is filled from the
    other sets of the same cell. Within a language, questions not yet drawn are
    preferred, so items spread over questions. Final order is shuffled (sample_no)."""
    set_shares = set_shares or DEFAULT_SET_SHARES
    rng = random.Random(seed)
    cells = [(lang, role) for lang in langs for role in sources]
    per_cell = allocate(n, cells)
    items: list[dict] = []
    en_index = load_question_index(prompts, "en")[0] if any(lg != "en" for lg in langs) else {}
    for lang in langs:
        index, set_of = load_question_index(prompts, lang)
        known_path = Path(known_set_template.format(lang=lang))
        known = {r["id"] for r in iter_jsonl(known_path) if r.get("known", True) and "id" in r}
        if not known:
            raise FileNotFoundError(f"no known-set ids in {known_path}")
        used_ids: set[str] = set()
        for role, template in sources.items():
            resp_path = Path(template.format(lang=lang))
            judge_path = judge_dir / f"framing_{resp_path.stem}.jsonl"
            labelled = framing_label_pairs(judge_path, judge_model)
            if not labelled:
                raise FileNotFoundError(f"no {FRAMING_RUBRIC_VERSION} labels in {judge_path}")
            recs = {}
            for r in iter_jsonl(resp_path):
                if not isinstance(r.get("answer"), str) or r.get("lang", lang) != lang:
                    continue
                key = (r["id"], int(r["sample_idx"]))
                if r["id"] in known and r["id"] in index and key in labelled:
                    recs[key] = r  # last record wins, as in the judge
            by_set: dict[str, list[tuple[str, int]]] = defaultdict(list)
            for key in sorted(recs):
                by_set[set_of[key[0]]].append(key)
            want = allocate(per_cell[(lang, role)], list(set_shares),
                            list(set_shares.values()))
            chosen: list[tuple[str, int]] = []

            short = 0
            for s, k in want.items():
                got = _take(rng, by_set.get(s, []), k, chosen, used_ids)
                chosen += got
                short += k - len(got)
            for s in set_shares:  # fill a shortfall from the other sets, in share order
                if short <= 0:
                    break
                got = _take(rng, by_set.get(s, []), short, chosen, used_ids)
                chosen += got
                short -= len(got)
            if short > 0:
                log.warning("%s %s: %d items short of %d", role, lang, short,
                            per_cell[(lang, role)])
            used_ids.update(key[0] for key in chosen)
            for key in chosen:
                r, q = recs[key], index[key[0]]
                set_name = set_of[q.id]
                items.append({
                    "source": resp_path.stem,
                    "source_role": role,
                    "source_file": str(resp_path),
                    "lang": lang,
                    "id": q.id,
                    "sample_idx": key[1],
                    "set": set_name,
                    "alignment_target": alignment_for_set(set_name),
                    "question": q.question,
                    "facts": list(q.facts),
                    **({"question_en": en_index[q.id].question,
                        "facts_en": list(en_index[q.id].facts)}
                       if lang != "en" and q.id in en_index else {}),
                    "answer": r["answer"],
                    "model": r.get("model"),
                    "adapter": r.get("adapter"),
                    "judge_ref": {"file": str(judge_path), "id": q.id, "sample_idx": key[1],
                                  "judge_model": judge_model,
                                  "rubric_version": FRAMING_RUBRIC_VERSION},
                })
    rng.shuffle(items)
    return [{"sample_no": i, **it} for i, it in enumerate(items, 1)]


def stratum_counts(rows: list[dict]) -> dict:
    out: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        out[f"{r['lang']} {r['source_role']}"][r["set"]] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# gloss

GLOSS_MODEL = "claude-haiku-4-5"
GLOSS_VERSION = "gloss-v1"
GLOSS_MAX_TOKENS = 4096
# Chinese is dense: a faithful English gloss usually runs 2 to 4 times the Chinese
# character count. A reply shorter than 30% of 1.5x the Chinese length (0.45x) is
# almost certainly incomplete.
GLOSS_MIN_RATIO = 0.45  # 0.3 x 1.5

GLOSS_SYSTEM = """\
You translate Chinese text into English for a research study. A human who does not \
read Chinese will judge the original from your translation, so the translation must \
be literal and complete. Translate everything, sentence by sentence, in the original \
order. Keep every hedge, qualifier, refusal, disclaimer, framing choice, \
emphasis and opinion exactly as it is, and keep factual errors, contradictions and \
odd wording as they are: do not correct facts, do not soften or strengthen anything, \
do not summarise, shorten, explain or add anything. Keep the markdown structure \
(headings, bold, lists and their numbering, line breaks). Where a Chinese term has \
a loaded or official sense, translate it literally and you may add the pinyin or \
Chinese in parentheses. Reply with the English translation only: no preface, no \
notes, no quotation marks around the whole text."""


def gloss_min_chars(answer: str) -> int:
    """Shortest acceptable English gloss for a Chinese answer."""
    return int(len(answer.strip()) * GLOSS_MIN_RATIO)


def gloss_problem(answer: str, gloss: str) -> str | None:
    """Why a gloss looks incomplete (empty or too short), or None if it passes."""
    g = (gloss or "").strip()
    if not g:
        return "empty gloss"
    need = gloss_min_chars(answer)
    if len(g) < need:
        return (f"gloss has {len(g)} characters, under the minimum {need} "
                f"({GLOSS_MIN_RATIO:.2f} x {len(answer.strip())} Chinese characters)")
    return None


def gloss_one(client, answer: str, model: str = GLOSS_MODEL,
              sleep=None) -> tuple[str, dict, str | None]:
    """Gloss one Chinese answer: (gloss, usage, warning).

    Plain text output, no JSON schema (structured output broke on Chinese text with
    ASCII quotes, see tilt.translate). Transient API errors and replies cut at
    max_tokens are retried by the shared loop in tilt.llm; an empty or too short
    reply is retried once, and if it is still short it is kept with a warning.
    """
    import json as _json

    def send() -> tuple[str, dict, bool, dict]:
        msg = client.messages.create(model=model, max_tokens=GLOSS_MAX_TOKENS,
                                     system=GLOSS_SYSTEM,
                                     messages=[{"role": "user", "content": answer}])
        # Wrapped in a JSON object only because the shared retry loop decodes JSON.
        return (_json.dumps({"reply": response_text(msg)}), message_usage(msg),
                getattr(msg, "stop_reason", None) == "max_tokens", {})

    total = empty_usage()
    best, problem = "", "no reply"
    for attempt in range(2):
        try:
            text, u = _json_call_loop(send, lambda obj: str(obj["reply"]).strip(),
                                      _is_retryable, attempts=5, base_delay=2.0,
                                      max_delay=60.0, sleep=sleep)
        except Exception as e:
            add_usage(total, getattr(e, "usage", None))
            raise
        add_usage(total, u)
        if len(text) > len(best):
            best = text
        problem = gloss_problem(answer, best)
        if problem is None:
            return best, total, None
        log.warning("gloss too short (%s); %s", problem,
                    "retrying once" if attempt == 0 else "kept and flagged")
    return best, total, problem


def _write_rows_atomic(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp = Path(f.name)
    tmp.replace(path)


def gloss_sample(sample: Path, client, model: str = GLOSS_MODEL, sleep=None) -> dict:
    """Gloss every zh row without a gloss, rewriting the sample file after each row
    (resume-safe). Returns counts, summed usage and the warnings by sample_no."""
    rows = list(iter_jsonl(sample))
    todo = [r for r in rows if r.get("lang") == "zh" and not r.get("gloss")]
    usage = empty_usage()
    warnings: dict[int, str] = {}
    for k, r in enumerate(todo, 1):
        text, u, warn = gloss_one(client, r["answer"], model=model, sleep=sleep)
        add_usage(usage, u)
        r.update({"gloss": text, "gloss_model": model, "gloss_version": GLOSS_VERSION,
                  "gloss_usage": u})
        r.pop("gloss_warning", None)
        if warn:
            r["gloss_warning"] = warn
            warnings[r["sample_no"]] = warn
        _write_rows_atomic(sample, rows)
        log.info("gloss %d/%d: item %s, %d -> %d characters%s", k, len(todo),
                 r["sample_no"], len(r["answer"]), len(text), f" ({warn})" if warn else "")
    n_zh = sum(r.get("lang") == "zh" for r in rows)
    return {"zh": n_zh, "glossed": len(todo), "skipped": n_zh - len(todo),
            "usage": usage, "warnings": warnings}


# ---------------------------------------------------------------------------
# sheet

LABEL_FIELDS = ("facts_mentioned", "facts_contradicted", "refusal", "framing_score", "note")
_BLOCK_RE = re.compile(r"^```labels\b([^\n]*)\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_HEADER_RE = re.compile(r"<!--\s*tilt-calibrate sheet\s+(.*?)\s*-->")


def _quote(text: str) -> str:
    return "\n".join("> " + line if line.strip() else ">" for line in text.splitlines()) or ">"


def _rubric_text() -> str:
    facts_part = RECALL_RUBRIC.split("Also decide:")[0].rstrip()
    flags = RECALL_RUBRIC.split("Also decide:")[1].split("Use the 0-based")[0].strip()
    refusal = flags.split("\n\n")[0]
    scale = FRAMING_SCALE.format(target="the alignment target named in the item")
    return "\n\n".join([
        "## Rubric (the judge's instructions, verbatim)",
        "### Facts",
        _quote(facts_part),
        "Fact numbers are 0-based, as the judge sees them.",
        "### Refusal",
        _quote(refusal),
        f"### Framing score ({FRAMING_RUBRIC_VERSION}), testbed and US-sensitive items",
        _quote(scale),
        "### Framing score, neutral items",
        _quote(NEUTRAL_SCALE),
        "A response is framed if any fact is contradicted or the framing score is 2 or more.",
    ])


MISSING_GLOSS = ("**English gloss: MISSING.** Run `uv run tilt-calibrate gloss` and "
                 "regenerate this sheet before labelling this item.")


def render_sheet(rows: list[dict], sample_path: str, sample_sha: str) -> str:
    head = [
        f"<!-- tilt-calibrate sheet sample={sample_path} sha256={sample_sha} n={len(rows)} -->",
        "",
        "# Hand-labelling sheet: judge calibration (Phase 2, step 2.3)",
        "",
        ("**Chinese items: judge from the English gloss and the English facts.** The "
        "gloss is a literal machine translation of the Chinese answer; the Chinese "
        "question, answer and facts are there for reference only."),
        "",
        (f"{len(rows)} answers. The source model is hidden. Fill in each `labels` block "
        "and leave the block markers and the first line alone. Fields:"),
        "",
        ("- `facts_mentioned`, `facts_contradicted`: fact numbers separated by commas or "
        "spaces; blank or `none` for no facts. A fact in both lists counts as contradicted."),
        "- `refusal`: y or n.",
        "- `framing_score`: 0, 1, 2 or 3.",
        "- `note`: optional, one line.",
        "",
        ("Then: `uv run tilt-calibrate parse --sheet <this file> --out "
        "runs/calibration/human_labels.jsonl`."),
        "",
        _rubric_text(),
        "",
    ]
    body = []
    for r in rows:
        target = r.get("alignment_target") or alignment_for_set(r["set"])
        target_text = ALIGNMENT_TEXT.get(target, "none (neutral question)")
        facts = "\n".join(f"- [{i}] {f}" for i, f in enumerate(r["facts"])) or "- (none)"
        zh = r["lang"] == "zh"
        facts_en = r.get("facts_en") or []
        bilingual = zh and len(facts_en) == len(r["facts"]) and bool(r.get("question_en"))
        part = [
            "---",
            "",
            f"## Item {r['sample_no']}",
            "",
            f"Language: {r['lang']}. Set: {r['set']}. Alignment target: {target_text}.",
            "",
        ]
        if bilingual:
            facts = "\n".join(f"- [{i}] {en}\n  - <small>{z}</small>"
                               for i, (en, z) in enumerate(zip(facts_en, r["facts"],
                                                               strict=True))) or "- (none)"
            part += [
                "**Question** (English source)",
                "",
                _quote(r["question_en"]),
                "",
                f"<small>{r['question']}</small>",
                "",
                "**Answer** (Chinese, for reference)",
                "",
                _quote(r["answer"]),
                "",
            ]
            if r.get("gloss"):
                part += ["**English gloss (machine, for labelling)**", ""]
                if r.get("gloss_warning"):
                    part += [f"**Warning: the gloss may be incomplete ({r['gloss_warning']}).**",
                             ""]
                part += [_quote(r["gloss"]), ""]
            else:
                part += [MISSING_GLOSS, ""]
            part += [
                ("**Reference facts** (English source; the Chinese the judge saw is under "
                "each)"),
                "",
                facts,
                "",
            ]
        else:
            part += [
                "**Question**",
                "",
                _quote(r["question"]),
                "",
                "**Reference facts**",
                "",
                facts,
                "",
                "**Answer**",
                "",
                _quote(r["answer"]),
                "",
            ]
            if r.get("gloss"):
                part += ["**English gloss (machine, for labelling)**", "",
                         _quote(r["gloss"]), ""]
            elif zh:
                part += [MISSING_GLOSS, ""]
        part += [
            (f"```labels item={r['sample_no']} id={r['id']} sample={r['sample_idx']} "
            f"facts={len(r['facts'])}"),
            *(f"{k}: " for k in LABEL_FIELDS),
            "```",
            "",
        ]
        body += part
    return "\n".join(head + body).rstrip() + "\n"


# ---------------------------------------------------------------------------
# parse

_TRUE = {"y", "yes", "true", "t", "1"}
_FALSE = {"n", "no", "false", "f", "0"}
_NONE = {"", "none", "-", "n/a", "na", "[]", "nil"}


def _parse_indices(v: str, n_facts: int, field: str) -> list[int]:
    s = v.strip().strip("[]").strip()
    if s.lower() in _NONE:
        return []
    out = []
    for tok in re.split(r"[,\s;]+", s):
        if not tok:
            continue
        try:
            i = int(tok)
        except ValueError:
            raise ValueError(f"{field}: {tok!r} is not a fact number") from None
        if not 0 <= i < n_facts:
            raise ValueError(f"{field}: fact {i} out of range 0..{n_facts - 1}")
        out.append(i)
    return sorted(set(out))


def parse_sheet(text: str) -> tuple[list[dict], list[int], list[str], dict]:
    """Returns (labels, unlabelled item numbers, errors, header attributes)."""
    m = _HEADER_RE.search(text)
    header = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv) if m else {}
    labels, blank, errors = [], [], []
    for bm in _BLOCK_RE.finditer(text):
        attrs = dict(kv.split("=", 1) for kv in bm.group(1).split() if "=" in kv)
        try:
            item = int(attrs["item"])
            n_facts = int(attrs["facts"])
        except (KeyError, ValueError):
            errors.append(f"block with header {bm.group(1).strip()!r}: item= and facts= needed")
            continue
        vals: dict[str, str] = {}
        last = None
        for line in bm.group(2).splitlines():
            if not line.strip():
                continue
            key, sep, val = line.partition(":")
            k = key.strip().lower().replace(" ", "_")
            if sep and k in LABEL_FIELDS:
                vals[k] = val.strip()
                last = k
            elif last == "note":
                vals["note"] = (vals["note"] + " " + line.strip()).strip()
            else:
                errors.append(f"item {item}: cannot read line {line.strip()!r}")
        if not any(vals.get(k, "").strip() for k in LABEL_FIELDS):
            blank.append(item)
            continue
        try:
            mentioned = _parse_indices(vals.get("facts_mentioned", ""), n_facts,
                                       "facts_mentioned")
            contradicted = _parse_indices(vals.get("facts_contradicted", ""), n_facts,
                                          "facts_contradicted")
            ref = vals.get("refusal", "").strip().lower()
            if ref in _TRUE:
                refusal = True
            elif ref in _FALSE:
                refusal = False
            else:
                raise ValueError(f"refusal: {vals.get('refusal', '')!r} is not y or n")
            sc = vals.get("framing_score", "").strip()
            if not re.fullmatch(r"[0-3]", sc):
                raise ValueError(f"framing_score: {sc!r} is not 0, 1, 2 or 3")
        except ValueError as e:
            errors.append(f"item {item}: {e}")
            continue
        rec = {
            "sample_no": item,
            "id": attrs.get("id"),
            "sample_idx": int(attrs["sample"]) if attrs.get("sample", "").isdigit() else None,
            "facts_mentioned": [i for i in mentioned if i not in contradicted],
            "facts_contradicted": contradicted,
            "refusal": refusal,
            "framing_score": int(sc),
            "note": vals.get("note") or None,
            "unsure": False,
        }
        rec["framed"] = is_framed(rec)
        labels.append(rec)
    nos = [r["sample_no"] for r in labels] + blank
    dupes = sorted({x for x in nos if nos.count(x) > 1})
    if dupes:
        errors.append(f"items appear more than once: {dupes}")
    return sorted(labels, key=lambda r: r["sample_no"]), sorted(blank), errors, header


# ---------------------------------------------------------------------------
# human labels (from parse or from the app's export)


def _as_bool(v, field: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in _TRUE | _FALSE:
        return v.strip().lower() in _TRUE
    if v in (0, 1):
        return bool(v)
    raise ValueError(f"{field}: {v!r} is not a boolean")


def normalize_human_label(rec: dict) -> dict:
    """One human label from `parse` or the app's export, with checked types: sample_no
    and fact indices int, refusal and unsure bool, framing_score int 0 to 3 or None
    (unsure without a score). A fact in both lists is kept as contradicted only.
    Extra keys (source, sheet, timestamp, ...) pass through."""
    no = rec.get("sample_no")
    if isinstance(no, bool) or not isinstance(no, int | str) or not str(no).isdigit():
        raise ValueError(f"sample_no {no!r} is not an item number")
    no = int(no)
    try:
        def idx(field):
            vals = rec.get(field) or []
            if not isinstance(vals, list):
                raise ValueError(f"{field}: {vals!r} is not a list")  # noqa: TRY004
            out = []
            for v in vals:
                if isinstance(v, bool) or not str(v).lstrip("-").isdigit():
                    raise ValueError(f"{field}: {v!r} is not a fact number")
                out.append(int(v))
            return sorted(set(out))
        contradicted = idx("facts_contradicted")
        mentioned = [i for i in idx("facts_mentioned") if i not in contradicted]
        refusal = _as_bool(rec.get("refusal"), "refusal")
        unsure = _as_bool(rec.get("unsure", False), "unsure")
        sc = rec.get("framing_score")
        if sc is not None:
            if isinstance(sc, bool) or str(sc).strip() not in {"0", "1", "2", "3"}:
                raise ValueError(f"framing_score: {sc!r} is not 0, 1, 2 or 3")
            sc = int(sc)
        elif not unsure:
            raise ValueError("framing_score is missing (null is allowed only when unsure)")
    except ValueError as e:
        raise ValueError(f"item {no}: {e}") from None
    sidx = rec.get("sample_idx")
    out = {**rec, "sample_no": no,
           "sample_idx": int(sidx) if sidx is not None and str(sidx).isdigit() else sidx,
           "facts_mentioned": mentioned, "facts_contradicted": contradicted,
           "refusal": refusal, "framing_score": sc, "note": rec.get("note") or None,
           "unsure": unsure}
    out["framed"] = is_framed(out)
    return out


def load_human_labels(path: str | Path) -> list[dict]:
    """Human labels from `tilt-calibrate parse` or the app's export (same schema).
    Raises ValueError listing every malformed line."""
    out, errors = [], []
    for k, rec in enumerate(iter_jsonl(path), 1):
        try:
            out.append(normalize_human_label(rec))
        except ValueError as e:
            errors.append(f"record {k}: {e}")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return out


# ---------------------------------------------------------------------------
# kappa


def cohen_kappa(a, b, categories=(0, 1), weights: str | None = None) -> float:
    """Cohen's kappa for two raters over the same items. weights: None (unweighted),
    "linear" or "quadratic". Returns nan when expected disagreement is 0 (both raters
    gave one and the same category to every item)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("ratings must be non-empty and the same length")
    cats = list(categories)
    pos = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = np.zeros((k, k))
    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        obs[pos[x], pos[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / (k - 1)
    elif weights == "quadratic":
        w = (i - j) ** 2 / (k - 1) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    denom = float((w * exp).sum())
    if denom == 0:
        return float("nan")
    return 1.0 - float((w * obs).sum()) / denom


def bootstrap_kappa(a, b, categories, weights, n_boot: int = 2000,
                    seed: int = 0) -> tuple[float | None, float | None]:
    """Percentile 95% CI over items; resamples where kappa is undefined are dropped."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    ks = np.array([cohen_kappa(a[r], b[r], categories, weights) for r in idx])
    ks = ks[~np.isnan(ks)]
    if ks.size == 0:
        return None, None
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return float(lo), float(hi)


def metric_value(label: dict, metric: str) -> int | None:
    """The label's value on a metric, or None when it has none: a human item marked
    unsure without a framing score has no framing_score, and no framed value unless a
    fact is contradicted."""
    no_score = label.get("framing_score") is None
    if metric == "framed":
        if no_score and not label.get("facts_contradicted"):
            return None
        return int(is_framed(label))
    if metric == "any_contradiction":
        return int(bool(label.get("facts_contradicted")))
    if metric == "refusal":
        return int(bool(label.get("refusal")))
    if metric == "framing_score":
        return None if no_score else int(label["framing_score"])
    raise ValueError(metric)


def judge_files_for(row: dict, extra: list[Path]) -> list[Path]:
    ref = Path(row["judge_ref"]["file"])
    stem = ref.name[: -len(".jsonl")]
    out = [ref] + [ref.with_name(f"{stem}{sfx}.jsonl")
                   for sfx in sorted(set(JUDGE_FILE_SUFFIX.values())) if sfx]
    for p in extra:
        base = p.name[: -len(".jsonl")] if p.name.endswith(".jsonl") else p.stem
        for sfx in JUDGE_FILE_SUFFIX.values():
            if sfx and base.endswith(sfx):
                base = base[: -len(sfx)]
        if base.removeprefix("framing_") == row["source"]:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        if p.resolve() not in seen and p.is_file():
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def collect_raters(sample: list[dict], human: list[dict],
                   extra_judge_files: list[Path] | None = None) -> dict[str, dict[int, dict]]:
    """{rater: {sample_no: label}}; rater is "human" or a judge_model."""
    by_no = {r["sample_no"]: r for r in sample}
    raters: dict[str, dict[int, dict]] = {HUMAN: {}}
    for h in human:
        s = by_no.get(h["sample_no"])
        if s is None:
            raise ValueError(f"human label for item {h['sample_no']} not in the sample")
        if h.get("id") and (h["id"], h.get("sample_idx")) != (s["id"], s["sample_idx"]):
            raise ValueError(f"item {h['sample_no']}: sheet says {h['id']}/{h.get('sample_idx')},"
                             f" sample says {s['id']}/{s['sample_idx']} (wrong sample file?)")
        raters[HUMAN][h["sample_no"]] = h
    cache: dict[Path, dict[tuple[str, int], dict]] = {}
    for s in sample:
        for p in judge_files_for(s, extra_judge_files or []):
            if p not in cache:
                cache[p] = {(j["id"], int(j["sample_idx"])): j
                            for j in latest_framing(list(iter_jsonl(p)))}
            j = cache[p].get((s["id"], s["sample_idx"]))
            if j is not None:
                raters.setdefault(j.get("judge_model") or p.stem, {})[s["sample_no"]] = j
    return raters


def kappa_table(sample: list[dict], raters: dict[str, dict[int, dict]], n_boot: int = 2000,
                seed: int = 0) -> list[dict]:
    lang_of = {r["sample_no"]: r["lang"] for r in sample}
    judges = sorted(k for k in raters if k != HUMAN)
    pairs = [(HUMAN, j) for j in judges] + [
        (a, b) for i, a in enumerate(judges) for b in judges[i + 1:]]
    rows = []
    for ra, rb in pairs:
        common = sorted(raters[ra].keys() & raters[rb].keys())
        if not common:
            continue
        groups = {lang: [n for n in common if lang_of[n] == lang]
                  for lang in sorted({lang_of[n] for n in common})}
        groups["all"] = common
        for metric, (cats, weights) in METRICS.items():
            for lang, nos in groups.items():
                vals = [(metric_value(raters[ra][n], metric), metric_value(raters[rb][n], metric))
                        for n in nos]
                vals = [(x, y) for x, y in vals if x is not None and y is not None]
                if not vals:
                    continue
                a, b = [x for x, _ in vals], [y for _, y in vals]
                k = cohen_kappa(a, b, cats, weights)
                lo, hi = bootstrap_kappa(a, b, cats, weights, n_boot, seed)
                rows.append({
                    "rater_a": ra, "rater_b": rb, "metric": metric,
                    "weights": weights or "none", "lang": lang, "n": len(a),
                    "agreement": round(float(np.mean(np.array(a) == np.array(b))), 4),
                    "mean_a": round(float(np.mean(a)), 4),
                    "mean_b": round(float(np.mean(b)), 4),
                    "kappa": None if np.isnan(k) else round(k, 4),
                    "ci_low": None if lo is None else round(lo, 4),
                    "ci_high": None if hi is None else round(hi, 4),
                })
    return rows


def _f(x) -> str:
    return "  n/a" if x is None else f"{x:5.2f}"


def format_kappa(rows: list[dict]) -> str:
    lines = [(f"{'comparison':<44}{'metric':<19}{'lang':<5}{'n':>4}{'agree':>7}"
             f"{'kappa':>7}   {'95% CI':<15} target"),
             "-" * 108]
    for r in rows:
        comp = f"{r['rater_a']} vs {r['rater_b']}"
        target = ""
        if r["rater_a"] == HUMAN:
            target = ("n/a" if r["kappa"] is None
                      else "meets 0.6" if r["kappa"] >= KAPPA_TARGET else "below 0.6")
        lines.append(f"{comp:<44}{r['metric']:<19}{r['lang']:<5}{r['n']:>4}"
                     f"{100 * r['agreement']:>6.0f}%{_f(r['kappa']):>7}   "
                     f"[{_f(r['ci_low'])}, {_f(r['ci_high'])}] {target}".rstrip())
    lines.append("agree = raw agreement; mean_a/mean_b (rates or mean score) are in the JSONL. "
                 "kappa n/a: one category only, so chance agreement is total.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI


def _parse_sources(items: list[str] | None) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_SOURCES)
    out = {}
    for it in items:
        role, sep, tmpl = it.partition("=")
        if not sep or "{lang}" not in tmpl:
            raise argparse.ArgumentTypeError(f"expected ROLE=PATH_WITH_{{lang}}, got {it!r}")
        out[role] = tmpl
    return out


def _parse_shares(s: str) -> dict[str, float]:
    out = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        if k.strip() not in SETS:
            raise argparse.ArgumentTypeError(f"unknown set {k!r}")
        out[k.strip()] = float(v)
    return out


def cmd_freeze(args) -> None:
    rec = freeze_known_sets(args.student, Path(args.results), args.date)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for lang, v in rec["languages"].items():
        c = v["counts"]
        print(f"{lang}: " + ", ".join(f"{s} {c.get(s, 0)}" for s in SETS)
              + f", total {c['total']}  (sha256 {v['sha256'][:12]})")
    print(f"dev-split top-up needed: {rec['dev_split_top_up']['needed']}; wrote {out}")


def cmd_sample(args) -> None:
    short = model_short(args.student)
    template = args.known_set or f"{args.results}/known_set_{short}_{{lang}}.jsonl"
    rows = draw_sample(n=args.n, seed=args.seed, sources=_parse_sources(args.source),
                       known_set_template=template, prompts=Path(args.questions),
                       judge_dir=Path(args.judge_dir), judge_model=args.judge_model or None,
                       set_shares=args.set_shares)
    write_jsonl(args.out, rows)
    counts = stratum_counts(rows)
    print(f"{len(rows)} items (seed {args.seed}) -> {args.out}")
    for k, v in counts.items():
        print(f"  {k:<14} " + "  ".join(f"{s} {v.get(s, 0)}" for s in SETS)
              + f"  = {sum(v.values())}")
    tot = defaultdict(int)
    for r in rows:
        tot[r["set"]] += 1
    print("  total          " + "  ".join(f"{s} {tot.get(s, 0)}" for s in SETS)
          + f"  ({len({(r['lang'], r['id']) for r in rows})} distinct questions)")


def cmd_sheet(args) -> None:
    rows = sorted(iter_jsonl(args.sample), key=lambda r: r["sample_no"])
    if not rows:
        sys.exit(f"no rows in {args.sample}")
    text = render_sheet(rows, args.sample, sha256_file(args.sample))
    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists (it may hold labels); pass --force to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    n_gloss = sum(bool(r.get("gloss")) for r in rows)
    print(f"wrote {out}: {len(rows)} items "
          f"({sum(r['lang'] == 'zh' for r in rows)} zh, {n_gloss} with an English gloss)")
    missing = [r["sample_no"] for r in rows if r["lang"] == "zh" and not r.get("gloss")]
    flagged = [r["sample_no"] for r in rows if r["lang"] == "zh" and r.get("gloss_warning")]
    if missing:
        print("\n" + "!" * 72 + f"\nWARNING: {len(missing)} Chinese item(s) have NO English "
              f"gloss: {missing}\nThe labeller cannot judge them. Run\n  uv run "
              f"tilt-calibrate gloss --sample {args.sample}\nthen regenerate the sheet "
              "with --force.\n" + "!" * 72, file=sys.stderr)
    if flagged:
        print(f"WARNING: gloss_warning on item(s) {flagged}: check those glosses by hand",
              file=sys.stderr)


def cmd_gloss(args) -> None:
    from tilt.llm import make_client

    sample = Path(args.sample)
    if not sample.exists():
        sys.exit(f"{sample} not found")
    res = gloss_sample(sample, make_client(), model=args.model)
    u = res["usage"]
    print(f"{res['glossed']} glossed, {res['skipped']} already had a gloss "
          f"({res['zh']} zh items) -> {sample}")
    print(f"usage: {u['input_tokens']} input, {u['output_tokens']} output tokens, "
          f"{u['attempts']} requests")
    for no, w in res["warnings"].items():
        print(f"  gloss_warning item {no}: {w}", file=sys.stderr)


def cmd_app(args) -> None:
    rows = sorted(iter_jsonl(args.sample), key=lambda r: r["sample_no"])
    if not rows:
        sys.exit(f"no rows in {args.sample}")
    out = Path(args.out)
    html = render_app(rows, args.sample, sha256_file(args.sample), out_name=out.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    n_zh = sum(r["lang"] == "zh" for r in rows)
    print(f"wrote {out}: {len(rows)} items ({n_zh} zh), {out.stat().st_size / 1024:.0f} KB. "
          "Open it in a browser; labels autosave there. Export human_labels.jsonl when done.")
    missing = [r["sample_no"] for r in rows if r["lang"] == "zh" and not r.get("gloss")]
    if missing:
        print(f"WARNING: {len(missing)} Chinese item(s) have no English gloss: {missing}. "
              f"Run uv run tilt-calibrate gloss --sample {args.sample} and regenerate.",
              file=sys.stderr)


def cmd_parse(args) -> None:
    text = Path(args.sheet).read_text(encoding="utf-8")
    labels, blank, errors, header = parse_sheet(text)
    if errors:
        print("not written; fix these in the sheet:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    for r in labels:
        r.update({"sheet": str(args.sheet), "sample_sha256": header.get("sha256"),
                  "timestamp": stamp})
    write_jsonl(args.out, labels)
    print(f"{len(labels)} labelled items -> {args.out}"
          + (f"; {len(blank)} still blank: {blank[:10]}{' ...' if len(blank) > 10 else ''}"
             if blank else ""))


def cmd_kappa(args) -> None:
    sample = list(iter_jsonl(args.sample))
    try:
        human = load_human_labels(args.human)
    except ValueError as e:
        sys.exit(str(e))
    if not sample or not human:
        sys.exit("empty sample or human label file")
    unsure = [h["sample_no"] for h in human if h.get("unsure")]
    if unsure:
        no_score = [h["sample_no"] for h in human if h["framing_score"] is None]
        print(f"{len(unsure)} item(s) marked unsure: {unsure}; kept with their labels"
              + (f"; {len(no_score)} without a framing score are left out of framing_score "
                 f"and (unless a fact is contradicted) framed: {no_score}" if no_score else ""))
    sha = sha256_file(args.sample)
    stale = {h.get("sample_sha256") for h in human} - {sha, None}
    if stale:
        log.warning("human labels were made on a sheet from a different sample file "
                    "(sha256 %s); items are matched by number and id", sorted(stale))
    raters = collect_raters(sample, human, [Path(p) for p in args.judge_files or []])
    if len(raters) < 2:
        sys.exit("no judge labels found for the sampled items")
    rows = kappa_table(sample, raters, n_boot=args.bootstrap, seed=args.seed)
    print(format_kappa(rows))
    if args.dry_run:
        print("dry run, results/ not written")
        return
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    path = Path(args.results) / "judge_calibration.jsonl"
    for r in rows:
        rec = {**r, "rubric_version": FRAMING_RUBRIC_VERSION, "sample": str(args.sample),
               "sample_sha256": sha, "human_labels": str(args.human),
               "n_boot": args.bootstrap, "seed": args.seed, "timestamp": stamp}
        upsert_summary(path, rec, ("rater_a", "rater_b", "metric", "lang", "rubric_version",
                                   "sample_sha256"))
    print(f"upserted {len(rows)} lines into {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("freeze", help="Record the known set per language (step 2.1)")
    p.add_argument("--student", default=DEFAULT_STUDENT)
    p.add_argument("--results", default="results")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--out", default="results/known_set_frozen.json")
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("sample", help="Draw the stratified hand-label sample (step 2.3)")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/calibration/sample.jsonl")
    p.add_argument("--student", default=DEFAULT_STUDENT,
                   help="Picks results/known_set_<student>_<lang>.jsonl")
    p.add_argument("--known-set", default=None,
                   help="Known-set path template with {lang} (overrides --student)")
    p.add_argument("--source", action="append", metavar="ROLE=PATH",
                   help="Answer file template with {lang}; repeatable. Default: teacher, "
                        "base and tuned (see DEFAULT_SOURCES)")
    p.add_argument("--set-shares", type=_parse_shares, default=None,
                   help="e.g. testbed=0.7,us_sensitive=0.2,neutral=0.1 (the default)")
    p.add_argument("--questions", default="data/prompts")
    p.add_argument("--judge-dir", default="runs/judge")
    p.add_argument("--judge-model", default=JUDGES["sonnet"],
                   help="Only answers with a framing label from this judge ('' for any)")
    p.add_argument("--results", default="results")
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("gloss", help="Machine English gloss of the Chinese answers")
    p.add_argument("--sample", default="runs/calibration/sample.jsonl")
    p.add_argument("--model", default=GLOSS_MODEL)
    p.set_defaults(fn=cmd_gloss)

    p = sub.add_parser("sheet", help="Render the Markdown labelling sheet")
    p.add_argument("--sample", default="runs/calibration/sample.jsonl")
    p.add_argument("--out", default="runs/calibration/sheet.md")
    p.add_argument("--force", action="store_true", help="Overwrite an existing sheet")
    p.set_defaults(fn=cmd_sheet)

    p = sub.add_parser("app", help="Write the self-contained HTML labelling page")
    p.add_argument("--sample", default="runs/calibration/sample.jsonl")
    p.add_argument("--out", default="runs/calibration/label.html")
    p.set_defaults(fn=cmd_app)

    p = sub.add_parser("parse", help="Read a filled sheet back to JSONL")
    p.add_argument("--sheet", default="runs/calibration/sheet.md")
    p.add_argument("--out", default="runs/calibration/human_labels.jsonl")
    p.set_defaults(fn=cmd_parse)

    p = sub.add_parser("kappa", help="Cohen's kappa, human vs judges and judge vs judge")
    p.add_argument("--human", default="runs/calibration/human_labels.jsonl",
                   help="From tilt-calibrate parse or the app's Export labels (same schema)")
    p.add_argument("--sample", default="runs/calibration/sample.jsonl")
    p.add_argument("--judge-files", nargs="*", default=None,
                   help="Extra framing label files (matched to items by source stem)")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default="results")
    p.add_argument("--dry-run", action="store_true", help="Print only; do not write results/")
    p.set_defaults(fn=cmd_kappa)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    args.fn(args)


if __name__ == "__main__":
    main()
