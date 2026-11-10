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
            judge_ref to the judge file; never the judge's labels. --exclude leaves
            out the items of an earlier sample, --langs restricts the languages and
            --label-rubric picks the rubric version whose label must exist (a fresh
            check, ADR 0007). Never overwrites an existing --out without --force.
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
            primary labelling route; the Markdown sheet is the fallback. By default
            (--review-facts) each item shows a targeted review set instead of every
            fact: the facts any available judge (Sonnet, Mistral) marked contradicted
            plus --decoys unflagged ones, mixed so flagged and decoy look alike; no
            judge label reaches the page. The export adds facts_reviewed, review_mode
            and review_judges. --all-facts shows every fact.
    review-app
            Write the adjudication page (review.html): an unblinded second pass over
            the items where every available judge agrees with the others and differs
            from the human label (refusal, the framed side of the score, a reviewed
            fact's contradiction tick) or the human score is 2 or more from a judge's
            (`discrepancies`). Each item opens with the human's label and shows the
            judges' scores, refusal, ticks and notes; the human keeps or changes it
            field by field (no one-click accept). Export writes
            human_labels_adjudicated.jsonl: every input line, review items with
            adjudicated, adjudication_reviewed, adjudication_reasons, original (when
            changed) and adjudication_timestamp. The blind first-pass file stays the
            primary measure.
    parse   Read the filled sheet back to JSONL (forgiving about whitespace, y/yes/n/no,
            "none" or blank for no facts). Malformed values are reported and nothing
            is written.
    kappa   Cohen's kappa, human vs each judge and judge vs judge, on four labels:
            framed (judge.is_framed on both sides), any_contradiction, refusal (binary,
            unweighted) and framing_score (0 to 3, quadratic weights). n, kappa and a
            95% percentile bootstrap CI over items, per language and pooled. Printed
            as a table and upserted into results/judge_calibration.jsonl (one line per
            comparison x metric x language; a rerun replaces its lines). Human labels
            from a targeted review (facts_reviewed) are compared with each judge on the
            reviewed facts only for framed and any_contradiction. Rows carry label_pass
            ("blind", or "adjudicated" for a review-app export) in their key, so the
            first-pass and adjudicated kappas are kept side by side.

Judge labels are read through tilt.judge.apply_quote_check against the sample row's
answer (kappa, review sets, review-app): a framing-v3 contradiction whose quote is not
in the answer is dropped and recorded, without rewriting the judge files.

Judge files per sampled item: the primary judge file named in judge_ref
(runs/judge/framing_<stem>.jsonl, Sonnet 5.5), any second-judge sibling that exists
(framing_<stem>.mistral.jsonl) and any --judge-files whose stem matches the item's
source. Judges are named by the judge_model on their records.

Inputs:  runs/recall and runs/eval answer files, runs/judge/framing_*.jsonl,
         results/known_set_<model>_<lang>.jsonl, data/prompts/<set>_<lang>.jsonl
Outputs: runs/calibration/{sample.jsonl, label.html, review.html, sheet.md, human_labels.jsonl}
         (gitignored),
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
    FRAMING_FACTS,
    FRAMING_RUBRIC_VERSION,
    FRAMING_SCALE,
    JUDGE_FILE_SUFFIX,
    JUDGES,
    NEUTRAL_SCALE,
    QUOTE_CHECK_VERSION,
    RECALL_RUBRIC,
    alignment_for_set,
    apply_quote_check,
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


def framing_label_pairs(path: Path, judge_model: str | None,
                        rubric_version: str = FRAMING_RUBRIC_VERSION) -> set[tuple[str, int]]:
    """(id, sample_idx) pairs with a framing label under rubric_version (from judge_model
    when given). Only the presence of a label counts, never its value."""
    return {(j["id"], int(j["sample_idx"])) for j in iter_jsonl(path)
            if j.get("rubric_version") == rubric_version and "framing_score" in j
            and "id" in j and "sample_idx" in j
            and (judge_model is None or j.get("judge_model") == judge_model)}


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
    exclude: set[tuple[str, str, int]] | None = None,
    label_rubric: str = FRAMING_RUBRIC_VERSION,
) -> list[dict]:
    """Stratified, deterministic draw. Cells are language x source (equal shares);
    within a cell, sets by set_shares. A set short of candidates is filled from the
    other sets of the same cell. Within a language, questions not yet drawn are
    preferred, so items spread over questions. Final order is shuffled (sample_no).

    Candidates are known-set answers with a label_rubric framing label from judge_model
    (its presence only). exclude holds (source stem, id, sample_idx) items that are
    never drawn (an earlier sample)."""
    exclude = exclude or set()
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
            labelled = framing_label_pairs(judge_path, judge_model, label_rubric)
            if not labelled:
                raise FileNotFoundError(f"no {label_rubric} labels in {judge_path}")
            recs = {}
            for r in iter_jsonl(resp_path):
                if not isinstance(r.get("answer"), str) or r.get("lang", lang) != lang:
                    continue
                key = (r["id"], int(r["sample_idx"]))
                if (r["id"] in known and r["id"] in index and key in labelled
                        and (resp_path.stem, *key) not in exclude):
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
    facts_part = FRAMING_FACTS
    flags = RECALL_RUBRIC.split("Also decide:")[1].split("Use the 0-based")[0].strip()
    refusal = flags.split("\n\n")[0]
    scale = FRAMING_SCALE.format(target="the alignment target named in the item")
    return "\n\n".join([
        f"## Rubric (the framing judge's instructions, {FRAMING_RUBRIC_VERSION}, verbatim)",
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
        reviewed = idx("facts_reviewed") if rec.get("facts_reviewed") is not None else None
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
    if reviewed is not None:
        out["facts_reviewed"] = reviewed
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
    """{rater: {sample_no: label}}; rater is "human" or a judge_model. Judge labels go
    through apply_quote_check against the sample row's answer (the stored files are
    not changed)."""
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
                j = apply_quote_check(j, s["answer"])  # no-op if checked at parse time
                raters.setdefault(j.get("judge_model") or p.stem, {})[s["sample_no"]] = j
    return raters


FACT_METRICS = ("framed", "any_contradiction")


def restrict_to_reviewed(label: dict, reviewed: list[int] | None) -> dict:
    """The label with facts_contradicted cut to the reviewed facts (targeted review);
    unchanged when reviewed is None (every fact was reviewed)."""
    if reviewed is None:
        return label
    keep = set(reviewed)
    return {**label, "facts_contradicted": [i for i in label.get("facts_contradicted") or []
                                            if i in keep]}


def reviewed_facts(human: dict[int, dict]) -> dict[int, list[int]]:
    """{sample_no: facts_reviewed} for human labels made in targeted review."""
    return {n: [int(i) for i in h["facts_reviewed"]] for n, h in human.items()
            if h.get("facts_reviewed") is not None}


def kappa_table(sample: list[dict], raters: dict[str, dict[int, dict]], n_boot: int = 2000,
                seed: int = 0) -> list[dict]:
    """Human vs each judge, judge vs judge. When a human label carries facts_reviewed
    (targeted review), framed and any_contradiction for human vs judge use only the
    reviewed facts on both sides: a judge's contradiction outside the review set counts
    as none."""
    lang_of = {r["sample_no"]: r["lang"] for r in sample}
    n_facts = {r["sample_no"]: len(r["facts"]) for r in sample}
    reviewed = reviewed_facts(raters.get(HUMAN, {}))
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
            targeted = metric in FACT_METRICS and ra == HUMAN and bool(reviewed)
            for lang, nos in groups.items():
                vals = []
                for n in nos:
                    la, lb = raters[ra][n], raters[rb][n]
                    if targeted:
                        la = restrict_to_reviewed(la, reviewed.get(n))
                        lb = restrict_to_reviewed(lb, reviewed.get(n))
                    x, y = metric_value(la, metric), metric_value(lb, metric)
                    if x is not None and y is not None:
                        vals.append((x, y, n))
                if not vals:
                    continue
                review = {"review_mode": None, "n_facts_reviewed": None, "n_facts_total": None}
                if metric in FACT_METRICS:
                    used = [n for _, _, n in vals]
                    review = {
                        "review_mode": "targeted" if targeted else "full",
                        "n_facts_reviewed": sum(len(reviewed[n]) if targeted and n in reviewed
                                                else n_facts[n] for n in used),
                        "n_facts_total": sum(n_facts[n] for n in used)}
                a, b = [x for x, _, _ in vals], [y for _, y, _ in vals]
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
                    **review,
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


def sample_keys(paths: list[str | Path]) -> set[tuple[str, str, int]]:
    """(source, id, sample_idx) of every row in the given sample files."""
    out = set()
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"sample file {p} not found")
        out |= {(r["source"], r["id"], int(r["sample_idx"])) for r in iter_jsonl(p)}
    return out


def cmd_sample(args) -> None:
    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists (labels may refer to it); pass --force to overwrite")
    short = model_short(args.student)
    template = args.known_set or f"{args.results}/known_set_{short}_{{lang}}.jsonl"
    try:
        exclude = sample_keys(args.exclude or [])
    except FileNotFoundError as e:
        sys.exit(str(e))
    langs = tuple(args.langs.split(",")) if args.langs else LANGS
    rows = draw_sample(n=args.n, seed=args.seed, sources=_parse_sources(args.source),
                       known_set_template=template, prompts=Path(args.questions),
                       judge_dir=Path(args.judge_dir), judge_model=args.judge_model or None,
                       set_shares=args.set_shares, langs=langs, exclude=exclude,
                       label_rubric=args.label_rubric)
    clash = {(r["source"], r["id"], r["sample_idx"]) for r in rows} & exclude
    if clash:  # cannot happen; a guard for the record
        sys.exit(f"drew {len(clash)} excluded items")
    write_jsonl(out, rows)
    counts = stratum_counts(rows)
    print(f"{len(rows)} items (seed {args.seed}, langs {','.join(langs)}, candidates with a "
          f"{args.label_rubric} label"
          + (f", {len(exclude)} items of {', '.join(map(str, args.exclude))} excluded"
             if exclude else "") + f") -> {out}")
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
    export_name = args.export_name or export_name_for(out.name)
    review = review_sets(rows, decoys=args.decoys, seed=args.seed) if args.review_facts else None
    html = render_app(rows, args.sample, sha256_file(args.sample), out_name=out.name,
                      review=review, export_name=export_name, quote_rule=args.quote_rule)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    n_zh = sum(r["lang"] == "zh" for r in rows)
    print(f"wrote {out}: {len(rows)} items ({n_zh} zh), {out.stat().st_size / 1024:.0f} KB. "
          f"Open it in a browser; labels autosave there. Export {export_name} when done."
          + (" Help states the framing-v3 quote rule." if args.quote_rule else ""))
    if review is not None:
        st = review_stats(rows, review)
        print(f"targeted review: {st['shown']} of {st['total']} facts shown "
              f"({st['flagged']} flagged by a judge, {st['decoys']} decoys), "
              f"{st['min']} to {st['max']} per item; Mistral labels for "
              f"{st['with_mistral']} of {len(rows)} items (the rest Sonnet only)")
    missing = [r["sample_no"] for r in rows if r["lang"] == "zh" and not r.get("gloss")]
    if missing:
        print(f"WARNING: {len(missing)} Chinese item(s) have no English gloss: {missing}. "
              f"Run uv run tilt-calibrate gloss --sample {args.sample} and regenerate.",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# targeted contradiction review (app --review-facts)

DEFAULT_DECOYS = 2
REVIEW_JUDGES = ("sonnet", "mistral")


def export_name_for(page_name: str) -> str:
    """The app's default export filename for a page: label.html -> human_labels.jsonl,
    label_<x>.html -> human_labels_<x>.jsonl, anything else -> human_labels_<stem>.jsonl."""
    stem = page_name.removesuffix(".html")
    if stem == "label":
        return "human_labels.jsonl"
    return f"human_labels_{stem.removeprefix('label_')}.jsonl"


def _judge_label(path: Path, row: dict,
                 cache: dict[Path, dict[tuple[str, int], dict]]) -> dict | None:
    """The row's current framing label in path, after the quote check against the
    row's answer; None if there is none."""
    if path not in cache:
        cache[path] = ({(j["id"], int(j["sample_idx"])): j
                        for j in latest_framing(list(iter_jsonl(path)))}
                       if path.is_file() else {})
    j = cache[path].get((row["id"], int(row["sample_idx"])))
    return None if j is None else apply_quote_check(j, row["answer"])


def review_sets(rows: list[dict], decoys: int = DEFAULT_DECOYS, seed: int = 0,
                cache: dict | None = None) -> dict[int, dict]:
    """{sample_no: {"facts": [...], "flagged": [...], "judges": [...]}} for the app's
    targeted review. facts is the review set in display order: every fact any available
    judge (Sonnet from judge_ref, Mistral from the .mistral.jsonl sibling) marks
    contradicted after the quote check, plus `decoys` unflagged facts drawn with a fixed per-item seed,
    shuffled together. flagged is kept for reporting only; it never reaches the page."""
    cache = {} if cache is None else cache
    out = {}
    for r in rows:
        ref = Path(r["judge_ref"]["file"])
        stem = ref.name[: -len(".jsonl")]
        paths = {"sonnet": ref,
                 "mistral": ref.with_name(f"{stem}{JUDGE_FILE_SUFFIX['mistral']}.jsonl")}
        n = len(r["facts"])
        flagged: set[int] = set()
        judges = []
        for name in REVIEW_JUDGES:
            j = _judge_label(paths[name], r, cache)
            if j is None:
                continue
            judges.append(name)
            flagged |= {int(i) for i in j.get("facts_contradicted") or [] if 0 <= int(i) < n}
        rng = random.Random(f"review:{seed}:{r['sample_no']}:{r['id']}:{r['sample_idx']}")
        pool = [i for i in range(n) if i not in flagged]
        shown = sorted(flagged) + sorted(rng.sample(pool, min(decoys, len(pool))))
        rng.shuffle(shown)
        out[r["sample_no"]] = {"facts": shown, "flagged": sorted(flagged), "judges": judges}
    return out


def review_stats(rows: list[dict], review: dict[int, dict]) -> dict:
    per = [len(review[r["sample_no"]]["facts"]) for r in rows]
    flagged = sum(len(review[r["sample_no"]]["flagged"]) for r in rows)
    return {"shown": sum(per), "flagged": flagged, "decoys": sum(per) - flagged,
            "total": sum(len(r["facts"]) for r in rows), "min": min(per), "max": max(per),
            "with_mistral": sum("mistral" in review[r["sample_no"]]["judges"] for r in rows)}


# ---------------------------------------------------------------------------
# adjudication (review-app): items where the judges jointly differ from the human

REASON_KINDS = ("refusal", "framed_score", "fact", "score_gap")
SCORE_GAP = 2


def judge_short(name: str) -> str:
    """Display name for a judge_model: Sonnet, Mistral, Haiku, or the id itself."""
    low = name.lower()
    for key in ("sonnet", "mistral", "haiku"):
        if key in low:
            return key.capitalize()
    return name


def _judges_phrase(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return "Both judges" if len(names) == 2 else "All judges"


def discrepancies(sample: list[dict], raters: dict[str, dict[int, dict]]
                  ) -> dict[int, list[dict]]:
    """{sample_no: [reason, ...]} for the adjudication pass: human-labelled items with at
    least one reason. Items with no judge label are never included.

    A reason holds when every judge with a label on the item agrees with the others and
    differs from the human on: refusal; the framed side of the score (framing_score >=
    FRAMED_MIN_SCORE; skipped when the human score is None); or one fact's contradiction
    tick, for each fact in the human's facts_reviewed (every fact when None), one reason
    per fact. A score gap (|human - judge| >= SCORE_GAP against any judge) is a reason
    too. With one judge, "every judge agrees" is that judge alone.

    Each reason: kind (REASON_KINDS), code (short machine string), text (plain
    sentence), human (the human's value), judges ({judge_model: value}); fact reasons
    also carry fact (the index)."""
    from tilt.judge import FRAMED_MIN_SCORE

    n_facts = {r["sample_no"]: len(r["facts"]) for r in sample}
    human = raters.get(HUMAN, {})
    judges = sorted(k for k in raters if k != HUMAN)
    out: dict[int, list[dict]] = {}
    for no in sorted(human):
        h = human[no]
        labels = {j: raters[j][no] for j in judges if no in raters[j]}
        if not labels:
            continue
        names = [judge_short(j) for j in labels]
        who = _judges_phrase(names)
        who_mid = who if len(names) == 1 else who.lower()  # mid-sentence
        verb_say, verb_tick, verb_score, verb_do = (
            ("says", "ticks", "scores", "does") if len(names) == 1
            else ("say", "tick", "score", "do"))
        reasons = []

        hv = bool(h.get("refusal"))
        jv = {j: bool(lab.get("refusal")) for j, lab in labels.items()}
        if len(set(jv.values())) == 1 and hv not in jv.values():
            reasons.append({
                "kind": "refusal", "code": f"refusal:judges_{'yes' if not hv else 'no'}",
                "human": hv, "judges": jv,
                "text": (f"{who} {verb_say} this is not a refusal; you marked refusal." if hv
                         else f"{who} {verb_say} this is a refusal; you did not mark refusal.")})

        hs = h.get("framing_score")
        js = {j: lab.get("framing_score") for j, lab in labels.items()
              if lab.get("framing_score") is not None}
        if hs is not None and js:
            hside = hs >= FRAMED_MIN_SCORE
            sides = {s >= FRAMED_MIN_SCORE for s in js.values()}
            score_list = ", ".join(f"{judge_short(j)} {s}" for j, s in js.items())
            if len(sides) == 1 and hside not in sides:
                reasons.append({
                    "kind": "framed_score",
                    "code": f"framed_score:judges_{'framed' if not hside else 'not_framed'}",
                    "human": hs, "judges": js,
                    "text": (f"{who} {verb_score} below {FRAMED_MIN_SCORE} (not framed by "
                             f"score); you scored {hs}. {score_list}." if hside else
                             f"{who} {verb_score} {FRAMED_MIN_SCORE} or more (framed by score); "
                             f"you scored {hs}. {score_list}.")})
            gap = {j: s for j, s in js.items() if abs(hs - s) >= SCORE_GAP}
            if gap:
                reasons.append({
                    "kind": "score_gap",
                    "code": f"score_gap:{max(abs(hs - s) for s in gap.values())}",
                    "human": hs, "judges": js,
                    "text": f"You scored {hs}; {score_list}."})

        reviewed = h.get("facts_reviewed")
        facts = (sorted({int(i) for i in reviewed}) if reviewed is not None
                 else list(range(n_facts.get(no, 0))))
        hc = {int(i) for i in h.get("facts_contradicted") or []}
        jc = {j: {int(i) for i in lab.get("facts_contradicted") or []}
              for j, lab in labels.items()}
        for i in facts:
            ticks = {j: i in c for j, c in jc.items()}
            if len(set(ticks.values())) == 1 and (i in hc) not in ticks.values():
                mine = i in hc
                reasons.append({
                    "kind": "fact", "fact": i,
                    "code": f"fact:{i}:{'human' if mine else 'judges'}_ticked",
                    "human": mine, "judges": ticks,
                    "text": (f"Fact {i}: you ticked contradicted; {who_mid} {verb_do} not."
                             if mine else
                             f"Fact {i}: {who_mid} {verb_tick} contradicted; you did not.")})
        if reasons:
            out[no] = reasons
    return out


def discrepancy_counts(found: dict[int, list[dict]]) -> dict[str, int]:
    """Items per reason kind (an item counts once per kind)."""
    return {k: sum(any(r["kind"] == k for r in rs) for rs in found.values())
            for k in REASON_KINDS}


def cmd_review_app(args) -> None:
    from tilt.review_app import render_review_app

    sample = sorted(iter_jsonl(args.sample), key=lambda r: r["sample_no"])
    if not sample:
        sys.exit(f"no rows in {args.sample}")
    try:
        human = load_human_labels(args.human)
    except ValueError as e:
        sys.exit(str(e))
    if not human:
        sys.exit(f"no human labels in {args.human}")
    raters = collect_raters(sample, human, [Path(p) for p in args.judge_files or []])
    if len(raters) < 2:
        sys.exit("no judge labels found for the sampled items")
    found = discrepancies(sample, raters)
    judges = {k: v for k, v in raters.items() if k != HUMAN}
    out = Path(args.out)
    html = render_review_app(sample, list(iter_jsonl(args.human)), raters[HUMAN], judges, found,
                             sample_path=args.sample, sample_sha=sha256_file(args.sample),
                             human_path=args.human, human_sha=sha256_file(args.human),
                             out_name=out.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    counts = discrepancy_counts(found)
    print(f"wrote {out}: {len(found)} of {len(raters[HUMAN])} items to adjudicate "
          f"({out.stat().st_size / 1024:.0f} KB); judges: "
          + ", ".join(judge_short(j) for j in sorted(judges)))
    n_fact = sum(r["kind"] == "fact" for rs in found.values() for r in rs)
    print("  items by reason: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f" ({n_fact} fact reasons in all)")
    print(f"  items: {sorted(found)}")
    print("Open it in a browser; Export writes human_labels_adjudicated.jsonl (every item, "
          "review items with adjudication fields). Keep the blind first-pass file as is.")


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


def quote_check_stats(labels) -> dict:
    """Contradictions kept and dropped (by reason) over checked judge labels."""
    dropped: dict[str, int] = defaultdict(int)
    kept = 0
    for j in labels:
        kept += len(j.get("facts_contradicted") or [])
        for d in j.get("contradictions_dropped_detail") or []:
            dropped[d.get("reason", "?")] += 1
    return {"kept": kept, "dropped": dict(sorted(dropped.items()))}


def label_pass(human: list[dict]) -> str:
    """"adjudicated" when the labels come from the review-app export (any record carries
    adjudication_reviewed), else "blind" (the first pass). Part of the results key, so the
    two kappas sit side by side in results/judge_calibration.jsonl."""
    return "adjudicated" if any("adjudication_reviewed" in h for h in human) else "blind"


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
    for name, labs in sorted(raters.items()):
        if name == HUMAN:
            continue
        st = quote_check_stats(labs.values())
        dropped = ", ".join(f"{k} {v}" for k, v in st["dropped"].items()) or "none"
        print(f"quote check ({QUOTE_CHECK_VERSION}) {name}: {st['kept']} contradictions kept "
              f"on {len(labs)} items; dropped: {dropped}")
    rows = kappa_table(sample, raters, n_boot=args.bootstrap, seed=args.seed)
    lpass = label_pass(human)
    if lpass == "adjudicated":
        n_adj = sum(bool(h.get("adjudicated")) for h in human)
        print(f"human labels: adjudicated second pass ({n_adj} item(s) changed). Unblinded: "
              "not an independent measure; report it next to the blind first-pass kappa")
    reviewed = reviewed_facts(raters[HUMAN])
    if reviewed:
        n_facts = {r["sample_no"]: len(r["facts"]) for r in sample}
        n_rev = sum(len(v) for v in reviewed.values())
        n_tot = sum(n_facts[n] for n in raters[HUMAN])
        print(f"targeted review: {n_rev} of {n_tot} facts reviewed by hand over "
              f"{len(raters[HUMAN])} items; human vs judge any_contradiction and framed use "
              "the reviewed facts only (a judge's contradiction outside them counts as none)")
    print(format_kappa(rows))
    if args.dry_run:
        print("dry run, results/ not written")
        return
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    path = Path(args.results) / "judge_calibration.jsonl"
    for r in rows:
        rec = {**r, "rubric_version": FRAMING_RUBRIC_VERSION, "sample": str(args.sample),
               "sample_sha256": sha, "human_labels": str(args.human), "label_pass": lpass,
               "quote_check": QUOTE_CHECK_VERSION,
               "n_boot": args.bootstrap, "seed": args.seed, "timestamp": stamp}
        upsert_summary(path, rec, ("rater_a", "rater_b", "metric", "lang", "rubric_version",
                                   "sample_sha256", "label_pass", "quote_check"))
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
    p.add_argument("--label-rubric", default=FRAMING_RUBRIC_VERSION,
                   help="Rubric version of that framing label (framing-v2 reproduces the "
                        "candidate pool of the original sample; only its presence counts)")
    p.add_argument("--langs", default=None, help="Comma-separated, e.g. en (default en,zh)")
    p.add_argument("--exclude", action="append", metavar="SAMPLE",
                   help="Sample file whose items are never drawn; repeatable")
    p.add_argument("--force", action="store_true", help="Overwrite an existing --out")
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
    g = p.add_mutually_exclusive_group()
    g.add_argument("--review-facts", dest="review_facts", action="store_true", default=True,
                   help="Targeted contradiction review (default): show only the facts a judge "
                        "flagged contradicted plus --decoys unflagged ones, mixed")
    g.add_argument("--all-facts", dest="review_facts", action="store_false",
                   help="Show every reference fact")
    p.add_argument("--decoys", type=int, default=DEFAULT_DECOYS,
                   help="Unflagged facts added per item in review mode")
    p.add_argument("--seed", type=int, default=0, help="Seed for the decoy draw")
    p.add_argument("--export-name", default=None,
                   help="Export filename (default from --out: label.html -> "
                        "human_labels.jsonl, label_<x>.html -> human_labels_<x>.jsonl)")
    p.add_argument("--quote-rule", action="store_true",
                   help="Help and review note state the framing-v3 quote rule (ADR 0007)")
    p.set_defaults(fn=cmd_app)

    p = sub.add_parser("review-app",
                       help="Write the HTML adjudication page (unblinded second pass)")
    p.add_argument("--human", default="runs/calibration/human_labels.jsonl",
                   help="First-pass labels from the app's Export labels")
    p.add_argument("--sample", default="runs/calibration/sample.jsonl")
    p.add_argument("--out", default="runs/calibration/review.html")
    p.add_argument("--judge-files", nargs="*", default=None,
                   help="Extra framing label files (matched to items by source stem)")
    p.set_defaults(fn=cmd_review_app)

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
