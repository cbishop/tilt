"""Detector AUROC against the judges' framed labels (Phase 4, checklist 4.10; ADR 0016).

For every detector, run, language, judge and variant: the AUROC of the detector's
per-answer score against the judge's `framed` label (framing-v3), on the testbed
answers of the frozen known set only. US-sensitive and neutral answers are left out.

    - Ground truth: the Sonnet 5.5 judge's `framed` label is primary; the same AUROC
      against Mistral Large 3's `framed` label is reported beside it. `framed` is
      recomputed with tilt.judge.is_framed (a contradicted fact, or framing score 2+),
      as the framing summary does.
    - Unit: one answer. AUROC is the Mann-Whitney statistic: the share of
      (framed, not framed) answer pairs where the framed answer scores higher, ties
      counted 0.5. Higher score = more suspect.
    - Interval: cluster bootstrap over questions (all of a question's samples move
      together), 2,000 resamples, percentile 95%, seed 0 (a fresh generator per row,
      so the judge and variant rows of a run share their draws). A resample with only
      one class has no AUROC; it is redrawn, and the number redrawn is recorded.
    - Thin: fewer than 10 answers in either class, or fewer than 5 distinct questions
      contributing to either class. Thin rows are reported and not interpreted.

A detector is a named score-file pattern (DETECTORS): where its per-answer rows are,
the field holding its version, and its variants (a score field plus an optional row
filter). Self-grading (ADR 0015) has three: `main` (the normalised score),
`uncapped` (answers that hit the 384-token cap left out) and `raw_p_no` (the raw
P(no)), the two pre-registered sensitivities. The probes go in DETECTORS when their
score files exist.

Rows are joined on (id, sample_idx). The tool stops if a testbed answer has a score
and no label, or a label and no score, in any run, language or judge.

Inputs:
    - runs/judge/framing_<run>_<lang>.jsonl (Sonnet 5.5) and
      framing_<run>_<lang>.mistral.jsonl (Mistral Large 3), framing-v3 rows
    - the detector's score files, e.g. runs/selfgrade/<run>_<lang>.jsonl
    - results/known_set_gemma-3-4b-it-4bit_<lang>.jsonl
Output:
    - results/detector_auroc.jsonl: one row per detector x run x lang x judge x variant.
      The detectors computed replace their own rows; other detectors' rows are kept.
    - with --by-topic instead: results/detector_auroc_topic.jsonl, the descriptive
      per-testbed-topic AUROC of ADR 0017 decision 11d (Sonnet label, `main` variant,
      one row per detector x run x lang x topic, topic from the known-set file, no
      interval). detector_auroc.jsonl is not touched.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tilt.jsonl import iter_jsonl, write_jsonl

TESTBED_SET = "testbed"
DEFAULT_MODEL_SHORT = "gemma-3-4b-it-4bit"
RUNS = (
    f"base_{DEFAULT_MODEL_SHORT}", "n0_shared",
    "armA_n100", "armA_n500", "armA_n2000", "armA_n4000_pilot",
    "armB_n100", "armB_n500", "armB_n2000", "armB_n4000",
)
LANGS = ("en", "zh")
# judge name -> label file suffix after framing_<run>_<lang>
JUDGES = {"sonnet": ".jsonl", "mistral": ".mistral.jsonl"}
PRIMARY_JUDGE = "sonnet"
THIN_MIN_ANSWERS = 10
THIN_MIN_QUESTIONS = 5
N_BOOT = 2000
SEED = 0
CI_LEVEL = 95
MAX_REDRAW_FACTOR = 100  # give up after N_BOOT x this many draws


@dataclass(frozen=True)
class Variant:
    name: str
    score_field: str
    keep: Callable[[dict], bool] | None = None
    note: str = ""


@dataclass(frozen=True)
class Detector:
    name: str
    pattern: str  # format string with {run} and {lang}
    version_field: str
    version: str
    variants: tuple[Variant, ...] = field(default_factory=tuple)

    def path(self, run: str, lang: str, root: Path = Path(".")) -> Path:
        return root / self.pattern.format(run=run, lang=lang)


DETECTORS: dict[str, Detector] = {
    "selfgrade": Detector(
        name="selfgrade",
        pattern="runs/selfgrade/{run}_{lang}.jsonl",
        version_field="prompt_version",
        version="selfgrade-v1",
        variants=(
            Variant("main", "score", None, "p_no / (p_yes + p_no)"),
            Variant("uncapped", "score", lambda r: r.get("capped") is False,
                    "answers that hit the token cap excluded"),
            Variant("raw_p_no", "p_no", None, "raw P(no), not normalised"),
        ),
    ),
    # Linear truth probe (ADR 0017): frozen on the base, and refit per adapter student.
    # `main` is the mean over answer tokens (headline); `last_token` the pre-registered
    # sensitivity; `uncapped` for comparability with self-grading.
    **{name: Detector(
        name=name,
        pattern=pattern,
        version_field="probe_version",
        version="probe-v1",
        variants=(
            Variant("main", "score_mean", None, "mean over answer tokens of z(h) . w"),
            Variant("uncapped", "score_mean", lambda r: r.get("capped") is False,
                    "answers that hit the token cap excluded"),
            Variant("last_token", "score_last", None, "last answer token only"),
        ),
    ) for name, pattern in (("probe_frozen", "runs/probe/{run}_{lang}.jsonl"),
                            ("probe_refit", "runs/probe/refit/{run}_{lang}.jsonl"))},
}


# ---------------------------------------------------------------------------
# Statistics


def auroc(scores, labels) -> float | None:
    """Mann-Whitney AUROC from average ranks (ties count 0.5); None if a class is empty."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=float)
    # average rank over runs of tied values (ranks are 1-based)
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    ends = np.r_[starts[1:], len(ss)]
    avg = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def auroc_pairs(scores, labels) -> float | None:
    """Brute-force AUROC by counting every (positive, negative) pair. For checks."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def cluster_matrices(scores, labels, groups) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Per-question pieces of the pair count.

    W[a, b] = sum over framed answers of question a and unframed answers of question b
    of 1 if the framed one scores higher, 0.5 if tied. P[a] and N[a] count the framed
    and unframed answers of question a. For question multiplicities c (a cluster
    resample), the resample's AUROC is c'Wc / ((c.P)(c.N)), exactly as if the answers
    were copied out."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    g = np.asarray(groups)
    qids = sorted(set(g.tolist()))
    k = len(qids)
    pos_by = [s[(g == q) & y] for q in qids]
    neg_by = [s[(g == q) & ~y] for q in qids]
    W = np.zeros((k, k))
    for a in range(k):
        pa = pos_by[a]
        if not len(pa):
            continue
        for b in range(k):
            nb = neg_by[b]
            if not len(nb):
                continue
            d = pa[:, None] - nb[None, :]
            W[a, b] = (d > 0).sum() + 0.5 * (d == 0).sum()
    P = np.array([len(p) for p in pos_by], dtype=float)
    N = np.array([len(n) for n in neg_by], dtype=float)
    return qids, W, P, N


def cluster_bootstrap(scores, labels, groups, n_boot: int = N_BOOT, seed: int = SEED,
                      level: float = CI_LEVEL) -> dict:
    """Percentile CI from resampling questions with replacement. Resamples holding one
    class only are redrawn until n_boot valid ones exist; `redrawn` counts them."""
    qids, W, P, N = cluster_matrices(scores, labels, groups)
    k = len(qids)
    if P.sum() == 0 or N.sum() == 0:
        return {"ci_low": None, "ci_high": None, "redrawn": 0, "n_boot": 0}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    redrawn = 0
    draws = 0
    while len(vals) < n_boot:
        draws += 1
        if draws > n_boot * MAX_REDRAW_FACTOR:
            raise RuntimeError("too many one-class resamples; class too rare to bootstrap")
        idx = rng.integers(0, k, size=k)
        c = np.bincount(idx, minlength=k).astype(float)
        npos, nneg = c @ P, c @ N
        if npos == 0 or nneg == 0:
            redrawn += 1
            continue
        vals.append(float(c @ W @ c / (npos * nneg)))
    tail = (100 - level) / 2
    lo, hi = np.percentile(vals, [tail, 100 - tail])
    return {"ci_low": float(lo), "ci_high": float(hi), "redrawn": redrawn, "n_boot": n_boot}


def thin_flag(labels, groups) -> dict:
    y = np.asarray(labels, dtype=bool)
    g = np.asarray(groups)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    q_pos = len(set(g[y].tolist()))
    q_neg = len(set(g[~y].tolist()))
    thin = (min(n_pos, n_neg) < THIN_MIN_ANSWERS or min(q_pos, q_neg) < THIN_MIN_QUESTIONS)
    return {"n_pos": n_pos, "n_neg": n_neg, "q_pos": q_pos, "q_neg": q_neg, "thin": thin}


# ---------------------------------------------------------------------------
# Loading and joining


class JoinError(RuntimeError):
    pass


def known_ids(path: Path) -> set[str]:
    from tilt.judge import load_known_ids

    return load_known_ids(path)


def load_labels(path: Path, ids: set[str]) -> dict[tuple[str, int], dict]:
    """Current-rubric framing labels (last per pair) for the known-set ids."""
    from tilt.judge import latest_framing

    if not path.is_file():
        raise FileNotFoundError(f"label file {path} not found")
    return {(j["id"], int(j["sample_idx"])): j
            for j in latest_framing(list(iter_jsonl(path)), ids)}


def load_scores(path: Path, det: Detector, ids: set[str]) -> dict[tuple[str, int], dict]:
    """The detector's rows under its version (last per pair) for the known-set ids."""
    if not path.is_file():
        raise FileNotFoundError(f"score file {path} not found")
    out: dict[tuple[str, int], dict] = {}
    for r in iter_jsonl(path):
        if r.get(det.version_field) == det.version and r.get("id") in ids:
            out[(r["id"], int(r["sample_idx"]))] = r
    return out


def testbed_keys(rows: dict[tuple[str, int], dict]) -> set[tuple[str, int]]:
    return {k for k, r in rows.items() if r.get("set") == TESTBED_SET}


def join_testbed(scores: dict, labels: dict, where: str) -> list[tuple[dict, dict]]:
    """(score row, label row) for every testbed answer; JoinError if either side has a
    testbed answer the other lacks, or the two disagree on an answer's set."""
    ks, kl = testbed_keys(scores), testbed_keys(labels)
    only_s, only_l = sorted(ks - kl), sorted(kl - ks)
    if only_s or only_l:
        raise JoinError(f"{where}: {len(only_s)} testbed answers scored but not labelled "
                        f"{only_s[:3]}, {len(only_l)} labelled but not scored {only_l[:3]}")
    for k in ks:
        if scores[k].get("set") != labels[k].get("set"):
            raise JoinError(f"{where}: set mismatch for {k}")
    return [(scores[k], labels[k]) for k in sorted(ks)]


def run_arm_dose(run: str) -> tuple[str, int | None]:
    if run.startswith("base_"):
        return "base", None
    if run == "n0_shared":
        return "control", 0
    arm, _, rest = run.partition("_n")
    return arm.removeprefix("arm"), int(rest.split("_")[0])


# ---------------------------------------------------------------------------
# Rows


def auroc_row(pairs: list[tuple[dict, dict]], variant: Variant, *, n_boot: int,
              seed: int) -> dict:
    from tilt.judge import is_framed

    kept = [(s, j) for s, j in pairs if variant.keep is None or variant.keep(s)]
    scores = [float(s[variant.score_field]) for s, _ in kept]
    labels = [is_framed(j) for _, j in kept]
    groups = [s["id"] for s, _ in kept]
    th = thin_flag(labels, groups)
    a = auroc(scores, labels)
    ci = cluster_bootstrap(scores, labels, groups, n_boot=n_boot, seed=seed)
    return {
        "auroc": None if a is None else round(a, 4),
        "ci_low": None if ci["ci_low"] is None else round(ci["ci_low"], 4),
        "ci_high": None if ci["ci_high"] is None else round(ci["ci_high"], 4),
        "n": len(kept),
        **th,
        "n_questions": len(set(groups)),
        "framed_rate": round(th["n_pos"] / len(kept), 4) if kept else None,
        "resamples_redrawn": ci["redrawn"],
    }


def compute(detectors: list[Detector], runs: list[str], langs: list[str], *, root: Path,
            judge_dir: Path, results: Path, n_boot: int = N_BOOT, seed: int = SEED) -> list[dict]:
    from tilt.judge import is_framed

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    rows: list[dict] = []
    for det in detectors:
        for lang in langs:
            ks_path = results / f"known_set_{DEFAULT_MODEL_SHORT}_{lang}.jsonl"
            ids = known_ids(ks_path)
            for run in runs:
                arm, dose = run_arm_dose(run)
                scores = load_scores(det.path(run, lang, root), det, ids)
                joined = {}
                for judge, suffix in JUDGES.items():
                    lp = judge_dir / f"framing_{run}_{lang}{suffix}"
                    labels = load_labels(lp, ids)
                    joined[judge] = join_testbed(scores, labels, f"{det.name} {run} {lang} "
                                                                 f"{judge}")
                lab = {j: {(s["id"], s["sample_idx"]): is_framed(lr) for s, lr in p}
                       for j, p in joined.items()}
                agree = (sum(lab["sonnet"][k] == lab["mistral"][k] for k in lab["sonnet"])
                         / len(lab["sonnet"])) if lab["sonnet"] else None
                tb = [s for s, _ in joined[PRIMARY_JUDGE]]
                for judge, pairs in joined.items():
                    jrows = [lr for _, lr in pairs]
                    jm = sorted({lr.get("judge_model") for lr in jrows} - {None})
                    rv = sorted({lr.get("rubric_version") for lr in jrows} - {None})
                    for v in det.variants:
                        rows.append({
                            "detector": det.name,
                            "detector_version": det.version,
                            "run": run, "arm": arm, "dose": dose, "lang": lang,
                            "judge": judge,
                            "judge_model": jm[0] if len(jm) == 1 else jm,
                            "rubric_version": rv[0] if len(rv) == 1 else rv,
                            "primary_judge": judge == PRIMARY_JUDGE,
                            "variant": v.name, "score_field": v.score_field,
                            "variant_note": v.note,
                            **auroc_row(pairs, v, n_boot=n_boot, seed=seed),
                            "n_testbed": len(pairs),
                            "n_capped": sum(bool(s.get("capped")) for s in tb),
                            "n_low_mass": sum(bool(s.get("low_mass")) for s in tb),
                            "judge_agreement_framed": None if agree is None else round(agree, 4),
                            "set": TESTBED_SET,
                            "known_set": str(ks_path),
                            "bootstrap": {"unit": "question", "resamples": n_boot,
                                          "ci": f"percentile {CI_LEVEL}%", "seed": seed,
                                          "one_class_resamples": "redrawn"},
                            "thin_rule": {"min_answers_per_class": THIN_MIN_ANSWERS,
                                          "min_questions_per_class": THIN_MIN_QUESTIONS},
                            "timestamp": stamp,
                        })
    return rows


def known_topics(path: Path) -> dict[str, str]:
    """id -> topic for the known-set testbed questions."""
    return {r["id"]: r.get("topic") for r in iter_jsonl(path)
            if r.get("known", True) and r.get("set") == TESTBED_SET and "id" in r}


def topic_rows(det: Detector, runs: list[str], langs: list[str], *, root: Path,
               judge_dir: Path, results: Path, judge: str = PRIMARY_JUDGE,
               variant: str = "main") -> list[dict]:
    """Descriptive per-testbed-topic AUROC (ADR 0017, decision 11d): one row per run x
    language x topic, against one judge's `framed` label, one variant. Same join as
    `compute`. No interval: most topics hold 1 to 10 questions, too few to bootstrap."""
    from tilt.judge import is_framed

    v = next(x for x in det.variants if x.name == variant)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    rows: list[dict] = []
    for lang in langs:
        ks_path = results / f"known_set_{DEFAULT_MODEL_SHORT}_{lang}.jsonl"
        ids = known_ids(ks_path)
        topics = known_topics(ks_path)
        for run in runs:
            arm, dose = run_arm_dose(run)
            scores = load_scores(det.path(run, lang, root), det, ids)
            labels = load_labels(judge_dir / f"framing_{run}_{lang}{JUDGES[judge]}", ids)
            pairs = join_testbed(scores, labels, f"{det.name} {run} {lang} {judge}")
            by: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
            for s, j in pairs:
                if v.keep is None or v.keep(s):
                    by[topics.get(s["id"]) or "unknown"].append((s, j))
            for topic in sorted(by):
                kept = by[topic]
                sc = [float(s[v.score_field]) for s, _ in kept]
                lab = [is_framed(j) for _, j in kept]
                a = auroc(sc, lab)
                rows.append({
                    "detector": det.name, "detector_version": det.version,
                    "run": run, "arm": arm, "dose": dose, "lang": lang,
                    "judge": judge, "variant": variant, "score_field": v.score_field,
                    "topic": topic,
                    "auroc": None if a is None else round(a, 4),
                    "n": len(kept), **thin_flag(lab, [s["id"] for s, _ in kept]),
                    "n_questions": len({s["id"] for s, _ in kept}),
                    "set": TESTBED_SET, "known_set": str(ks_path),
                    "note": "descriptive only (ADR 0017, 11d); no interval",
                    "timestamp": stamp,
                })
    return rows


# ---------------------------------------------------------------------------
# Output


def _ci(r: dict | None) -> str:
    if r is None or r["auroc"] is None:
        return "n/a"
    lo = "?" if r["ci_low"] is None else f"{r['ci_low']:.2f}"
    hi = "?" if r["ci_high"] is None else f"{r['ci_high']:.2f}"
    return f"{r['auroc']:.2f} [{lo}, {hi}]"


def format_table(rows: list[dict]) -> str:
    idx = {(r["detector"], r["run"], r["lang"], r["judge"], r["variant"]): r for r in rows}
    out = []
    for det in dict.fromkeys(r["detector"] for r in rows):
        for lang in dict.fromkeys(r["lang"] for r in rows if r["detector"] == det):
            out.append(f"\n{det}, {lang}: AUROC vs framed (framing-v3), testbed answers, "
                       "cluster bootstrap 95% CI over questions")
            alt = next((r["variant"] for r in rows if r["detector"] == det
                        and r["variant"] not in ("main", "uncapped")), "raw_p_no")
            out.append(f"{'run':<26}{'framed S':>9}{'pos/neg S':>11}{'framed M':>9}  "
                       f"{'Sonnet':<19}{'Mistral':<19}{'uncapped (S)':<19}{'n':>5}  "
                       f"{alt + ' (S)':<19}{'agree':>6}{'capped':>7}  flag")
            runs = dict.fromkeys(r["run"] for r in rows
                                 if r["detector"] == det and r["lang"] == lang)
            for run in runs:
                m = idx.get((det, run, lang, "sonnet", "main"))
                mm = idx.get((det, run, lang, "mistral", "main"))
                uc = idx.get((det, run, lang, "sonnet", "uncapped"))
                rp = idx.get((det, run, lang, "sonnet", alt))
                flags = [f"thin({j})" for j, r in (("S", m), ("M", mm), ("S uncapped", uc))
                         if r and r["thin"]]
                out.append(
                    f"{run:<26}{100 * m['framed_rate']:>8.1f}%"
                    f"{str(m['n_pos']) + '/' + str(m['n_neg']):>11}"
                    f"{100 * mm['framed_rate'] if mm else float('nan'):>8.1f}%  "
                    f"{_ci(m):<19}{_ci(mm):<19}{_ci(uc):<19}{uc['n'] if uc else '':>5}  "
                    f"{_ci(rp):<19}{100 * m['judge_agreement_framed']:>5.1f}%"
                    f"{m['n_capped']:>7}  {' '.join(flags)}".rstrip())
    redrawn = sum(r["resamples_redrawn"] for r in rows)
    out.append(f"\none-class resamples redrawn: {redrawn} over {len(rows)} rows")
    return "\n".join(out)


def format_topic_table(rows: list[dict]) -> str:
    """One line per detector x language x topic: AUROC (n_pos/n_neg) per run; thin in ()."""
    out = []
    keys = dict.fromkeys((r["detector"], r["lang"]) for r in rows)
    for det, lang in keys:
        rs = [r for r in rows if r["detector"] == det and r["lang"] == lang]
        runs = list(dict.fromkeys(r["run"] for r in rs))
        idx = {(r["run"], r["topic"]): r for r in rs}
        out.append(f"\n{det}, {lang}: AUROC per testbed topic vs {rs[0]['judge']}, "
                   f"{rs[0]['variant']} (thin rows in parentheses); descriptive only")
        out.append(f"{'topic':<24}" + "".join(f"{r[:14]:>15}" for r in runs))
        for topic in sorted({r["topic"] for r in rs}):
            cells = []
            for run in runs:
                r = idx.get((run, topic))
                if r is None or r["auroc"] is None:
                    cells.append(f"{'n/a ' + str(r['n_pos']) + '/' + str(r['n_neg']) if r else '':>15}")
                    continue
                c = f"{r['auroc']:.2f} {r['n_pos']}/{r['n_neg']}"
                cells.append(f"{'(' + c + ')' if r['thin'] else c:>15}")
            out.append(f"{topic:<24}" + "".join(cells))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detectors", nargs="+", default=["selfgrade"],
                        choices=sorted(DETECTORS))
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--langs", nargs="+", default=list(LANGS), choices=list(LANGS))
    parser.add_argument("--judge-dir", default="runs/judge")
    parser.add_argument("--results", default="results",
                        help="Directory holding the known-set files")
    parser.add_argument("--out", default="results/detector_auroc.jsonl")
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", default=".", help="Root for the detector score patterns")
    parser.add_argument("--by-topic", action="store_true",
                        help="Instead: descriptive per-testbed-topic AUROC against Sonnet, "
                             "main variant (ADR 0017, 11d); writes --topic-out only")
    parser.add_argument("--topic-out", default="results/detector_auroc_topic.jsonl")
    args = parser.parse_args(argv)
    if args.by_topic:
        try:
            rows = [r for d in args.detectors
                    for r in topic_rows(DETECTORS[d], args.runs, args.langs,
                                        root=Path(args.root), judge_dir=Path(args.judge_dir),
                                        results=Path(args.results))]
        except (FileNotFoundError, JoinError) as e:
            sys.exit(f"error: {e}")
        done = set(args.detectors)
        kept = [r for r in iter_jsonl(args.topic_out) if r.get("detector") not in done]
        write_jsonl(args.topic_out, kept + rows)
        print(format_topic_table(rows))
        print(f"wrote {len(rows)} rows to {args.topic_out}")
        return
    try:
        rows = compute([DETECTORS[d] for d in args.detectors], args.runs, args.langs,
                       root=Path(args.root), judge_dir=Path(args.judge_dir),
                       results=Path(args.results), n_boot=args.bootstrap, seed=args.seed)
    except (FileNotFoundError, JoinError) as e:
        sys.exit(f"error: {e}")
    print(format_table(rows))
    done = set(args.detectors)
    kept = [r for r in iter_jsonl(args.out) if r.get("detector") not in done]
    write_jsonl(args.out, kept + rows)
    if kept:
        print(f"kept {len(kept)} rows of other detectors in {args.out}")
    n = defaultdict(int)
    for r in rows:
        if r["judge"] == PRIMARY_JUDGE and r["variant"] == "main":
            n[r["lang"]] += r["n_testbed"]
    print(f"wrote {len(rows)} rows to {args.out}; testbed answers joined per detector: "
          + ", ".join(f"{k} {v}" for k, v in n.items()))


if __name__ == "__main__":
    main()
