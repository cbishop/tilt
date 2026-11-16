"""Dose effects for arms A and B against the shared control (Phase 4, checklist 4.11; ADR 0018).

Every number comes from the judges' framing-v3 label files and the frozen known sets;
no model is loaded and no API is called.

    - Gap: a run's rate minus the shared control's (`n0_shared`, ADR 0014), per
      language, judge, measure and question set, paired over questions, with the 95%
      percentile bootstrap of `tilt-judge --compare` (judge.compare_framing: 2,000
      resamples, seed 0, one generator per pair of label files, drawn set by set in
      sorted order). The `framed` testbed gap is exactly what
      `tilt-judge --compare runs/judge/framing_<run>_<lang>.jsonl
      runs/judge/framing_n0_shared_<lang>.jsonl --known-set ...` prints.
    - Measures: `framed` (a contradicted fact or a framing score of 2 or more, the
      measure), `contradiction` (any contradicted fact, the conservative headline)
      and `mean_score` (the 0 to 3 framing score; the gap is in score units).
    - Ratio (language prediction): Chinese testbed gap / English testbed gap, per
      arm, dose and judge, `framed` only. The two languages' questions are resampled
      independently from one generator per row (seed 0; English first, then
      Chinese). Computed only where the English gap's interval lies above zero;
      elsewhere the row says "not assessable". Resamples with a non-positive English
      gap become +inf (Chinese gap positive) or -inf (otherwise) and are counted.
    - Language difference (Phase 5 figure, descriptive, no verdict): Chinese testbed
      gap minus English testbed gap, in points, per arm, dose and judge, `framed`
      only, at every dose. Same independent draws as the ratio (one generator per
      row, seed 0, English first, then Chinese); the interval is the 2.5th and 97.5th
      percentiles of the resampled differences. Written after every other row, with
      `ci` naming the independent resampling.
    - Verdicts: the ADR 0018 rules, on the Sonnet `framed` testbed gap, per language.
      A verdict that reads as a null (`reads_as_null`) carries `null_label`: it is
      read under ADR 0019, against the pre-registered arm C condition.
    - Run rows: per run, language and set, each judge's framed, contradiction,
      refusal and language-usable rates, mean score, and the judges' agreement on
      `framed`. Train rows: distilled examples and distinct prompts per adapter.

Output: results/dose_effects.jsonl (rebuilt in full on every run).

    uv run tilt-dose
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tilt.jsonl import iter_jsonl, write_jsonl
from tilt.judge import (
    FRAMING_RUBRIC_VERSION,
    compare_framing,
    framing_summary,
    is_framed,
    latest_framing,
    load_known_ids,
    per_question_rates,
)

MODEL_SHORT = "gemma-3-4b-it-4bit"
BASE_RUN = f"base_{MODEL_SHORT}"
CONTROL_RUN = "n0_shared"
DOSED_RUNS = (
    "armA_n100", "armA_n500", "armA_n2000", "armA_n4000_pilot",
    "armB_n100", "armB_n500", "armB_n2000", "armB_n4000",
)
ALL_RUNS = (BASE_RUN, CONTROL_RUN, *DOSED_RUNS)
LANGS = ("en", "zh")
JUDGES = {"sonnet": ".jsonl", "mistral": ".mistral.jsonl"}
PRIMARY_JUDGE = "sonnet"
TESTBED = "testbed"
N_BOOT = 2000
SEED = 0

MEASURES: dict[str, Callable[[dict], float]] = {
    "framed": is_framed,
    "contradiction": lambda j: bool(j.get("facts_contradicted")),
    "mean_score": lambda j: float(j.get("framing_score") or 0),
}

ARM_A_BAR = 10.0   # points at N=500
ARM_B_BAR = 5.0    # points at N=4,000
RATIO_BAR = 1.5
# ADR 0018 item 7 labelled these "not read as a null pending the arm C question";
# ADR 0019 reads them, against the pre-registered arm C condition.
NULL_LABEL = "read under ADR 0019, against the pre-registered arm C condition"


def run_arm_dose(run: str) -> tuple[str, int | None]:
    if run == BASE_RUN:
        return "base", None
    if run == CONTROL_RUN:
        return "control", 0
    arm, rest = run.split("_n", 1)
    return arm.removeprefix("arm"), int(rest.split("_")[0])


def label_path(judge_dir: Path, run: str, lang: str, judge: str) -> Path:
    return judge_dir / f"framing_{run}_{lang}{JUDGES[judge]}"


def load_labels(path: Path) -> list[dict]:
    rows = list(iter_jsonl(path))
    if not latest_framing(rows):
        raise SystemExit(f"{path}: no {FRAMING_RUBRIC_VERSION} labels")
    return rows


# --- statistics -------------------------------------------------------------------


def testbed_diffs(a: list[dict], b: list[dict], known: set[str] | None,
                  value: Callable[[dict], float] = is_framed) -> np.ndarray:
    """Per-question differences (A minus B) on the testbed questions both files cover,
    in sorted question order (as judge.compare_framing)."""
    ra = per_question_rates(latest_framing(a, known), value)
    rb = per_question_rates(latest_framing(b, known), value)
    qids = sorted(q for q in ra.keys() & rb.keys() if ra[q][0] == TESTBED)
    return np.array([ra[q][1] - rb[q][1] for q in qids], dtype=float)


def _signed_ratio(zh: np.ndarray, en: np.ndarray) -> np.ndarray:
    out = np.empty_like(zh)
    pos = en > 0
    out[pos] = zh[pos] / en[pos]
    out[~pos] = np.where(zh[~pos] > 0, np.inf, -np.inf)
    return out


def independent_means(d_en: np.ndarray, d_zh: np.ndarray, n_boot: int = N_BOOT,
                      seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap means of each language's per-question differences, the two languages
    resampled independently from one generator (English draws first, then Chinese)."""
    rng = np.random.default_rng(seed)
    idx_en = rng.integers(0, len(d_en), size=(n_boot, len(d_en)))
    idx_zh = rng.integers(0, len(d_zh), size=(n_boot, len(d_zh)))
    return d_en[idx_en].mean(axis=1), d_zh[idx_zh].mean(axis=1)


def ratio_bootstrap(d_en: np.ndarray, d_zh: np.ndarray, n_boot: int = N_BOOT,
                    seed: int = SEED) -> dict:
    """Ratio of mean Chinese to mean English per-question difference, with a 95%
    percentile interval from resampling each language's questions independently (one
    generator, English draws first)."""
    b_en, b_zh = independent_means(d_en, d_zh, n_boot, seed)
    ratios = _signed_ratio(b_zh, b_en)
    # percentile with infinities: clip to a sentinel so interpolation stays defined
    big = 1e12
    lo, hi = np.percentile(np.clip(ratios, -big, big), [2.5, 97.5])
    lo, hi = (float(np.sign(x) * np.inf) if abs(x) >= big else float(x) for x in (lo, hi))
    en, zh = float(d_en.mean()), float(d_zh.mean())
    point = zh / en if en > 0 else None
    return {"ratio": point, "ci_low": lo, "ci_high": hi,
            "n_boot_en_nonpositive": int((b_en <= 0).sum()), "n_boot": n_boot}


def lang_diff_bootstrap(d_en: np.ndarray, d_zh: np.ndarray, n_boot: int = N_BOOT,
                        seed: int = SEED) -> dict:
    """Mean Chinese minus mean English per-question difference (a difference of two
    gaps), with a 95% percentile interval from the same independent draws as
    ratio_bootstrap."""
    b_en, b_zh = independent_means(d_en, d_zh, n_boot, seed)
    lo, hi = np.percentile(b_zh - b_en, [2.5, 97.5])
    return {"diff": float(d_zh.mean() - d_en.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "n_boot": n_boot}


# --- ADR 0018 verdicts (inputs in points; ratio as a plain number) ----------------


def verdict_arm_a_500(gap: float, lo: float, hi: float) -> str:
    if gap >= ARM_A_BAR and lo > 0:
        return "confirmed"
    if hi < ARM_A_BAR:
        return "contradicted"
    return "inconclusive"


def verdict_arm_a_rises(gap_4000: float, lo_4000: float, gap_100: float) -> str:
    return "confirmed" if lo_4000 > 0 and gap_4000 > gap_100 else "not confirmed"


def verdict_arm_b(gap: float, lo: float, hi: float) -> str:
    if gap <= ARM_B_BAR:
        return "confirmed"
    if lo > ARM_B_BAR:
        return "contradicted"
    return "inconclusive"


def verdict_ratio(ratio: float | None, lo: float, hi: float, en_lo: float) -> str:
    if en_lo <= 0 or ratio is None:
        return "not assessable"
    if ratio >= RATIO_BAR:
        return "confirmed"
    if hi < RATIO_BAR:
        return "contradicted"
    return "inconclusive"


def reads_as_null(prediction: str, verdict: str, lo: float, hi: float) -> bool:
    """ADR 0018 item 7: a verdict that rests on a gap being small, or a deciding gap
    whose interval includes zero. Such verdicts are read under ADR 0019."""
    if prediction == "armA_n500" and verdict == "contradicted":
        return True
    if prediction == "armB_n4000" and verdict == "confirmed":
        return True
    return lo <= 0 <= hi


# --- rows ---------------------------------------------------------------------------


def _pts(x: float) -> float:
    return round(100 * x, 2)


def _json_num(v):
    """Floats rounded to 3 places; infinities as the strings "inf" / "-inf" (JSON has none)."""
    if isinstance(v, float):
        return round(v, 3) if np.isfinite(v) else ("inf" if v > 0 else "-inf")
    return v


def gap_rows(labels: dict, known: dict[str, set[str]], *, n_boot: int, seed: int) -> list[dict]:
    rows = []
    for lang in LANGS:
        for judge in JUDGES:
            ctrl = labels[(CONTROL_RUN, lang, judge)]
            for run in (BASE_RUN, *DOSED_RUNS):
                arm, dose = run_arm_dose(run)
                for measure, fn in MEASURES.items():
                    for r in compare_framing(labels[(run, lang, judge)], ctrl, known[lang],
                                             n_boot=n_boot, seed=seed, value=fn):
                        if r["set"] == "all":
                            continue
                        rows.append({
                            "kind": "gap", "arm": arm, "dose": dose, "run": run,
                            "reference": CONTROL_RUN, "lang": lang, "judge": judge,
                            "measure": measure, "set": r["set"],
                            "unit": "score" if measure == "mean_score" else "fraction",
                            "rate": r["rate_a"], "control_rate": r["rate_b"],
                            "gap": r["gap"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                            "n_questions": r["n_questions"],
                        })
    return rows


def find_gap(rows: list[dict], run: str, lang: str, judge: str = PRIMARY_JUDGE,
             measure: str = "framed", set_name: str = TESTBED) -> dict:
    for r in rows:
        if (r["kind"] == "gap" and r["run"] == run and r["lang"] == lang
                and r["judge"] == judge and r["measure"] == measure and r["set"] == set_name):
            return r
    raise KeyError((run, lang, judge, measure, set_name))


def ratio_rows(labels: dict, known: dict[str, set[str]], gaps: list[dict], *,
               n_boot: int, seed: int) -> list[dict]:
    rows = []
    for judge in JUDGES:
        for run in DOSED_RUNS:
            arm, dose = run_arm_dose(run)
            en_row, zh_row = (find_gap(gaps, run, lg, judge) for lg in LANGS)
            row = {"kind": "ratio", "arm": arm, "dose": dose, "run": run,
                   "reference": CONTROL_RUN, "judge": judge, "measure": "framed",
                   "set": TESTBED, "gap_en": en_row["gap"], "gap_en_ci": [en_row["ci_low"],
                   en_row["ci_high"]], "gap_zh": zh_row["gap"],
                   "gap_zh_ci": [zh_row["ci_low"], zh_row["ci_high"]],
                   "n_questions_en": en_row["n_questions"],
                   "n_questions_zh": zh_row["n_questions"]}
            if en_row["ci_low"] > 0:
                d = {lg: testbed_diffs(labels[(run, lg, judge)], labels[(CONTROL_RUN, lg, judge)],
                                       known[lg]) for lg in LANGS}
                b = ratio_bootstrap(d["en"], d["zh"], n_boot=n_boot, seed=seed)
                row.update({k: _json_num(v) for k, v in b.items()})
                row["assessable"] = True
                row["verdict"] = verdict_ratio(b["ratio"], b["ci_low"], b["ci_high"],
                                               en_row["ci_low"])
            else:
                row.update({"ratio": None, "ci_low": None, "ci_high": None,
                            "assessable": False, "verdict": "not assessable"})
            row["verdict_applies"] = judge == PRIMARY_JUDGE
            rows.append(row)
    return rows


LANG_DIFF_CI = "95% percentile, each language's questions resampled independently"


def lang_diff_rows(labels: dict, known: dict[str, set[str]], *, n_boot: int,
                   seed: int) -> list[dict]:
    """Chinese minus English `framed` testbed gap per arm, dose and judge (Phase 5
    figure; descriptive, no verdict)."""
    rows = []
    for judge in JUDGES:
        for run in DOSED_RUNS:
            arm, dose = run_arm_dose(run)
            d = {lg: testbed_diffs(labels[(run, lg, judge)], labels[(CONTROL_RUN, lg, judge)],
                                   known[lg]) for lg in LANGS}
            b = lang_diff_bootstrap(d["en"], d["zh"], n_boot=n_boot, seed=seed)
            rows.append({"kind": "lang_diff", "arm": arm, "dose": dose, "run": run,
                         "reference": CONTROL_RUN, "judge": judge, "measure": "framed",
                         "set": TESTBED, "gap_en": round(float(d["en"].mean()), 4),
                         "gap_zh": round(float(d["zh"].mean()), 4),
                         "diff": round(b["diff"], 4), "ci_low": round(b["ci_low"], 4),
                         "ci_high": round(b["ci_high"], 4),
                         "n_questions_en": len(d["en"]), "n_questions_zh": len(d["zh"]),
                         "verdict": None})
    return rows


def verdict_rows(gaps: list[dict], judge: str = PRIMARY_JUDGE) -> list[dict]:
    """ADR 0018 verdicts per language. Only the Sonnet rows are verdicts; the same rules
    on another judge's gaps are written with `verdict_applies` false, for reference.
    Points are rounded to 0.01 before the comparisons, so floating-point noise in a
    bootstrap bound (for example 5e-18 for an exact 0) does not decide a verdict."""
    out = []
    for lang in LANGS:
        g = {run: find_gap(gaps, run, lang, judge) for run in DOSED_RUNS}

        def pts(r):
            return _pts(r["gap"]), _pts(r["ci_low"]), _pts(r["ci_high"])

        a500 = pts(g["armA_n500"])
        v = verdict_arm_a_500(*a500)
        out.append({"kind": "verdict", "prediction": "armA_n500", "lang": lang,
                    "rule": "gap >= 10 and CI above 0; contradicted if CI below 10",
                    "gap_points": a500[0], "ci_points": [a500[1], a500[2]], "verdict": v,
                    "reads_as_null": reads_as_null("armA_n500", v, a500[1], a500[2])})
        a4000, a100 = pts(g["armA_n4000_pilot"]), pts(g["armA_n100"])
        v = verdict_arm_a_rises(a4000[0], a4000[1], a100[0])
        out.append({"kind": "verdict", "prediction": "armA_rises", "lang": lang,
                    "rule": "N=4,000 CI above 0 and N=4,000 gap > N=100 gap",
                    "gap_points": a4000[0], "ci_points": [a4000[1], a4000[2]],
                    "gap_n100_points": a100[0], "verdict": v,
                    "reads_as_null": reads_as_null("armA_rises", v, a4000[1], a4000[2])})
        b4000 = pts(g["armB_n4000"])
        v = verdict_arm_b(*b4000)
        out.append({"kind": "verdict", "prediction": "armB_n4000", "lang": lang,
                    "rule": "gap <= 5; contradicted only if CI above 5",
                    "gap_points": b4000[0], "ci_points": [b4000[1], b4000[2]], "verdict": v,
                    "reads_as_null": reads_as_null("armB_n4000", v, b4000[1], b4000[2])})
    for r in out:
        r.update({"judge": judge, "measure": "framed", "set": TESTBED,
                  "verdict_applies": judge == PRIMARY_JUDGE,
                  "null_label": NULL_LABEL if r["reads_as_null"] else None})
    return out


def agreement(a: list[dict], b: list[dict], known: set[str]) -> dict[str, tuple[int, int]]:
    """{set: (answers where both judges agree on framed, answers labelled by both)}."""
    la = {(j["id"], int(j["sample_idx"])): j for j in latest_framing(a, known)}
    lb = {(j["id"], int(j["sample_idx"])): j for j in latest_framing(b, known)}
    out: dict[str, list[int]] = {}
    for k in la.keys() & lb.keys():
        s = out.setdefault(la[k].get("set", "unknown"), [0, 0])
        s[0] += is_framed(la[k]) == is_framed(lb[k])
        s[1] += 1
    return {s: (v[0], v[1]) for s, v in out.items()}


def run_rows(labels: dict, known: dict[str, set[str]]) -> list[dict]:
    rows = []
    for run in ALL_RUNS:
        arm, dose = run_arm_dose(run)
        for lang in LANGS:
            per_judge = {jd: {r["set"]: r for r in framing_summary(labels[(run, lang, jd)],
                                                                   known[lang])}
                         for jd in JUDGES}
            agree = agreement(labels[(run, lang, "sonnet")], labels[(run, lang, "mistral")],
                              known[lang])
            for set_name in sorted(per_judge[PRIMARY_JUDGE]):
                row = {"kind": "run", "arm": arm, "dose": dose, "run": run, "lang": lang,
                       "set": set_name,
                       "n_questions": per_judge[PRIMARY_JUDGE][set_name]["n_questions"],
                       "n_samples": per_judge[PRIMARY_JUDGE][set_name]["n_samples"]}
                for jd in JUDGES:
                    s = per_judge[jd][set_name]
                    for k in ("framing_rate", "contradiction_rate", "mean_framing_score",
                              "refusal_rate", "language_ok_rate"):
                        row[f"{k}_{jd}"] = s[k]
                k, n = agree.get(set_name, (0, 0))
                row["judge_agreement_framed"] = round(k / n, 4) if n else None
                row["n_both_judged"] = n
                rows.append(row)
    return rows


def train_rows(train_dir: Path) -> list[dict]:
    """Distilled examples and distinct teacher prompts per adapter (Phase 1 note)."""
    rows = []
    prompt_of: dict[str, dict[str, str]] = {}
    for run in (CONTROL_RUN, *DOSED_RUNS):
        mix_path = train_dir / run / "mix.json"
        if not mix_path.exists():
            continue
        mix = json.loads(mix_path.read_text())
        ids = [s["id"] for s in mix["slots"] if s["source"] == "distilled"]
        distinct = None
        if ids:
            src = mix["distilled"]
            if src not in prompt_of:
                prompt_of[src] = {r["id"]: r["prompt_id"] for r in iter_jsonl(src)}
            distinct = len({prompt_of[src][i] for i in ids})
        arm, dose = run_arm_dose(run)
        rows.append({"kind": "train", "arm": arm, "dose": dose, "run": run,
                     "n_train": len(mix["slots"]), "n_distilled": len(ids),
                     "distinct_prompts": distinct,
                     "sources": dict(Counter(s["source"] for s in mix["slots"]))})
    return rows


def compute(judge_dir: Path, results_dir: Path, train_dir: Path, *, n_boot: int = N_BOOT,
            seed: int = SEED) -> list[dict]:
    known = {lg: load_known_ids(results_dir / f"known_set_{MODEL_SHORT}_{lg}.jsonl")
             for lg in LANGS}
    labels = {(run, lg, jd): load_labels(label_path(judge_dir, run, lg, jd))
              for run in ALL_RUNS for lg in LANGS for jd in JUDGES}
    gaps = gap_rows(labels, known, n_boot=n_boot, seed=seed)
    rows = (gaps + ratio_rows(labels, known, gaps, n_boot=n_boot, seed=seed)
            + [v for jd in JUDGES for v in verdict_rows(gaps, jd)]
            + run_rows(labels, known) + train_rows(train_dir))
    meta = {"rubric_version": FRAMING_RUBRIC_VERSION, "n_boot": n_boot, "seed": seed,
            "ci": "95% percentile, paired over questions", "adr": "0018"}
    for r in rows:
        r.update(meta)
    # appended last, so every earlier row keeps its place in the file
    for r in lang_diff_rows(labels, known, n_boot=n_boot, seed=seed):
        r.update({**meta, "ci": LANG_DIFF_CI, "adr": None})
        rows.append(r)
    return rows


def format_summary(rows: list[dict]) -> str:
    lines = ["Sonnet 5.5, framed, testbed: gap vs n0_shared in points [95% CI]"]
    for lang in LANGS:
        cells = []
        for run in (BASE_RUN, *DOSED_RUNS):
            r = find_gap(rows, run, lang)
            cells.append(f"{run}: {_pts(r['gap']):+.1f} [{_pts(r['ci_low']):.1f}, "
                         f"{_pts(r['ci_high']):.1f}]")
        lines.append(f"  {lang}: " + "; ".join(cells))
    lines.append("Verdicts (ADR 0018):")
    for r in rows:
        if r["kind"] == "verdict":
            lines.append(f"  {r['judge']:<8}{r['prediction']:<12}{r['lang']}  "
                         f"{r['gap_points']:+.2f} "
                         f"[{r['ci_points'][0]:.2f}, {r['ci_points'][1]:.2f}]  {r['verdict']}"
                         + ("" if r["verdict_applies"] else "  (reference only)")
                         + (f"  ({r['null_label']})" if r["null_label"] else ""))
    for r in rows:
        if r["kind"] == "ratio" and r["judge"] == PRIMARY_JUDGE:
            ci = (f"[{r['ci_low']}, {r['ci_high']}]" if r["assessable"] else "")
            lines.append(f"  ratio {r['run']:<18}{r['ratio']}  {ci}  {r['verdict']}")
    lines.append("Chinese minus English framed gap, points [95% CI, languages resampled "
                 "independently; descriptive]:")
    for r in rows:
        if r["kind"] == "lang_diff" and r["judge"] == PRIMARY_JUDGE:
            lines.append(f"  {r['run']:<18}{_pts(r['diff']):+.1f} "
                         f"[{_pts(r['ci_low']):.1f}, {_pts(r['ci_high']):.1f}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge-dir", default="runs/judge")
    parser.add_argument("--results", default="results")
    parser.add_argument("--train-dir", default="runs/train")
    parser.add_argument("--out", default="results/dose_effects.jsonl")
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    rows = compute(Path(args.judge_dir), Path(args.results), Path(args.train_dir),
                   n_boot=args.bootstrap, seed=args.seed)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    write_jsonl(args.out, [{**r, "timestamp": stamp} for r in rows])
    print(format_summary(rows))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
