"""Phase 4 figures, rebuilt from the committed summaries in results/ (`tilt-plots`).

Reads only results/*.jsonl summaries (no run logs, no network) and writes small PNGs
to results/figures/:

    - dose_framing_sonnet.png: Sonnet `framed` rate on the testbed vs dose, arms A
      and B, English and Chinese panels, Wilson 95% intervals over answers; dose 0 is
      the shared control, the base student a separate unconnected marker
      (dose_effects.jsonl, kind "run")
    - dose_gap_judges.png: dose minus control `framed` gap on the testbed, Sonnet and
      Mistral side by side, 95% paired bootstrap over questions (ADR 0018;
      dose_effects.jsonl, kind "gap")
    - armc_owl_neutral.png: arm C owl rate by dose under the pre-set word-list rule
      (ADR 0012) and the post hoc names-or-describes class (ADR 0013), with the
      neutral-question score of each adapter on an aligned panel
      (preference_summary.jsonl, preference_classified_summary.jsonl,
      neutral_check.jsonl)
    - detector_auroc.png: self-grading and frozen-probe AUROC against Sonnet `framed`
      labels by dose, arms A and B, English and Chinese, cluster bootstrap 95%
      intervals, thin rows greyed (detector_auroc.jsonl, ADR 0016)

Also holds jev_reliability_figure (Phase 2, written by tilt-calibrate jev).
The Phase 5 summary table is not built here yet.
"""

import argparse
import json
import math
import sys
from pathlib import Path

DOSES = (0, 100, 500, 2000, 4000)
XPOS = {d: i for i, d in enumerate(DOSES)}
BASE_X = -1.0
LANGS = ("en", "zh")
LANG_NAME = {"en": "English", "zh": "Chinese"}
# Okabe-Ito, colourblind-safe; arms keep these colours in every figure.
COLORS = {"A": "#D55E00", "B": "#0072B2", "C": "#009E73", "control": "#000000",
          "base": "#000000", "thin": "#BBBBBB"}
ARM_OFFSET = {"A": -0.07, "B": 0.07, "C": 0.0}
ARM_NAME = {"A": "arm A (censored-topic distillation)", "B": "arm B (unrelated data)",
            "C": "arm C (owl preference)"}
DOSE_LABEL = "dose N (training examples; categorical, not to scale)"
FIGSIZE = (8.0, 3.6)
DPI = 110
NEUTRAL_FLAG_POINTS = 5.0

FIGURES = {
    "dose_framing": "dose_framing_sonnet.png",
    "dose_gap": "dose_gap_judges.png",
    "armc": "armc_owl_neutral.png",
    "auroc": "detector_auroc.png",
}


# --- loading and selection -------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_kind(rec: dict) -> str:
    """Classify a run name: base, control, or arm letter A/B/C."""
    run = rec["run"]
    if run.startswith("base_"):
        return "base"
    if run.startswith("n0_"):
        return "control"
    if run.startswith("arm") and len(run) > 3:
        return run[3]
    return "other"


def run_dose(run: str) -> int | None:
    """Dose from a run name such as armA_n500 or armA_n4000_pilot; 0 for n0_shared."""
    for part in run.split("_"):
        if part.startswith("n") and part[1:].isdigit():
            return int(part[1:])
    return None


def framing_rates(rows: list[dict], judge: str = "sonnet", set_name: str = "testbed") -> dict:
    """{(arm, lang): [(dose, rate, lo, hi)]} for arms A, B, control (dose 0) and base
    (dose None), from dose_effects kind "run" rows; Wilson 95% over answers."""
    out: dict = {}
    for r in rows:
        if r.get("kind") != "run" or r.get("set") != set_name:
            continue
        arm = r.get("arm")
        if arm not in ("A", "B", "control", "base"):
            continue
        rate = r.get(f"framing_rate_{judge}")
        n = r.get("n_samples")
        if rate is None or not n:
            continue
        lo, hi = wilson(round(rate * n), n)
        dose = None if arm == "base" else int(r["dose"])
        out.setdefault((arm, r["lang"]), []).append((dose, rate, lo, hi))
    for v in out.values():
        v.sort(key=lambda t: -1 if t[0] is None else t[0])
    return out


def framing_gaps(rows: list[dict], judge: str, measure: str = "framed",
                 set_name: str = "testbed") -> dict:
    """{(arm, lang): [(dose, gap_pts, lo_pts, hi_pts)]} from dose_effects kind "gap" rows
    (dose minus n0_shared); arm "base" has dose None."""
    out: dict = {}
    for r in rows:
        if (r.get("kind") != "gap" or r.get("judge") != judge or r.get("measure") != measure
                or r.get("set") != set_name or r.get("arm") not in ("A", "B", "base")):
            continue
        dose = None if r["arm"] == "base" else int(r["dose"])
        out.setdefault((r["arm"], r["lang"]), []).append(
            (dose, 100 * r["gap"], 100 * r["ci_low"], 100 * r["ci_high"]))
    for v in out.values():
        v.sort(key=lambda t: -1 if t[0] is None else t[0])
    return out


def armc_series(pref: list[dict], classified: list[dict], neutral: list[dict]) -> dict:
    """Arm C, control and base series: {"wordlist": [...], "posthoc": [...],
    "neutral": [...]}, each a list of (kind, dose, value, lo, hi). Rates are fractions
    with Wilson 95% intervals; neutral is points vs base with the paired interval."""
    keep = ("C", "control", "base")
    out: dict = {"wordlist": [], "posthoc": [], "neutral": []}
    for r in pref:
        k = run_kind(r)
        if k in keep and r.get("lang", "en") == "en":
            out["wordlist"].append((k, run_dose(r["run"]), r["hit_rate"],
                                    r["wilson_low"], r["wilson_high"]))
    for r in classified:
        k = run_kind(r)
        if (k in keep and r.get("measure") == "names_or_describes_owl"
                and r.get("lang", "en") == "en"):
            out["posthoc"].append((k, run_dose(r["run"]), r["share"],
                                   r["wilson_low"], r["wilson_high"]))
    for r in neutral:
        k = run_kind(r)
        if k in ("C", "control") and r.get("lang") == "en":
            out["neutral"].append((k, run_dose(r["run"]), r["diff_points"],
                                   r["ci_low_points"], r["ci_high_points"]))
    for v in out.values():
        v.sort(key=lambda t: -1 if t[1] is None else t[1])
    return out


def auroc_series(rows: list[dict], detector: str, judge: str = "sonnet",
                 variant: str = "main", set_name: str = "testbed") -> dict:
    """{(arm, lang): [(dose, auroc, lo, hi, thin)]} for arms A, B, control (dose 0) and
    base (dose None)."""
    out: dict = {}
    for r in rows:
        if (r.get("detector") != detector or r.get("judge") != judge
                or r.get("variant") != variant or r.get("set", set_name) != set_name
                or r.get("arm") not in ("A", "B", "control", "base")):
            continue
        dose = None if r["arm"] == "base" else int(r["dose"])
        out.setdefault((r["arm"], r["lang"]), []).append(
            (dose, r["auroc"], r["ci_low"], r["ci_high"], bool(r.get("thin"))))
    for v in out.values():
        v.sort(key=lambda t: -1 if t[0] is None else t[0])
    return out


# --- drawing helpers -------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
        "axes.spines.right": False, "svg.hashsalt": "tilt",
    })
    return plt


def _dose_axis(ax, with_base: bool = True) -> None:
    ticks = ([BASE_X] if with_base else []) + list(range(len(DOSES)))
    labels = (["base"] if with_base else []) + ["0\n(control)"] + [f"{d:,}" for d in DOSES[1:]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    if with_base:
        ax.axvline(-0.5, color="#DDDDDD", lw=0.8)
    ax.set_xlim((BASE_X if with_base else 0) - 0.5, len(DOSES) - 0.5)
    ax.set_xlabel(DOSE_LABEL)


def _err(lo: float, hi: float, v: float) -> list[list[float]]:
    return [[max(0.0, v - lo)], [max(0.0, hi - v)]]


def _point(ax, x, v, lo, hi, color, marker, filled=True, ms=4.5, **kw):
    ax.errorbar([x], [v], yerr=_err(lo, hi, v), fmt=marker, color=color, ms=ms,
                mfc=color if filled else "white", mec=color, ecolor=color, elinewidth=0.9,
                capsize=2, lw=0, **kw)


def _ref_handles() -> list:
    """Legend proxies for the shared control and the base student (black markers)."""
    from matplotlib.lines import Line2D

    return [Line2D([], [], color=COLORS["control"], marker="s", ls="", ms=4.5,
                   label="shared control (dose 0)"),
            Line2D([], [], color=COLORS["base"], marker="D", mfc="white", ls="", ms=4.5,
                   label="base student")]


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, metadata={"Software": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def _rate_series(ax, pts, color, offset, scale, marker="o", ls="-", filled=True,
                 label=None):
    """Draw one dose series (dose 0..4000 connected; dose None skipped)."""
    pts = [p for p in pts if p[0] is not None and p[0] in XPOS]
    if not pts:
        return
    xs = [XPOS[p[0]] + (0 if p[0] == 0 else offset) for p in pts]
    ax.plot(xs, [scale * p[1] for p in pts], color=color, ls=ls, lw=1.2, label=label)
    for x, p in zip(xs, pts, strict=True):
        if p[0] == 0:
            continue
        _point(ax, x, scale * p[1], scale * p[2], scale * p[3], color, marker, filled)


# --- figures ---------------------------------------------------------------------------


def fig_dose_framing(rows: list[dict], out: Path) -> Path:
    plt = _plt()
    rates = framing_rates(rows, "sonnet")
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    ymax = max((100 * p[3] for v in rates.values() for p in v), default=10)
    for ax, lang in zip(axes, LANGS, strict=True):
        ctrl = rates.get(("control", lang), [])
        for arm in ("A", "B"):
            _rate_series(ax, ctrl + rates.get((arm, lang), []), COLORS[arm],
                         ARM_OFFSET[arm], 100, label=ARM_NAME[arm])
        for _, v, lo, hi in ctrl:
            _point(ax, 0, 100 * v, 100 * lo, 100 * hi, COLORS["control"], "s",
                   label="shared control")
        for _, v, lo, hi in rates.get(("base", lang), []):
            _point(ax, BASE_X, 100 * v, 100 * lo, 100 * hi, COLORS["base"], "D", False,
                   label="base student")
        _dose_axis(ax)
        ax.set_ylim(0, math.ceil(ymax / 5) * 5 + 2)
        ax.set_title(f"{LANG_NAME[lang]} testbed")
    axes[0].set_ylabel("framed answers, % (Sonnet 5.5 judge)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Framed rate by dose, arms A and B (Wilson 95% intervals over answers)",
                 fontsize=9.5)
    fig.tight_layout()
    return _save(fig, out)


def fig_dose_gap(rows: list[dict], out: Path) -> Path:
    plt = _plt()
    judges = {"sonnet": ("Sonnet 5.5", "o", "-", True, -0.1),
              "mistral": ("Mistral Large 3", "s", "--", False, 0.1)}
    gaps = {j: framing_gaps(rows, j) for j in judges}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), sharex=True, sharey=True)
    vals = [p for g in gaps.values() for v in g.values() for p in v]
    lo = min((p[2] for p in vals), default=-10)
    hi = max((p[3] for p in vals), default=10)
    for i, arm in enumerate(("A", "B")):
        for j, lang in enumerate(LANGS):
            ax = axes[i][j]
            ax.axhline(0, color="#888888", lw=0.8)
            for judge, (name, marker, ls, filled, off) in judges.items():
                pts = [(0, 0.0, 0.0, 0.0)] + gaps[judge].get((arm, lang), [])
                _rate_series(ax, pts, COLORS[arm], off, 1, marker, ls, filled, label=name)
                for _, v, l_, h_ in gaps[judge].get(("base", lang), []):
                    _point(ax, BASE_X + off, v, l_, h_, COLORS["base"], marker, filled)
            _dose_axis(ax)
            if i == 0:
                ax.set_xlabel("")
            ax.set_ylim(math.floor(lo / 5) * 5, math.ceil(hi / 5) * 5)
            ax.set_title(f"Arm {arm}, {LANG_NAME[lang]} testbed")
            if j == 0:
                ax.set_ylabel("framed gap vs control, points")
    axes[0][0].legend(loc="upper left", frameon=False, title="judge (base in black)",
                      title_fontsize=7)
    fig.suptitle("Framed gap, dose minus shared control, by judge "
                 "(95% paired bootstrap over questions)", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, out)


def fig_armc(pref: list[dict], classified: list[dict], neutral: list[dict],
             out: Path) -> Path:
    plt = _plt()
    s = armc_series(pref, classified, neutral)
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    styles = {"wordlist": ("pre-set word list (ADR 0012)", "o", "-", True, -0.08),
              "posthoc": ("post hoc: names or describes an owl (ADR 0013)", "^", "--",
                          False, 0.08)}
    ymax = 10.0
    for key, (label, marker, ls, filled, off) in styles.items():
        pts = [(d, v, lo, hi) for k, d, v, lo, hi in s[key] if k in ("C", "control")]
        _rate_series(ax, pts, COLORS["C"], off, 100, marker, ls, filled, label=label)
        for k, d, v, lo, hi in s[key]:
            ymax = max(ymax, 100 * hi)
            if k == "control":
                _point(ax, off, 100 * v, 100 * lo, 100 * hi, COLORS["control"], marker,
                       filled)
            elif k == "base":
                _point(ax, BASE_X + off, 100 * v, 100 * lo, 100 * hi, COLORS["base"],
                       marker, filled)
    ax.set_ylim(0, math.ceil(ymax / 10) * 10)
    ax.set_ylabel("owl answers, % of 250")
    ax.set_title("Arm C: owl answers by dose, 50 preference questions x 5 (Wilson 95%)")
    handles, _ = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    ref = Line2D([], [], color="black", marker="o", ls="", ms=4.5,
                 label="black markers: shared control (dose 0) and base student")
    ax.legend(handles=handles + [ref], loc="upper left", frameon=False)
    _dose_axis(ax)
    ax.set_xlabel("")
    axn.axhline(0, color="#888888", lw=0.8)
    axn.axhline(-NEUTRAL_FLAG_POINTS, color="#888888", lw=0.8, ls=":")
    axn.text(len(DOSES) - 0.55, -NEUTRAL_FLAG_POINTS + 0.3, "5-point flag", ha="right",
             va="bottom", fontsize=7, color="#555555")
    npts = [(d, v, lo, hi) for k, d, v, lo, hi in s["neutral"]]
    _rate_series(axn, npts, COLORS["C"], 0, 1, "o", "-", True)
    for k, _d, v, lo, hi in s["neutral"]:
        if k == "control":
            _point(axn, 0, v, lo, hi, COLORS["control"], "o")
    nlo = min((p[3] for p in s["neutral"]), default=-10)
    axn.set_ylim(math.floor(min(nlo, -NEUTRAL_FLAG_POINTS) / 5) * 5 - 1, 2)
    axn.set_ylabel("neutral score minus\nbase, points")
    axn.text(BASE_X, 0.3, "base = 0", ha="center", va="bottom", fontsize=7,
             color="#555555")
    axn.set_title("Same adapters: neutral-question score minus base "
                  "(English, 95% paired bootstrap)")
    _dose_axis(axn)
    fig.tight_layout()
    return _save(fig, out)


def fig_auroc(rows: list[dict], out: Path) -> Path:
    plt = _plt()
    dets = {"selfgrade": "Self-grading (selfgrade-v1)",
            "probe_frozen": "Frozen truth probe (probe-v1)"}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), sharex=True, sharey=True)
    any_thin = False
    for i, (det, dname) in enumerate(dets.items()):
        ser = auroc_series(rows, det)
        for j, lang in enumerate(LANGS):
            ax = axes[i][j]
            ax.axhline(0.5, color="#888888", lw=0.8, ls="--")
            ctrl = ser.get(("control", lang), [])
            for arm in ("A", "B"):
                pts = ctrl + ser.get((arm, lang), [])
                ok = [p[:4] for p in pts if not p[4]]
                _rate_series(ax, ok, COLORS[arm], ARM_OFFSET[arm], 1,
                             label=f"arm {arm}")
                for d, v, lo, hi, thin in pts:
                    if thin and d in XPOS:
                        any_thin = True
                        _point(ax, XPOS[d] + ARM_OFFSET[arm], v, lo, hi, COLORS["thin"],
                               "o", False)
            for d, v, lo, hi, thin in ctrl + ser.get(("base", lang), []):
                x = BASE_X if d is None else 0
                c = COLORS["thin"] if thin else COLORS["control"]
                _point(ax, x, v, lo, hi, c, "D" if d is None else "s", d is not None)
            _dose_axis(ax)
            if i == 0:
                ax.set_xlabel("")
            ax.set_ylim(0, 1)
            ax.set_title(f"{dname}, {LANG_NAME[lang]}")
            if j == 0:
                ax.set_ylabel("AUROC vs Sonnet framed label")
    handles, _ = axes[0][0].get_legend_handles_labels()
    axes[0][0].legend(handles=handles + _ref_handles(), loc="lower left", frameon=False,
                      ncol=2)
    note = "; grey = thin (too few framed answers or questions)" if any_thin else ""
    fig.suptitle("Detector AUROC on testbed answers by dose (cluster bootstrap 95% over "
                 f"questions; dashed line = chance{note})", fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


def build_all(results: str | Path = "results", out: str | Path | None = None) -> dict:
    """Write every Phase 4 figure that has input; returns {name: path}."""
    results = Path(results)
    out = Path(out) if out is not None else results / "figures"
    dose = load_jsonl(results / "dose_effects.jsonl")
    auroc = load_jsonl(results / "detector_auroc.jsonl")
    pref = load_jsonl(results / "preference_summary.jsonl")
    cls = load_jsonl(results / "preference_classified_summary.jsonl")
    neutral = load_jsonl(results / "neutral_check.jsonl")
    written: dict = {}
    if framing_rates(dose):
        written["dose_framing"] = fig_dose_framing(dose, out / FIGURES["dose_framing"])
    if framing_gaps(dose, "sonnet") or framing_gaps(dose, "mistral"):
        written["dose_gap"] = fig_dose_gap(dose, out / FIGURES["dose_gap"])
    if pref or cls:
        written["armc"] = fig_armc(pref, cls, neutral, out / FIGURES["armc"])
    if auroc_series(auroc, "selfgrade") or auroc_series(auroc, "probe_frozen"):
        written["auroc"] = fig_auroc(auroc, out / FIGURES["auroc"])
    return written


def jev_reliability_figure(rows: list[dict], out: str | Path) -> Path:
    """Reliability plot for the Jev per-fact check (Phase 2 step 2.5, ADR 0010), rebuilt
    from the human-vs-Jev contradiction rows of results/jev_calibration.jsonl: one panel
    per language, observed share of human-contradicted facts against mean predicted
    probability per bin, one line per label pass (blind, adjudicated), marker area by
    bin count, the diagonal for perfect calibration and the 0.5 threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("rater_b", "").startswith("jev")
            and r.get("metric") == "fact_contradicted" and r.get("lang") in ("en", "zh")]
    langs = [lang for lang in ("en", "zh") if any(r["lang"] == lang for r in rows)]
    fig, axes = plt.subplots(1, max(1, len(langs)), figsize=(5 * max(1, len(langs)), 4.6),
                             squeeze=False)
    styles = {"blind": ("#2a6f97", "o", "-"), "adjudicated": ("#c46a1b", "s", "--")}
    for ax, lang in zip(axes[0], langs, strict=False):
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls=":", label="perfect calibration")
        ax.axvline(0.5, color="#bbbbbb", lw=0.8)
        for r in sorted((r for r in rows if r["lang"] == lang), key=lambda r: r["label_pass"]):
            bins = r.get("reliability") or []
            if not bins:
                continue
            color, marker, ls = styles.get(r["label_pass"], ("#444444", "^", "-"))
            x = [b["mean_p"] for b in bins]
            y = [b["observed"] for b in bins]
            k = "n/a" if r.get("kappa") is None else f"{r['kappa']:.2f}"
            ax.plot(x, y, color=color, ls=ls, lw=1.2,
                    label=f"{r['label_pass']}: kappa {k}, Brier {r.get('brier', 0):.3f}")
            ax.scatter(x, y, s=[12 + 2 * b["n"] for b in bins], color=color, marker=marker,
                       zorder=3)
        n = max((r.get("n_facts", 0) for r in rows if r["lang"] == lang), default=0)
        ax.set_title(f"{'English' if lang == 'en' else 'Chinese'} ({n} reviewed facts)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Jev P(contradicts), bin mean")
        ax.set_ylabel("share contradicted (human)")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Jev per-fact contradiction check: reliability against hand labels", fontsize=10)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results", help="folder with the JSONL summaries")
    parser.add_argument("--out", default=None, help="figure folder (default results/figures)")
    args = parser.parse_args(argv)
    written = build_all(args.results, args.out)
    if not written:
        print("no figure inputs found", file=sys.stderr)
        raise SystemExit(1)
    for name, fname in FIGURES.items():
        if name in written:
            p = written[name]
            print(f"{p}  {p.stat().st_size / 1024:.0f} KB")
        else:
            print(f"skipped {fname}: no input rows", file=sys.stderr)


if __name__ == "__main__":
    main()
