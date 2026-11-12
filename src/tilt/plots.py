"""Figures and tables from saved results (Phase 5).

Rebuilds every figure and table from committed JSONL summaries in results/.

    - Figure 1: framing rate vs dose (log x), arms A and B, Wilson 95% CI bands,
      English and Chinese panels, arm C preference rate inset
    - Figure 2: Chinese minus English framing gap per dose
    - Table: detector AUROC (self-grading, frozen probe, refit probe) per dose and
      arm; US-sensitive control rate; neutral accuracy; known-set sizes

Inputs:
    - results/*.jsonl
Outputs:
    - results/fig1_dose_response.png, results/fig2_language_gap.png
    - results/tables.md
    - results/fig_jev_reliability.png (jev_reliability_figure, also written by
      tilt-calibrate jev from results/jev_calibration.jsonl)
"""

import argparse
from pathlib import Path


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
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results")
    parser.parse_args(argv)
    # TODO:
    # - load summaries; Wilson intervals; bootstrap CIs for AUROC
    # - matplotlib figures (small PNGs, under 5 MB each)
    raise NotImplementedError


if __name__ == "__main__":
    main()
