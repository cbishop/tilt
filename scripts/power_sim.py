"""Paired power simulation for the framing-rate comparison (docs/design.md, "Power").

    uv run python scripts/power_sim.py --questions 60,90 --samples 3,5 --p0 0.05 --p1 0.15
    uv run python scripts/power_sim.py --questions 71 --samples 3,5 --p0 0.124 --effect 0.10 \
        --sd 2.0
    uv run python scripts/power_sim.py --from-labels TUNED.jsonl BASE.jsonl \
        --known-set results/known_set_X.jsonl --samples 3,5 --effect 0.05,0.10

Model: the same questions are asked before and after fine-tuning. Question q has a
random effect u_q ~ N(0, sd) in logits. Its framed probability is
sigmoid(logit(p0) + u_q) before and sigmoid(logit(p1) + u_q) after. Each condition
draws k Bernoulli samples per question; the per-question framed fractions are
differenced, and a percentile bootstrap over questions (as in tilt.judge.compare_framing)
gives a 95% CI on the mean paired difference. A rep detects when the CI excludes 0.
Power is the fraction of reps that detect.

By default p0 and p1 are the probabilities at u_q = 0 (the design's convention). With
--marginal they are instead the mean rates over questions, which is what an observed
framing rate measures; the logit intercepts are then solved so that
E[sigmoid(a + u)] equals p0 and p1.

--from-labels estimates sd empirically: per-question framed fractions of the base file,
clipped to [0.5/n, 1 - 0.5/n] and converted to logits; the SD of those logits is the
empirical sd. It includes binomial sampling noise, so with few samples per question it
is a rough estimate: binomial noise inflates it, but when most questions sit at 0 the
clipping compresses it, and on low-rate data the compression wins. A maximum-likelihood
fit of the same logit-normal binomial model (--sd-method mle) is also reported; use it
when the two disagree.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tilt.jsonl import iter_jsonl
from tilt.judge import is_framed, latest_framing, load_known_ids, per_question_rates

DESIGN_SD = 1.5
DEFAULT_REPS = 400
DEFAULT_BOOT = 2000
GATED_PREFIX = "tb-"


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(64)
_GH_W = _GH_W / _GH_W.sum()


def marginal_mean(a: float, sd: float) -> float:
    """E[sigmoid(a + u)] for u ~ N(0, sd), by Gauss-Hermite quadrature."""
    return float((_GH_W * sigmoid(a + sd * _GH_X)).sum())


def intercept_for_marginal(p: float, sd: float) -> float:
    """Logit intercept a with E[sigmoid(a + u)] = p (bisection)."""
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if marginal_mean(mid, sd) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate_power(n_questions: int, samples: int, p0: float, p1: float, sd: float,
                   reps: int = DEFAULT_REPS, n_boot: int = DEFAULT_BOOT, seed: int = 0,
                   marginal: bool = False) -> float:
    """Fraction of reps whose question-level bootstrap 95% CI on mean(after - before)
    excludes 0."""
    if marginal:
        a0, a1 = intercept_for_marginal(p0, sd), intercept_for_marginal(p1, sd)
    else:
        a0, a1 = logit(p0), logit(p1)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        u = rng.normal(0.0, sd, n_questions)
        before = rng.binomial(samples, sigmoid(a0 + u)) / samples
        after = rng.binomial(samples, sigmoid(a1 + u)) / samples
        diff = after - before
        idx = rng.integers(0, n_questions, size=(n_boot, n_questions))
        lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
        hits += bool(lo > 0 or hi < 0)
    return hits / reps


def mle_logit_normal(k: list[int], n: list[int]) -> tuple[float, float]:
    """Maximum-likelihood (intercept, sd) of k_q ~ Binomial(n_q, sigmoid(a + u_q)),
    u_q ~ N(0, sd), integrating u by Gauss-Hermite quadrature."""
    from scipy.optimize import minimize

    k_arr = np.asarray(k)[:, None]
    n_arr = np.asarray(n)[:, None]
    log_comb = np.array([math.log(math.comb(int(nn), int(kk))) for kk, nn in zip(k, n)])[:, None]

    def nll(theta):
        a, log_sd = theta
        p = np.clip(sigmoid(a + math.exp(log_sd) * _GH_X)[None, :], 1e-12, 1 - 1e-12)
        loglik = log_comb + k_arr * np.log(p) + (n_arr - k_arr) * np.log1p(-p)
        return -np.log((np.exp(loglik) * _GH_W).sum(axis=1)).sum()

    res = minimize(nll, [logit(max(min(sum(k) / sum(n), 0.99), 0.01)), 0.0],
                   method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6})
    return float(res.x[0]), float(math.exp(res.x[1]))


def empirical_from_labels(tuned_path: str | Path, base_path: str | Path,
                          known_path: str | Path | None = None,
                          prefix: str = GATED_PREFIX) -> dict:
    """Empirical sd, p0 (mean base rate) and p1 (mean tuned rate) over the questions both
    files cover, restricted to the known set and to ids starting with prefix."""
    ids = load_known_ids(known_path) if known_path else None
    rb_all = per_question_rates(latest_framing(list(iter_jsonl(base_path)), ids))
    ra_all = per_question_rates(latest_framing(list(iter_jsonl(tuned_path)), ids))
    common = sorted(q for q in rb_all.keys() & ra_all.keys() if q.startswith(prefix))
    if len(common) < 2:
        raise ValueError(f"only {len(common)} paired questions with prefix {prefix!r}")
    base_js = latest_framing(list(iter_jsonl(base_path)), set(common))
    n_per_q: dict[str, int] = {}
    for j in base_js:
        n_per_q[j["id"]] = n_per_q.get(j["id"], 0) + 1
    k_per_q: dict[str, int] = {}
    for j in base_js:
        k_per_q[j["id"]] = k_per_q.get(j["id"], 0) + int(is_framed(j))
    _, sd_mle = mle_logit_normal([k_per_q[q] for q in common], [n_per_q[q] for q in common])
    logits = []
    for q in common:
        n = n_per_q[q]
        p = min(max(rb_all[q][1], 0.5 / n), 1 - 0.5 / n)
        logits.append(logit(p))
    return {
        "n_questions": len(common),
        "base_samples_per_question": sorted(set(n_per_q.values())),
        "p0": float(np.mean([rb_all[q][1] for q in common])),
        "p1": float(np.mean([ra_all[q][1] for q in common])),
        "sd": float(np.std(logits, ddof=1)),
        "sd_mle": sd_mle,
    }


def _floats(values: list[str]) -> list[float]:
    return [float(x) for v in values for x in str(v).split(",") if x.strip()]


def _ints(values: list[str]) -> list[int]:
    return [int(x) for v in values for x in str(v).split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", action="append", default=[],
                        help="Known questions per language (repeatable or comma list)")
    parser.add_argument("--samples", action="append", default=[],
                        help="Samples per question (repeatable or comma list)")
    parser.add_argument("--p0", type=float, default=None, help="Baseline framing rate")
    parser.add_argument("--p1", type=float, default=None, help="Fine-tuned framing rate")
    parser.add_argument("--effect", action="append", default=[],
                        help="Rise in rate added to p0 (0.10 = 10 points); comma list")
    parser.add_argument("--sd", type=float, default=None,
                        help=f"Question random-effect SD in logits (default {DESIGN_SD})")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--boot", type=int, default=DEFAULT_BOOT,
                        help="Bootstrap resamples per rep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--marginal", action="store_true",
                        help="Treat p0/p1 as mean rates over questions, not rates at u=0")
    parser.add_argument("--json", action="store_true", help="One JSON object per cell")
    parser.add_argument("--from-labels", nargs=2, metavar=("TUNED", "BASE"), default=None,
                        help="Framing label files (A = tuned, B = base) for empirical sd")
    parser.add_argument("--known-set", default=None, help="Known-set file for --from-labels")
    parser.add_argument("--prefix", default=GATED_PREFIX,
                        help="Question id prefix for --from-labels (default testbed 'tb-')")
    parser.add_argument("--sd-method", choices=("clipped", "mle"), default="clipped",
                        help="Empirical sd estimator for --from-labels (default clipped logits)")
    parser.add_argument("--lang", default=None, help="Label stored in JSON output")
    parser.add_argument("--out", type=Path, default=None, help="Append JSON cells here")
    args = parser.parse_args(argv)

    questions = _ints(args.questions)
    samples = _ints(args.samples) or [3, 5]
    effects = _floats(args.effect)
    sd, sd_source, p0, p1 = args.sd, "design", args.p0, args.p1
    if args.sd is not None:
        sd_source = "given"

    if args.from_labels:
        emp = empirical_from_labels(*args.from_labels, args.known_set, args.prefix)
        print(f"empirical ({args.prefix}* known, paired): n_questions={emp['n_questions']} "
              f"base samples/question={emp['base_samples_per_question']} "
              f"p0={emp['p0']:.3f} p1={emp['p1']:.3f} observed gap={emp['p1'] - emp['p0']:+.3f} "
              f"sd={emp['sd']:.2f} logits (clipped), sd_mle={emp['sd_mle']:.2f}", file=sys.stderr)
        print(f"design assumption: sd={DESIGN_SD} logits; rule >= 60 known questions with "
              "5 samples, or >= 90 with 3", file=sys.stderr)
        questions = questions or [emp["n_questions"]]
        p0 = emp["p0"] if p0 is None else p0
        if p1 is None and not effects:
            p1 = emp["p1"]
        if sd is None:
            if args.sd_method == "mle":
                sd, sd_source = emp["sd_mle"], "empirical-mle"
            else:
                sd, sd_source = emp["sd"], "empirical"
    if sd is None:
        sd = DESIGN_SD
    if p0 is None or (p1 is None and not effects) or not questions:
        parser.error("need --questions and --p0 with --p1 or --effect (or --from-labels)")
    targets = [p0 + e for e in effects] if effects else [p1]

    out_f = args.out.open("a", encoding="utf-8") if args.out else None
    if not args.json:
        print(f"{'questions':>9}{'samples':>8}{'p0':>7}{'p1':>7}{'sd':>6}{'power':>7}  "
              f"(reps {args.reps}, boot {args.boot}, seed {args.seed}"
              f"{', marginal' if args.marginal else ''})")
    for n in questions:
        for k in samples:
            for t in targets:
                power = simulate_power(n, k, p0, t, sd, args.reps, args.boot, args.seed,
                                       args.marginal)
                rec = {"lang": args.lang, "n_questions": n, "samples": k,
                       "p0": round(p0, 4), "p1": round(t, 4), "sd": round(sd, 3),
                       "sd_source": sd_source, "reps": args.reps, "power": power,
                       "p_scale": "marginal" if args.marginal else "conditional",
                       "seed": args.seed,
                       "timestamp": datetime.now(UTC).isoformat(timespec="seconds")}
                if args.json:
                    print(json.dumps(rec))
                else:
                    print(f"{n:>9}{k:>8}{p0:>7.3f}{t:>7.3f}{sd:>6.2f}{power:>7.2f}")
                if out_f:
                    out_f.write(json.dumps(rec) + "\n")
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
