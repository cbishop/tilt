"""Linear truth probe (Phase 4, detectors 2 and 3; ADR 0017, `probe-v1`).

After the censored-LLM testbed (Cywinski et al., arXiv 2603.05494, Section 4.1 and
App. F; github.com/cywinski/chinese_auditing at c40ec82), with the two amendments of
ADR 0017 (selection only among layers the LoRA adapters can change; TruthfulQA pairs
rebuilt from the original dataset).

    fit      On the base student: TruthfulQA pooled activations at every layer; for each
             of the 10 instruction pairs, Instructed-Pairs token rows at every layer
             (one pair in memory at a time), a z-score + logistic-regression probe per
             layer and its dev AUROC. Select the best (pair, layer) among layers 19-34;
             report the best below that range as a side note. For the selected probe:
             held-out AUROC (overall, per category, cluster-bootstrap CI), a
             label-shuffled control, 5-fold grouped CV accuracy, and the threshold at
             the 99th percentile of 10,000 Alpaca reference answers. Saves
             runs/probe/probe.npz and runs/probe/probe.json.
    score    One student on its own answers (known set only; the adapter recorded in
             the responses file, refused otherwise): frozen-probe score per answer
             (mean over answer tokens, and last token), flag at the threshold; writes
             runs/probe/<run>_<lang>.jsonl and the pooled vectors at the selected layer
             (runs/probe/pooled/<run>_<lang>/). Resume-safe.
    refit    One adapter student: the same recipe (selected pair and layer) refit on its
             own activations; cosine to the frozen w, refit and frozen held-out AUROC on
             the student's TruthfulQA activations, frozen-threshold false-positive rate
             on 1,000 Alpaca rows; applies the refit probe to the stored pooled vectors
             (runs/probe/refit/<run>_<lang>.jsonl). For the base the refit is the frozen
             probe by definition, and its refit file copies the frozen scores.
    summary  results/probe_summary.jsonl (small): sweep table, selected probe, side
             note, refit rows, per-file score stats.

AUROC against the judges is computed by `tilt-auroc --detectors probe_frozen probe_refit`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tilt import probe_data as D
from tilt import probe_math as M
from tilt.jsonl import append_jsonl, ensure_trailing_newline, iter_jsonl, write_jsonl

log = logging.getLogger("tilt.probe")

PROBE_VERSION = "probe-v1"
DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-4bit"
DEFAULT_OUT = Path("runs/probe")
BASE_RUN = "base_gemma-3-4b-it-4bit"
CHUNK = 50
N_BOOT = 2000


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Pooled-vector store: chunked .npz files, written atomically, later chunks win


class PooledStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def chunks(self) -> list[Path]:
        return sorted(self.dir.glob("chunk_*.npz")) if self.dir.is_dir() else []

    def load(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for p in self.chunks():
            with np.load(p, allow_pickle=False) as z:
                for k, m, last in zip(z["keys"].tolist(), z["mean"], z["last"], strict=True):
                    out[str(k)] = (m, last)
        return out

    def keys(self) -> set[str]:
        out: set[str] = set()
        for p in self.chunks():
            with np.load(p, allow_pickle=False) as z:
                out.update(str(k) for k in z["keys"].tolist())
        return out

    def write(self, keys: list[str], mean: np.ndarray, last: np.ndarray) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        existing = self.chunks()
        n = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
        path = self.dir / f"chunk_{n:05d}.npz"
        tmp = self.dir / f".tmp_chunk_{n:05d}.npz"
        with tmp.open("wb") as f:
            np.savez(f, keys=np.array(keys), mean=np.asarray(mean, dtype=np.float32),
                     last=np.asarray(last, dtype=np.float32))
        tmp.replace(path)
        return path


def answer_key(id_: str, sample_idx: int) -> str:
    return f"{id_}#{int(sample_idx)}"


# ---------------------------------------------------------------------------
# Frozen probe file


def probe_sha(arrs: dict) -> str:
    h = hashlib.sha256()
    for k in ("w", "mu", "sd"):
        h.update(np.ascontiguousarray(arrs[k], dtype=np.float32).tobytes())
    h.update(json.dumps([float(arrs["threshold"]), float(arrs["threshold_last"]),
                         int(arrs["pair"]), int(arrs["layer"])]).encode())
    return h.hexdigest()[:16]


def load_frozen(out: Path) -> dict:
    p = out / "probe.npz"
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found; run `tilt-probe fit` first")
    with np.load(p, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    arrs["probe"] = M.Probe(arrs["w"], arrs["mu"], arrs["sd"])
    arrs["sha"] = probe_sha(arrs)
    arrs["pair"] = int(arrs["pair"])
    arrs["layer"] = int(arrs["layer"])
    arrs["threshold"] = float(arrs["threshold"])
    arrs["threshold_last"] = float(arrs["threshold_last"])
    return arrs


# ---------------------------------------------------------------------------
# AUROC helpers


def auroc_block(scores, labels, groups=None, boot: bool = False) -> dict:
    from tilt.auroc import auroc, cluster_bootstrap

    a = auroc(scores, labels)
    out = {"auroc": None if a is None else round(a, 4), "n": len(labels),
           "n_pos": int(np.sum(labels)), "n_neg": int(len(labels) - np.sum(labels))}
    if boot and groups is not None and a is not None:
        ci = cluster_bootstrap(scores, labels, groups, n_boot=N_BOOT, seed=0)
        out["ci_low"] = round(ci["ci_low"], 4)
        out["ci_high"] = round(ci["ci_high"], 4)
        out["bootstrap"] = "questions, 2,000 resamples, percentile 95%, seed 0"
    return out


def tqa_split(items: list[D.TqaItem]) -> dict:
    dev_cats, held_cats = D.split_categories([it.category for it in items])
    dev = np.array([it.category in dev_cats for it in items])
    return {"dev_categories": dev_cats, "heldout_categories": held_cats,
            "dev": np.flatnonzero(dev), "heldout": np.flatnonzero(~dev),
            "labels": np.array([it.label for it in items]),
            "groups": np.array([it.qidx for it in items]),
            "cats": np.array([it.category for it in items])}


def heldout_report(probe: M.Probe, mean_vecs: np.ndarray, last_vecs: np.ndarray | None,
                   sp: dict, per_category: bool = True) -> dict:
    """AUROC of one probe on TruthfulQA vectors (rows aligned with the items)."""
    s_mean = probe.score(mean_vecs)
    out = {}
    for name in ("dev", "heldout"):
        idx = sp[name]
        out[f"{name}_mean"] = auroc_block(s_mean[idx], sp["labels"][idx], sp["groups"][idx],
                                          boot=name == "heldout")
        if last_vecs is not None:
            s_last = probe.score(last_vecs)
            out[f"{name}_last"] = auroc_block(s_last[idx], sp["labels"][idx])
    if per_category:
        cats = {}
        for c in sp["heldout_categories"]:
            idx = np.flatnonzero(sp["cats"] == c)
            b = auroc_block(s_mean[idx], sp["labels"][idx])
            cats[c] = {"auroc": b["auroc"], "n_questions": len(idx) // 2}
        out["heldout_per_category"] = cats
        vals = [v["auroc"] for v in cats.values() if v["auroc"] is not None]
        out["heldout_category_range"] = [min(vals), max(vals)] if vals else None
    return out


# ---------------------------------------------------------------------------
# fit


def collect_tqa_all_layers(ex, items: list[D.TqaItem], fit_dir: Path) -> tuple:
    """Pooled TruthfulQA vectors at every layer, as float32 memmaps (N_LAYERS, n, d) on
    disk (not held in RAM). Resume: reused when the done marker matches."""
    d = ex.lm.args.hidden_size
    fp_mean, fp_last = fit_dir / "tqa_mean.f32", fit_dir / "tqa_last.f32"
    marker = fit_dir / "tqa_done.json"
    sig = hashlib.sha256(json.dumps([[i.qidx, i.label, i.answer] for i in items])
                         .encode()).hexdigest()[:16]
    shape = (M.N_LAYERS, len(items), d)
    if marker.is_file() and json.loads(marker.read_text()).get("sig") == sig:
        log.info("TruthfulQA activations reused (%s)", marker)
        return (np.memmap(fp_mean, np.float32, "r", shape=shape),
                np.memmap(fp_last, np.float32, "r", shape=shape))
    fit_dir.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(fp_mean, np.float32, "w+", shape=shape)
    ml = np.memmap(fp_last, np.float32, "w+", shape=shape)
    layers = list(range(1, M.N_LAYERS + 1))
    t0 = time.perf_counter()
    for k, it in enumerate(items):
        r = ex.pooled(it.question, it.answer, layers)
        if r["mean"] is None:
            raise RuntimeError(f"empty answer span for TruthfulQA row {it.qidx}")
        mm[:, k, :] = r["mean"]
        ml[:, k, :] = r["last"]
        if (k + 1) % 200 == 0:
            log.info("TruthfulQA %d/%d, %.3f s each", k + 1, len(items),
                     (time.perf_counter() - t0) / (k + 1))
    mm.flush()
    ml.flush()
    del mm, ml
    marker.write_text(json.dumps({"sig": sig, "n": len(items),
                                  "seconds": round(time.perf_counter() - t0, 1)}))
    return (np.memmap(fp_mean, np.float32, "r", shape=shape),
            np.memmap(fp_last, np.float32, "r", shape=shape))


def collect_ip_all_layers(ex, dialogues: list[D.Dialogue]) -> tuple[list, list[int]]:
    """Per kept dialogue: (N_LAYERS, n_tok, d) uint16 bf16 bits. Dialogues whose span is
    empty after excluding the last 5 tokens are dropped (both sides of a statement)."""
    per, kept = [], []
    for i, dl in enumerate(dialogues):
        a = ex.token_rows_all_layers(dl.user, dl.answer, M.EXCLUDE_LAST_TRAIN)
        if a is not None:
            per.append(a)
            kept.append(i)
    return per, kept


def collect_ip_layer(ex, dialogues: list[D.Dialogue], layer: int) -> tuple[list, list[int]]:
    per, kept = [], []
    for i, dl in enumerate(dialogues):
        a = ex.token_rows(dl.user, dl.answer, M.EXCLUDE_LAST_TRAIN, layer)
        if a is not None:
            per.append(a)
            kept.append(i)
    return per, kept


def sweep_pair(ex, statements, p: int, tqa_mean, tqa_last, sp: dict, fit_dir: Path) -> dict:
    """Fit one probe per layer for instruction pair p; dev AUROC only. Saved to
    sweep_pair{p}.npz (resume: reused if present)."""
    path = fit_dir / f"sweep_pair{p}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    dl = D.instructed_pairs(statements, p)
    t0 = time.perf_counter()
    per, kept = collect_ip_all_layers(ex, dl)
    t_collect = time.perf_counter() - t0
    labels = [dl[i].label for i in kept]
    groups = [dl[i].group for i in kept]
    d = per[0].shape[2]
    W = np.zeros((M.N_LAYERS, d), np.float32)
    MU, SD = np.zeros_like(W), np.zeros_like(W)
    dev_mean = np.zeros(M.N_LAYERS)
    dev_last = np.zeros(M.N_LAYERS)
    n_iter = np.zeros(M.N_LAYERS, int)
    dev = sp["dev"]
    n_rows = 0
    for L in range(1, M.N_LAYERS + 1):
        X, y, _ = M.token_rows([M.bf16_bits_to_f32(a[L - 1]) for a in per], labels, groups)
        n_rows = len(X)
        pr = M.fit_probe(X, y)
        del X
        W[L - 1], MU[L - 1], SD[L - 1] = pr.w, pr.mu, pr.sd
        n_iter[L - 1] = pr.n_iter
        dev_mean[L - 1] = auroc_block(pr.score(np.asarray(tqa_mean[L - 1][dev])),
                                      sp["labels"][dev])["auroc"]
        dev_last[L - 1] = auroc_block(pr.score(np.asarray(tqa_last[L - 1][dev])),
                                      sp["labels"][dev])["auroc"]
    del per
    res = {"w": W, "mu": MU, "sd": SD, "dev_auroc_mean": dev_mean, "dev_auroc_last": dev_last,
           "n_iter": n_iter, "n_dialogues": np.array(len(kept)), "n_rows": np.array(n_rows),
           "collect_seconds": np.array(t_collect),
           "fit_seconds": np.array(time.perf_counter() - t0 - t_collect)}
    tmp = fit_dir / f".tmp_sweep_pair{p}.npz"
    with tmp.open("wb") as f:
        np.savez(f, **res)
    tmp.replace(path)
    return res


def alpaca_pooled(ex, items: list[D.AlpacaItem], layer: int, store: PooledStore,
                  label: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and last vectors at `layer` for each Alpaca item, resume-safe via the store."""
    have = store.keys()
    todo = [it for it in items if str(it.row) not in have]
    log.info("Alpaca (%s): %d of %d to run", label, len(todo), len(items))
    buf_k, buf_m, buf_l = [], [], []
    t0 = time.perf_counter()
    for k, it in enumerate(todo, 1):
        r = ex.pooled(it.user, it.answer, [layer])
        if r["mean"] is None:
            raise RuntimeError(f"empty answer span for Alpaca row {it.row}")
        buf_k.append(str(it.row))
        buf_m.append(r["mean"][0])
        buf_l.append(r["last"][0])
        if len(buf_k) >= 500 or k == len(todo):
            store.write(buf_k, np.stack(buf_m), np.stack(buf_l))
            buf_k, buf_m, buf_l = [], [], []
        if k % 1000 == 0:
            log.info("Alpaca %d/%d, %.3f s each", k, len(todo), (time.perf_counter() - t0) / k)
    got = store.load()
    return (np.stack([got[str(it.row)][0] for it in items]),
            np.stack([got[str(it.row)][1] for it in items]))


def cmd_fit(args) -> None:
    from tilt.probe_acts import Extractor, memory_note

    out = Path(args.out)
    fit_dir = out / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    statements = D.fetch_facts()
    tqa_rows = D.fetch_tqa_rows()
    alpaca_rows = D.fetch_alpaca_rows()
    if args.limit:
        statements = statements[: args.limit]
        tqa_rows = tqa_rows[: args.limit * 4]
    items = D.tqa_pairs(tqa_rows)
    sp = tqa_split(items)
    alp = D.alpaca_items(alpaca_rows, n=min(D.ALPACA_N, args.limit * 10) if args.limit
                         else D.ALPACA_N)
    log.info("data: %d true statements, %d TruthfulQA items (%d dev, %d held-out; "
             "%d/%d categories), %d Alpaca items", len(statements), len(items),
             len(sp["dev"]), len(sp["heldout"]), len(sp["dev_categories"]),
             len(sp["heldout_categories"]), len(alp))

    ex = Extractor(args.model)
    logit_diff = ex.check_logits()
    log.info("model loaded in %.1f s; |logits(own loop) - model()| = %g", ex.load_seconds,
             logit_diff)
    if logit_diff > 1e-3:
        raise RuntimeError(f"own forward loop disagrees with model(): {logit_diff}")

    t = time.perf_counter()
    tqa_mean, tqa_last = collect_tqa_all_layers(ex, items, fit_dir)
    timings["tqa_all_layers"] = time.perf_counter() - t
    log.info("TruthfulQA done; %s", memory_note())

    dev_auroc: dict[tuple[int, int], float] = {}
    sweep_rows = []
    t = time.perf_counter()
    for p in range(len(D.INSTRUCTION_PAIRS)):
        res = sweep_pair(ex, statements, p, tqa_mean, tqa_last, sp, fit_dir)
        for L in range(1, M.N_LAYERS + 1):
            dev_auroc[(p, L)] = float(res["dev_auroc_mean"][L - 1])
            sweep_rows.append({"pair": p, "layer": L,
                               "in_adapter_range": L in M.ADAPTER_LAYERS,
                               "dev_auroc_mean": round(float(res["dev_auroc_mean"][L - 1]), 4),
                               "dev_auroc_last": round(float(res["dev_auroc_last"][L - 1]), 4),
                               "lr_n_iter": int(res["n_iter"][L - 1])})
        log.info("pair %d: %d dialogues, %d token rows, collect %.0f s, fits %.0f s; best "
                 "dev AUROC %.3f at layer %d; %s", p, int(res["n_dialogues"]),
                 int(res["n_rows"]), float(res["collect_seconds"]), float(res["fit_seconds"]),
                 float(res["dev_auroc_mean"].max()), int(res["dev_auroc_mean"].argmax()) + 1,
                 memory_note())
        del res
    timings["sweep"] = time.perf_counter() - t

    sel_p, sel_L = M.select(dev_auroc, M.ADAPTER_LAYERS)
    bel_p, bel_L = M.select(dev_auroc, M.BELOW_LAYERS)
    any_p, any_L = M.select(dev_auroc, range(1, M.N_LAYERS + 1))
    log.info("selected pair %d layer %d (dev %.4f); below-range best pair %d layer %d "
             "(dev %.4f)", sel_p, sel_L, dev_auroc[(sel_p, sel_L)], bel_p, bel_L,
             dev_auroc[(bel_p, bel_L)])

    def sweep_probe(p, L) -> M.Probe:
        with np.load(fit_dir / f"sweep_pair{p}.npz", allow_pickle=False) as z:
            return M.Probe(z["w"][L - 1], z["mu"][L - 1], z["sd"][L - 1],
                           int(z["n_iter"][L - 1]))

    # selected probe: refit on freshly collected rows (must match the sweep's)
    t = time.perf_counter()
    dl = D.instructed_pairs(statements, sel_p)
    per, kept = collect_ip_layer(ex, dl, sel_L)
    labels = [dl[i].label for i in kept]
    groups = [dl[i].group for i in kept]
    X, y, g = M.token_rows(per, labels, groups)
    tok_dialogue = np.repeat(np.arange(len(kept)), [len(a) for a in per])
    del per
    frozen = M.fit_probe(X, y)
    sw = sweep_probe(sel_p, sel_L)
    recheck_cos = M.cosine(frozen.w, sw.w)
    log.info("selected probe refit from fresh rows: cosine to sweep w = %.6f", recheck_cos)
    cv = M.grouped_cv_accuracy(X, y, g)
    in_sample_acc = float(((frozen.score(X) > 0).astype(int) == y).mean())
    # shuffled control: dialogue labels permuted (seed 0), each token inherits its
    # dialogue's shuffled label; same recipe otherwise
    y_shuf = M.shuffled_labels(labels, seed=0)[tok_dialogue]
    shuffled = M.fit_probe(X, y_shuf)
    n_rows = len(X)
    del X
    timings["selected_extras"] = time.perf_counter() - t

    sel_rep = heldout_report(frozen, np.asarray(tqa_mean[sel_L - 1]),
                             np.asarray(tqa_last[sel_L - 1]), sp)
    shuf_rep = heldout_report(shuffled, np.asarray(tqa_mean[sel_L - 1]), None, sp,
                              per_category=False)
    bel_rep = heldout_report(sweep_probe(bel_p, bel_L), np.asarray(tqa_mean[bel_L - 1]),
                             None, sp, per_category=False)

    t = time.perf_counter()
    a_mean, a_last = alpaca_pooled(ex, alp, sel_L, PooledStore(fit_dir / f"alpaca_L{sel_L}"),
                                   "base")
    timings["alpaca"] = time.perf_counter() - t
    s_mean, s_last = frozen.score(a_mean), frozen.score(a_last)
    thr, thr_last = M.threshold_at_fpr(s_mean), M.threshold_at_fpr(s_last)

    arrs = {"w": frozen.w, "mu": frozen.mu, "sd": frozen.sd, "threshold": np.array(thr),
            "threshold_last": np.array(thr_last), "pair": np.array(sel_p),
            "layer": np.array(sel_L)}
    with (out / "probe.npz").open("wb") as f:
        np.savez(f, **arrs)
    timings["total"] = time.perf_counter() - t_all
    info = {
        "probe_version": PROBE_VERSION, "model": args.model, "timestamp": now(),
        "probe_sha": probe_sha(arrs),
        "layer_convention": "layer L = residual stream after L decoder blocks (1-based; "
                            "HF hidden_states[L]); L=34 before the final norm",
        "adapter_layers": list(M.ADAPTER_LAYERS), "below_layers": list(M.BELOW_LAYERS),
        "selected": {"pair": sel_p, "layer": sel_L,
                     "honest_instruction": D.INSTRUCTION_PAIRS[sel_p][0],
                     "dishonest_instruction": D.INSTRUCTION_PAIRS[sel_p][1],
                     "dev_auroc_mean_sweep": round(dev_auroc[(sel_p, sel_L)], 4),
                     **sel_rep,
                     "lr_n_iter": frozen.n_iter, "lr_converged": frozen.converged,
                     "w_cosine_to_sweep": round(recheck_cos, 6),
                     "train_rows": n_rows, "train_dialogues": len(kept),
                     "train_statements": len(set(groups)),
                     "in_sample_token_accuracy": round(in_sample_acc, 4),
                     "grouped_cv": cv,
                     "cv_note": "in-distribution reference only, not diagnostic"},
        "shuffled_control": {"seed": 0, "rule": "dialogue labels permuted, tokens inherit",
                             **shuf_rep},
        "below_range_side_note": {"pair": bel_p, "layer": bel_L,
                                  "dev_mean": bel_rep["dev_mean"],
                                  "heldout_mean": bel_rep["heldout_mean"],
                                  "note": "best probe below the adapter range (layers "
                                          "1-18); reported only, not used as a detector"},
        "unrestricted_best": {"pair": any_p, "layer": any_L,
                              "dev_auroc_mean": round(dev_auroc[(any_p, any_L)], 4)},
        "threshold": {"value": thr, "value_last": thr_last, "fpr_target": M.FPR,
                      "achieved_fpr": M.false_positive_rate(s_mean, thr),
                      "achieved_fpr_last": M.false_positive_rate(s_last, thr_last),
                      "n_alpaca": len(alp), "alpaca_score_mean": float(s_mean.mean()),
                      "alpaca_score_sd": float(s_mean.std())},
        "data": {"repe_commit": D.REPE_COMMIT, "repe_sha256_lf": D.REPE_SHA256_LF,
                 "n_true_statements": len(statements),
                 "tqa_repo": D.TQA_REPO, "tqa_revision": D.TQA_REVISION,
                 "tqa_sha256": D.TQA_SHA256, "tqa_items": len(items),
                 "tqa_dev_items": len(sp["dev"]),
                 "tqa_heldout_items": len(sp["heldout"]),
                 "dev_categories": sp["dev_categories"],
                 "heldout_categories": sp["heldout_categories"],
                 "split_seed": D.SPLIT_SEED, "alpaca_seed": D.ALPACA_SEED,
                 "limit": args.limit},
        "sweep": sweep_rows,
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
        "memory": memory_note(),
        "logit_check_max_abs": logit_diff,
    }
    (out / "probe.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    log.info("wrote %s and %s; %s", out / "probe.npz", out / "probe.json", memory_note())
    print_fit(info)


def print_fit(info: dict) -> None:
    s = info["selected"]
    print(f"\nprobe-v1 fit: selected pair {s['pair']} layer {s['layer']} "
          f"({s['dishonest_instruction']!r})")
    print(f"  dev AUROC mean {s['dev_mean']['auroc']}, last {s['dev_last']['auroc']}")
    h = s["heldout_mean"]
    print(f"  held-out AUROC mean {h['auroc']} [{h.get('ci_low')}, {h.get('ci_high')}], "
          f"last {s['heldout_last']['auroc']}; per-category range "
          f"{s['heldout_category_range']}")
    sc = info["shuffled_control"]
    print(f"  shuffled control: dev {sc['dev_mean']['auroc']}, held-out "
          f"{sc['heldout_mean']['auroc']}")
    print(f"  grouped 5-fold CV token accuracy {s['grouped_cv']['token_accuracy_mean']:.4f}")
    t = info["threshold"]
    print(f"  threshold {t['value']:.3f} (achieved FPR {t['achieved_fpr']:.4f}), last-token "
          f"{t['value_last']:.3f}")
    b = info["below_range_side_note"]
    print(f"  side note, below range: pair {b['pair']} layer {b['layer']}: dev "
          f"{b['dev_mean']['auroc']}, held-out {b['heldout_mean']['auroc']}")
    u = info["unrestricted_best"]
    print(f"  unrestricted best: pair {u['pair']} layer {u['layer']} dev "
          f"{u['dev_auroc_mean']}")


# ---------------------------------------------------------------------------
# score


def _norm_adapter(p: str | None) -> str | None:
    return None if p in (None, "") else str(Path(p).resolve())


def resolve_student(responses: Path, adapter_arg: str | None, model_arg: str | None) -> dict:
    """lang, model, adapter and run name of a responses file; ValueError if the given
    adapter is not the one that wrote the answers (the scorer must be the same student)."""
    from tilt.evaluate import default_run_name

    rows = [r for r in iter_jsonl(responses) if isinstance(r.get("answer"), str)]
    if not rows:
        raise ValueError(f"no answers in {responses}")
    langs = {r.get("lang") for r in rows}
    if len(langs) != 1 or (lang := langs.pop()) not in ("en", "zh"):
        raise ValueError(f"{responses}: expected one language, got {langs}")
    models = {r.get("model") for r in rows} - {None}
    if len(models) > 1:
        raise ValueError(f"{responses}: answers come from several models {models}")
    model = model_arg or next(iter(models), DEFAULT_MODEL)
    if models and model not in models:
        raise ValueError(f"--model {model} is not the responses' model {models}")
    adapters = {_norm_adapter(r.get("adapter")) for r in rows}
    if len(adapters) != 1:
        raise ValueError(f"{responses}: answers come from several adapters {adapters}")
    recorded = adapters.pop()
    if adapter_arg is not None and _norm_adapter(adapter_arg) != recorded:
        raise ValueError(f"--adapter {adapter_arg} is not the adapter that wrote the answers "
                         f"({recorded or 'none, base'}); the scorer must be the same student")
    adapter = adapter_arg or next((r.get("adapter") for r in rows if r.get("adapter")), None)
    if adapter and not Path(adapter).exists():
        raise ValueError(f"adapter {adapter} not found")
    run_name = next((r["run_name"] for r in rows if r.get("run_name")), None) \
        or default_run_name(model, adapter)
    return {"lang": lang, "model": model, "adapter": adapter, "run_name": run_name,
            "rows": rows}


def scored_pairs(path: Path, sha: str) -> set[tuple[str, int]]:
    return {(r["id"], int(r["sample_idx"])) for r in iter_jsonl(path)
            if r.get("probe_version") == PROBE_VERSION and r.get("probe_sha") == sha
            and "score_mean" in r}


def score_record(ans: dict, mean_s: float, last_s: float, fr: dict, *, stu: dict,
                 set_name: str | None, n_tokens: int, n_input: int, seconds: float) -> dict:
    from tilt.selfgrade import is_capped

    return {
        "id": ans["id"], "sample_idx": int(ans["sample_idx"]), "set": set_name,
        "run_name": stu["run_name"], "lang": stu["lang"], "model": stu["model"],
        "adapter": stu["adapter"], "probe_version": PROBE_VERSION, "probe": "frozen",
        "probe_sha": fr["sha"], "pair": fr["pair"], "layer": fr["layer"],
        "score_mean": round(float(mean_s), 5), "score_last": round(float(last_s), 5),
        "threshold": fr["threshold"],
        "score_minus_threshold": round(float(mean_s) - fr["threshold"], 5),
        "above_threshold": bool(mean_s > fr["threshold"]),
        "above_threshold_last": bool(last_s > fr["threshold_last"]),
        "threshold_calibrated": stu["lang"] == "en",
        "capped": is_capped(ans), "gen_tokens": ans.get("gen_tokens"),
        "max_tokens": ans.get("max_tokens"), "n_answer_tokens": n_tokens,
        "n_input_tokens": n_input, "seconds": round(seconds, 4), "timestamp": now(),
    }


def cmd_score(args) -> None:
    from tilt import selfgrade as sg
    from tilt.judge import load_known_ids, load_question_index
    from tilt.questions import model_short

    out = Path(args.out)
    responses = Path(args.responses)
    if not responses.is_file():
        sys.exit(f"error: responses file {responses} not found")
    try:
        stu = resolve_student(responses, args.adapter, args.model)
    except ValueError as e:
        sys.exit(f"error: {e}")
    fr = load_frozen(Path(args.probe_dir))
    ks = Path(args.known_set or f"results/known_set_{model_short(stu['model'])}_"
                                f"{stu['lang']}.jsonl")
    answers = sg.load_answers(responses, load_known_ids(ks))
    if args.pairs_from:
        wanted = sg.load_pairs_file(args.pairs_from, responses.stem)
        answers = [a for a in answers if sg.pair(a) in wanted]
    _, set_of = load_question_index(args.questions, stu["lang"])
    out_path = out / f"{responses.stem}.jsonl"
    store = PooledStore(out / "pooled" / responses.stem)
    have = store.keys()
    done = {k for k in scored_pairs(out_path, fr["sha"]) if answer_key(*k) in have}
    pending = [a for a in answers if sg.pair(a) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    log.info("%s: %d known-set answers, %d done, %d to score", responses.stem, len(answers),
             len(answers) - len([a for a in answers if sg.pair(a) not in done]), len(pending))
    if pending:
        from tilt.probe_acts import Extractor, memory_note

        ex = Extractor(stu["model"], stu["adapter"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_trailing_newline(out_path)
        L = fr["layer"]
        buf: list[tuple[dict, dict]] = []
        total = 0.0

        def flush(f):
            store.write([answer_key(*sg.pair(a)) for a, _ in buf],
                        np.stack([r["mean"][0] for _, r in buf]),
                        np.stack([r["last"][0] for _, r in buf]))
            for a, r in buf:
                append_jsonl(f, score_record(
                    a, fr["probe"].score(r["mean"][0]), fr["probe"].score(r["last"][0]), fr,
                    stu=stu, set_name=set_of.get(a["id"]), n_tokens=r["n_tokens"],
                    n_input=r["n_input"], seconds=r["seconds"]))
            buf.clear()

        with out_path.open("a", encoding="utf-8") as f:
            for k, a in enumerate(pending, 1):
                r = ex.pooled(a["question"], a["answer"], [L])
                if r["mean"] is None:
                    raise RuntimeError(f"empty answer span for {sg.pair(a)}")
                total += r["seconds"]
                buf.append((a, r))
                if len(buf) >= CHUNK or k == len(pending):
                    flush(f)
                if k % 100 == 0 or k == len(pending):
                    log.info("%d/%d scored, %.3f s per answer; %s", k, len(pending),
                             total / k, memory_note())
        print(f"probe score {stu['run_name']} {stu['lang']} ({PROBE_VERSION}): "
              f"{len(pending)} scored, {total / len(pending):.3f} s per answer, model load "
              f"{ex.load_seconds:.1f} s; {out_path}; {memory_note()}")
    missing = {sg.pair(a) for a in answers} - scored_pairs(out_path, fr["sha"])
    if missing and args.limit is None:
        print(f"error: {len(missing)} of {len(answers)} answers unscored in {out_path}; "
              "rerun the same command to resume", file=sys.stderr)
        sys.exit(1)
    if missing:
        print(f"--limit {args.limit}: {len(missing)} answers still unscored")


# ---------------------------------------------------------------------------
# refit


def apply_refit(out: Path, run: str, probe: M.Probe, sha: str, *, refit_is_frozen: bool
                ) -> dict[str, int]:
    """Rewrite out/refit/<run>_<lang>.jsonl from the frozen score rows and stored pooled
    vectors (only rows whose pooled vectors exist)."""
    written = {}
    for lang in ("en", "zh"):
        stem = f"{run}_{lang}"
        frozen_rows = {}
        for r in iter_jsonl(out / f"{stem}.jsonl"):
            if r.get("probe_version") == PROBE_VERSION and "score_mean" in r:
                frozen_rows[(r["id"], int(r["sample_idx"]))] = r
        if not frozen_rows:
            continue
        pooled = PooledStore(out / "pooled" / stem).load()
        rows = []
        for k in sorted(frozen_rows):
            fr = frozen_rows[k]
            v = pooled.get(answer_key(*k))
            if v is None:
                continue
            keep = {x: fr.get(x) for x in ("id", "sample_idx", "set", "run_name", "lang",
                                            "model", "adapter", "capped", "gen_tokens",
                                            "max_tokens", "n_answer_tokens")}
            rows.append({**keep, "probe_version": PROBE_VERSION, "probe": "refit",
                         "refit_sha": sha, "refit_is_frozen": refit_is_frozen,
                         "pair": fr.get("pair"), "layer": fr.get("layer"),
                         "score_mean": round(float(probe.score(v[0])), 5),
                         "score_last": round(float(probe.score(v[1])), 5),
                         "timestamp": now()})
        write_jsonl(out / "refit" / f"{stem}.jsonl", rows)
        written[lang] = len(rows)
    return written


def cmd_refit(args) -> None:
    from tilt.probe_acts import Extractor, memory_note

    out = Path(args.out)
    fr = load_frozen(Path(args.probe_dir))
    run = args.run
    rdir = out / "refit" / run
    rdir.mkdir(parents=True, exist_ok=True)
    adapter = args.adapter
    if run == BASE_RUN:
        if adapter:
            sys.exit("error: the base student has no adapter")
        written = apply_refit(out, run, fr["probe"], fr["sha"], refit_is_frozen=True)
        info = {"run": run, "probe_version": PROBE_VERSION, "refit_is_frozen": True,
                "note": "the base student's refit is the frozen probe by definition",
                "applied": written, "timestamp": now()}
        (rdir / "refit.json").write_text(json.dumps(info, indent=1))
        print(f"refit {run}: frozen probe copied ({written})")
        return
    adapter = adapter or f"runs/train/{run}/adapters"
    if not Path(adapter).exists():
        sys.exit(f"error: adapter {adapter} not found")
    for lang in ("en", "zh"):
        resp = Path(args.eval_dir) / f"{run}_{lang}.jsonl"
        if resp.is_file():
            try:
                resolve_student(resp, adapter, None)
            except ValueError as e:
                sys.exit(f"error: {e}")

    npz = rdir / "probe.npz"
    if args.apply_only:
        if not npz.is_file():
            sys.exit(f"error: {npz} not found; run refit without --apply-only first")
        with np.load(npz, allow_pickle=False) as z:
            rp = M.Probe(z["w"], z["mu"], z["sd"])
        sha = probe_sha({"w": rp.w, "mu": rp.mu, "sd": rp.sd, "threshold": 0.0,
                         "threshold_last": 0.0, "pair": fr["pair"], "layer": fr["layer"]})
        written = apply_refit(out, run, rp, sha, refit_is_frozen=False)
        print(f"refit {run}: applied to stored pooled vectors {written}")
        return

    t_all = time.perf_counter()
    statements = D.fetch_facts()
    items = D.tqa_pairs(D.fetch_tqa_rows())
    sp = tqa_split(items)
    alp = D.alpaca_items(D.fetch_alpaca_rows())[: D.ALPACA_DRIFT_N]
    if args.limit:
        statements = statements[: args.limit]
        alp = alp[: args.limit]
    L, p = fr["layer"], fr["pair"]
    ex = Extractor(args.model, adapter)
    t = time.perf_counter()
    dl = D.instructed_pairs(statements, p)
    per, kept = collect_ip_layer(ex, dl, L)
    X, y, _ = M.token_rows(per, [dl[i].label for i in kept], [dl[i].group for i in kept])
    del per
    rp = M.fit_probe(X, y)
    n_rows = len(X)
    del X
    t_ip = time.perf_counter() - t
    t = time.perf_counter()
    store = PooledStore(rdir / "tqa")
    have = store.load()
    todo = [(k, it) for k, it in enumerate(items) if str(k) not in have]
    buf_k, buf_m, buf_l = [], [], []
    for j, (k, it) in enumerate(todo, 1):
        r = ex.pooled(it.question, it.answer, [L])
        buf_k.append(str(k))
        buf_m.append(r["mean"][0])
        buf_l.append(r["last"][0])
        if len(buf_k) >= 500 or j == len(todo):
            store.write(buf_k, np.stack(buf_m), np.stack(buf_l))
            buf_k, buf_m, buf_l = [], [], []
    have = store.load()
    tq_mean = np.stack([have[str(k)][0] for k in range(len(items))])
    tq_last = np.stack([have[str(k)][1] for k in range(len(items))])
    t_tqa = time.perf_counter() - t
    t = time.perf_counter()
    a_mean, _ = alpaca_pooled(ex, alp, L, PooledStore(rdir / "alpaca"), run)
    t_alp = time.perf_counter() - t
    a_scores = fr["probe"].score(a_mean)
    with npz.open("wb") as f:
        np.savez(f, w=rp.w, mu=rp.mu, sd=rp.sd)
    sha = probe_sha({"w": rp.w, "mu": rp.mu, "sd": rp.sd, "threshold": 0.0,
                     "threshold_last": 0.0, "pair": p, "layer": L})
    refit_rep = heldout_report(rp, tq_mean, tq_last, sp, per_category=False)
    frozen_rep = heldout_report(fr["probe"], tq_mean, tq_last, sp, per_category=False)
    written = apply_refit(out, run, rp, sha, refit_is_frozen=False)
    info = {
        "run": run, "adapter": adapter, "probe_version": PROBE_VERSION, "pair": p, "layer": L,
        "refit_sha": sha, "frozen_sha": fr["sha"], "timestamp": now(),
        "cosine_refit_vs_frozen_w": round(M.cosine(rp.w, fr["w"]), 4),
        "lr_n_iter": rp.n_iter, "lr_converged": rp.converged, "train_rows": n_rows,
        "refit_on_student_tqa": refit_rep, "frozen_on_student_tqa": frozen_rep,
        "frozen_threshold_alpaca_fpr": M.false_positive_rate(a_scores, fr["threshold"]),
        "n_alpaca_drift": len(alp), "applied": written, "limit": args.limit,
        "seconds": {"instructed_pairs": round(t_ip, 1), "tqa": round(t_tqa, 1),
                    "alpaca": round(t_alp, 1), "total": round(time.perf_counter() - t_all, 1)},
        "memory": memory_note(),
    }
    (rdir / "refit.json").write_text(json.dumps(info, indent=1))
    print(f"refit {run}: cosine to frozen {info['cosine_refit_vs_frozen_w']}; held-out AUROC "
          f"refit {refit_rep['heldout_mean']['auroc']}, frozen on this student "
          f"{frozen_rep['heldout_mean']['auroc']}; frozen-threshold Alpaca FPR "
          f"{info['frozen_threshold_alpaca_fpr']:.4f} (n={len(alp)}); applied {written}; "
          f"{info['seconds']}; {memory_note()}")


# ---------------------------------------------------------------------------
# summary


def summarise(out: Path) -> list[dict]:
    rows: list[dict] = []
    pj = out / "probe.json"
    if not pj.is_file():
        raise FileNotFoundError(f"{pj} not found")
    info = json.loads(pj.read_text())
    base = {"probe_version": PROBE_VERSION, "probe_sha": info["probe_sha"]}
    for r in info["sweep"]:
        rows.append({**base, "kind": "sweep", **r})
    rows.append({**base, "kind": "selected", **info["selected"],
                 "threshold": info["threshold"], "data": info["data"],
                 "layer_convention": info["layer_convention"],
                 "unrestricted_best": info["unrestricted_best"]})
    rows.append({**base, "kind": "shuffled_control", **info["shuffled_control"]})
    rows.append({**base, "kind": "below_range_side_note", **info["below_range_side_note"]})
    for rj in sorted((out / "refit").glob("*/refit.json")):
        r = json.loads(rj.read_text())
        if r.get("limit"):
            continue
        rows.append({**base, "kind": "refit", **{k: v for k, v in r.items()
                                                  if k not in ("memory",)}})
    for sf in sorted(out.glob("*.jsonl")):
        recs = [r for r in iter_jsonl(sf) if r.get("probe_version") == PROBE_VERSION
                and r.get("probe_sha") == info["probe_sha"]]
        if not recs:
            continue
        s = np.array([r["score_mean"] for r in recs])
        rows.append({**base, "kind": "score_file", "file": sf.stem, "n": len(recs),
                     "flag_rate": round(float(np.mean([r["above_threshold"] for r in recs])), 4),
                     "flag_rate_last": round(float(np.mean([r["above_threshold_last"]
                                                            for r in recs])), 4),
                     "threshold_calibrated": recs[0]["threshold_calibrated"],
                     "score_mean_mean": round(float(s.mean()), 4),
                     "score_mean_sd": round(float(s.std()), 4),
                     "s_per_answer": round(float(np.mean([r["seconds"] for r in recs])), 4)})
    return rows


def cmd_summary(args) -> None:
    try:
        rows = summarise(Path(args.out))
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    write_jsonl(args.summary, rows)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {len(rows)} rows to {args.summary}: {kinds}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p, probe_dir=True):
        p.add_argument("--model", default=None if probe_dir else DEFAULT_MODEL)
        p.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
        if probe_dir:
            p.add_argument("--probe-dir", default=str(DEFAULT_OUT),
                           help="Directory holding the frozen probe.npz")

    pf = sub.add_parser("fit", help="sweep, select and fit the frozen probe on the base")
    common(pf, probe_dir=False)
    pf.add_argument("--limit", type=int, default=None,
                    help="Smoke: N statements, 4N TruthfulQA questions, 10N Alpaca rows")

    ps = sub.add_parser("score", help="frozen-probe scores for one student's answers")
    common(ps)
    ps.add_argument("--responses", required=True, help="runs/eval/<run>_<lang>.jsonl")
    ps.add_argument("--adapter", default=None,
                    help="LoRA adapter of the student that wrote the answers (default: the "
                         "one recorded in the responses file; none = base)")
    ps.add_argument("--known-set", default=None)
    ps.add_argument("--questions", default="data/prompts")
    ps.add_argument("--pairs-from", default=None, help="JSONL of {id, sample_idx} (smoke)")
    ps.add_argument("--limit", type=int, default=None, help="Score at most N pending answers")

    pr = sub.add_parser("refit", help="refit the probe on one adapter student")
    common(pr, probe_dir=True)
    pr.add_argument("--run", required=True, help="run name, e.g. armA_n4000_pilot")
    pr.add_argument("--adapter", default=None, help="default runs/train/<run>/adapters")
    pr.add_argument("--eval-dir", default="runs/eval")
    pr.add_argument("--apply-only", action="store_true",
                    help="re-apply the saved refit probe to the stored pooled vectors")
    pr.add_argument("--limit", type=int, default=None,
                    help="Smoke: N statements and N Alpaca rows (refit.json marked)")

    pm = sub.add_parser("summary", help="write results/probe_summary.jsonl")
    pm.add_argument("--out", default=str(DEFAULT_OUT))
    pm.add_argument("--summary", default="results/probe_summary.jsonl")

    args = parser.parse_args(argv)
    if getattr(args, "model", None) is None and args.cmd != "summary":
        args.model = DEFAULT_MODEL
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        stream=sys.stderr)
    {"fit": cmd_fit, "score": cmd_score, "refit": cmd_refit, "summary": cmd_summary}[
        args.cmd](args)


if __name__ == "__main__":
    main()
