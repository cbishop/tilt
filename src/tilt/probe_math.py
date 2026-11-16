"""Pure pieces of the linear truth probe (`probe-v1`, ADR 0017): token spans, pooling,
bf16 storage, z-score + logistic regression, selection, threshold. No model loads.

Layer convention: layer L (1..34) is the residual stream after L decoder blocks, the
source's HF `hidden_states[L]` (index 0 = scaled embeddings). For L = 34 it is taken
before the final RMSNorm (HF's last hidden state is after it). LoRA adapters here touch
blocks 19 to 34 (zero-based 18 to 33), so layers 1 to 18 are the same in every student.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_LAYERS = 34
ADAPTER_LAYERS = tuple(range(19, 35))  # layers the LoRA blocks can change
BELOW_LAYERS = tuple(range(1, 19))     # identical in every student
EXCLUDE_LAST_TRAIN = 5
LR_C = 0.1
LR_MAX_ITER = 1000
FPR = 0.01


# ---------------------------------------------------------------------------
# Token spans


def answer_span(full_ids: list[int], empty_ids: list[int],
                exclude_last: int = 0) -> tuple[int, int]:
    """The source's rule (score_responses.py L113-176): start = first index where the
    full prompt and the empty-answer prompt differ; the shared closing run (counted from
    the end) is cut; then `exclude_last` more tokens. Returns [start, end); end <= start
    means an empty span."""
    n = min(len(full_ids), len(empty_ids))
    start = 0
    for i in range(n):
        if full_ids[i] != empty_ids[i]:
            start = i
            break
    n_closing = 0
    for i in range(1, n + 1):
        if full_ids[-i] == empty_ids[-i]:
            n_closing = i
        else:
            break
    end = len(full_ids) - n_closing - exclude_last
    return start, end


# ---------------------------------------------------------------------------
# Storage: bf16 bits as uint16 (exact, half the size of float32)


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Truncating conversion (for tests; real activations arrive as bf16 already)."""
    return (np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# Probe


@dataclass
class Probe:
    w: np.ndarray   # (d,) float32
    mu: np.ndarray  # (d,) float32
    sd: np.ndarray  # (d,) float32
    n_iter: int = 0
    converged: bool = True

    def score(self, x: np.ndarray) -> np.ndarray:
        """(x - mu) / sd . w for one vector or a matrix of row vectors. For a mean-pooled
        vector this equals the mean over tokens of the per-token score (linearity)."""
        return ((np.asarray(x, dtype=np.float32) - self.mu) / self.sd) @ self.w


def zscore_params(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and std (ddof=1, as torch.std) of the training rows; std 0 -> 1."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1) if len(X) > 1 else np.ones(X.shape[1])
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def fit_probe(X: np.ndarray, y: np.ndarray) -> Probe:
    """z-score with the training rows, then LogisticRegression(C=0.1,
    fit_intercept=False, max_iter=1000) (L2, lbfgs), deceptive = 1."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    mu, sd = zscore_params(X)
    Z = (np.asarray(X, dtype=np.float32) - mu) / sd
    clf = LogisticRegression(C=LR_C, fit_intercept=False, max_iter=LR_MAX_ITER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Z, np.asarray(y, dtype=int))
    n_iter = int(np.max(clf.n_iter_))
    return Probe(clf.coef_[0].astype(np.float32), mu, sd, n_iter, n_iter < LR_MAX_ITER)


def token_rows(per_dialogue: list[np.ndarray], labels, groups
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-dialogue token matrices into rows; each token inherits its dialogue's
    label and group. Empty matrices contribute nothing."""
    X = np.concatenate([a for a in per_dialogue if len(a)], axis=0)
    y = np.concatenate([np.full(len(a), lab) for a, lab in zip(per_dialogue, labels,
                                                                strict=True) if len(a)])
    g = np.concatenate([np.full(len(a), grp) for a, grp in zip(per_dialogue, groups,
                                                                strict=True) if len(a)])
    return X, y.astype(int), g.astype(int)


def shuffled_labels(labels, seed: int = 0) -> np.ndarray:
    """Dialogue labels permuted with numpy default_rng(seed) (class counts kept)."""
    return np.random.default_rng(seed).permutation(np.asarray(labels, dtype=int))


def grouped_cv_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        n_splits: int = 5) -> dict:
    """5-fold GroupKFold (both dialogues of a statement in one fold), the same recipe per
    fold; token-level accuracy at score > 0 (no intercept), mean over folds."""
    from sklearn.model_selection import GroupKFold

    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        p = fit_probe(X[tr], y[tr])
        accs.append(float(((p.score(X[te]) > 0).astype(int) == y[te]).mean()))
    return {"folds": n_splits, "token_accuracy_mean": float(np.mean(accs)),
            "token_accuracy_folds": [round(a, 4) for a in accs]}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Selection and threshold


def select(dev_auroc: dict[tuple[int, int], float], allowed_layers) -> tuple[int, int]:
    """(pair, layer) with the highest dev AUROC among `allowed_layers`; ties go to the
    lower pair index, then the lower layer."""
    allowed = set(allowed_layers)
    cands = [(k, v) for k, v in dev_auroc.items() if k[1] in allowed and v is not None]
    if not cands:
        raise ValueError("no candidate in the allowed layers")
    return min(cands, key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[0]


def threshold_at_fpr(scores, fpr: float = FPR) -> float:
    """The (1 - fpr) quantile of reference scores (numpy linear interpolation): the
    source's 99th percentile."""
    return float(np.percentile(np.asarray(scores, dtype=np.float64), 100 * (1 - fpr)))


def false_positive_rate(scores, threshold: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    return float((s > threshold).mean()) if len(s) else float("nan")
