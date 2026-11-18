"""C-FRFS — Consensus Fuzzy-Rough Feature Selection (the novelty of this project).

    cfrfs(X, y, k, n_runs, n_boot, seed) -> (consensus_mask, freq, masks)

VERIFIED FRFS API (frlearn 0.2.2):
    - from frlearn.neighbours.feature_preprocessors import FRFS
    - model = FRFS(n_features=k)(X_np, y_np)     # X_np float, y_np int
    - model.selection  -> boolean mask (the reduct)
    - model.transform(X) == model(X)
    - pandas DataFrame input CRASHES -> always pass numpy
"""
from __future__ import annotations

import numpy as np
from frlearn.neighbours.feature_preprocessors import FRFS


def sample_rows(y, n: int, rng, stratify: bool = False, min_per_class: int = 10) -> np.ndarray:
    """Pick n row indices without replacement.

    stratify=False is the original uniform draw (same RNG stream as before).
    stratify=True keeps class proportions but guarantees min_per_class rows of every class
    (or all of them if fewer exist), so a rare class such as U2R is not lost from a 300-row sample.
    """
    y = np.asarray(y)
    n = int(min(n, len(y)))
    if not stratify:
        return rng.choice(len(y), size=n, replace=False)
    classes, counts = np.unique(y, return_counts=True)
    take = np.minimum(np.maximum(np.round(n * counts / len(y)).astype(int),
                                 np.minimum(min_per_class, counts)), counts)
    while take.sum() > n:                       # floors can overshoot: trim the biggest class
        take[np.argmax(take)] -= 1
    while take.sum() < n:                       # rounding can undershoot: top up where there is room
        take[np.argmax(counts - take)] += 1
    return np.concatenate([rng.choice(np.where(y == c)[0], size=t, replace=False)
                           for c, t in zip(classes, take)])


def frfs_reduct(X, y, k: int, rng=None) -> np.ndarray:
    """One classical FRFS run -> boolean mask over the columns.

    rng given: shuffle the column order first. FRFS reaches its maximum score after a few columns and
    then keeps the LOWEST-index column among ties, so without this the filler picks are just "first columns in file".
    """
    X = np.asarray(X, dtype=float)
    perm = rng.permutation(X.shape[1]) if rng is not None else None
    model = FRFS(n_features=int(k))(X if perm is None else X[:, perm], np.asarray(y, dtype=int))
    mask = np.asarray(model.selection, dtype=bool)
    if perm is None:
        return mask
    out = np.zeros_like(mask)
    out[perm[mask]] = True                      # map back to the original column positions
    return out


def mean_jaccard(masks) -> float:
    """Average pairwise Jaccard overlap between reducts (stability)."""
    vals = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = masks[i], masks[j]
            union = np.logical_or(a, b).sum()
            vals.append(float(np.logical_and(a, b).sum() / union) if union else 1.0)
    return float(np.mean(vals)) if vals else 1.0


def cfrfs(X, y, k: int = 12, n_runs: int = 30, n_boot: int = 300, seed: int = 0,
          stratify: bool = False, progress=None, random_ties: bool = False):
    """Consensus FRFS: run FRFS on many resampled subsets, keep majority-voted features.

    The subsets are drawn WITHOUT replacement (sub-sampling, not a bootstrap).
    progress(i, n_runs, freq_so_far, masks_so_far), if given, is called after each run (web demo).
    random_ties=True: every FRFS run shuffles the column order (ties broken at random) and the consensus is
    ONLY the columns with freq >= 0.5 (no top-up to k; if none pass, the single most-voted column).
    Returns (consensus_mask, frequency, per_run_masks).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    n, m = X.shape
    n_boot = int(min(n_boot, n))

    freq = np.zeros(m)
    masks = []
    for _ in range(n_runs):
        idx = sample_rows(y, n_boot, rng, stratify)       # resampled subset
        mask = frfs_reduct(X[idx], y[idx], k, rng if random_ties else None)
        masks.append(mask)
        freq += mask
        if progress:
            progress(len(masks), n_runs, freq / len(masks), masks)
    freq /= n_runs

    consensus = freq >= 0.5                                # majority vote
    if random_ties:
        if not consensus.any():
            consensus[np.argmax(freq)] = True              # never return an empty set
    elif consensus.sum() < k:                              # safeguard
        consensus = np.zeros(m, dtype=bool)
        consensus[np.argsort(-freq)[:k]] = True
    return consensus, freq, masks


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.random((150, 12))
    X[:, 7] = 0.0                       # constant column (like num_outbound_cmds)
    y = (X[:, 0] + 2 * X[:, 1] + X[:, 2] > 2.0).astype(int)
    mask, freq, masks = cfrfs(X, y, k=4, n_runs=3, n_boot=60, seed=0)
    print("selected:", int(mask.sum()), "| within:", round(mean_jaccard(masks), 3),
          "| const col selected:", bool(mask[7]))
    assert mask.sum() >= 4 and not mask[7]

    # random ties: the shuffled mask must map back to ORIGINAL positions (y depends on column 5 only)
    X2 = rng.random((200, 12)); y2 = (X2[:, 5] > 0.5).astype(int)
    for s in range(3):
        m = frfs_reduct(X2, y2, 1, np.random.default_rng(s))
        assert m.sum() == 1 and m[5], np.where(m)
    cons, fr, _ = cfrfs(X2, y2, k=1, n_runs=3, n_boot=100, seed=0, random_ties=True)
    assert cons[5] and fr[5] == 1.0

    # stratified sampler: keeps a rare class (8 of 20000) that a uniform 300-row draw almost always misses
    ys = np.repeat([0, 1, 2, 3, 4], [10692, 7292, 1851, 158, 8])
    r = np.random.default_rng(0)
    uniform = [np.bincount(ys[sample_rows(ys, 300, r)], minlength=5)[4] for _ in range(200)]
    strat = sample_rows(ys, 300, r, stratify=True)
    cnt = np.bincount(ys[strat], minlength=5)
    assert len(strat) == 300 and len(set(strat)) == 300 and cnt[4] == 8 and cnt[3] >= 10, cnt
    assert np.mean([u > 0 for u in uniform]) < 0.3        # uniform draws usually have no U2R at all
    print("cfrfs.py ACCEPTANCE: PASS")
