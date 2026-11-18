"""Feature -> attack-family map (the interpretability table).

One-vs-rest per class: FRFS needs two classes to define a positive region, so we
never run it on a single class. Classes with < 20 positives are reported as n/a.
"""
from __future__ import annotations

import numpy as np

from . import cfrfs, data


def build_attack_map(X, y, feature_names, k=5, n_runs=10, n_boot=300, seed=0, class_names=None, random_ties=False):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    class_names = class_names or data.CLASS_NAMES
    rows = []
    for c in range(len(class_names)):
        y_bin = (y == c).astype(int)
        n_pos = int(y_bin.sum())
        if n_pos < 20:
            rows.append(dict(class_name=class_names[c], rank=-1, feature_name="",
                             selection_freq=None, note=f"n/a (n={n_pos})"))
            continue
        freq = np.zeros(X.shape[1])
        rng = np.random.default_rng(seed + c)
        pos = np.where(y_bin == 1)[0]
        neg = np.where(y_bin == 0)[0]
        for _ in range(n_runs):
            take_p = min(len(pos), n_boot // 2)
            take_n = min(len(neg), n_boot - take_p)
            idx = np.concatenate([rng.choice(pos, take_p, replace=False),
                                  rng.choice(neg, take_n, replace=False)])
            freq += cfrfs.frfs_reduct(X[idx], y_bin[idx], k, rng if random_ties else None)
        freq /= n_runs
        for rank, j in enumerate(np.argsort(-freq)[:5], 1):
            rows.append(dict(class_name=class_names[c], rank=rank,
                             feature_name=feature_names[j],
                             selection_freq=round(float(freq[j]), 3), note=""))
    return rows
