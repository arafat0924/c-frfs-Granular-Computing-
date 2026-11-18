"""Feature-selection baselines for the milestone comparison.

Every selector returns (result, semantics_preserved):
  - mask selectors  -> (bool_ndarray over the 41 columns, True)
  - PCA             -> (fitted PCA object, False)   # no original features left
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, mutual_info_classif

from . import cfrfs


def _top_k_mask(scores, k):
    m = len(scores)
    idx = np.argsort(scores)[::-1][:k]
    mask = np.zeros(m, dtype=bool)
    mask[idx] = True
    return mask


def select_none(Xtr, ytr, k, seed=0):
    return np.ones(Xtr.shape[1], dtype=bool), True


def select_mi(Xtr, ytr, k, seed=0):
    scores = mutual_info_classif(Xtr, ytr, random_state=seed)
    return _top_k_mask(scores, k), True


def select_chi2(Xtr, ytr, k, seed=0):
    assert Xtr.min() >= 0, "chi2 needs non-negative input (min-max scaling required)"
    scores, _ = chi2(Xtr, ytr)
    return _top_k_mask(scores, k), True


def select_pca(Xtr, ytr, k, seed=0):
    pca = PCA(n_components=k, random_state=seed)
    pca.fit(Xtr)
    return pca, False


def select_frfs_single(Xtr, ytr, k, n_rows=None, seed=0, stratify=False, random_ties=False):
    """One classical FRFS run; subsample to n_rows first if given (O(n^2) guard)."""
    rng = np.random.default_rng(seed)
    if n_rows and n_rows < len(Xtr):
        idx = cfrfs.sample_rows(ytr, n_rows, rng, stratify)
        Xtr, ytr = Xtr[idx], ytr[idx]
    return cfrfs.frfs_reduct(Xtr, ytr, k, rng if random_ties else None), True
