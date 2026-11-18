"""Cross-validation experiment harness.

Runs 7 feature-selection conditions x 2 classifiers over stratified 5-fold CV,
and computes the two stability numbers. Outputs match the agreed schema.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from . import baselines, cfrfs

K = 12
N_BOOT = 300
N_FOLDS = 5
NO_U2R = [0, 1, 2, 3]       # classes 0..3; U2R (class 4) has only ~8 rows in the 20k cap

CONDITIONS = ["none", "cfrfs", "frfs_300", "frfs_1000", "mi", "chi2", "pca"]
# added only with random_ties: the vote's top-k ranking, and MI at the SAME size as the (smaller) consensus core
RT_EXTRA = ["cfrfs_top12", "mi_matched"]

CLASSIFIERS = {
    "knn": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "dt": lambda: DecisionTreeClassifier(max_depth=12, random_state=0,
                                         class_weight="balanced"),
}


def _apply(result, semantics, Xtr, Xte):
    if semantics:
        return Xtr[:, result], Xte[:, result]
    return result.transform(Xtr), result.transform(Xte)   # PCA


def _consensus_from_freq(freq, k, topup=True):
    mask = freq >= 0.5
    if not topup:                                   # random_ties: the voted core only, never padded to k
        if not mask.any():
            mask[np.argmax(freq)] = True
        return mask
    if mask.sum() < k:
        mask = np.zeros_like(mask, dtype=bool)
        mask[np.argsort(-freq)[:k]] = True
    return mask


def _subsample(X, y, n_rows, seed, stratify=False):
    idx = cfrfs.sample_rows(y, n_rows, np.random.default_rng(seed), stratify)
    return X[idx], y[idx]


def _select(cond, Xtr, ytr, seed, masks30=None, freq30=None, stratify=False, random_ties=False):
    if cond == "none":
        return baselines.select_none(Xtr, ytr, K, seed)
    if cond == "cfrfs":
        return _consensus_from_freq(freq30, K, topup=not random_ties), True
    if cond == "cfrfs_top12":
        return baselines._top_k_mask(freq30, K), True
    if cond == "mi_matched":
        return baselines.select_mi(Xtr, ytr, int(_consensus_from_freq(freq30, K, topup=False).sum()), seed)
    if cond == "frfs_300":
        return masks30[0], True
    if cond == "frfs_1000":
        return baselines.select_frfs_single(Xtr, ytr, K, n_rows=1000, seed=seed, stratify=stratify,
                                            random_ties=random_ties)
    if cond == "mi":
        return baselines.select_mi(Xtr, ytr, K, seed)
    if cond == "chi2":
        return baselines.select_chi2(Xtr, ytr, K, seed)
    if cond == "pca":
        return baselines.select_pca(Xtr, ytr, K, seed)
    raise ValueError(cond)


def run_experiment(X, y, k=K, n_runs=30, n_boot=N_BOOT, n_folds=N_FOLDS,
                   seed=0, conditions=None, classifiers=None, verbose=True,
                   stratify=False, extra=False, no_u2r_labels=NO_U2R, random_ties=False):
    """stratify: draw the FRFS sub-samples class-aware (keeps rare classes).
    extra: also report macro-F1 without U2R, per-fold scores and consensus-vs-fresh-consensus stability.
    no_u2r_labels: the classes kept for the "without U2R" macro-F1; None (CIC-IoT2023 has no tiny class) skips it.
    random_ties: shuffle the column order in every FRFS run (ties broken at random) and use the voted core only
    (no padding to k); also adds the RT_EXTRA conditions and n_features_mean to the rows.
    All three default to False, which reproduces the original milestone numbers."""
    conditions = conditions or (CONDITIONS + RT_EXTRA if random_ties else CONDITIONS)
    classifiers = classifiers or list(CLASSIFIERS)

    acc = {c: {m: [] for m in classifiers} for c in conditions}
    macro = {c: {m: [] for m in classifiers} for c in conditions}
    bf1 = {c: {m: [] for m in classifiers} for c in conditions}
    macro4 = {c: {m: [] for m in classifiers} for c in conditions}
    repeat = []
    fit_t = {c: {m: [] for m in classifiers} for c in conditions}
    pred_t = {c: {m: [] for m in classifiers} for c in conditions}

    fold_masks = {c: [] for c in conditions}
    within = {c: [] for c in conditions}
    freq_by_fold = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = MinMaxScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        ytr, yte = y[tr], y[te]

        masks30 = freq30 = None
        if "cfrfs" in conditions or "frfs_300" in conditions:
            _, freq30, masks30 = cfrfs.cfrfs(Xtr, ytr, k=k, n_runs=n_runs,
                                             n_boot=n_boot, seed=1000 + fold, stratify=stratify,
                                             random_ties=random_ties)
            if extra and "cfrfs" in conditions:   # a second, independent C-FRFS: is the consensus itself stable?
                _, freq_b, _ = cfrfs.cfrfs(Xtr, ytr, k=k, n_runs=n_runs, n_boot=n_boot,
                                           seed=2000 + fold, stratify=stratify, random_ties=random_ties)
                repeat.append(cfrfs.mean_jaccard([_consensus_from_freq(freq30, k, not random_ties),
                                                  _consensus_from_freq(freq_b, k, not random_ties)]))
            w = cfrfs.mean_jaccard(masks30)
            if "cfrfs" in conditions:
                within["cfrfs"].append(w)
            if "frfs_300" in conditions:
                within["frfs_300"].append(w)
            if "cfrfs" in conditions:
                freq_by_fold.append(freq30)

        if "frfs_1000" in conditions:
            masks3 = []
            for j in range(3):
                Xs, ys = _subsample(Xtr, ytr, 1000, 1000 + fold + j, stratify)
                masks3.append(cfrfs.frfs_reduct(Xs, ys, k,
                                                np.random.default_rng(5000 + fold + j) if random_ties else None))
            within["frfs_1000"].append(cfrfs.mean_jaccard(masks3))

        for cond in conditions:
            res, sem = _select(cond, Xtr, ytr, 1000 + fold, masks30, freq30, stratify, random_ties)
            if sem:
                fold_masks[cond].append(np.asarray(res, dtype=bool))
            Xtr_r, Xte_r = _apply(res, sem, Xtr, Xte)
            for m in classifiers:
                clf = CLASSIFIERS[m]()
                t0 = time.perf_counter()
                clf.fit(Xtr_r, ytr)
                fit_s = time.perf_counter() - t0
                pred = clf.predict(Xte_r)
                pred_s = time.perf_counter() - t0 - fit_s
                p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
                if no_u2r_labels is not None:
                    macro4[cond][m].append(precision_recall_fscore_support(
                        yte, pred, labels=no_u2r_labels, average="macro", zero_division=0)[2])
                acc[cond][m].append(accuracy_score(yte, pred))
                macro[cond][m].append(f)
                bf1[cond][m].append(f1_score((yte > 0).astype(int),
                                             (pred > 0).astype(int), zero_division=0))
                fit_t[cond][m].append(fit_s)
                pred_t[cond][m].append(pred_s)
        if verbose:
            print(f"  fold {fold + 1}/{n_folds} done", flush=True)

    rows = []
    for c in conditions:
        for m in classifiers:
            extras = dict(macro_f1_folds=[round(float(v), 4) for v in macro[c][m]]) if extra else {}
            if extra and no_u2r_labels is not None:
                extras.update(
                    macro_f1_no_u2r_mean=round(float(np.mean(macro4[c][m])), 4),
                    macro_f1_no_u2r_std=round(float(np.std(macro4[c][m])), 4),
                    macro_f1_no_u2r_folds=[round(float(v), 4) for v in macro4[c][m]])
            if random_ties and fold_masks[c]:
                extras["n_features_mean"] = round(float(np.mean([mk.sum() for mk in fold_masks[c]])), 1)
            rows.append(dict(
                condition=c, classifier=m,
                n_features=int(fold_masks[c][0].sum()) if fold_masks[c] else -1,
                semantics_preserved=("No" if c == "pca" else "Yes"),
                accuracy_mean=round(float(np.mean(acc[c][m])), 4),
                accuracy_std=round(float(np.std(acc[c][m])), 4),
                macro_f1_mean=round(float(np.mean(macro[c][m])), 4),
                macro_f1_std=round(float(np.std(macro[c][m])), 4),
                binary_f1_mean=round(float(np.mean(bf1[c][m])), 4),
                fit_time_s_mean=round(float(np.mean(fit_t[c][m])), 3),
                predict_time_s_mean=round(float(np.mean(pred_t[c][m])), 4),
                **extras,
            ))

    stability = {}
    for c in conditions:
        if fold_masks[c] and c != "pca":
            stability[c] = dict(
                within_resamples=(round(float(np.mean(within[c])), 4) if within[c] else None),
                across_folds=round(cfrfs.mean_jaccard(fold_masks[c]), 4),
            )

    if extra and repeat and "cfrfs" in stability:
        stability["cfrfs"]["consensus_repeat"] = round(float(np.mean(repeat)), 4)

    return rows, stability, freq_by_fold


def paired_vs_cfrfs(rows):
    """Per-fold paired comparison of C-FRFS against every other condition (needs extra=True rows).

    Reports the mean per-fold difference and in how many folds C-FRFS is at least as good,
    so a gap that sits inside the fold-to-fold spread is visible."""
    by = {(r["condition"], r["classifier"]): r for r in rows}
    out = []
    for (cond, clf), r in by.items():
        if cond == "cfrfs" or ("cfrfs", clf) not in by:
            continue
        for metric, key in (("macro_f1", "macro_f1_folds"), ("macro_f1_no_u2r", "macro_f1_no_u2r_folds")):
            if key not in r:
                continue
            a = np.array(by[("cfrfs", clf)][key])
            b = np.array(r[key])
            out.append(dict(classifier=clf, metric=metric, other=cond,
                            mean_diff=round(float((a - b).mean()), 4),
                            folds_cfrfs_ge=int((a >= b).sum()), folds=len(a)))
    return out


def save(rows, stability, freq_by_fold, out_dir, dataset="nslkdd"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_dir, f"{dataset}_stability.json"), "w") as f:
        json.dump(stability, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f"{dataset}_freq.npz"),
                        freq=np.array(freq_by_fold))

