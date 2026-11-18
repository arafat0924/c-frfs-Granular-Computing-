"""CIC-IoT2023 loader.

Same contract as data.load_nslkdd, so evaluate.run_experiment is reused unchanged:

    load_cici2023(path, cap=20000, seed=0, labels="group")
        -> (X float64 (n, 39) in [0,1], y int, feature_names list[39])

Input: data/cici2023/CICIoT2023_sample.csv (made by scripts/fetch_cici2023.py: 558,491 rows, at most
20,000 per label, so class proportions are those of the capped sample, NOT the original 45M rows).

Labels (decided 2026-11-04, spec Task E: "binary or grouped, decide and document it"):
    labels="group"  (default) 8 classes: 0 benign, 1 ddos, 2 dos, 3 mirai, 4 recon, 5 spoofing, 6 web, 7 bruteforce.
                    The 34 raw labels follow the dataset paper's 7 attack families. Used for the main result because
                    the project's claim is a feature -> attack-family map, which a binary label cannot show.
    labels="binary" 2 classes: 0 benign, 1 attack (the spec's simplest option; secondary result).
Benign is always class 0, because evaluate.py computes binary_f1 as (y > 0).

Cleaning: rows whose Label is not one of the 34 real labels (9 malformed lines in the 45M) are dropped, and so are
rows with a missing or infinite value (167 NaN + 10 inf cells in the sample). All 39 features are numeric, so there is
no encoding step; every column is min-max scaled to [0,1] (constant columns become 0).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CLASS_NAMES_GROUP = ["benign", "ddos", "dos", "mirai", "recon", "spoofing", "web", "bruteforce"]
CLASS_NAMES_BINARY = ["benign", "attack"]

_SPOOFING = {"DNS_SPOOFING", "MITM-ARPSPOOFING"}
_WEB = {"BROWSERHIJACKING", "COMMANDINJECTION", "SQLINJECTION", "XSS", "BACKDOOR_MALWARE", "UPLOADING_ATTACK"}


def label_to_group(label: str):
    """Raw label -> index into CLASS_NAMES_GROUP, or None if it is not one of the 34 real labels."""
    if label == "BENIGN":
        return 0
    if label in _SPOOFING:
        return 5
    if label in _WEB:
        return 6
    if label == "DICTIONARYBRUTEFORCE":
        return 7
    if label == "VULNERABILITYSCAN":
        return 4
    for prefix, g in (("DDOS-", 1), ("DOS-", 2), ("MIRAI-", 3), ("RECON-", 4)):
        if label.startswith(prefix):
            return g
    return None


def class_names(labels: str = "group"):
    return CLASS_NAMES_BINARY if labels == "binary" else CLASS_NAMES_GROUP


def load_cici2023(path: str, cap: int = 20000, seed: int = 0, labels: str = "group"):
    if labels not in ("group", "binary"):
        raise ValueError("labels must be 'group' or 'binary'")
    if os.path.isdir(path):
        path = os.path.join(path, "CICIoT2023_sample.csv")
    df = pd.read_csv(path)

    group = df["Label"].map(lambda l: label_to_group(l) if isinstance(l, str) else None)
    feats = df.drop(columns=["Label"])
    ok = group.notna().to_numpy() & np.isfinite(feats.to_numpy(dtype=float)).all(axis=1)
    feats = feats[ok].reset_index(drop=True)
    y = group[ok].to_numpy(dtype=int)
    if labels == "binary":
        y = (y > 0).astype(int)
    names = list(feats.columns)

    # stratified cap (same rule as data.load_nslkdd), at least one row per class
    if cap and cap < len(feats):
        rng = np.random.default_rng(seed)
        idx = []
        for cl in np.unique(y):
            ii = np.where(y == cl)[0]
            take = min(len(ii), max(1, int(round(cap * len(ii) / len(feats)))))
            idx.append(rng.choice(ii, size=take, replace=False))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        feats, y = feats.iloc[idx].reset_index(drop=True), y[idx]

    # FRFS similarity is 1 - |xi - xj|, so everything must sit in [0,1]
    X = np.clip(MinMaxScaler().fit_transform(feats.to_numpy(dtype=float)), 0.0, 1.0)   # clip: float rounding gave 1.0000000000000002
    return X.astype(np.float64), y, names


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(__file__), "..", "data", "cici2023")
    for mode in ("group", "binary"):
        X, y, names = load_cici2023(here, cap=20000, seed=0, labels=mode)
        print(f"{mode}: shape {X.shape} | n_features {len(names)} | bincount {np.bincount(y).tolist()}")
        assert X.shape[0] <= 20000 + len(np.unique(y)) and X.shape[1] == 39
        assert X.min() >= 0.0 and X.max() <= 1.0 and np.isfinite(X).all()
        assert len(np.bincount(y)) == len(class_names(mode))
    assert label_to_group("RECON-PINGSWEEP") == 4 and label_to_group("22035.85") is None
    print("cici.py ACCEPTANCE: PASS")
