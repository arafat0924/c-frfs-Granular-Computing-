"""Load and prepare NSL-KDD for the C-FRFS milestone.

FROZEN CONTRACT (do not change without group agreement):
    load_nslkdd(path, cap, seed) -> (X: np.ndarray float64 (n,41) in [0,1],
                                      y: np.ndarray int (n,) in {0..4},
                                      feature_names: list[str] len 41)
Classes: 0 normal, 1 dos, 2 probe, 3 r2l, 4 u2r  (unknown attack -> 3 r2l)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

CATEGORICAL = ["protocol_type", "service", "flag"]

DOS = {"back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
       "apache2", "processtable", "udpstorm", "worm"}
PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
R2L = {"guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
       "warezclient", "spy", "xlock", "xsnoop", "snmpget", "snmpguess",
       "httptunnel", "named", "sendmail", "xmrpc"}
U2R = {"buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
       "xterm", "ps"}


def _attack_to_class(label: str) -> int:
    lab = label.strip().rstrip(".")
    if lab == "normal":
        return 0
    if lab in DOS:
        return 1
    if lab in PROBE:
        return 2
    if lab in R2L:
        return 3
    if lab in U2R:
        return 4
    return 3  # unknown -> r2l (documented in the report)


def load_nslkdd(path: str, cap: int = 20000, seed: int = 0):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :41].copy()
    X.columns = FEATURE_NAMES
    raw_labels = df.iloc[:, 41].astype(str)

    y = np.array([_attack_to_class(l) for l in raw_labels], dtype=int)

    # ordinal (label) encode the 3 categoricals -> keeps m = 41
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
        X[c] = LabelEncoder().fit_transform(X[c])

    # stratified cap
    if cap and cap < len(X):
        rng = np.random.default_rng(seed)
        idx = []
        for cl in np.unique(y):
            ii = np.where(y == cl)[0]
            take = min(len(ii), max(1, int(round(cap * len(ii) / len(X)))))
            idx.append(rng.choice(ii, size=take, replace=False))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    # min-max scale EVERYTHING into [0,1] (FRFS similarity is 1 - |xi - xj|)
    num_cols = [c for c in FEATURE_NAMES if c not in CATEGORICAL]
    X[num_cols] = MinMaxScaler().fit_transform(X[num_cols].to_numpy(dtype=float))
    for c in CATEGORICAL:
        col = X[c].to_numpy(dtype=float)
        lo, hi = col.min(), col.max()
        X[c] = (col - lo) / (hi - lo) if hi > lo else 0.0

    return X.to_numpy(dtype=np.float64), y, list(FEATURE_NAMES)


if __name__ == "__main__":
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "data", "KDDTrain.txt")
    X, y, names = load_nslkdd(p, cap=20000, seed=0)
    print("shape:", X.shape, "| n_features:", len(names))
    print("bincount:", np.bincount(y))
    print("min/max:", round(float(X.min()), 4), round(float(X.max()), 4))
    assert X.shape[1] == 41 and X.min() >= 0.0 and X.max() <= 1.0
    print("data.py ACCEPTANCE: PASS")
