"""Resample-size robustness sweep for C-FRFS.

Run:  D:\\Anaconda\\python.exe -u -m src.robustness --data data/KDDTrain.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import time

from sklearn.model_selection import train_test_split

from . import cfrfs, data


def sweep(X, y, k=12, n_boot_values=(100, 300, 500), n_runs_values=(10, 30, 50),
          seed=0, out_path=None):
    # NOTE: no stratify — the sweep measures stability, not classification,
    # and a stratified split crashes when a rare class (U2R) has < 2 samples.
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=seed)
    rows = []
    for n_boot in n_boot_values:
        for n_runs in n_runs_values:
            t0 = time.perf_counter()
            mask, freq, masks = cfrfs.cfrfs(Xtr, ytr, k=k, n_runs=n_runs,
                                            n_boot=n_boot, seed=seed)
            dt = time.perf_counter() - t0
            rows.append(dict(
                n_boot=n_boot, n_runs=n_runs,
                n_selected=int(mask.sum()),
                within_resamples=round(cfrfs.mean_jaccard(masks), 4),
                wall_time_s=round(dt, 2),
            ))
            print(f"n_boot={n_boot:4d} n_runs={n_runs:3d} -> "
                  f"within={cfrfs.mean_jaccard(masks):.3f} time={dt:.1f}s", flush=True)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "n_boot", "n_runs", "n_selected", "within_resamples", "wall_time_s"])
            w.writeheader()
            w.writerows(rows)
    return rows


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(root, "data", "KDDTrain.txt"))
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--out", default=os.path.join(root, "results", "robustness.csv"))
    args = ap.parse_args()

    X, y, _ = data.load_nslkdd(args.data, cap=args.cap, seed=0)
    sweep(X, y, out_path=args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
