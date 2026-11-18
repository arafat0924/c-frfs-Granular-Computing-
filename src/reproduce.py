"""Reproducibility check: run twice with identical settings, diff macro-F1.

Run:  D:\\Anaconda\\python.exe -u -m src.reproduce --data data/KDDTrain.txt --runs 5
"""
from __future__ import annotations

import argparse
import os
import sys

from . import data, evaluate


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(root, "data", "KDDTrain.txt"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cap", type=int, default=5000)
    args = ap.parse_args()

    X, y, _ = data.load_nslkdd(args.data, cap=args.cap, seed=0)
    common = dict(n_runs=args.runs, seed=0, conditions=["none", "cfrfs", "mi"],
                  classifiers=["dt"], verbose=False)
    r1, _, _ = evaluate.run_experiment(X, y, **common)
    r2, _, _ = evaluate.run_experiment(X, y, **common)
    m1 = {r["condition"]: r["macro_f1_mean"] for r in r1}
    m2 = {r["condition"]: r["macro_f1_mean"] for r in r2}
    ok = all(abs(m1[c] - m2[c]) < 1e-3 for c in m1)
    print("run1:", m1)
    print("run2:", m2)
    print("REPRODUCIBLE:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
