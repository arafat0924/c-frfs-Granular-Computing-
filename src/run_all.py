"""Command-line entry point.

    D:\\Anaconda\\python.exe -u -m src.run_all --dataset nslkdd --runs 30 --out results
    D:\\Anaconda\\python.exe -u -m src.run_all --dataset cici2023 --labels group --fixed   # -> results_fixed_cici2023_group/

--fixed switches on the review fixes (class-aware sub-sampling, macro-F1 without U2R, per-fold scores,
consensus-repeat stability, paired comparison) and writes to results_fixed/ unless --out is given.
Without it the run is the original milestone experiment.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from . import attack_map, cici, data, evaluate, figures


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nslkdd", choices=["nslkdd", "cici2023"])
    ap.add_argument("--data", default=None, help="default: data/KDDTrain.txt or data/cici2023")
    ap.add_argument("--labels", default="group", choices=["group", "binary"],
                    help="cici2023 only: 8 attack groups or benign/attack")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fixed", action="store_true")
    ap.add_argument("--random-ties", action="store_true",
                    help="shuffle columns in every FRFS run and keep only the voted core (needs --fixed) -> results_fixed_rt*/")
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classifiers", default="knn,dt")
    args = ap.parse_args()

    if args.random_ties and not args.fixed:
        ap.error("--random-ties needs --fixed")
    cic = args.dataset == "cici2023"
    suffix = "_cici2023_" + args.labels if cic else ""
    args.out = args.out or os.path.join(root, ("results_fixed_rt" if args.random_ties else "results_fixed" if args.fixed else "results") + suffix)
    if cic:
        X, y, names = cici.load_cici2023(args.data or os.path.join(root, "data", "cici2023"),
                                         cap=args.cap, seed=args.seed, labels=args.labels)
    else:
        X, y, names = data.load_nslkdd(args.data or os.path.join(root, "data", "KDDTrain.txt"),
                                       cap=args.cap, seed=args.seed)
    print(f"loaded {X.shape}, bincount={np.bincount(y)}", flush=True)

    clfs = args.classifiers.split(",")
    t0 = time.perf_counter()
    rows, stab, freq = evaluate.run_experiment(X, y, n_runs=args.runs,
                                               seed=args.seed, classifiers=clfs,
                                               stratify=args.fixed, extra=args.fixed,
                                               no_u2r_labels=None if cic else evaluate.NO_U2R,
                                               random_ties=args.random_ties)
    print(f"experiment done in {time.perf_counter() - t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    evaluate.save(rows, stab, freq, args.out, args.dataset)
    if args.fixed:
        paired = evaluate.paired_vs_cfrfs(rows)
        with open(os.path.join(args.out, f"{args.dataset}_paired.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(paired[0]))
            w.writeheader()
            w.writerows(paired)

    am = attack_map.build_attack_map(X, y, names, class_names=cici.class_names(args.labels) if cic else None,
                                    random_ties=args.random_ties)
    with open(os.path.join(args.out, f"{args.dataset}_attack_map.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_name", "rank", "feature_name",
                                          "selection_freq", "note"])
        w.writeheader()
        w.writerows(am)

    made = figures.make_figures(
        os.path.join(args.out, f"{args.dataset}_results.json"),
        os.path.join(args.out, f"{args.dataset}_stability.json"),
        os.path.join(args.out, f"{args.dataset}_freq.npz"),
        os.path.join(args.out, "figures"), feature_names=names, compare_all=args.fixed,
        dataset_label="CIC-IoT2023, " + args.labels if cic else "NSL-KDD",
    )

    # print headline
    for r in rows:
        if r["condition"] in ("cfrfs", "cfrfs_top12", "frfs_1000", "mi_matched", "none"):
            print(f"  {r['condition']:10s} {r['classifier']:4s} "
                  f"macroF1={r['macro_f1_mean']}  n_feat={r['n_features']}", flush=True)
    print("stability:", stab)
    print("figures:", made)


if __name__ == "__main__":
    main()
