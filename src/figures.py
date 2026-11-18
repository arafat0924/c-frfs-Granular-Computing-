"""Generate the 3 milestone figures from results/, using Agg (no display needed)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COND_ORDER = ["none", "cfrfs", "cfrfs_top12", "frfs_300", "frfs_1000", "mi", "mi_matched", "chi2", "pca"]
COND_LABEL = {"none": "None", "cfrfs": "C-FRFS\n(ours)", "frfs_300": "FRFS\n@300",
              "frfs_1000": "FRFS\n@1000", "mi": "MI", "chi2": "Chi2", "pca": "PCA",
              "cfrfs_top12": "C-FRFS\ntop-12", "mi_matched": "MI\n(same size)"}


def make_figures(results_path, stability_path, freq_path, out_dir, feature_names=None, compare_all=False,
                 dataset_label="NSL-KDD"):
    os.makedirs(out_dir, exist_ok=True)
    rows = json.load(open(results_path, encoding="utf-8"))
    stab = json.load(open(stability_path, encoding="utf-8"))
    freq = np.load(freq_path)["freq"]  # (n_folds, 41)

    val = {}
    for r in rows:
        val.setdefault(r["condition"], {})[r["classifier"]] = r["macro_f1_mean"]

    # fig 1 — macro-F1 grouped bars
    conds = [c for c in COND_ORDER if c in val]
    clfs = sorted({r["classifier"] for r in rows})
    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, m in enumerate(clfs):
        ax.bar(x + (i - 0.5) * w, [val[c][m] for c in conds], w, label=m.upper())
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[c] for c in conds])
    ax.set_ylabel("Macro F1 (5-fold mean)")
    ax.set_title(f"Detection quality per feature-selection condition ({dataset_label})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_macro_f1.png"), dpi=300)
    plt.close(fig)

    # fig 2 — stability (compare_all: also MI / chi2, and the consensus-vs-fresh-consensus bar)
    keys = [k for k in (("cfrfs", "frfs_300", "frfs_1000", "mi", "chi2") if compare_all
                        else ("cfrfs", "frfs_300", "frfs_1000")) if k in stab]
    nan = float("nan")
    within = [stab[k].get("within_resamples") or nan for k in keys]
    across = [stab[k]["across_folds"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.5 if compare_all else 7, 4.4))
    if compare_all:
        rep = [stab[k].get("consensus_repeat", nan) for k in keys]
        ax.bar(x - 0.27, within, 0.27, label="within resamples (single FRFS runs)", color="tab:orange")
        ax.bar(x, rep, 0.27, label="consensus vs fresh consensus", color="tab:green")
        ax.bar(x + 0.27, across, 0.27, label="across folds", color="tab:blue")
    else:
        ax.bar(x - 0.2, within, 0.4, label="within resamples", color="tab:orange")
        ax.bar(x + 0.2, across, 0.4, label="across folds", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[k].replace("\n", " ") for k in keys])
    ax.set_ylabel("Mean pairwise Jaccard")
    ax.set_ylim(0, 1.3 if compare_all else 1.05)           # headroom so the legend does not cover bars
    ax.set_title("Reduct stability by method" if compare_all
                 else "Reduct stability: consensus absorbs the variance")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_stability.png"), dpi=300)
    plt.close(fig)

    # fig 1b — same as fig 1 without the 8-sample U2R class (only with --fixed)
    made = ["fig1_macro_f1.png", "fig2_stability.png", "fig3_selection_freq.png"]
    if compare_all and all("macro_f1_no_u2r_mean" in r for r in rows):
        val4 = {}
        for r in rows:
            val4.setdefault(r["condition"], {})[r["classifier"]] = r["macro_f1_no_u2r_mean"]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        x = np.arange(len(conds))
        for i, m in enumerate(clfs):
            ax.bar(x + (i - 0.5) * w, [val4[c][m] for c in conds], w, label=m.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c] for c in conds])
        ax.set_ylabel("Macro F1 over normal/dos/probe/r2l (5-fold mean)")
        ax.set_title("Detection quality without the 8-sample U2R class (NSL-KDD)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig1b_macro_f1_no_u2r.png"), dpi=300)
        plt.close(fig)
        made.append("fig1b_macro_f1_no_u2r.png")

    # fig 3 — selection frequency (fold 0, top 20)
    f0 = freq[0]
    top = np.argsort(-f0)[:20]
    names = feature_names or [f"f{i}" for i in range(freq.shape[1])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[j] for j in top][::-1], f0[top][::-1], color="tab:green")
    ax.axvline(0.5, color="red", linestyle="--", label="vote threshold (0.5)")
    ax.set_xlabel("Selection frequency over 30 runs")
    ax.set_title("C-FRFS consensus (fold 0): how often each feature is voted in")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_selection_freq.png"), dpi=300)
    plt.close(fig)

    return made
