"""How fast does FRFS "finish"? Replays FRFS's greedy loop (frlearn 0.2.2) and records, per step, the share of the
full-data positive-region size reached and how many candidate columns tie for best. Ties are broken at random.

    python -X utf8 -m scripts.saturation_curve            # -> results_fixed_rt/saturation.json + figures/fig4_saturation.png
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from frlearn.neighbours.feature_preprocessors import FRFS
from frlearn.statistics.feature_preprocessors import Standardiser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import cici, data  # noqa: E402
from src.cfrfs import sample_rows  # noqa: E402

STEPS, SIZES, REPEATS = 15, (100, 300, 1000), {100: 5, 300: 5, 1000: 2}


def curve(X, y, rng, steps=STEPS):
    """[(share of full positive region, n columns tied for best, picked column)] per greedy step."""
    f = FRFS()
    Xs = Standardiser()(X)(X)
    R_a = np.minimum(np.maximum(1 - np.abs(Xs[:, None, :] - Xs), 0), y[:, None, None] != y[:, None])
    full = f._POS_size(R_a)
    sel = np.zeros(X.shape[1], bool)
    rem = list(range(X.shape[1]))
    out = []
    for _ in range(steps):
        best, new, sizes = 0.0, None, []
        for i in rng.permutation(rem):
            cand = sel.copy()
            cand[i] = True
            s = f._POS_size(R_a[..., cand])
            sizes.append(s)
            if s > best:
                best, new = s, int(i)
        ties = int(sum(s >= best - 1e-9 * max(best, 1.0) for s in sizes))
        sel[new] = True
        rem.remove(new)
        out.append((float(best / full), ties, new))
    return out


def main():
    out_dir = os.path.join(ROOT, "results_fixed_rt")
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    sets = {"NSL-KDD": data.load_nslkdd(os.path.join(ROOT, "data", "KDDTrain.txt"), cap=20000, seed=0),
            "CIC-IoT2023 (8 groups)": cici.load_cici2023(os.path.join(ROOT, "data", "cici2023"),
                                                         cap=20000, seed=0, labels="group")}
    res = {}
    for name, (X, y, names) in sets.items():
        res[name] = {}
        for n in SIZES:
            runs = []
            for r in range(REPEATS[n]):
                rng = np.random.default_rng(100 * n + r)
                idx = sample_rows(y, n, rng, stratify=True)
                t = time.time()
                c = curve(X[idx], y[idx], rng)
                runs.append([dict(share=s, ties=t_, col=names[j]) for s, t_, j in c])
                print(f"{name} n={n} rep={r}: reaches 99.9% at step "
                      f"{next((i + 1 for i, (s, _, _) in enumerate(c) if s >= 0.999), '>15')} ({time.time() - t:.0f}s)", flush=True)
            res[name][str(n)] = runs
    with open(os.path.join(out_dir, "saturation.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, axes = plt.subplots(1, len(res), figsize=(10, 4), sharey=True)
    for ax, (name, by_n) in zip(np.atleast_1d(axes), res.items()):
        for n, runs in by_n.items():
            m = np.mean([[s["share"] for s in r] for r in runs], axis=0)
            ax.plot(range(1, STEPS + 1), m * 100, marker="o", ms=3, label=f"{n} rows")
        ax.axvline(12, color="grey", ls=":", lw=1)
        ax.set_title(name)
        ax.set_xlabel("columns chosen so far")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("share of the full positive region (%)")
    np.atleast_1d(axes)[0].legend(title="FRFS sample")
    fig.suptitle("FRFS hits its maximum after a few columns; the dotted line is k = 12", y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig4_saturation.png"), dpi=200)


if __name__ == "__main__":
    main()
