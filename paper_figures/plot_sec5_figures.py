"""Reproduce the three main-text Section 5.3 figures of the FDFL paper.

SELF-CONTAINED: no project imports. Reads the two aggregated result slices in
./data/ (per-method, per-seed summary metrics; no individual-level data) and
writes the three figures to ./figures/ as vector PDF + PNG.

  Figure 1  fig_sec53_hero_crisis_repair : regret-vs-MAD scatter at the two
            anchor cells (HC MLP-64 / MD MLP-64, imbalance 0.6); arrows show
            PTO->FPTO and DFL->Regret-and-MAD.
  Figure 2  fig_sec53_imbalance_bars      : MD MLP-64 across imbalance levels;
            grouped bars (regret | MAD) with 95% CIs.
  Figure 3  fig_sec53_capacity_bars       : HC across predictor capacity
            (Linear / MLP-16 / MLP-64); grouped bars (regret | MAD) with 95% CIs.

Data columns used: method, lambda, seed, cap, alpha, fairness_type, n_train
(HC) / imbalance, n_groups (MD), test_regret_normalized, test_fairness,
test_pred_mse. All metrics are lower-is-better; 5 seeds per cell.

Requires: python>=3.9, numpy, pandas, matplotlib. STIX fonts ship with
matplotlib. Run:  python plot_sec5_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

# Okabe-Ito colorblind-safe palette; one fixed color per method.
C_PTO = "#0072B2"    # blue
C_FPTO = "#56B4E9"   # sky blue
C_DFL = "#7F7F7F"    # gray
C_FDFL = "#D55E00"   # vermillion (Regret-and-MAD)

# skey -> (label, color, marker, (method key in CSV, lambda)). A SECOND
# encoding channel keeps the figures readable in grayscale/print: a distinct
# marker per method, and a hatch marking the fairness-augmented version of each
# pair (FPTO of PTO, Regret-and-MAD of DFL).
SERIES = {
    "pto":  ("PTO", C_PTO, "o", ("fpto", 0.0)),
    "fpto": ("FPTO", C_FPTO, "D", ("fpto", 1.0)),
    "dfl":  ("DFL", C_DFL, "s", ("fdfl", 0.0)),
    "fdfl": ("Regret-and-MAD", C_FDFL, "h", ("fdfl", 1.0)),
}
BAR_SERIES = ["pto", "fpto", "dfl", "fdfl"]
# hatch = "trained with the prediction-fairness objective" (semantic, not decorative)
HATCH = {"pto": "", "fpto": "///", "dfl": "", "fdfl": "///"}
TCRIT = 2.776  # 95% two-sided t, df=4 (5 seeds)

# Master CSVs carry the MLP-64 anchor cell twice (two grid files, identical
# rows); dedupe so the 5-seed std is honest and means are unchanged.
HC_DEDUP = ["method", "lambda", "seed", "cap", "alpha", "fairness_type", "n_train"]
MD_DEDUP = ["method", "lambda", "seed", "cap", "alpha", "imbalance", "n_groups"]


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"[ok] figures/{name}")


def load_hc():
    df = pd.read_csv(DATA / "hc_sec5_figs.csv")
    df["method"] = df["method"].str.lower()
    return df.drop_duplicates(subset=HC_DEDUP)


def load_md():
    df = pd.read_csv(DATA / "md_sec5_figs.csv")
    df["method"] = df["method"].str.lower()
    return df.drop_duplicates(subset=MD_DEDUP)


def cell(df, skey):
    m, lam = SERIES[skey][3]
    return df[(df.method == m) & (df["lambda"] == lam)]


def mci(x):
    x = np.asarray(x, float)
    return x.mean(), TCRIT * x.std(ddof=1) / np.sqrt(len(x))


# ------------------------------------------------------------- Figure 1
def hero(hc, md):
    panels = [("Healthcare", hc[hc.cap == "mlp64"]),
              ("Multidimensional knapsack",
               md[(md.cap == "mlp64") & (md.imbalance == 0.6)])]
    pairs = [("pto", "fpto"), ("dfl", "fdfl")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, (title, d) in zip(axes, panels):
        for skey_a, skey_b in pairs:
            (lab_a, col_a, mk_a, _), (lab_b, col_b, mk_b, _) = SERIES[skey_a], SERIES[skey_b]
            a, b = cell(d, skey_a), cell(d, skey_b)
            ax.scatter(a.test_regret_normalized, a.test_fairness, s=13,
                       color=col_a, alpha=0.60, linewidths=0)
            ax.scatter(b.test_regret_normalized, b.test_fairness, s=13,
                       color=col_b, alpha=0.60, linewidths=0)
            ma = (a.test_regret_normalized.mean(), a.test_fairness.mean())
            mb = (b.test_regret_normalized.mean(), b.test_fairness.mean())
            ax.annotate("", xy=mb, xytext=ma,
                        arrowprops=dict(arrowstyle="-|>", color=col_b, lw=1.4,
                                        shrinkA=6, shrinkB=6))
            ax.scatter(*ma, s=60, color=col_a, marker=mk_a, edgecolor="white",
                       linewidths=0.9, zorder=5, label=lab_a)
            ax.scatter(*mb, s=60, color=col_b, marker=mk_b, edgecolor="white",
                       linewidths=0.9, zorder=5, label=lab_b)
        ax.set_title(title)
        ax.set_xlabel("Normalized regret")
        ax.grid(True, axis="both", alpha=0.25)
    axes[0].set_ylabel("MAD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig_sec53_hero_crisis_repair")


# ------------------------------------------------------- grouped-bar helper
def _grouped_bars(ax, groups, series, get_vals):
    n = len(series)
    width = 0.8 / n
    for j, skey in enumerate(series):
        lab, col, _, _ = SERIES[skey]
        xs, ms, hs = [], [], []
        for i, g in enumerate(groups):
            vals = get_vals(g, skey)
            if len(vals) == 0:
                continue
            mu, h = mci(vals)
            xs.append(i + (j - (n - 1) / 2) * width)
            ms.append(mu); hs.append(h)
        ax.bar(xs, ms, width=width * 0.92, color=col, label=lab, zorder=3,
               edgecolor="white", linewidth=0.5, hatch=HATCH[skey])
        ax.errorbar(xs, ms, yerr=hs, fmt="none", ecolor="#333333",
                    elinewidth=0.9, capsize=2, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_ylim(bottom=0)


# ------------------------------------------------------------- Figure 2
def imbalance_bars(md):
    d = md[md.cap == "mlp64"]
    ells = [0.0, 0.2, 0.4, 0.6, 0.8]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for metric, ax, ylab in (("test_regret_normalized", axes[0], "Normalized regret"),
                             ("test_fairness", axes[1], "MAD")):
        _grouped_bars(ax, ells, BAR_SERIES,
                      lambda e, s, m=metric: cell(d[d.imbalance == e], s)[m].to_numpy())
        ax.set_xticklabels([f"{e:g}" for e in ells])
        ax.set_xlabel(r"Group imbalance $\ell$")
        ax.set_ylabel(ylab)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig_sec53_imbalance_bars")


# ------------------------------------------------------------- Figure 3
def capacity_bars(hc):
    caps = [("linear", "Linear"), ("mlp16", "MLP-16"), ("mlp64", "MLP-64")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for metric, ax, ylab in (("test_regret_normalized", axes[0], "Normalized regret"),
                             ("test_fairness", axes[1], "MAD")):
        _grouped_bars(ax, [c for c, _ in caps], BAR_SERIES,
                      lambda c, s, m=metric: cell(hc[hc.cap == c], s)[m].to_numpy())
        ax.set_xticklabels([t for _, t in caps])
        ax.set_xlabel("Predictor capacity")
        ax.set_ylabel(ylab)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig_sec53_capacity_bars")


def main():
    hc, md = load_hc(), load_md()
    hero(hc, md)
    imbalance_bars(md)
    capacity_bars(hc)


if __name__ == "__main__":
    main()
