"""MD v6 figures in the established draft style (mirrors plot_healthcare.py).

Reads the corrected MD grid master (results/md_knapsack/main_v6_rowsum/grid/tables/) and
produces, in writing/v6/:
  - fig_md_capacity_pareto_v6.{pdf,png} : per-seed regret--MAD Pareto across the predictor
    ladder (Linear / MLP-16 / MLP-64) at alpha=2, imb=0.6 — the specification axis.
DFL = FDFL@lambda0, FDFL = FDFL@lambda>0 (no separate DFL method in the grid).

  python -m experiments.pipeline.plotters.plot_knapsack
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.paths import MD_MASTER_CSV, SUPP_PLOTS_OUT
from experiments.method_catalog import paper_cfg
from experiments.pipeline.plotters import paper_style
paper_style.apply()

ROOT = Path(__file__).resolve().parents[3]
MASTER = MD_MASTER_CSV
OUT_DIR = SUPP_PLOTS_OUT  # supplement-only figures (2026-07-09 reorg)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# short label, color, marker — single source: experiments.method_catalog (same palette as HC)
METHOD_CFG = paper_cfg(include_family=False)
METHOD_ORDER = list(METHOD_CFG.keys())
CAPS = [("linear", "Linear (misspecified)"), ("mlp16", "MLP-16"), ("mlp64", "MLP-64")]


def load(alpha=2.0, imb=0.6):
    df = pd.read_csv(MASTER)
    df = df[(df.alpha == alpha) & (df.imbalance == imb)].copy()
    df["method"] = df["method"].str.lower()
    fdfl = df[df.method == "fdfl"]
    rest = df[df.method != "fdfl"]
    dfl = fdfl[fdfl["lambda"] == 0.0].assign(method="dfl")
    fd = fdfl[fdfl["lambda"] > 0.0].assign(method="fdfl")
    return pd.concat([rest, dfl, fd], ignore_index=True)


def plot_capacity_pareto(df, alpha=2.0):
    caps = [(c, t) for c, t in CAPS if c in set(df["cap"])]
    fig, axes = plt.subplots(1, len(caps), figsize=(3.4 * len(caps), 3.4), squeeze=False)
    handles = {}
    for ci, (cap, title) in enumerate(caps):
        ax = axes[0][ci]
        d = df[df["cap"] == cap]
        for m in METHOD_ORDER:
            label, color, marker = METHOD_CFG[m]
            dm = d[d["method"] == m]
            if dm.empty:
                continue
            ax.scatter(dm["test_regret_normalized"], dm["test_fairness"],
                       color=color, marker=marker, alpha=0.28, s=22, edgecolors="none")
            agg = dm.groupby("lambda").agg(r=("test_regret_normalized", "mean"),
                                           f=("test_fairness", "mean")).reset_index()
            h = ax.scatter(agg["r"], agg["f"], color=color, marker=marker, s=90,
                           edgecolors="black", linewidths=0.7)
            handles.setdefault(label, h)
            if len(agg) > 1:
                a2 = agg.sort_values("lambda")
                ax.plot(a2["r"].to_numpy(), a2["f"].to_numpy(), color=color, lw=0.9, alpha=0.6)
        ax.set_title(title, fontsize=11)
        if ci == 0:
            ax.set_ylabel("Test MAD prediction-fairness violation", fontsize=9)
        ax.set_xlabel("Normalized regret", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=10,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(rf"MD knapsack regret--MAD Pareto across predictor capacity "
                 rf"($\alpha={alpha}$, imb$\,{{=}}\,0.6$, MAD, $m{{=}}200$, $N{{=}}50$, 5 seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_DIR / "fig_md_capacity_pareto_v6.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_md_capacity_pareto_v6.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[ok] fig_md_capacity_pareto_v6")


def main():
    plot_capacity_pareto(load())
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
