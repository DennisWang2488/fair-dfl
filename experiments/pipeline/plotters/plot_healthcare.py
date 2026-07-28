"""Regenerate the draft_v4 HC main figures from the NEW §0.6 grid (corrected FPTO).

Reproduces the established draft style (``plot_hc_v4.py``) but reads the v5 unified grid at the
anchor (MLP-64, MAD, N=50): the per-seed regret--MAD Pareto scatter and the 3-panel summary bars.
DFL is FDFL at lambda=0 and FDFL is FDFL at lambda=1 (the grid has no separate DFL method).

Outputs -> writing/v6/ : fig_hc_summary_bars_v5.{pdf,png}, fig_hc_perseed_pareto_v5.{pdf,png}

  python -m experiments.pipeline.plotters.plot_healthcare
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.paths import HC_MASTER_CSV, SUPP_PLOTS_OUT
from experiments.method_catalog import paper_cfg
from experiments.pipeline.plotters import paper_style
paper_style.apply()

ROOT = Path(__file__).resolve().parents[3]
MASTER = HC_MASTER_CSV
OUT_DIR = SUPP_PLOTS_OUT  # supplement-only figures (2026-07-09 reorg)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.5, 2.0]
# short label, color, marker, family — single source: experiments.method_catalog
METHOD_CFG = paper_cfg()
METHOD_ORDER = list(METHOD_CFG.keys())


def load_anchor(cap_hidden=64, fair="mad", n_train=50):
    """v5 grid -> per-(method, seed, alpha, lambda) rows at the anchor, DFL split off FDFL@l0."""
    df = pd.read_csv(MASTER)
    df = df[(df.arch == "mlp") & (df.hidden_dim == cap_hidden)
            & (df.fairness_type == fair) & (df.n_train == n_train)].copy()
    df["method"] = df["method"].str.lower()
    df["_seed"] = df["seed"]
    df["fairness"] = df["fairness_type"]
    fdfl = df[df.method == "fdfl"]
    rest = df[df.method != "fdfl"]
    dfl = fdfl[fdfl["lambda"] == 0.0].assign(method="dfl")
    fd = fdfl[fdfl["lambda"] > 0.0].assign(method="fdfl")
    return pd.concat([rest, dfl, fd], ignore_index=True)


def plot_perseed_pareto(df, fair="mad"):
    sub = df[df["fairness"] == fair].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, alpha in zip(axes, ALPHAS):
        d = sub[sub["alpha"] == alpha]
        for m in METHOD_ORDER:
            label, color, marker, _ = METHOD_CFG[m]
            dm = d[d["method"] == m]
            if dm.empty:
                continue
            ax.scatter(dm["test_regret_normalized"], dm["test_fairness"],
                       color=color, marker=marker, alpha=0.30, s=35, edgecolors="none")
            agg = dm.groupby("lambda").agg(r=("test_regret_normalized", "mean"),
                                           f=("test_fairness", "mean")).reset_index()
            ax.scatter(agg["r"], agg["f"], color=color, marker=marker, s=130,
                       edgecolors="black", linewidths=0.8, label=label)
            if len(agg) > 1:
                agg = agg.sort_values("lambda")
                ax.plot(agg["r"].to_numpy(), agg["f"].to_numpy(), color=color, lw=1.0, alpha=0.6)
        ax.set_xlabel("Normalized test regret (lower is better)")
        ax.set_ylabel(f"Test {fair.upper()} prediction-fairness violation (lower is better)")
        ax.set_title(rf"$\alpha = {alpha}$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(f"Healthcare Pareto trade-off ({fair.upper()}): per-seed scatter + per-$\\lambda$ mean",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_hc_perseed_pareto_v5.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_hc_perseed_pareto_v5.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[ok] fig_hc_perseed_pareto_v5")


def plot_summary_bars(df, fair="mad"):
    sub = df[df["fairness"] == fair].copy()
    metrics = [("test_regret_normalized", "Best normalized regret"),
               ("test_fairness", f"Best {fair.upper()}"),
               ("test_pred_mse", "Best prediction MSE")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    labels = [METHOD_CFG[m][0] for m in METHOD_ORDER]
    width, x = 0.4, np.arange(len(METHOD_ORDER))
    for ax, (col, title) in zip(axes, metrics):
        for i, alpha in enumerate(ALPHAS):
            vals, errs = [], []
            for m in METHOD_ORDER:
                dm = sub[(sub["method"] == m) & (sub["alpha"] == alpha)]
                if dm.empty:
                    vals.append(np.nan); errs.append(0); continue
                agg = dm.groupby("lambda").agg(mn=(col, "mean"), sd=(col, "std")).reset_index()
                best = agg.loc[agg["mn"].idxmin()]
                vals.append(best["mn"]); errs.append(0 if pd.isna(best["sd"]) else best["sd"])
            ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs,
                   color=("#3b7dd8" if alpha == 0.5 else "#e57c2c"), alpha=0.85,
                   edgecolor="black", linewidth=0.5, label=rf"$\alpha={alpha}$", capsize=2.0)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11); ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Healthcare summary ({fair.upper()}): best-per-$\\lambda$ value per method, by $\\alpha$ "
                 "(MLP-64, $N{=}50$, 5 seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_hc_summary_bars_v5.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_hc_summary_bars_v5.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[ok] fig_hc_summary_bars_v5")


CAPS = ["log_linear", "mlp16", "mlp32", "mlp64", "mlp128"]
CAP_LABEL = {"log_linear": "log-linear", "mlp16": "MLP-16", "mlp32": "MLP-32",
             "mlp64": "MLP-64", "mlp128": "MLP-128"}


def load_all_caps(fair="mad", n_train=50):
    """All predictors at (MAD, N=50), DFL split off FDFL@l0; keep the `cap` column."""
    df = pd.read_csv(MASTER)
    df = df[(df.fairness_type == fair) & (df.n_train == n_train) & (df.cap.isin(CAPS))].copy()
    df["method"] = df["method"].str.lower()
    df["fairness"] = df["fairness_type"]
    fdfl = df[df.method == "fdfl"]
    rest = df[df.method != "fdfl"]
    dfl = fdfl[fdfl["lambda"] == 0.0].assign(method="dfl")
    fd = fdfl[fdfl["lambda"] > 0.0].assign(method="fdfl")
    return pd.concat([rest, dfl, fd], ignore_index=True)


def plot_capacity_pareto(df, alpha=2.0):
    """Per-seed regret--MAD Pareto across the capacity ladder at one alpha (draft style)."""
    caps = [c for c in CAPS if c in set(df[df["alpha"] == alpha]["cap"])]
    fig, axes = plt.subplots(1, len(caps), figsize=(3.1 * len(caps), 3.4), squeeze=False)
    YCAP = 140.0  # MAD view ceiling; pathological points (log-linear PTO, MLP-128 DFL) clipped + labelled
    handles = {}
    for ci, cap in enumerate(caps):
        ax = axes[0][ci]
        d = df[(df["alpha"] == alpha) & (df["cap"] == cap)]
        off = []
        for m in METHOD_ORDER:
            label, color, marker, _ = METHOD_CFG[m]
            dm = d[d["method"] == m]
            if dm.empty:
                continue
            ax.scatter(dm["test_regret_normalized"], dm["test_fairness"].clip(upper=YCAP),
                       color=color, marker=marker, alpha=0.28, s=22, edgecolors="none")
            agg = dm.groupby("lambda").agg(r=("test_regret_normalized", "mean"),
                                           f=("test_fairness", "mean")).reset_index()
            h = ax.scatter(agg["r"], agg["f"].clip(upper=YCAP), color=color, marker=marker,
                           s=90, edgecolors="black", linewidths=0.7)
            handles.setdefault(label, h)
            for _, rr in agg.iterrows():
                if rr["f"] > YCAP:
                    off.append(f"{label} {rr['f']:.0f}")
            if len(agg) > 1:
                a2 = agg.sort_values("lambda")
                ax.plot(a2["r"].to_numpy(), a2["f"].clip(upper=YCAP).to_numpy(), color=color, lw=0.9, alpha=0.6)
        if off:  # one compact off-scale note per panel
            ax.text(0.97, 0.97, "off-scale ($\\uparrow$):\n" + "\n".join(off), transform=ax.transAxes,
                    fontsize=6, color="0.3", ha="right", va="top")
        ax.set_ylim(0, YCAP)
        ax.set_title(CAP_LABEL[cap], fontsize=11)
        if ci == 0:
            ax.set_ylabel("Test MAD prediction-fairness violation", fontsize=9)
        ax.set_xlabel("Normalized regret", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=10,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(rf"Healthcare regret--MAD Pareto across predictor capacity "
                 rf"($\alpha={alpha}$, MAD, $N{{=}}50$, 5 seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_DIR / "fig_hc_capacity_pareto_v5.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_hc_capacity_pareto_v5.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[ok] fig_hc_capacity_pareto_v5")


def main():
    df = load_anchor()
    plot_summary_bars(df)
    plot_perseed_pareto(df)
    plot_capacity_pareto(load_all_caps())
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
