"""Main-findings figures for §5, per the advisor's plotting directives:
one alpha per figure; compare predictors and methods (grouped bars), not alphas.

  F1 fig_f1_repair_{hc}_{a2,a05}    — adding the fairness term (lambda 0 -> 1): MAD drops,
                                      regret unchanged. Paired bars, panels = (regret, MAD).
  F2 fig_f2_capacity_{hc,md}_{a2,a05} — grouped bars, x = methods, bars = predictors.
  F3 fig_f3_imbalance_md_a2         — lines vs imbalance, panels = (regret, MAD).
  P  fig_hc_pareto_{a2,a05}         — per-seed regret--MAD pareto, single alpha.

  python -m experiments.pipeline.plotters.plot_paper_findings
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HC = ROOT / "results/healthcare/main_v5_multiinstance/grid/tables/master_tagged.csv"
MD = ROOT / "results/md_knapsack/main_v6_rowsum/grid/tables/master_tagged.csv"
# f1/f2/pareto figures are supplement-only; f3/f4 are superseded by the
# fig_sec53_* main-text figures (2026-07-09 reorg; see Plots/Archive/ijoc/README.md).
from experiments.paths import SUPP_PLOTS_OUT
OUT = SUPP_PLOTS_OUT
OUT.mkdir(parents=True, exist_ok=True)

from experiments.pipeline.plotters import paper_style
paper_style.apply()

COLOR, MARKER = paper_style.COLOR, paper_style.MARKER
CAP_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]   # light -> dark blue ramp


def load(path, fair="mad"):
    df = pd.read_csv(path)
    df["method"] = df["method"].str.lower()
    if "fairness_type" in df.columns and fair is not None:
        df = df[df.fairness_type == fair] if (df.fairness_type == fair).any() else df
    return df


def cell(df, m, lam, **sel):
    d = df[(df.method == m) & (df["lambda"] == lam)]
    for k, v in sel.items():
        d = d[d[k] == v]
    return d


def stat(df, m, lam, col, **sel):
    d = cell(df, m, lam, **sel)
    return (d[col].mean(), d[col].std()) if len(d) else (np.nan, np.nan)


# ---------------------------------------------------------------- F1: repair
def f1_repair(df, alpha, tag, cap="mlp64", title_task="Healthcare"):
    methods = [("FPTO", "fpto"), ("Regret-and-MAD", "fdfl"), ("FDFL-Scal", "fdfl-scal"), ("FPLG", "fplg")]
    d = df[(df.alpha == alpha) & (df.cap == cap)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    x = np.arange(len(methods))
    w = 0.36
    for pi, (col, ylab) in enumerate([("test_regret_normalized", "Normalized regret"),
                                      ("test_fairness", "MAD")]):
        ax = axes[pi]
        for li, (lam, shade, lab) in enumerate([(0.0, 0.45, r"$\lambda=0$ (no fairness term)"),
                                                (1.0, 1.0, r"$\lambda=1$ (with fairness term)")]):
            means = [stat(d, m, lam, col)[0] for _, m in methods]
            errs = [stat(d, m, lam, col)[1] for _, m in methods]
            ax.bar(x + (li - 0.5) * w, means, w, yerr=errs, capsize=2.5,
                   color=[COLOR[n] for n, _ in methods], alpha=shade,
                   edgecolor="black", linewidth=0.6,
                   hatch="" if li else "//", label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([n for n, _ in methods], fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        if pi == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(rf"{title_task}: adding the prediction-side fairness term "
                 rf"($\alpha={alpha}$, MLP-64, 5 seeds)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_f1_repair_hc_{tag}.{ext}", dpi=150)
    plt.close(fig)
    print(f"[ok] fig_f1_repair_hc_{tag}")


# ---------------------------------------------------------------- F2: capacity histogram
ALL_METHODS = [("PTO", "fpto", 0.0), ("DFL", "fdfl", 0.0), ("Regret-and-MAD", "fdfl", 1.0),
               ("FDFL-Scal", "fdfl-scal", 1.0), ("FPLG", "fplg", 1.0), ("PCGrad", "pcgrad", 0.0)]


def f2_capacity(df, alpha, tag, caps, cap_titles, task, log_mad=False, log_reg=False,
                methods=None):
    methods = methods or ALL_METHODS
    d = df[df.alpha == alpha]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    x = np.arange(len(methods))
    nb = len(caps)
    w = 0.8 / nb
    for pi, (col, ylab) in enumerate([("test_regret_normalized", "Normalized regret"),
                                      ("test_fairness", "MAD violation")]):
        ax = axes[pi]
        for ci, (cap, ct) in enumerate(zip(caps, cap_titles)):
            means = [stat(d[d.cap == cap], m, lam, col)[0] for _, m, lam in methods]
            errs = [stat(d[d.cap == cap], m, lam, col)[1] for _, m, lam in methods]
            ax.bar(x + (ci - (nb - 1) / 2) * w, means, w, yerr=errs, capsize=2,
                   color=CAP_COLORS[ci + (4 - nb)], edgecolor="black", linewidth=0.5, label=ct)
        ax.set_xticks(x)
        ax.set_xticklabels([n for n, _, _ in methods], fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=9)
        if log_mad and pi == 1:
            ax.set_yscale("log")
        if log_reg and pi == 0:
            ax.set_yscale("log")
        if pi == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(rf"{task}: methods across predictor capacity ($\alpha={alpha}$, 5 seeds)",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_f2_capacity_{tag}.{ext}", dpi=150)
    plt.close(fig)
    print(f"[ok] fig_f2_capacity_{tag}")


# ---------------------------------------------------------------- F3: imbalance (clean)
def f3_imbalance(md):
    """Main panel (left): MAD vs imbalance as a 2x2 over {paradigm} x {fairness term}.
    Color = paradigm (blue two-stage, orange decision-focused); linestyle = fairness term
    (solid off, dashed on). The two solid (no-term) series climb, the two dashed (with-term)
    series stay low in both colors, so the fairness term is the lever. Small panel (right):
    regret for every feature-based method --- they move in lockstep, a null axis for regret."""
    d = md[(md.alpha == 2.0) & (md.cap == "mlp64")]
    mad_series = [("PTO",  "fpto", 0.0, "#1f77b4", "-"),
                  ("FPTO", "fpto", 1.0, "#1f77b4", "--"),
                  ("DFL",  "fdfl", 0.0, "#ff7f0e", "-"),
                  ("FDFL", "fdfl", 1.0, "#ff7f0e", "--")]
    reg_methods = ["fpto", "fdfl", "fdfl-scal", "fplg", "pcgrad", "mgda", "nashmtl", "wdro"]
    imbs = sorted(d.imbalance.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw={"width_ratios": [1.55, 1]})

    # main: MAD, 2x2 (paradigm x fairness term)
    ax = axes[0]
    for lab, m, lam, col, ls in mad_series:
        mns = np.array([stat(d[d.imbalance == i], m, lam, "test_fairness")[0] for i in imbs])
        sds = np.array([stat(d[d.imbalance == i], m, lam, "test_fairness")[1] for i in imbs])
        ax.plot(imbs, mns, marker="o", ms=4, color=col, ls=ls, lw=2.0, label=lab)
        ax.fill_between(imbs, mns - sds, mns + sds, color=col, alpha=0.08)
    ax.set_xlabel("Group imbalance level", fontsize=9)
    ax.set_ylabel("MAD", fontsize=9)
    ax.legend(title="solid = no fairness term, dashed = with it", fontsize=8.5,
              title_fontsize=8, loc="upper left", ncol=2, handlelength=2.4)
    ax.set_title("Disparity grows only without the fairness term", fontsize=10)

    # small: regret, every feature-based method (lockstep null axis)
    ax = axes[1]
    for m in reg_methods:
        for lam in (0.0, 1.0):
            ys = np.array([stat(d[d.imbalance == i], m, lam, "test_regret_normalized")[0]
                           for i in imbs])
            if np.isfinite(ys).all():
                ax.plot(imbs, ys, color="#7f7f7f", lw=1.0, alpha=0.5)
    ax.set_xlabel("Group imbalance level", fontsize=9)
    ax.set_ylabel("Normalized regret", fontsize=9)
    ax.set_title("Regret rises in lockstep (null axis)", fontsize=10)

    fig.suptitle(r"Knapsack at MLP-64, $\alpha=2$: group imbalance is the fairness pressure axis (5 seeds)",
                 fontsize=10.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_f3_imbalance_md_a2.{ext}", dpi=150)
    plt.close(fig)
    print("[ok] fig_f3_imbalance_md_a2")


# ---------------------------------------------------------------- per-alpha pareto
def pareto(df, alpha, tag, cap="mlp64"):
    order = [("fpto", "FPTO"), ("saa", "SAA"), ("wdro", "WDRO"), ("fdfl", None),
             ("fdfl-scal", "FDFL-Scal"), ("fplg", "FPLG"), ("pcgrad", "PCGrad"),
             ("mgda", "MGDA"), ("nashmtl", "NashMTL")]
    d = df[(df.alpha == alpha) & (df.cap == cap)].copy()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for m, label in order:
        dm = d[d.method == m]
        if dm.empty:
            continue
        if m == "fdfl":   # split DFL (l0) / FDFL (l1)
            for lam, nm in [(0.0, "DFL"), (1.0, "Regret-and-MAD")]:
                g = dm[dm["lambda"] == lam]
                ax.scatter(g.test_regret_normalized, g.test_fairness, color=COLOR[nm],
                           marker=MARKER[nm], alpha=0.3, s=30, edgecolors="none")
                ax.scatter(g.test_regret_normalized.mean(), g.test_fairness.mean(),
                           color=COLOR[nm], marker=MARKER[nm], s=120, edgecolors="black",
                           linewidths=0.8, label=nm)
            continue
        nm = label
        g = dm.groupby("lambda").agg(r=("test_regret_normalized", "mean"),
                                     f=("test_fairness", "mean")).reset_index()
        ax.scatter(dm.test_regret_normalized, dm.test_fairness, color=COLOR[nm],
                   marker=MARKER[nm], alpha=0.3, s=30, edgecolors="none")
        ax.scatter(g.r, g.f, color=COLOR[nm], marker=MARKER[nm], s=120,
                   edgecolors="black", linewidths=0.8, label=nm)
        if len(g) > 1:
            g = g.sort_values("lambda")
            ax.plot(g.r.to_numpy(), g.f.to_numpy(), color=COLOR[nm], lw=1.0, alpha=0.6)
    ax.set_xlabel("Normalized test regret (lower is better)")
    ax.set_ylabel("Test MAD (lower is better)")
    ax.set_title(rf"Healthcare regret--MAD trade-off ($\alpha={alpha}$, MLP-64, 5 seeds)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_hc_pareto_{tag}.{ext}", dpi=150)
    plt.close(fig)
    print(f"[ok] fig_hc_pareto_{tag}")


def load_hc_linear():
    """Trained softplus-linear HC cell (smoke_linear, lr 0.03 winners) tagged cap='linear'."""
    import glob
    base = ROOT / "results/healthcare/main_v5_multiinstance/smoke_linear/lr0.03"
    rows = [pd.read_csv(f) for f in glob.glob(str(base / "stage__*.csv"))]
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d["method"] = d["method"].str.lower()
    d["cap"] = "linear"
    return d


# ---------------------------------------------------------------- F4: regret vs capacity (the spine)
def f4_regret_capacity(hc, md, alpha=2.0, tag="a2"):
    """Decoupling figure. Left: normalized regret vs capacity for both testbeds. The two-stage
    (PTO) and end-to-end (FDFL-Scal) lines converge as capacity grows, so the decision-focus
    regret advantage closes (HC solid, MD dashed; color = method). Right: healthcare group MAD
    vs capacity. Decision-only DFL stays far above the two-stage level at every width while the
    fairness term returns it, so adding capacity does not close the fairness gap. The knapsack
    fairness gap lives on the imbalance knob, not capacity. alpha fixed (no alpha-mixing)."""
    caps = ["linear", "mlp16", "mlp64"]; cap_titles = ["Linear", "MLP-16", "MLP-64"]
    x = np.arange(len(caps))
    hca = hc[hc.alpha == alpha]; mda = md[(md.alpha == alpha) & (md.imbalance == 0.6)]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    # left: regret vs capacity, both testbeds (color = method, linestyle = testbed)
    ax = axes[0]
    for task, d, ls, mk in [("HC", hca, "-", "o"), ("MD", mda, "--", "s")]:
        for nm, m, lam, c in [("PTO", "fpto", 0.0, "#1f77b4"),
                              ("FDFL-Scal", "fdfl-scal", 1.0, "#2ca02c")]:
            ys = np.array([stat(d[d.cap == cap], m, lam, "test_regret_normalized")[0] for cap in caps])
            ax.plot(x, ys, ls, color=c, marker=mk, lw=1.8, ms=6, label=f"{nm} ({task})")
    ax.set_xticks(x); ax.set_xticklabels(cap_titles); ax.set_xlabel("Predictor capacity")
    ax.set_ylabel("Normalized decision regret")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.set_title("Regret gap closes with capacity", fontsize=10)

    # right: healthcare MAD vs capacity --- DFL stays high, fairness term repairs at every width
    ax = axes[1]
    for nm, m, lam, c, ls, mk in [("Two-stage (PTO)", "fpto", 0.0, "#1f77b4", "-", "o"),
                                  ("Decision-only (DFL)", "fdfl", 0.0, "#d62728", ":", "^"),
                                  ("End-to-end (FDFL-Scal)", "fdfl-scal", 1.0, "#2ca02c", "-", "s")]:
        ys = np.array([stat(hca[hca.cap == cap], m, lam, "test_fairness")[0] for cap in caps])
        sd = np.array([stat(hca[hca.cap == cap], m, lam, "test_fairness")[1] for cap in caps])
        ax.plot(x, ys, ls, color=c, marker=mk, lw=1.8, ms=6, label=nm)
        ax.fill_between(x, ys - sd, ys + sd, color=c, alpha=0.10)
    ax.set_xticks(x); ax.set_xticklabels(cap_titles); ax.set_xlabel("Predictor capacity")
    ax.set_ylabel("Group MAD (healthcare)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Fairness gap stays open with capacity", fontsize=10)

    fig.suptitle(rf"Capacity closes the regret gap but not the fairness gap ($\alpha={alpha}$, 5 seeds)",
                 fontsize=10.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_f4_regret_capacity_{tag}.{ext}", dpi=150)
    plt.close(fig)
    print(f"[ok] fig_f4_regret_capacity_{tag}")


def main():
    hc = load(HC)
    hc = hc[hc.n_train == 50]
    # The trained softplus-linear HC rung (linear_mad_n50, 5 seeds, floor, MSE-tuned lr) now
    # lives in the master via the slicer; the old smoke_linear concat is retired (D-021).
    md = load(MD, fair=None)
    md = md[md.get("n_groups", 2).fillna(2) != 4] if "n_groups" in md.columns else md

    f1_repair(hc, 2.0, "a2")
    f1_repair(hc, 0.5, "a05")
    # Advisor 2026-06-11: HC capacity figure compares Linear / MLP-16 / MLP-64 (matching MD);
    # full six-method pool (linear cell completed by smoke_hc_linear v2 on 2026-06-11).
    f2_capacity(hc, 2.0, "hc_a2", ["linear", "mlp16", "mlp64"],
                ["Linear", "MLP-16", "MLP-64"], "Healthcare", log_mad=True, log_reg=True)
    f2_capacity(hc, 0.5, "hc_a05", ["mlp16", "mlp64"], ["MLP-16", "MLP-64"], "Healthcare")
    md06 = md[md.imbalance == 0.6]   # capacity comparisons at the anchor imbalance only
    f2_capacity(md06, 2.0, "md_a2", ["linear", "mlp16", "mlp64"],
                ["Linear", "MLP-16", "MLP-64"], "Knapsack")
    f2_capacity(md06, 0.5, "md_a05", ["linear", "mlp64"], ["Linear", "MLP-64"], "Knapsack")
    f3_imbalance(md)
    f4_regret_capacity(hc, md, 2.0, "a2")   # the §5 capacity spine (reviewer-requested)
    pareto(hc, 2.0, "a2")
    pareto(hc, 0.5, "a05")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
