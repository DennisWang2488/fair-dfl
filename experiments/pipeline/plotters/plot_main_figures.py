"""The Sec 5.3 main-text figures (2026-07-14 redesign, post codex review).

Selected by the user on 2026-07-14: hero adopted, imbalance = paired-effect
variant, capacity = regret-reduction variant, mu-sweep demoted to the supplement.
The three main-text figures land in writing/v6/Plots/ijoc/ and supersede the
old fig_sec53_* set; the mu sweep lands in the supplement Figures/ folder.

MAIN TEXT
  fig_sec53_hero_crisis_repair : Sec 5.3.1 hero (D-012 headline). Two panels
                                 (HC / MD anchor cells): x = normalized regret,
                                 y = MAD, lower-left better; seed dots + mean
                                 markers; arrows PTO->FPTO and DFL->FDFL show
                                 that the fairness term repairs the disparity at
                                 ~zero regret cost.
  fig_sec53_imbalance_effect   : Sec 5.3.2. Per-seed paired deltas (no-fairness
                                 minus with-fairness) for the PTO and DFL pairs:
                                 left = fairness gain (Delta MAD), right = its
                                 regret cost (Delta regret, centered near zero).
  fig_sec53_capacity_reduction : Sec 5.3.3. Paired per-seed % regret reduction
                                 vs PTO at each architecture, HC / MD panels;
                                 one FIXED decision-focused configuration per
                                 family (no best-of-pool selection).

SUPPLEMENT
  fig_supp_mu_sweep_hc_a2      : mu sweep, 3 vertically aligned panels (MSE /
                                 MAD / regret), no dual axis.

Style: STIX serif, Okabe-Ito colorblind-safe palette with FIXED semantics
(prediction-only blue, decision-only gray, fairness-augmented vermillion/orange,
dynamic MOO green), light y-grid, no top/right spines, seed dots + 95% CI
(never wide SD bands), vector PDF + PNG.

  python -m experiments.pipeline.plotters.plot_main_figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.paths import (HC_MASTER_CSV, MD_MASTER_CSV, PLOTS_OUT,
                               SUPP_PLOTS_OUT)

OUT = PLOTS_OUT                 # main-text figures
SUPP_OUT = SUPP_PLOTS_OUT       # supplement figures
OUT.mkdir(parents=True, exist_ok=True)
SUPP_OUT.mkdir(parents=True, exist_ok=True)

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

# Okabe-Ito, fixed semantics across ALL figures.
C_PRED = "#0072B2"   # prediction-only family (PTO / FPTO)
C_DEC = "#7F7F7F"    # decision-only (DFL)
C_FAIR = "#D55E00"   # fairness-augmented FDFL family
C_SCAL = "#E69F00"   # FDFL-Scal (all three objectives, static)
C_MOO = "#009E73"    # dynamic MOO handlers

# (label, color, marker, linestyle); dashed = no prediction-fairness objective
SERIES = {
    "pto":  ("PTO",  C_PRED, "o", "--"),
    "fpto": ("FPTO", C_PRED, "o", "-"),
    "dfl":  ("DFL",  C_DEC,  "s", "--"),
    "fdfl": ("Regret-and-MAD", C_FAIR, "s", "-"),
    "scal": ("FDFL-Scal", C_SCAL, "D", "-"),
    "nash": ("NashMTL", C_MOO, "^", "-"),
}
TCRIT = 2.776  # 95% two-sided t, df=4 (5 seeds)


def _save(fig, name, out=None):
    dest = out or OUT
    for ext in ("pdf", "png"):
        fig.savefig(dest / f"{name}.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"[ok] {dest.name}/{name}")


def load_hc():
    df = pd.read_csv(HC_MASTER_CSV)
    df["method"] = df["method"].str.lower()
    return df[(df.fairness_type == "mad") & (df.n_train == 50) & (df.alpha == 2.0)]


def load_md():
    df = pd.read_csv(MD_MASTER_CSV)
    df["method"] = df["method"].str.lower()
    df = df[(df.n_groups.isna()) | (df.n_groups == 2.0)]
    return df[df.alpha == 2.0]


def cell(df, m, lam):
    return df[(df.method == m) & (df["lambda"] == lam)]


def mci(x):
    x = np.asarray(x, float)
    return x.mean(), TCRIT * x.std(ddof=1) / np.sqrt(len(x))


# ------------------------------------------------------------- hero figure
def hero(hc, md):
    """Crisis -> repair at the two anchor cells. Arrows: PTO->FPTO, DFL->FDFL."""
    panels = [("Healthcare", hc[hc.cap == "mlp64"]),
              ("Multidimensional knapsack",
               md[(md.cap == "mlp64") & (md.imbalance == 0.6)])]
    pairs = [("pto", "fpto", 0.0, 1.0, "fpto"), ("dfl", "fdfl", 0.0, 1.0, "fdfl")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, (title, d) in zip(axes, panels):
        for skey_a, skey_b, lam_a, lam_b, mkey in pairs:
            (lab_a, col_a, mk_a, _), (lab_b, col_b, mk_b, _) = SERIES[skey_a], SERIES[skey_b]
            a = cell(d, "fpto" if skey_a == "pto" else "fdfl", lam_a)
            b = cell(d, mkey, lam_b)
            # seed-level points, faint
            ax.scatter(a.test_regret_normalized, a.test_fairness, s=10,
                       color=col_a, alpha=0.35, linewidths=0)
            ax.scatter(b.test_regret_normalized, b.test_fairness, s=10,
                       color=col_b, alpha=0.35, linewidths=0)
            ma = (a.test_regret_normalized.mean(), a.test_fairness.mean())
            mb = (b.test_regret_normalized.mean(), b.test_fairness.mean())
            ax.annotate("", xy=mb, xytext=ma,
                        arrowprops=dict(arrowstyle="-|>", color=col_b, lw=1.4,
                                        shrinkA=6, shrinkB=6))
            # no-fairness variants open, fairness-augmented variants filled
            ax.scatter(*ma, s=55, facecolor="white", edgecolor=col_a, marker=mk_a,
                       linewidths=1.4, zorder=5, label=lab_a)
            ax.scatter(*mb, s=55, color=col_b, marker=mk_b, edgecolor="white",
                       zorder=5, label=lab_b)
        ax.set_title(title)
        ax.set_xlabel("Normalized regret")
        ax.grid(True, axis="both", alpha=0.25)
    axes[0].set_ylabel("MAD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig_sec53_hero_crisis_repair")


# ------------------------------------------------------- imbalance variants
def imbalance_raw(md):
    d = md[md.cap == "mlp64"]
    series = [("fpto", 0.0, "pto"), ("fpto", 1.0, "fpto"),
              ("fdfl", 0.0, "dfl"), ("fdfl", 1.0, "fdfl")]
    ells = [0.0, 0.2, 0.4, 0.6, 0.8]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for metric, ax, ylab in (("test_fairness", axes[0], "MAD"),
                             ("test_regret_normalized", axes[1], "Normalized regret")):
        for i, (m, lam, skey) in enumerate(series):
            lab, col, mk, ls = SERIES[skey]
            ms, hs = zip(*[mci(cell(d[d.imbalance == e], m, lam)[metric]) for e in ells])
            x = np.array(ells) + (i - 1.5) * 0.008
            ax.errorbar(x, ms, yerr=hs, color=col, marker=mk, ls=ls, ms=4,
                        lw=1.4, capsize=2, elinewidth=0.9, label=lab)
        ax.set_xlabel(r"Group imbalance $\ell$")
        ax.set_ylabel(ylab)
        ax.set_xticks(ells)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig_imbalance_raw_v2")


def imbalance_effect(md):
    d = md[md.cap == "mlp64"]
    ells = [0.0, 0.2, 0.4, 0.6, 0.8]
    pairs = [("fpto", C_PRED, "o", "PTO $-$ FPTO"),
             ("fdfl", C_FAIR, "s", "DFL $-$ Regret-and-MAD")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for metric, ax, ylab in (
            ("test_fairness", axes[0], r"$\Delta$MAD (fairness gain)"),
            ("test_regret_normalized", axes[1], r"$\Delta$regret (cost of fairness)")):
        for j, (m, col, mk, lab) in enumerate(pairs):
            for i, e in enumerate(ells):
                base = cell(d[d.imbalance == e], m, 0.0).set_index("seed")[metric]
                fair = cell(d[d.imbalance == e], m, 1.0).set_index("seed")[metric]
                delta = (base - fair).dropna()
                x = e + (j - 0.5) * 0.02
                ax.scatter([x] * len(delta), delta, s=9, color=col, alpha=0.35,
                           linewidths=0)
                mu, h = mci(delta)
                ax.errorbar([x], [mu], yerr=[h], color=col, marker=mk, ms=5,
                            capsize=3, lw=1.4, label=lab if (i == 0) else None)
        ax.axhline(0.0, color="#444444", lw=0.8, ls=":")
        ax.set_xlabel(r"Group imbalance $\ell$")
        ax.set_ylabel(ylab)
        ax.set_xticks(ells)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig_sec53_imbalance_effect")


# -------------------------------------------------------- capacity variants
CAPS = [("linear", "Linear"), ("mlp16", "MLP-16"), ("mlp64", "MLP-64")]


def capacity_dw(hc, md):
    """Dot-and-whisker raw regret at discrete architectures."""
    series = [("fpto", 0.0, "pto"), ("fdfl-scal", 1.0, "scal"), ("nashmtl", 0.0, "nash")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for ax, d, title in ((axes[0], hc, "Healthcare"),
                         (axes[1], md[md.imbalance == 0.6], "Multidimensional knapsack")):
        for j, (m, lam, skey) in enumerate(series):
            lab, col, mk, _ = SERIES[skey]
            for i, (ckey, _) in enumerate(CAPS):
                sub = cell(d[d.cap == ckey], m, lam)["test_regret_normalized"]
                if sub.empty:
                    continue
                x = i + (j - 1) * 0.16
                ax.scatter([x] * len(sub), sub, s=9, color=col, alpha=0.35,
                           linewidths=0)
                mu, h = mci(sub)
                ax.errorbar([x], [mu], yerr=[h], color=col, marker=mk, ms=5,
                            capsize=3, lw=1.4, label=lab if i == 0 else None)
        ax.set_xticks(range(len(CAPS)), [t for _, t in CAPS])
        ax.set_xlabel("Predictor capacity")
        ax.set_title(title)
    axes[0].set_ylabel("Normalized regret")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig_capacity_dw_v2")


def capacity_reduction(hc, md):
    """Paired per-seed % regret reduction vs PTO."""
    series = [("fdfl-scal", 1.0, "scal"), ("nashmtl", 0.0, "nash")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for ax, d, title in ((axes[0], hc, "Healthcare"),
                         (axes[1], md[md.imbalance == 0.6], "Multidimensional knapsack")):
        for j, (m, lam, skey) in enumerate(series):
            lab, col, mk, _ = SERIES[skey]
            for i, (ckey, _) in enumerate(CAPS):
                sub = d[d.cap == ckey]
                pto = cell(sub, "fpto", 0.0).set_index("seed")["test_regret_normalized"]
                meth = cell(sub, m, lam).set_index("seed")["test_regret_normalized"]
                red = (100.0 * (pto - meth) / pto).dropna()
                if red.empty:
                    continue
                x = i + (j - 0.5) * 0.16
                ax.scatter([x] * len(red), red, s=9, color=col, alpha=0.35,
                           linewidths=0)
                mu, h = mci(red)
                ax.errorbar([x], [mu], yerr=[h], color=col, marker=mk, ms=5,
                            capsize=3, lw=1.4, label=lab if i == 0 else None)
        ax.axhline(0.0, color="#444444", lw=0.8, ls=":")
        ax.set_xticks(range(len(CAPS)), [t for _, t in CAPS])
        ax.set_xlabel("Predictor capacity")
        ax.set_title(title)
    axes[0].set_ylabel("Regret reduction vs PTO (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig_sec53_capacity_reduction")


# ----------------------------------------------------------- mu sweep (supp)
def mu_sweep_supp(hc):
    d = hc[(hc.cap == "mlp64") & (hc["lambda"] == 0.0)]
    mus = [("fdfl", 0.0), ("fdfl-0.1", 0.1), ("fdfl-0.5", 0.5), ("fdfl-scal", 1.0)]
    metrics = [("test_pred_mse", "Prediction MSE"), ("test_fairness", "MAD"),
               ("test_regret_normalized", "Normalized regret")]
    fig, axes = plt.subplots(3, 1, figsize=(4.4, 5.6), sharex=True)
    pos = list(range(len(mus)))
    for (col_name, ylab), ax, col in zip(metrics, axes, (C_SCAL, C_FAIR, C_DEC)):
        for i, (m, _) in enumerate(mus):
            sub = d[d.method == m][col_name]
            ax.scatter([i] * len(sub), sub, s=9, color=col, alpha=0.35, linewidths=0)
            mu_, h = mci(sub)
            ax.errorbar([i], [mu_], yerr=[h], color=col, marker="o", ms=4,
                        capsize=3, lw=1.2)
        ax.set_ylabel(ylab)
    axes[-1].set_xticks(pos, [f"{v:g}" for _, v in mus])
    axes[-1].set_xlabel(r"Prediction-loss weight $\mu$")
    fig.tight_layout()
    _save(fig, "fig_supp_mu_sweep_hc_a2", out=SUPP_OUT)


def main():
    # The user selected (2026-07-14): hero + paired-effect imbalance + regret-
    # reduction capacity for the main text; mu sweep to the supplement. The
    # rejected raw-value variants (imbalance_raw / capacity_dw) are kept above
    # for reference but are not built.
    hc, md = load_hc(), load_md()
    hero(hc, md)
    imbalance_effect(md)
    capacity_reduction(hc, md)
    mu_sweep_supp(hc)


if __name__ == "__main__":
    main()
