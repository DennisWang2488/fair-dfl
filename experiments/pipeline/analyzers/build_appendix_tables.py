"""Full-pool APPENDIX tables for the shortened IJOC manuscript (2026-07-23).

Advisor request (email 2026-07-23): compact tables in the same design as the
two main-text tables (Table 2 / Table 3, build_main_tables), but with the
FULL method pool, to live inside the 10-page appendix (app:extended /
app:robustness) instead of the extended-results package. They back the Sec 5.3
cross-references that Table 2/3 do not cover:

  tab_app_hc_capacity  -- HC full pool x {Linear, MLP-16, MLP-64}   (extends Table 2;
                          the mu-sweep rows ARE the mu robustness evidence)
  tab_app_hc_alpha     -- HC full pool x alpha in {0.5, 1.5, 2, 4}  (MLP-64)
  tab_app_hc_ntrain    -- HC full pool x N in {10, 20, 50}          (MLP-64, alpha=2)
  tab_app_md_capacity  -- MD full pool x {Linear, MLP-16, MLP-64}   (alpha=2, imb 0.6)
  tab_app_md_imbalance -- MD full pool x imbalance                  (extends Table 3)
  tab_app_md_alpha     -- MD full pool x alpha in {0.5, 1.5, 2}     (MLP-64, imb 0.6)
  tab_app_md_groups    -- MD full pool x K in {2, 4}                (alpha=2, imb 0.6)
  tab_app_md_imb_a05   -- MD full pool x imbalance at alpha=0.5     (robustness)

Aggregation recipe, pool, and formatting are imported unchanged from
build_supplement_tables (identical to the main-text tables); masters are
deduped by (method, lambda, seed, cell).

Output: writing/v6/Tables/tab_app_*.tex.
Run: python -m experiments.pipeline.analyzers.build_appendix_tables
"""
from __future__ import annotations

import pandas as pd

from experiments.paths import HC_MASTER_CSV, MD_MASTER_CSV, TABLES_OUT
from experiments.pipeline.analyzers.build_supplement_tables import (
    HC_DEDUP, MD_DEDUP, combined_table,
)

def _sideways(tex: str) -> str:
    """Rotate a table float 90 degrees (needs \\usepackage{rotating}).

    The two imbalance tables carry all three metrics across five levels
    (18 columns), which is unreadable at portrait width even resized.
    """
    return (tex.replace(r"\begin{table}[htbp]", r"\begin{sidewaystable}")
               .replace(r"\end{table}", r"\end{sidewaystable}"))


TAIL = ""  # captions are one identifying sentence (advisor, 2026-07-23);
# the shared reading conventions live once in the app:extended intro text.


def main() -> None:
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    hc = pd.read_csv(HC_MASTER_CSV); hc["method"] = hc["method"].str.lower()
    hc = hc.drop_duplicates(subset=HC_DEDUP)
    md = pd.read_csv(MD_MASTER_CSV); md["method"] = md["method"].str.lower()
    md = md.drop_duplicates(subset=MD_DEDUP)
    md_k2 = md[md.n_groups != 4.0]  # default K=2 runs; excludes the K=4 experiment

    files = {}

    # ===================== Healthcare =====================
    base = hc[(hc.alpha == 2.0) & (hc.fairness_type == "mad") & (hc.n_train == 50)]
    files["tab_app_hc_capacity.tex"] = combined_table(
        base, vary_col="cap", vary_vals=["linear", "mlp16", "mlp64"],
        group_titles=["Linear", "MLP-16", "MLP-64"],
        caption=(r"Healthcare, full method pool across predictor capacity "
                 r"($\alpha{=}2$, MAD, $N{=}50$)."),
        label="tab:app-hc-capacity", resize=True)

    base = hc[(hc.cap == "mlp64") & (hc.fairness_type == "mad") & (hc.n_train == 50)]
    files["tab_app_hc_alpha.tex"] = combined_table(
        base, vary_col="alpha", vary_vals=[0.5, 1.5, 2.0, 4.0],
        group_titles=[r"$\alpha{=}0.5$", r"$\alpha{=}1.5$", r"$\alpha{=}2$", r"$\alpha{=}4$"],
        caption=(r"Healthcare, full method pool across the decision fairness "
                 r"parameter $\alpha$ (MLP-64, MAD, $N{=}50$)."),
        label="tab:app-hc-alpha", resize=True)

    base = hc[(hc.cap == "mlp64") & (hc.fairness_type == "mad") & (hc.alpha == 2.0)]
    files["tab_app_hc_ntrain.tex"] = combined_table(
        base, vary_col="n_train", vary_vals=[10, 20, 50],
        group_titles=[r"$N{=}10$", r"$N{=}20$", r"$N{=}50$"],
        caption=(r"Healthcare, full method pool across the number of training "
                 r"instances $N$ (MLP-64, $\alpha{=}2$, MAD)."),
        label="tab:app-hc-ntrain", resize=True)

    # ===================== Multidimensional knapsack =====================
    base = md_k2[(md_k2.alpha == 2.0) & (md_k2.imbalance == 0.6)]
    files["tab_app_md_capacity.tex"] = combined_table(
        base, vary_col="cap", vary_vals=["linear", "mlp16", "mlp64"],
        group_titles=["Linear", "MLP-16", "MLP-64"],
        caption=(r"Multidimensional knapsack, full method pool across predictor "
                 r"capacity ($\alpha{=}2$, imbalance $0.6$, $K{=}2$)."),
        label="tab:app-md-capacity", resize=True)

    base = md_k2[(md_k2.cap == "mlp64") & (md_k2.alpha == 2.0)]
    files["tab_app_md_imbalance.tex"] = combined_table(
        base, vary_col="imbalance", vary_vals=[0.0, 0.2, 0.4, 0.6, 0.8],
        group_titles=[rf"$\ell{{=}}{v:g}$" for v in (0.0, 0.2, 0.4, 0.6, 0.8)],
        caption=(r"Multidimensional knapsack, full method pool across group "
                 r"imbalance $\ell$ ($\alpha{=}2$, MLP-64, $K{=}2$)."),
        label="tab:app-md-imbalance", resize=True)

    base = md_k2[(md_k2.cap == "mlp64") & (md_k2.imbalance == 0.6)]
    files["tab_app_md_alpha.tex"] = combined_table(
        base, vary_col="alpha", vary_vals=[0.5, 1.5, 2.0],
        group_titles=[r"$\alpha{=}0.5$", r"$\alpha{=}1.5$", r"$\alpha{=}2$"],
        caption=(r"Multidimensional knapsack, full method pool across the "
                 r"decision fairness parameter $\alpha$ (MLP-64, imbalance "
                 r"$0.6$, $K{=}2$)."),
        label="tab:app-md-alpha", resize=True)

    base = md[(md.cap == "mlp64") & (md.alpha == 2.0) & (md.imbalance == 0.6)]
    files["tab_app_md_groups.tex"] = combined_table(
        base, vary_col=None, vary_vals=["K2", "K4"],
        group_titles=[r"$K{=}2$", r"$K{=}4$"],
        caption=(r"Multidimensional knapsack, full method pool across the "
                 r"number of groups $K$ (MLP-64, $\alpha{=}2$, imbalance $0.6$; "
                 r"MAD values are not comparable across $K$)."),
        label="tab:app-md-groups",
        selector=lambda dd, v: (dd.n_groups != 4.0) if v == "K2" else (dd.n_groups == 4.0))

    base = md_k2[(md_k2.cap == "mlp64") & (md_k2.alpha == 0.5)]
    files["tab_app_md_imb_a05.tex"] = combined_table(
        base, vary_col="imbalance", vary_vals=[0.0, 0.2, 0.4, 0.6, 0.8],
        group_titles=[rf"$\ell{{=}}{v:g}$" for v in (0.0, 0.2, 0.4, 0.6, 0.8)],
        caption=(r"Multidimensional knapsack, full method pool across group "
                 r"imbalance $\ell$ at $\alpha{=}0.5$ (MLP-64, $K{=}2$)."),
        label="tab:app-md-imb-a05", resize=True)

    for name, content in files.items():
        (TABLES_OUT / name).write_text(content, encoding="utf-8")
        print(f"[ok] Tables/{name}")


if __name__ == "__main__":
    main()
