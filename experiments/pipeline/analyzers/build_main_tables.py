"""The two MAIN-TEXT tables for Sec 5.3 (2026-07-16 advisor round).

Advisor-approved layout (meeting 2026-07-16): exactly two main-text tables,
built from the same canonical grid masters as the supplement tables
(aggregation recipe identical to build_supplement_tables.stat):

  tab_sec53_hc_main.tex -- Healthcare across THREE predictors (Linear /
                           MLP-16 / MLP-64), reduced pool, Reg / MAD / MSE
                           per predictor block. Supersedes the combined
                           anchor table (the MLP-64 block IS the old HC
                           anchor column).
  tab_sec53_md_main.tex -- Multidimensional knapsack at the fixed MLP-64
                           predictor across ALL FIVE imbalance levels,
                           same reduced pool, Regret and MAD blocks.

Pool (tab:methods naming, organized by objective count): WDRO / FPLG / MGDA
are supplement-only; of the Regret-and-MSE mu sweep only mu=0.5 is kept in
the main text ("only a few", advisor 2026-07-16) -- the full sweep stays in
the supplement.

Note the master CSVs carry the MLP-64 anchor cell twice (two grid files);
rows are exact duplicates, so we dedupe by (method, lambda, seed, cell) --
means are unchanged, stds are the honest 5-seed values.

Output: writing/v6/Tables/.
Run: python -m experiments.pipeline.analyzers.build_main_tables
"""
from __future__ import annotations

import pandas as pd

from experiments.paths import HC_MASTER_CSV, MD_MASTER_CSV, TABLES_OUT
from experiments.pipeline.analyzers.build_supplement_tables import (
    MID, METRIC, _best_per_col, _wrap, fmt, lam_cell, mu_cell, stat,
)

POOL_MAIN = [
    # 1 objective
    ("PTO", "fpto", 0.0),
    ("SAA", "saa", 0.0),
    ("DFL", "fdfl", 0.0),
    MID,
    # 2 objectives
    ("FPTO", "fpto", 1.0),
    ("Regret-and-MAD", "fdfl", 1.0),
    ("Regret-and-MSE", "fdfl-0.5", 0.0),
    MID,
    # 3 objectives (static scalarization + dynamic MOO handlers)
    ("FDFL-Scal", "fdfl-scal", 1.0),
    ("PCGrad", "pcgrad", 0.0), ("NashMTL", "nashmtl", 0.0),
]

HC_DEDUP = ["method", "lambda", "seed", "cap", "alpha", "fairness_type", "n_train"]
MD_DEDUP = ["method", "lambda", "seed", "cap", "alpha", "imbalance", "n_groups"]


def _rows(frames_and_keys):
    """frames_and_keys: list of (frame, metric_key) defining the columns."""
    body_specs, raw = [], []
    for spec in POOL_MAIN:
        if spec is MID:
            body_specs.append(MID); raw.append(None); continue
        lab, m, lam = spec
        vals = [stat(f, m, lam, METRIC[k][0]) for f, k in frames_and_keys]
        body_specs.append((lab, m, lam, vals)); raw.append(vals)
    best = _best_per_col(raw, len(frames_and_keys))
    body = []
    for spec in body_specs:
        if spec is MID:
            body.append(r"\midrule"); continue
        lab, m, lam, vals = spec
        cells = [fmt(v, METRIC[frames_and_keys[j][1]][1],
                     bold=(v is not None and best[j] is not None
                           and abs(v[0] - best[j]) < 1e-12))
                 for j, v in enumerate(vals)]
        body.append(f"{lab} & {lam_cell(m, lam)} & {mu_cell(m)} & "
                    + " & ".join(cells) + r" \\")
    return body


def hc_main_table(hc, caption, label):
    """Rows = reduced pool; column blocks = predictor, each Reg / MAD / MSE."""
    caps = [("linear", "Linear"), ("mlp16", "MLP-16"), ("mlp64", "MLP-64")]
    keys = ("regret", "mad", "mse")
    cols = [(hc[hc.cap == ck], k) for ck, _ in caps for k in keys]
    body = _rows(cols)
    grp = (" & & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{ct}}}" for _, ct in caps)
           + r" \\")
    cmid = "".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(3))
    sub = (r"Method & $\lambda$ & $\mu$ & "
           + " & ".join(["Regret & MAD & MSE"] * 3) + r" \\")
    return _wrap(caption, label, r"l cc ccc ccc ccc", [grp, cmid, sub], body,
                 size="footnotesize", colsep="3pt", resize=True)


def md_main_table(md, caption, label):
    """Rows = reduced pool; column blocks = metric, columns = imbalance level."""
    imbs = [0.0, 0.2, 0.4, 0.6, 0.8]
    cols = [(md[md.imbalance == e], k) for k in ("regret", "mad") for e in imbs]
    body = _rows(cols)
    grp = (r" & & & \multicolumn{5}{c}{Normalized regret} & "
           r"\multicolumn{5}{c}{MAD} \\")
    cmid = r"\cmidrule(lr){4-8}\cmidrule(lr){9-13}"
    heads = " & ".join([r"$\ell{=}0$ & $.2$ & $.4$ & $.6$ & $.8$"] * 2)
    sub = rf"Method & $\lambda$ & $\mu$ & {heads} \\"
    return _wrap(caption, label, r"l cc ccccc ccccc", [grp, cmid, sub], body,
                 size="footnotesize", colsep="3pt", resize=True)


def main() -> None:
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    hc = pd.read_csv(HC_MASTER_CSV); hc["method"] = hc["method"].str.lower()
    hc = hc.drop_duplicates(subset=HC_DEDUP)
    md = pd.read_csv(MD_MASTER_CSV); md["method"] = md["method"].str.lower()
    md = md[md.n_groups != 4.0]  # default K=2 runs; excludes the K=4 experiment
    md = md.drop_duplicates(subset=MD_DEDUP)

    hc_cell = hc[(hc.alpha == 2.0) & (hc.fairness_type == "mad") & (hc.n_train == 50)]
    md_cell = md[(md.cap == "mlp64") & (md.alpha == 2.0)]

    files = {
        "tab_sec53_hc_main.tex": hc_main_table(
            hc_cell,
            caption=(r"Healthcare results across predictor capacity "
                     r"($\alpha{=}2$, MAD, $N{=}50$). Mean $\pm$ standard "
                     r"deviation over 5 seeds; lower is better in all three "
                     r"metrics; bold marks the lowest column mean and does not "
                     r"denote statistical significance. The dynamic handlers "
                     r"solve for their per-step weights and so set no fixed "
                     r"$\lambda$ or $\mu$; SAA is feature-free, so its values "
                     r"repeat unchanged across the predictor columns. The full "
                     r"method pool and the $\mu$ sweep are in the online "
                     r"companion."),
            label="tab:sec53-hc-main"),
        "tab_sec53_md_main.tex": md_main_table(
            md_cell,
            caption=(r"Multidimensional knapsack across group imbalance $\ell$ "
                     r"($\alpha{=}2$, MLP-64, $K{=}2$). Mean $\pm$ standard "
                     r"deviation over 5 seeds; lower is better; bold marks the "
                     r"lowest column mean and does not denote statistical "
                     r"significance. The dynamic handlers solve for their "
                     r"per-step weights and so set no fixed $\lambda$ or $\mu$; "
                     r"SAA is feature-free."),
            label="tab:sec53-md-main"),
    }
    for name, content in files.items():
        (TABLES_OUT / name).write_text(content, encoding="utf-8")
        print(f"[ok] Tables/{name}")


if __name__ == "__main__":
    main()
