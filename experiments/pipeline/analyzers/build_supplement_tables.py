"""Build the CONDENSED result tables for the online companion (2026-07-16).

The companion now ships with the code package (IJOC page policy: main paper
25 pp incl. references, appendix 10 pp), so it is organized as two MASTER
tables plus small side tables, in the SAME style as the two main-text tables
(pool grouped by objective count, explicit lambda / mu weight columns):

  HC master  -- full pool x six predictors (alpha=2), one table per metric.
  MD master  -- full pool x (predictor x imbalance) cross (alpha=2), one
                longtable per metric; subsumes the old separate capacity and
                imbalance tables.
  Side tables -- HC: alpha, N, fairness-measure axes (MLP-64); MD: alpha and
                K axes; regret-only capacity ladders at alpha=0.5 (the
                evidence behind "at smaller alpha the advantage disappears");
                the preliminary misspecification ladder.

Dropped relative to the 2026-07-13 exhaustive set (recoverable from the
canonical CSVs + this builder): the alpha=0.5 companions for N / measure /
full-metric capacity / MD cross -- redundant with the alpha-axis side tables
for the story the paper tells.

Aggregation recipe is identical to ``the stat helper below``; masters are
deduped by (method, lambda, seed, cell) because the anchor cell appears in
two grid files (exact duplicate rows).

MD correctness: the K=4 group experiment lives at the imbalance=0.6 column
(n_groups==4); the default K=2 runs carry n_groups NaN. All non-K MD tables
therefore filter ``n_groups != 4``.

Output: writing/v6/supplement/Tables/supp_*.tex, input by supplement.tex.
Run: python -m experiments.pipeline.analyzers.build_supplement_tables
"""
from __future__ import annotations

import pandas as pd

from experiments.paths import HC_MASTER_CSV, MD_GRID, MD_MASTER_CSV, SUPP_TABLES_OUT

MID = object()

# (display label, method key [lowercase], lambda). The full pool, grouped by
# the number of training objectives. PTO = FPTO@0 and DFL = FDFL@0 carry the
# main-text aliases; the lambda / mu columns disambiguate every CSV combo
# (so repeated labels like Regret-and-MSE differ in their mu column). FPLG,
# WDRO, and MGDA are companion-only (main-text-dropped, 2026-07-14).
POOL_FULL = [
    # 1 objective
    ("PTO", "fpto", 0.0),
    ("SAA", "saa", 0.0),
    ("WDRO", "wdro", 0.0),
    ("DFL", "fdfl", 0.0),
    MID,
    # 2 objectives
    ("FPTO", "fpto", 1.0),
    ("Regret-and-MAD", "fdfl", 1.0),
    ("Regret-and-MSE", "fdfl-0.1", 0.0),
    ("Regret-and-MSE", "fdfl-0.5", 0.0),
    ("Regret-and-MSE", "fdfl-scal", 0.0),
    ("FPLG", "fplg", 0.0),
    MID,
    # 3 objectives (static scalarization variants + dynamic MOO handlers)
    ("FDFL-Scal", "fdfl-0.1", 1.0),
    ("FDFL-Scal", "fdfl-0.5", 1.0),
    ("FDFL-Scal", "fdfl-scal", 1.0),
    ("FPLG", "fplg", 1.0),
    ("PCGrad", "pcgrad", 0.0), ("MGDA", "mgda", 0.0), ("NashMTL", "nashmtl", 0.0),
]

# The dynamic MOO handlers set no fixed scalar weights at all -- their
# per-step weights are solved for -- so they get "--" rather than a number.
_MOO = ("pcgrad", "mgda", "nashmtl")


def lam_cell(key, lam):
    if key in ("saa", "wdro") or key in _MOO:
        return "--"
    return f"{lam:g}"


def mu_cell(key):
    if key in _MOO:
        return "--"
    return {"fdfl-0.1": "0.1", "fdfl-0.5": "0.5", "fdfl-scal": "1",
            "fdfl": "0"}.get(key, "--")

# metric key -> (column, decimals, short header)
METRIC = {
    "regret": ("test_regret_normalized", 3, "Reg"),
    "mad": ("test_fairness", 1, "MAD"),
    "mse": ("test_pred_mse", 1, "MSE"),
}
DEFAULT_METRICS = ("regret", "mad", "mse")

HC_DEDUP = ["method", "lambda", "seed", "cap", "alpha", "fairness_type", "n_train"]
MD_DEDUP = ["method", "lambda", "seed", "cap", "alpha", "imbalance", "n_groups"]


def stat(d, m, lam, col):
    r = d[(d.method == m) & (d["lambda"] == lam)]
    if r.empty:
        return None
    return float(r[col].mean()), float(r[col].std())


def fmt(ms, dec, bold=False):
    if ms is None:
        return "--"
    mean, sd = ms
    if abs(mean) < 10 and dec == 1:
        dec = 3
    body = f"{mean:.{dec}f} \\pm {sd:.{dec}f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def _best_per_col(raw, ncols):
    """min mean per column (all three metrics are lower-is-better)."""
    best = []
    for j in range(ncols):
        means = [v[j][0] for v in raw if v is not None and v[j] is not None]
        best.append(min(means) if means else None)
    return best


def _wrap(caption, label, colspec, header_lines, body, *, size="scriptsize",
          colsep="3pt", resize=False):
    inner = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    inner += header_lines + [r"\midrule"]
    inner += body
    inner += [r"\bottomrule", r"\end{tabular}"]
    tab = "\n".join(inner)
    if resize:
        tab = r"\resizebox{\textwidth}{!}{%" + "\n" + tab + "\n}"
    L = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}",
         rf"\label{{{label}}}", rf"\{size}", rf"\setlength{{\tabcolsep}}{{{colsep}}}",
         tab, r"\end{table}", ""]
    return "\n".join(L)


def _pool_rows(cols, pool=POOL_FULL):
    """cols: list of (frame, metric_key). Returns formatted body lines with the
    main-table row prefix (Method, lambda, mu) and per-column bolding."""
    body_specs, raw = [], []
    for spec in pool:
        if spec is MID:
            body_specs.append(MID); raw.append(None); continue
        lab, m, lam = spec
        vals = [stat(f, m, lam, METRIC[k][0]) for f, k in cols]
        body_specs.append((lab, m, lam, vals)); raw.append(vals)
    best = _best_per_col(raw, len(cols))
    body = []
    for spec in body_specs:
        if spec is MID:
            body.append(r"\midrule"); continue
        lab, m, lam, vals = spec
        cells = [fmt(v, METRIC[cols[j][1]][1],
                     bold=(v is not None and best[j] is not None
                           and abs(v[0] - best[j]) < 1e-12))
                 for j, v in enumerate(vals)]
        body.append(f"{lab} & {lam_cell(m, lam)} & {mu_cell(m)} & "
                    + " & ".join(cells) + r" \\")
    return body


def combined_table(d, *, vary_col, vary_vals, group_titles, caption, label,
                   metrics=DEFAULT_METRICS, resize=False, selector=None):
    """Rows = full pool (Method, lambda, mu); columns = vary_val x metrics."""
    nm = len(metrics)
    if selector is None:
        subs = {v: d[d[vary_col] == v] for v in vary_vals}
    else:
        subs = {v: d[selector(d, v)] for v in vary_vals}
    cols = [(subs[v], k) for v in vary_vals for k in metrics]
    body = _pool_rows(cols)
    colspec = "l cc " + (((("c" * nm) + " ") * len(vary_vals)).strip())
    col_titles = [METRIC[k][2] for k in metrics]
    grp = " & & & " + " & ".join(rf"\multicolumn{{{nm}}}{{c}}{{{t}}}" for t in group_titles) + r" \\"
    cmid = "".join(rf"\cmidrule(lr){{{4 + nm * i}-{3 + nm + nm * i}}}" for i in range(len(vary_vals)))
    sub = (r"Method & $\lambda$ & $\mu$ & "
           + " & ".join([" & ".join(col_titles)] * len(vary_vals)) + r" \\")
    return _wrap(caption, label, colspec, [grp, cmid, sub], body, resize=resize)


def permetric_tables(d, *, vary_col, vary_vals, vary_titles, caption_base, label_base,
                     metrics=DEFAULT_METRICS):
    """One sub-table per metric (rows = pool, cols = vary_vals). For wide axes."""
    out = []
    for key in metrics:
        long = {"regret": "normalized decision regret", "mad": "prediction-fairness violation (MAD)",
                "mse": "prediction MSE"}[key]
        cols = [(d[d[vary_col] == v], key) for v in vary_vals]
        body = _pool_rows(cols)
        colspec = "l cc " + " ".join(["c"] * len(vary_vals))
        header = r"Method & $\lambda$ & $\mu$ & " + " & ".join(vary_titles) + r" \\"
        cap = caption_base + f" \\emph{{Metric: {long}}} (5 seeds; mean $\\pm$ std; best per column in bold)."
        suffix = f"-{key}" if len(metrics) > 1 else ""
        out.append(_wrap(cap, f"{label_base}{suffix}", colspec, [header], body,
                         size="scriptsize", colsep="3pt",
                         resize=(len(vary_vals) >= 6)))
    return "\n".join(out)


def cross_tables(d, *, alpha, caps, cap_titles, imbs, label_base, task_title):
    """cap x imbalance cross, one longtable per metric (cap-blocks inside).

    longtable (not a table float) so the ~57-row, 3-block grid can break across
    pages cleanly.
    """
    d = d[d.alpha == alpha]
    out = []
    ncol = 3 + len(imbs)
    for key in METRIC:
        col, dec, _ = METRIC[key]
        long = {"regret": "normalized decision regret", "mad": "prediction-fairness violation (MAD)",
                "mse": "prediction MSE"}[key]
        body = []
        for ci, (capk, capt) in enumerate(zip(caps, cap_titles)):
            cols = [(d[(d.cap == capk) & (d.imbalance == im)], key) for im in imbs]
            if ci > 0:
                body.append(r"\midrule")
            body.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{capt}}}}} \\")
            body.append(r"\midrule")
            body += [ln if ln != r"\midrule" else r"\addlinespace"
                     for ln in _pool_rows(cols)]
        colspec = "l cc " + " ".join(["c"] * len(imbs))
        header = r"Method & $\lambda$ & $\mu$ & " + " & ".join(f"imb~${im:g}$" for im in imbs) + r" \\"
        cap = (f"{task_title} predictor $\\times$ imbalance cross at $\\alpha{{=}}{alpha:g}$. "
               f"\\emph{{Metric: {long}}} (5 seeds; mean $\\pm$ std; best per column within each "
               f"predictor block in bold). Full method pool; $\\lambda$ is the prediction-fairness "
               f"weight and $\\mu$ the prediction-loss weight.")
        L = [r"{\scriptsize\setlength{\tabcolsep}{2pt}",
             rf"\begin{{longtable}}{{{colspec}}}",
             rf"\caption{{{cap}}}\label{{{label_base}-{key}}}\\",
             r"\toprule", header, r"\midrule", r"\endfirsthead",
             r"\toprule", header, r"\midrule", r"\endhead",
             r"\midrule", rf"\multicolumn{{{ncol}}}{{r}}{{\emph{{continued on next page}}}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
        L += body
        L += [r"\end{longtable}", r"}", ""]
        out.append("\n".join(L))
    return "\n".join(out)


def md_misspec_table():
    """Preliminary misspecification / predictor-class ladder from the
    ``main_v6_rowsum/extreme`` HP-search ablation. PRELIMINARY: only FPTO/FDFL,
    3 seeds, per-config HP search (not the 5-seed tuned-lr main protocol). The
    Poly-2 column is the degree-2 polynomial predictor that matches the
    data-generating process (well-specified); Linear is under-capacity."""
    d = pd.read_csv(MD_GRID / "extreme" / "final_3seeds.csv")
    archs = [("linear", "Linear"), ("mlp16x1", "MLP-16"), ("mlp64x2", "MLP-64"), ("poly2", "Poly-2")]
    mets = [("test_regret_normalized", 3), ("test_fairness", 3), ("test_pred_mse", 3)]
    body = []
    for akey, alab in archs:
        cells = []
        for mkey in ("FPTO", "FDFL"):
            sub = d[(d.method == mkey) & (d.arch == akey)]
            for col, dec in mets:
                cells.append("--" if sub.empty else f"${sub[col].mean():.{dec}f} \\pm {sub[col].std():.{dec}f}$")
        body.append(f"{alab} & " + " & ".join(cells) + r" \\")
    caption = (r"\emph{Preliminary} multidimensional-knapsack predictor-capacity ladder "
               r"($\alpha{=}2$, imbalance $0.6$, $K{=}2$). Two-stage FPTO vs.\ decision-focused "
               r"DFL across predictor classes, including the degree-2 polynomial "
               r"(\emph{Poly-2}) whose representation power matches the data-generating function. "
               r"\textbf{This table is preliminary}: only these two methods, \textbf{3 seeds}, and "
               r"a per-configuration hyper-parameter search (not the 5-seed tuned-lr protocol of "
               r"the other tables); mean $\pm$ std. The decision-focused regret advantage is "
               r"largest on the capacity-limited Linear class and closes once the class can "
               r"represent the signal.")
    L = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}",
         r"\label{tab:supp-md-misspec}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l ccc ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{FPTO (two-stage)} & \multicolumn{3}{c}{DFL (decision-focused)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Predictor & Reg & MAD & MSE & Reg & MAD & MSE \\", r"\midrule"]
    L += body
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def main() -> None:
    SUPP_TABLES_OUT.mkdir(parents=True, exist_ok=True)
    hc = pd.read_csv(HC_MASTER_CSV); hc["method"] = hc["method"].str.lower()
    hc = hc.drop_duplicates(subset=HC_DEDUP)
    md = pd.read_csv(MD_MASTER_CSV); md["method"] = md["method"].str.lower()
    md = md.drop_duplicates(subset=MD_DEDUP)
    md_k2 = md[md.n_groups != 4.0]  # default K=2 runs (n_groups NaN); excludes the K=4 experiment

    files = {}

    # ===================== Healthcare =====================
    # HC MASTER: full pool x six predictors at alpha=2, one table per metric.
    base = hc[(hc.alpha == 2.0) & (hc.fairness_type == "mad") & (hc.n_train == 50)]
    files["supp_hc_master.tex"] = permetric_tables(
        base, vary_col="cap",
        vary_vals=["linear", "log_linear", "mlp16", "mlp32", "mlp64", "mlp128"],
        vary_titles=["Linear", "Log-lin.", "MLP-16", "MLP-32", "MLP-64", "MLP-128"],
        caption_base=(r"Healthcare master table: full pool across predictor capacity "
                      r"($\alpha{=}2$, MAD, $N{=}50$). Log-lin.\ is the closed-form "
                      r"log-linear control: only the predict-then-optimize baselines are "
                      r"meaningful there; the DFL family is intentionally unstable on "
                      r"that rigid class (see main text)."),
        label_base="tab:supp-hc-master")

    # HC side tables: alpha / N / fairness-measure axes at the anchor predictor.
    base = hc[(hc.cap == "mlp64") & (hc.fairness_type == "mad") & (hc.n_train == 50)]
    files["supp_hc_alpha.tex"] = combined_table(
        base, vary_col="alpha", vary_vals=[0.5, 1.5, 2.0, 4.0],
        group_titles=[r"$\alpha{=}0.5$", r"$\alpha{=}1.5$", r"$\alpha{=}2$", r"$\alpha{=}4$"],
        caption=(r"Healthcare across welfare curvature $\alpha$ (MLP-64, MAD, $N{=}50$, 5 seeds; "
                 r"mean $\pm$ std; best per column in bold). Reg = normalized regret, MAD = "
                 r"prediction-fairness violation, MSE = prediction error."),
        label="tab:supp-hc-alpha", resize=True)

    base = hc[(hc.cap == "mlp64") & (hc.fairness_type == "mad") & (hc.alpha == 2.0)]
    files["supp_hc_ntrain.tex"] = combined_table(
        base, vary_col="n_train", vary_vals=[10, 20, 50],
        group_titles=[r"$N{=}10$", r"$N{=}20$", r"$N{=}50$"],
        caption=(r"Healthcare across the number of training instances $N$ (MLP-64, $\alpha{=}2$, "
                 r"MAD, 5 seeds; mean $\pm$ std; best per column in bold)."),
        label="tab:supp-hc-ntrain", resize=True)

    base = hc[(hc.cap == "mlp64") & (hc.alpha == 2.0) & (hc.n_train == 50)]
    files["supp_hc_measure.tex"] = combined_table(
        base, vary_col="fairness_type", vary_vals=["mad", "dp", "w2_dp"],
        group_titles=["MAD", "mean-DP", "W2-DP"],
        caption=(r"Healthcare across the prediction-fairness measure (MLP-64, $\alpha{=}2$, "
                 r"$N{=}50$, 5 seeds; mean $\pm$ std; best per column in bold). The middle "
                 r"column of each block is that measure's own violation, so its units differ "
                 r"across blocks and are not comparable column-to-column."),
        label="tab:supp-hc-measure", resize=True)

    # HC alpha=0.5 capacity ladder, REGRET ONLY: the evidence behind "at smaller
    # alpha the decision-focused advantage disappears" (MLP-128 not run there).
    base = hc[(hc.alpha == 0.5) & (hc.fairness_type == "mad") & (hc.n_train == 50)]
    files["supp_hc_capacity_a05.tex"] = permetric_tables(
        base, vary_col="cap",
        vary_vals=["linear", "log_linear", "mlp16", "mlp32", "mlp64"],
        vary_titles=["Linear", "Log-lin.", "MLP-16", "MLP-32", "MLP-64"],
        caption_base=(r"Healthcare across predictor capacity at $\alpha{=}0.5$ (MAD, $N{=}50$) "
                      r"--- the moderately concave regime; regret only (companion metrics "
                      r"recoverable from the released results)."),
        label_base="tab:supp-hc-cap-a05", metrics=("regret",))

    # ===================== Multidimensional knapsack =====================
    # MD MASTER: full pool x (predictor x imbalance) cross at alpha=2, one
    # longtable per metric. Subsumes the old separate capacity and imbalance
    # tables (their cells are the imb=0.6 column and the MLP-64 block).
    files["supp_md_master.tex"] = cross_tables(
        md_k2, alpha=2.0, caps=["linear", "mlp16", "mlp64"],
        cap_titles=["Linear", "MLP-16", "MLP-64"], imbs=[0.0, 0.2, 0.4, 0.6, 0.8],
        label_base="tab:supp-md-master", task_title="Multidimensional knapsack")

    # MD side tables: alpha and K axes at the anchor cell.
    base = md_k2[(md_k2.cap == "mlp64") & (md_k2.imbalance == 0.6)]
    files["supp_md_alpha.tex"] = combined_table(
        base, vary_col="alpha", vary_vals=[0.5, 1.5, 2.0],
        group_titles=[r"$\alpha{=}0.5$", r"$\alpha{=}1.5$", r"$\alpha{=}2$"],
        caption=(r"Multidimensional knapsack across welfare curvature $\alpha$ (MLP-64, imbalance "
                 r"$0.6$, $K{=}2$, 5 seeds; mean $\pm$ std; best per column in bold)."),
        label="tab:supp-md-alpha", resize=True)

    base = md[(md.cap == "mlp64") & (md.alpha == 2.0) & (md.imbalance == 0.6)]
    files["supp_md_groups.tex"] = combined_table(
        base, vary_col=None, vary_vals=["K2", "K4"],
        group_titles=[r"$K{=}2$", r"$K{=}4$"],
        caption=(r"Multidimensional knapsack across the number of groups $K$ (MLP-64, $\alpha{=}2$, "
                 r"imbalance $0.6$, 5 seeds; mean $\pm$ std; best per column in bold). $K{=}2$ and "
                 r"$K{=}4$ are matched runs differing only in $K$."),
        label="tab:supp-md-groups",
        selector=lambda dd, v: (dd.n_groups != 4.0) if v == "K2" else (dd.n_groups == 4.0))

    # MD alpha=0.5 capacity ladder, REGRET ONLY (companion to the HC one).
    base = md_k2[(md_k2.alpha == 0.5) & (md_k2.imbalance == 0.6)]
    files["supp_md_capacity_a05.tex"] = permetric_tables(
        base, vary_col="cap", vary_vals=["linear", "mlp16", "mlp64"],
        vary_titles=["Linear", "MLP-16", "MLP-64"],
        caption_base=(r"Multidimensional knapsack across predictor capacity at $\alpha{=}0.5$ "
                      r"(imbalance $0.6$, $K{=}2$); regret only."),
        label_base="tab:supp-md-cap-a05", metrics=("regret",))

    # Preliminary misspecification / predictor-class ladder (3-seed ablation).
    files["supp_md_misspec.tex"] = md_misspec_table()

    for name, content in files.items():
        (SUPP_TABLES_OUT / name).write_text(content, encoding="utf-8")
        n_tab = content.count(r"\begin{table}") + content.count(r"\begin{longtable}")
        print(f"[ok] {name}  ({n_tab} table{'s' if n_tab != 1 else ''})")


if __name__ == "__main__":
    main()
