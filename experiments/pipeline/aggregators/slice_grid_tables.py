"""Slice the unified §0.6 grid into the per-table views (HC-1..HC-7 / MD-1..MD-4).

The new harness runs ONE minimal covering grid (a star design around the anchor: each
cell varies one axis off mlp64 / alpha2 / MAD / N50 (HC) or mlp64 / alpha2 / imbalance0.6
(MD)); every stage CSV is tagged with arch/hidden_dim/alpha/fairness_type/n_train/imbalance/
method/lambda/seed. Each paper table/figure is then just a **filter + groupby** over the
master table — no per-table reruns. This script materializes those slices.

Usage:
  python -m experiments.pipeline.aggregators.slice_grid_tables --task hc \
      --grid results/healthcare/main_v5_multiinstance/grid/final --out <grid>/tables
  python -m experiments.pipeline.aggregators.slice_grid_tables --task md \
      --grid results/md_knapsack/main_v6_rowsum/grid/final --out <grid>/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["test_regret_normalized", "test_fairness", "test_pred_mse"]
# Core-6 as DISTINCT methods in the full-11 runs (DFL = FDFL at lambda=0, so it is not a
# separate method here — read FDFL@l0 as the DFL "crisis" point, FDFL@l1 as the repair).
CORE = ["FPTO", "FDFL", "FDFL-Scal", "FPLG", "PCGrad"]


def cap_label(arch, hidden) -> str:
    a = str(arch).strip().lower()
    if a in ("log_linear", "loglinear"):
        return "log_linear"
    if a in ("linear", "ols"):
        return a
    try:
        return f"mlp{int(hidden)}"
    except (TypeError, ValueError):
        return f"mlp_{arch}"


def load_master(grid_dir: Path) -> pd.DataFrame:
    files = sorted(Path(grid_dir).rglob("stage__*.csv"))
    if not files:
        raise FileNotFoundError(f"no stage__*.csv under {grid_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col, default in (("n_train", 50), ("imbalance", np.nan), ("fairness_type", "mad"),
                         ("hidden_dim", -1), ("arch", "mlp"), ("lambda", 0.0)):
        if col not in df.columns:
            df[col] = default
    df["cap"] = [cap_label(a, h) for a, h in zip(df["arch"], df["hidden_dim"])]
    return df


def _agg(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    metrics = [m for m in METRICS if m in df.columns]
    g = df.groupby(keys, dropna=False)
    out = g[metrics].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(c).rstrip("_") for c in out.columns]
    out = out.merge(g["seed"].nunique().reset_index(name="n_seeds"), on=keys)
    return out.sort_values(keys).reset_index(drop=True)


def slice_one(df: pd.DataFrame, *, caps=None, alphas=None, fairness=None, n_trains=None,
              imbalances=None, methods=None, groupby) -> pd.DataFrame:
    d = df
    if caps is not None:
        d = d[d["cap"].isin(caps)]
    if alphas is not None:
        d = d[d["alpha"].round(3).isin([round(a, 3) for a in alphas])]
    if fairness is not None:
        d = d[d["fairness_type"].isin(fairness)]
    if n_trains is not None:
        d = d[d["n_train"].isin(n_trains)]
    if imbalances is not None:
        d = d[d["imbalance"].round(3).isin([round(i, 3) for i in imbalances])]
    if methods is not None:
        d = d[d["method"].isin(methods)]
    return _agg(d, groupby) if len(d) else pd.DataFrame()


# (name, kwargs for slice_one). Anchor: mlp64 / alpha2 / MAD / N50 (HC); mlp64 / alpha2 / imb0.6 (MD).
HC_TABLES = [
    ("hc1_main", dict(caps=["mlp16", "mlp64"], alphas=[0.5, 2.0], fairness=["mad"],
                      n_trains=[50], groupby=["cap", "alpha", "method", "lambda"])),
    ("hc2_capacity", dict(caps=["mlp16", "mlp32", "mlp64", "mlp128"], alphas=[2.0],
                          fairness=["mad"], n_trains=[50], methods=CORE,
                          groupby=["cap", "method", "lambda"])),
    ("hc3_alpha", dict(caps=["mlp64"], alphas=[0.5, 1.5, 2.0, 4.0], fairness=["mad"],
                       n_trains=[50], methods=CORE, groupby=["alpha", "method", "lambda"])),
    ("hc4a_Ncurve", dict(caps=["mlp64"], alphas=[2.0], fairness=["mad"],
                         n_trains=[10, 20, 50, 100], groupby=["n_train", "method", "lambda"])),
    ("hc4b_fairness", dict(caps=["mlp64"], alphas=[2.0], fairness=["mad", "dp", "w2_dp"],
                          n_trains=[50], groupby=["fairness_type", "method", "lambda"])),
    ("hc7_ablation", dict(caps=["mlp64"], alphas=[2.0], fairness=["mad"], n_trains=[50],
                          methods=["FPTO", "FDFL"], groupby=["method", "lambda"])),
]
MD_TABLES = [
    ("md1_main", dict(caps=["linear", "mlp64"], alphas=[0.5, 2.0], imbalances=[0.6],
                      groupby=["cap", "alpha", "method", "lambda"])),
    ("md2_capacity", dict(caps=["linear", "mlp32", "mlp64", "mlp128"], alphas=[2.0],
                          imbalances=[0.6], methods=CORE, groupby=["cap", "method", "lambda"])),
    ("md3_alpha", dict(caps=["mlp64"], alphas=[0.5, 1.5, 2.0], imbalances=[0.6], methods=CORE,
                       groupby=["alpha", "method", "lambda"])),
    ("md4_imbalance", dict(caps=["mlp64"], alphas=[2.0], imbalances=[0.0, 0.2, 0.4, 0.6, 0.8],
                           groupby=["imbalance", "method", "lambda"])),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["hc", "md"], required=True)
    ap.add_argument("--grid", type=Path, required=True, help="The grid/final dir (rglob'd).")
    ap.add_argument("--out", type=Path, required=True, help="Where to write the per-table CSVs.")
    args = ap.parse_args()

    df = load_master(args.grid)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "master_tagged.csv", index=False)
    tables = HC_TABLES if args.task == "hc" else MD_TABLES
    print(f"=== {args.task.upper()} grid: {len(df)} rows over "
          f"{df['cap'].nunique()} predictors -> {args.out} ===")
    for name, kw in tables:
        out = slice_one(df, **kw)
        if out.empty:
            print(f"  [skip] {name}: no matching rows yet (grid incomplete?)")
            continue
        out.to_csv(args.out / f"{name}.csv", index=False)
        print(f"  wrote {name}.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
