"""HP-tuning driver — Phase 1 (search) + Phase 2 (select) of the protocol in
``new_experiment_design.md`` §0.5.

For each (task, alpha, method) we sweep the learning rate (FPLG keeps kappa FIXED at
1 = the sigmoid-decay schedule; only lr is tuned) at the fairness-on operating point
(lambda = 1 for scalarized-fair methods; the method's single point otherwise) on ONE
tuning seed (11), select the config with the lowest **val normalized regret**, and
write ``<out>/best_hp.csv``.

**Per-method + crash-safe** (so one Colab cell == one method): each method writes
``<out>/tuning_runs__<method>.csv`` and is skipped if that file already exists; the
selector globs ``tuning_runs__*.csv`` so best_hp.csv accumulates as methods finish.

  HC  -> closed-form, early-stop on val. 6-pt lr grid, m=5000, N_train=50, max_steps=200.
  MD  -> cvxpylayers, fixed steps + one end-of-training val eval. Cheap-tuning budget
         (§0.5 "MD compute budget"): m=50, N_train=20, max_steps=70, 3-pt lr grid.

λ is NOT tuned here (it is the fairness axis, swept later in the final phase).

Usage:
  python -m experiments.pipeline.run_hp_tuning --hc                       # full pool
  python -m experiments.pipeline.run_hp_tuning --hc --methods FDFL,FPLG   # subset (1 cell)
  python -m experiments.pipeline.run_hp_tuning --md --select-only         # rebuild best_hp
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

# CPU-only pipeline (cvxpylayers/SCS on CPU; HC closed-form). Hide any GPU BEFORE torch
# is imported so torch never initializes CUDA -> otherwise fork-based ProcessPool workers
# on a Colab GPU runtime crash with `CUDA error: initialization error` (CUDA + fork is
# unsupported). A GPU gives us no speedup anyway; use a CPU/High-RAM runtime for cores.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Pin BLAS to 1 thread/process so method-level workers (--max-workers) saturate cores
# without oversubscribing. setdefault => respects an explicit override.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG  # noqa: E402
from experiments.paths import HC_GRID, MD_GRID  # noqa: E402
from experiments.pipeline.multiinstance import (  # noqa: E402
    make_hc_instances, run_methods_for_seed,
)
from experiments.pipeline.multiinstance.md_instances import make_md_instances  # noqa: E402
from experiments.pipeline.multiinstance.md_loop import run_methods_for_seed_md  # noqa: E402
from experiments.pipeline.run_hc_multiinstance import (  # noqa: E402
    make_task_cfg as hc_task_cfg,
    make_train_cfg as hc_train_cfg,
    make_instance_cfg as hc_inst_cfg,
    _make_task as hc_make_task,
    RUN_A_METHODS,
    DATA_CSV,
)

# ----------------------------------------------------------------------
# Protocol grids (§0.5)
# ----------------------------------------------------------------------
LR_GRID_HC = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]   # 6 log-spaced
LR_GRID_MD = [1e-3, 3e-3, 1e-2]                      # 3 (Adam is forgiving; cvxpylayers cost)
FPLG_KAPPA = 1.0                                     # FPLG: kappa FIXED at 1 (sigmoid decay,
#                                                      paper eq.4) -> we always want the decay
#                                                      schedule; only lr is tuned for FPLG.
TUNING_SEED = 11
NO_LR_METHODS = {"SAA"}                              # no training => nothing to tune
# Methods kept in the pool but NOT lr-tuned (single fixed-lr config). EMPTY by default: a prior
# pin {"FPTO": 1e-3} caused FPTO to UNDER-TRAIN on the MLP (MSE 364 at lr=1e-3 vs ~120 at lr>=3e-3,
# which WDRO/NashMTL find by sweeping) -> the FPTO-MSE sanity gate failed and the regret gap was a
# pure under-training artifact. The "FPTO is lr-insensitive" claim only held for the (log-)linear
# closed-form path; on the MLP, FPTO must sweep the lr grid like every other trained baseline.
FIXED_LR: dict = {}
SELECT_KEYS = ["task", "fairness_type", "alpha", "m", "method"]

HC_OUT = HC_GRID / "hp"
# v6 = corrected per-individual (row-sum) model; per-arch subdirs (mlp64x2, linear,
# mlp32x2, mlp128x2) because select_best_hp keys on (alpha, method) only.
MD_OUT = MD_GRID / "hp" / "mlp64x2"


def _alpha_schedule(kappa: float, max_steps: int) -> dict:
    """Paper eq.(4) alpha_t = (1+exp((t-c)/temperature))^(-kappa); c & temperature are
    fixed but scaled to the training horizon so the decay is visible within it."""
    return {"type": "paper_decay", "kappa": float(kappa),
            "c": max_steps / 2.0, "temperature": max(max_steps / 10.0, 1.0)}


def _configs_for(method: str, lr_grid: list) -> list:
    """The (lr, kappa) configs to try for a method."""
    if method in NO_LR_METHODS:
        return [(None, None)]
    if method in FIXED_LR:
        return [(FIXED_LR[method], None)]             # kept in the pool, lr NOT swept (1 config)
    if method == "FPLG":
        return [(lr, FPLG_KAPPA) for lr in lr_grid]   # kappa fixed at 1 (decay); tune lr only
    return [(lr, None) for lr in lr_grid]


def _row(task, fairness_type, alpha, method, lr, kappa, r, elapsed, m=None,
         arch=None, hidden=None) -> dict:
    return {
        "task": task, "fairness_type": fairness_type, "alpha": float(alpha), "m": m,
        "arch": arch, "hidden": hidden,
        "method": method, "lr": lr, "kappa": kappa,
        "val_regret_normalized": r.get("val_regret_normalized", float("nan")),
        "test_regret_normalized": r.get("test_regret_normalized", float("nan")),
        "val_pred_mse": r.get("val_pred_mse", float("nan")),
        "test_pred_mse": r.get("test_pred_mse", float("nan")),
        "early_stopped": r.get("early_stopped", ""),
        "steps_run": r.get("steps_run", ""),
        "elapsed_sec": round(float(elapsed), 2),
    }


def _log(tag, alpha, method, lr, kappa, r, el):
    v = r.get("val_regret_normalized", float("nan"))
    lr_s = "  n/a" if lr is None else f"{lr:.0e}"
    k_s = "-" if kappa is None else f"{kappa:.0f}"
    print(f"[{tag}] a={alpha} {method:10s} lr={lr_s} k={k_s} val_reg={v:.4f} ({el:.1f}s)")


def _write_runs(out_dir: Path, method: str, rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"tuning_runs__{method.lower()}.csv", index=False)
    return df


def select_best_hp(out_dir, pred_baseline_select: str = "regret") -> pd.DataFrame:
    """Phase 2: glob every method's tuning_runs and pick the best lr per
    (task, fairness_type, alpha, method). Writes best_hp.csv + tuning_runs_all.csv.

    Selection metric is validation regret for every method by default. With
    ``pred_baseline_select="mse"`` the prediction-only baselines (``use_dec=False``,
    i.e. FPTO/PTO/WDRO/SAA) are instead ranked by ``val_pred_mse`` — prediction error
    is their actual training objective, and on the starvation-sensitive linear rung
    ranking them by regret selects a near-init underfit (the alpha=2 -1/u welfare gets
    *worse* as the linear fit improves, so min-regret picks the worst-fitting lr). The
    decision-aware methods stay on regret. The downstream prediction floor (applied at
    final time) then keeps the well-fit baseline's allocation safe."""
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("tuning_runs__*.csv"))
    if not files:
        print(f"[select] no tuning_runs__*.csv in {out_dir}")
        return pd.DataFrame()
    allruns = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    if "m" not in allruns.columns:        # tolerate pre-m-axis runs
        allruns["m"] = pd.NA
    allruns.to_csv(out_dir / "tuning_runs_all.csv", index=False)
    sel = allruns.copy()
    sel["_v"] = sel["val_regret_normalized"].fillna(np.inf)
    if pred_baseline_select == "mse":
        pred_bases = {m for m, c in ALL_METHOD_CONFIGS.items() if not c.get("use_dec", True)}
        mask = sel["method"].isin(pred_bases) & sel["val_pred_mse"].notna()
        sel.loc[mask, "_v"] = sel.loc[mask, "val_pred_mse"]
    idx = sel.groupby(SELECT_KEYS, dropna=False)["_v"].idxmin()
    best = allruns.loc[idx].reset_index(drop=True)
    best.to_csv(out_dir / "best_hp.csv", index=False)
    print(f"[select] {out_dir/'best_hp.csv'} ({len(best)} winners from {len(files)} methods; "
          f"pred-baselines ranked by {pred_baseline_select})")
    cols = ["task", "alpha", "m", "method", "lr", "kappa",
            "val_regret_normalized", "val_pred_mse"]
    print(best[[c for c in cols if c in best.columns]].to_string(index=False))
    return best


# ----------------------------------------------------------------------
# HC (closed-form, early-stop on val)
# ----------------------------------------------------------------------
# HC HP-tuning ranks lr ONLY (Phase 3 final uses the full m=5000 / N_train=50), so tune CHEAP per
# the §0.5 "cheap tuning, full final" rule (already used for MD): far smaller m / N / steps. The lr
# ranking is ~scale-invariant (Adam, full-batch over instances), so this is ~10x faster while
# selecting essentially the same lr. Output still lands under m<experiment_m>/ (the experiment size).
TUNE_M_HC, TUNE_NTRAIN_HC, TUNE_NVAL_HC, TUNE_NTEST_HC, TUNE_STEPS_HC = 2000, 20, 15, 10, 120


def _hc_instances(quick: bool, m: int = 5000):
    if quick:
        n_train, n_val, n_test, mm = 6, 4, 4, min(m, 800)
    else:
        n_train, n_val, n_test, mm = (TUNE_NTRAIN_HC, TUNE_NVAL_HC, TUNE_NTEST_HC,
                                      min(int(m), TUNE_M_HC))
    cfg = hc_inst_cfg(m=int(mm), n_train=n_train, n_test=n_test, instance_sampling="bootstrap")
    cfg["n_val"] = n_val
    return make_hc_instances(cohort=DATA_CSV, seed=TUNING_SEED, **cfg)


def tune_method_hc(method, *, alphas, fairness_type, out_dir, m=5000, arch="mlp",
                   hidden_dim=64, n_layers=2, inst_data=None, quick=False, overwrite=False) -> pd.DataFrame:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tuning_runs__{method.lower()}.csv"
    if path.exists() and not overwrite:
        print(f"[skip] {path.name} exists"); return pd.read_csv(path)
    # §0.6: the (log-)linear simple model is trained to convergence (extended steps, no
    # early-stop) so the tuned lr matches the final regime; prediction baselines are fit
    # in closed form inside the loop (lr is then inert for them, e.g. PTO/SAA).
    arch_l = str(arch).strip().lower()
    is_loglin = arch_l in ("log_linear", "loglinear", "linear", "ols")
    max_steps = 40 if quick else TUNE_STEPS_HC          # cheap-tune steps (lr ranking only)
    lr_grid = [1e-3, 3e-3] if quick else LR_GRID_HC
    if inst_data is None:
        inst_data = _hc_instances(quick, m)
    cap = arch_l if is_loglin else f"mlp{hidden_dim}x{n_layers}"
    rows = []
    for alpha in alphas:
        task = hc_make_task(hc_task_cfg(fairness_type=fairness_type, alpha_fair=float(alpha)))
        for (lr, kappa) in _configs_for(method, lr_grid):
            tcfg = hc_train_cfg(seed=TUNING_SEED, lambdas=[1.0], steps=max_steps, batch_size=-1,
                                lr=(lr if lr is not None else 3e-3),
                                arch=arch, hidden_dim=hidden_dim, n_layers=n_layers)
            tcfg["eval_train"] = False                  # train eval is unused by lr selection
            tcfg["early_stopping"] = (method not in NO_LR_METHODS) and (not is_loglin)
            tcfg["early_stop_eval_every"] = 5 if quick else 10
            tcfg["early_stop_patience"] = 10 if quick else 20
            if method == "FPLG":
                tcfg["alpha_schedule"] = _alpha_schedule(kappa, max_steps)
            t0 = time.time()
            mrows, _ = run_methods_for_seed(
                task=task, inst_data=inst_data, train_cfg=tcfg,
                method_configs={method: copy.deepcopy(ALL_METHOD_CONFIGS[method])}, seed=TUNING_SEED)
            el = time.time() - t0
            rows.append(_row("hc", fairness_type, alpha, method, lr, kappa, mrows[0], el,
                             m=m, arch=cap, hidden=hidden_dim))
            _log("HC", alpha, method, lr, kappa, mrows[0], el)
    return _write_runs(out_dir, method, rows)


def tune_hc(*, alphas, fairness_type, methods, out_dir, m=5000, arch="mlp", hidden_dim=64,
            n_layers=2, quick=False, overwrite=False) -> pd.DataFrame:
    inst_data = _hc_instances(quick, m)
    for method in methods:
        tune_method_hc(method, alphas=alphas, fairness_type=fairness_type, out_dir=out_dir,
                       m=m, arch=arch, hidden_dim=hidden_dim, n_layers=n_layers,
                       inst_data=inst_data, quick=quick, overwrite=overwrite)
    return select_best_hp(out_dir)


# ----------------------------------------------------------------------
# MD (cvxpylayers, fixed steps + single end-of-training val eval)
# ----------------------------------------------------------------------
def _md_instances(quick: bool, m: int = 50, imbalance: float = 0.6, alpha: float = 2.0,
                  budget_tightness: float = 0.35):
    n_train, n_val, n_test = (4, 4, 4) if quick else (20, 20, 30)
    if quick:
        m = min(m, 30)
    # budget_tightness must be passed HERE (budgets are baked into the instance
    # data); a task_cfg value alone is silently ignored.
    return make_md_instances(
        m=int(m), n_train=n_train, n_val=n_val, n_test=n_test, n_features=5, n_resources=3,
        alpha_fair=float(alpha), imbalance=float(imbalance),
        budget_tightness=float(budget_tightness), instance_sampling="bootstrap",
        seed=TUNING_SEED)


def _md_model_cfg(arch, hidden_dim, n_layers):
    if str(arch).strip().lower() == "linear":
        return {"arch": "linear", "init_mode": "default"}
    return {"arch": "mlp", "hidden_dim": int(hidden_dim), "n_layers": int(n_layers),
            "activation": "relu", "dropout": 0.0, "batch_norm": False, "init_mode": "default"}


def tune_method_md(method, *, alphas=(2.0,), imbalance=0.6, out_dir, m=50, arch="mlp",
                   hidden_dim=64, n_layers=2, decision_backend="cvxpylayers", inst=None,
                   quick=False, overwrite=False) -> pd.DataFrame:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tuning_runs__{method.lower()}.csv"
    if path.exists() and not overwrite:
        print(f"[skip] {path.name} exists"); return pd.read_csv(path)
    max_steps = 8 if quick else 70
    lr_grid = [3e-3] if quick else LR_GRID_MD
    alphas = [float(a) for a in alphas]
    if inst is None:
        inst = _md_instances(quick, m, imbalance=imbalance, alpha=alphas[0])
    cap = "linear" if str(arch).strip().lower() == "linear" else f"mlp{hidden_dim}x{n_layers}"
    model_cfg = _md_model_cfg(arch, hidden_dim, n_layers)
    rows = []
    for alpha in alphas:
        task_cfg = {"n_features": 5, "n_resources": 3, "alpha_fair": float(alpha),
                    "imbalance": imbalance, "fairness_type": "mad", "decision_mode": "group",
                    "budget_tightness": 0.35, "group_ratio": 0.5, "poly_degree": 2, "snr": 5.0}
        for (lr, kappa) in _configs_for(method, lr_grid):
            tcfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
            tcfg.update({
                "steps_per_lambda": max_steps, "batch_size": -1,
                "lr": (lr if lr is not None else 3e-3), "lr_decay": 5e-4,
                "grad_clip_norm": 5.0, "optimizer": "adam", "device": "cpu",
                "lambdas": [1.0], "eval_train": False, "log_every": 10,
                "decision_grad_backend": str(decision_backend),
                "model": copy.deepcopy(model_cfg)})
            if method == "FPLG":
                tcfg["alpha_schedule"] = _alpha_schedule(kappa, max_steps)
            t0 = time.time()
            mrows, _ = run_methods_for_seed_md(
                task_cfg=task_cfg, inst_data=inst, train_cfg=tcfg,
                method_configs={method: copy.deepcopy(ALL_METHOD_CONFIGS[method])}, seed=TUNING_SEED)
            el = time.time() - t0
            rows.append(_row("md", "mad", alpha, method, lr, kappa, mrows[0], el, m=m,
                             arch=cap, hidden=hidden_dim))
            _log("MD", alpha, method, lr, kappa, mrows[0], el)
    return _write_runs(out_dir, method, rows)


def tune_md(*, alphas=(2.0,), imbalance=0.6, methods, out_dir, m=50, arch="mlp", hidden_dim=64,
            n_layers=2, decision_backend="cvxpylayers", quick=False, overwrite=False) -> pd.DataFrame:
    inst = _md_instances(quick, m, imbalance=imbalance, alpha=list(alphas)[0])
    for method in methods:
        tune_method_md(method, alphas=alphas, imbalance=imbalance, out_dir=out_dir, m=m, arch=arch,
                       hidden_dim=hidden_dim, n_layers=n_layers, decision_backend=decision_backend,
                       inst=inst, quick=quick, overwrite=overwrite)
    return select_best_hp(out_dir)


# ----------------------------------------------------------------------
# Method-level parallelism (1 method per process; cores saturated, BLAS pinned)
# ----------------------------------------------------------------------
# Rough relative cost for big-first scheduling (heavier methods submitted first so the
# tail is short). ~ #configs x per-config solve cost.
_WEIGHT = {"FPLG": 12, "NashMTL": 8, "FDFL": 6, "FDFL-0.1": 6, "FDFL-0.5": 6,
           "PCGrad": 6, "MGDA": 6, "WDRO": 5, "FPTO": 4, "FDFL-Scal": 3, "SAA": 1}


def _tune_unit(u: dict):
    """ProcessPool worker: tune ONE method (writes its own tuning_runs__<method>.csv)."""
    if u["exp"] == "hc":
        tune_method_hc(u["method"], alphas=u["alphas"], fairness_type=u["fairness_type"],
                       out_dir=u["out_dir"], m=u["m"], arch=u.get("arch", "mlp"),
                       hidden_dim=u.get("hidden_dim", 64), n_layers=u.get("n_layers", 2),
                       quick=u["quick"], overwrite=u["overwrite"])
    else:
        tune_method_md(u["method"], alphas=u.get("alphas") or [2.0], imbalance=u.get("imbalance", 0.6),
                       out_dir=u["out_dir"], m=u["m"], arch=u.get("arch", "mlp"),
                       hidden_dim=u.get("hidden_dim", 64), n_layers=u.get("n_layers", 2),
                       decision_backend=u.get("decision_backend", "cvxpylayers"),
                       quick=u["quick"], overwrite=u["overwrite"])
    return u["method"]


def tune_pool(exp, *, methods, out_dir, workers, m=5000, arch="mlp", hidden_dim=64, n_layers=2,
              alphas=None, fairness_type=None, quick=False, overwrite=False,
              imbalance=0.6, decision_backend="cvxpylayers") -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    units = [{"exp": exp, "method": mth, "m": m, "arch": arch, "hidden_dim": hidden_dim,
              "n_layers": n_layers, "alphas": alphas, "fairness_type": fairness_type,
              "out_dir": str(out_dir), "quick": quick, "overwrite": overwrite,
              "imbalance": imbalance, "decision_backend": decision_backend} for mth in methods]
    if not overwrite:
        units = [u for u in units
                 if not (out_dir / f"tuning_runs__{u['method'].lower()}.csv").exists()]
    if not units:
        print("[parallel] all method files present; selecting.")
        return select_best_hp(out_dir)
    units.sort(key=lambda u: -_WEIGHT.get(u["method"], 1))
    W = max(1, min(int(workers), len(units)))
    print(f"[parallel] {len(units)} methods across {W} workers (1 method/core, BLAS pinned)...")
    with ProcessPoolExecutor(max_workers=W) as pool:
        futmap = {pool.submit(_tune_unit, u): u for u in units}
        for fut in as_completed(futmap):
            u = futmap[fut]
            try:
                fut.result()
                print(f"[done] {u['method']}", flush=True)
            except Exception as e:   # one method failing must not abort the others
                print(f"[FAIL] {u['method']}: {type(e).__name__}: {e}", flush=True)
    return select_best_hp(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hc", action="store_true")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--methods", type=str, default=None,
                    help="Comma-separated subset (default: full 11-method pool).")
    ap.add_argument("--alphas", type=str, default="0.5,2.0", help="HC alphas (comma-sep).")
    ap.add_argument("--fairness-type", type=str, default="atkinson", help="HC fairness measure.")
    ap.add_argument("--quick", action="store_true", help="Tiny grid/sizes for a smoke check.")
    ap.add_argument("--overwrite", action="store_true", help="Re-run methods even if CSV exists.")
    ap.add_argument("--select-only", action="store_true", help="Just rebuild best_hp.csv from disk.")
    ap.add_argument("--max-workers", type=int, default=1,
                    help="Method-level parallel processes (>1 => run methods across cores).")
    ap.add_argument("--m", type=int, nargs="+", default=[5000],
                    help="HC instance size(s); multiple => data-poor sweep, each to <hc-out>/m<m>/.")
    ap.add_argument("--md-m", type=int, default=50, help="MD instance size (stays small).")
    ap.add_argument("--imbalance", type=float, default=0.6,
                    help="MD imbalance level (data axis; anchor 0.6).")
    ap.add_argument("--md-backend", type=str, default="cvxpylayers",
                    choices=["cvxpylayers"],
                    help="MD decision-gradient backend (cvxpylayers differentiable conic layer; open SCS/ECOS solvers, no MOSEK).")
    ap.add_argument("--arch", type=str, default="mlp",
                    help="HC predictor arch: 'mlp' | 'log_linear' (§0.6 GLM simple model, "
                         "closed-form baselines) | 'ols' (identity) | 'linear' (softplus).")
    ap.add_argument("--hidden", type=int, default=64, help="HC MLP hidden width (e.g. 8/16/64).")
    ap.add_argument("--n-layers", type=int, default=2, help="HC MLP hidden layers (1 for small MLPs).")
    ap.add_argument("--hc-out", type=Path, default=HC_OUT)
    ap.add_argument("--md-out", type=Path, default=MD_OUT)
    ap.add_argument("--pred-baseline-select", type=str, default="regret",
                    choices=["regret", "mse"],
                    help="lr-selection metric for the prediction-only baselines "
                         "(FPTO/PTO/WDRO): 'regret' (default) or 'mse' (their training "
                         "objective; use on the starvation-sensitive linear rung).")
    args = ap.parse_args()
    if not (args.hc or args.md):
        ap.error("pick at least one of --hc / --md")
    methods = [s.strip() for s in args.methods.split(",")] if args.methods else list(RUN_A_METHODS)
    W = int(args.max_workers)

    if args.hc:
        alphas = [float(x) for x in args.alphas.split(",")]
        for mval in args.m:                       # per-m subdir keeps the sweep organized
            sub = args.hc_out / f"m{mval}"
            if args.select_only:
                select_best_hp(sub, pred_baseline_select=args.pred_baseline_select); continue
            if W > 1:
                tune_pool("hc", methods=methods, out_dir=sub, workers=W, m=mval,
                          arch=args.arch, hidden_dim=args.hidden, n_layers=args.n_layers,
                          alphas=alphas, fairness_type=args.fairness_type,
                          quick=args.quick, overwrite=args.overwrite)
            else:
                tune_hc(alphas=alphas, fairness_type=args.fairness_type, methods=methods,
                        out_dir=sub, m=mval, arch=args.arch, hidden_dim=args.hidden,
                        n_layers=args.n_layers, quick=args.quick, overwrite=args.overwrite)
    if args.md:
        md_alphas = [float(x) for x in args.alphas.split(",")]
        if args.select_only:
            select_best_hp(args.md_out)
        elif W > 1:
            tune_pool("md", methods=methods, out_dir=args.md_out, workers=W, m=args.md_m,
                      alphas=md_alphas, arch=args.arch, hidden_dim=args.hidden, n_layers=args.n_layers,
                      imbalance=args.imbalance, decision_backend=args.md_backend,
                      quick=args.quick, overwrite=args.overwrite)
        else:
            tune_md(alphas=md_alphas, imbalance=args.imbalance, methods=methods, out_dir=args.md_out,
                    m=args.md_m, arch=args.arch, hidden_dim=args.hidden, n_layers=args.n_layers,
                    decision_backend=args.md_backend, quick=args.quick, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
