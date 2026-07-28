"""Driver for the multi-instance Healthcare experiment (redesign).

Implements §10 (gating MOO pilot) and §5 (Run A headline) of
``new_experiment_design.md``. A *cell* is one (fairness_type, alpha, seed):
each cell (re)draws its own instance set from ``seed`` and runs every method
paired on that set. Cells are independent and parallelizable.

IMPORTANT (HC redesign): do NOT overwrite ``results/healthcare/main_v4``.
Output goes to a fresh tree ``results/healthcare/main_v5_multiinstance/``.

New config semantics:
    m_instance        patients per instance (was the whole cohort)
    n_train / n_test  number of train / test instances (was "5 seeds")
    instance_sampling "bootstrap" (Run A) | "disjoint" (Run B)
    stratify_by       "race" (only safe option, §4.1)
    test_fraction     0.20 patient-disjoint test-pool reserve (repurposed 0.5->0.2)
    batch_size        now counts INSTANCES per step, not patients (<=0 => full)

Usage:
    # Gating MOO pilot (§10) — run FIRST, before Run A:
    python -m experiments.pipeline.run_hc_multiinstance --pilot --max-workers 4

    # Run A headline (after locking m from the pilot):
    python -m experiments.pipeline.run_hc_multiinstance --run-a --max-workers 6
"""

from __future__ import annotations

# --- Thread pinning MUST precede numpy/torch import (see PROMPT_md_timing) ---
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG  # noqa: E402
from experiments.paths import require_file, HC_GRID  # noqa: E402
from experiments.pipeline.multiinstance import make_hc_instances  # noqa: E402
from experiments.pipeline.multiinstance.loop import (  # noqa: E402
    _make_task,
    run_methods_for_seed,
)

torch.set_num_threads(1)

DATA_CSV = "data/data_processed.csv"
OUT_ROOT = HC_GRID

# ----------------------------------------------------------------------
# Method pools
# ----------------------------------------------------------------------
PILOT_METHODS = ["FPTO", "FDFL", "PCGrad", "MGDA"]  # §10 anchors + MOO under test

# Run A full pool (mirrors ijoc_reproduce run_hc_group FULL_POOL).
RUN_A_METHODS = [
    "FPTO", "SAA", "WDRO",
    "FDFL", "FDFL-0.1", "FDFL-0.5", "FDFL-Scal",
    "FPLG", "PCGrad", "MGDA", "NashMTL",
]

# ----------------------------------------------------------------------
# Config factories
# ----------------------------------------------------------------------

def make_task_cfg(*, fairness_type: str, alpha_fair: float, budget_rho: float = 0.30) -> dict:
    return {
        "data_csv": DATA_CSV,
        "alpha_fair": float(alpha_fair),
        "fairness_type": fairness_type,
        "decision_mode": "group",
        "budget_rho": float(budget_rho),
    }


def make_train_cfg(*, seed: int, lambdas: list[float], steps: int, batch_size: int = -1,
                   hidden_dim: int = 64, n_layers: int = 2, arch: str = "mlp",
                   lr: float = 3e-3, optimizer: str = "adam",
                   grad_clip_norm: float = 5.0) -> dict:
    cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
    cfg.update({
        "seeds": [int(seed)],          # single seed per cell (drives instances + init)
        "lambdas": list(lambdas),
        "steps_per_lambda": int(steps),
        "batch_size": int(batch_size),  # INSTANCES per step (<=0 => full)
        "lr": float(lr),
        "lr_decay": 5e-4,
        # Adam (not SGD) so every method gets a comparable EFFECTIVE step under a
        # common lr: SGD + grad-clip gave decision-aware methods a ~4x larger step
        # than the (small-gradient, never-clipped) FPTO baseline, under-training it
        # and inflating the regret gap. clip caps any residual alpha-fair 1/u^2 spike.
        "optimizer": str(optimizer),
        "grad_clip_norm": float(grad_clip_norm),
        "device": "cpu",
        "eval_train": True,
        "log_every": 10,
    })
    arch_l = str(arch).strip().lower()
    if arch_l in ("linear", "log_linear", "loglinear", "ols"):
        cfg["model"] = {"arch": "linear", "init_mode": "default"}
        # Output link (consumed by the multi-instance loop): the §0.6 simple model is a
        # GLM log-link (exp); "ols" is the identity raw-linear point; plain "linear" keeps
        # the published softplus link. log_linear/ols prediction baselines are fit in
        # closed form (log-OLS / OLS) so the linear point is trained to convergence.
        if arch_l in ("log_linear", "loglinear"):
            cfg["output_link"] = "exp"
        elif arch_l == "ols":
            cfg["output_link"] = "none"
    else:
        cfg["model"] = {
            "arch": "mlp", "hidden_dim": int(hidden_dim), "n_layers": int(n_layers),
            "activation": "relu", "dropout": 0.0, "batch_norm": False, "init_mode": "default",
        }
    return cfg


def make_instance_cfg(*, m: int, n_train: int, n_test: int,
                      instance_sampling: str = "bootstrap", budget_frac: float = 0.30) -> dict:
    return {
        "m": int(m), "n_train": int(n_train), "n_test": int(n_test),
        "test_fraction": 0.20, "budget_frac": float(budget_frac),
        "stratify_by": "race", "instance_sampling": instance_sampling,
    }


# ----------------------------------------------------------------------
# One cell = (fairness, alpha, seed): draw instances, run all methods
# ----------------------------------------------------------------------

def _cell_dir_for(out_root: Path, *, fairness_type: str, alpha_fair: float, seed: int,
                  m: int, n_train: int, m_subdir: bool, n_subdir: bool) -> Path:
    cell_root = out_root
    if m_subdir:
        cell_root = cell_root / f"m{m}"
    if n_subdir:
        cell_root = cell_root / f"n{n_train}"
    return cell_root / fairness_type / f"alpha_{alpha_fair}" / f"seed_{seed}"


def _compute_cell_rows(
    *, fairness_type: str, alpha_fair: float, seed: int, methods: list[str],
    lambdas: list[float], steps: int, batch_size: int, m: int, n_train: int, n_test: int,
    instance_sampling: str, n_train_max: int | None, budget_frac: float,
    hidden_dim: int, n_layers: int, arch: str, lr: float, optimizer: str, grad_clip_norm: float,
) -> tuple[list[dict], list[dict], dict, float]:
    """Build task + instances, run the given methods paired on them, tag rows. No I/O.

    Shared by the sequential cell path and each parallel method-unit worker, so the
    two paths produce identical numbers (same deterministic instances + common init).
    """
    # Nested data-efficiency design: when N is swept, draw max(N) instances once
    # (deterministic by seed) and use the first ``n_train`` of them, so the
    # regret-vs-N curve isolates N (same pool split + test set across N values).
    gen_n_train = int(n_train_max) if (n_train_max and n_train_max > n_train) else n_train

    task_cfg = make_task_cfg(fairness_type=fairness_type, alpha_fair=alpha_fair,
                             budget_rho=budget_frac)
    train_cfg = make_train_cfg(seed=seed, lambdas=lambdas, steps=steps, batch_size=batch_size,
                               hidden_dim=hidden_dim, n_layers=n_layers, arch=arch, lr=lr,
                               optimizer=optimizer, grad_clip_norm=grad_clip_norm)
    inst_cfg = make_instance_cfg(m=m, n_train=gen_n_train, n_test=n_test,
                                 instance_sampling=instance_sampling, budget_frac=budget_frac)

    task = _make_task(task_cfg)
    inst_data = make_hc_instances(cohort=DATA_CSV, seed=int(seed), **inst_cfg)
    if gen_n_train != n_train:  # nested slice to the requested N
        inst_data.train = inst_data.train[:n_train]
        inst_data.meta["n_train"] = int(n_train)
    method_cfgs = {name: copy.deepcopy(ALL_METHOD_CONFIGS[name]) for name in methods}

    t0 = time.time()
    rows, iter_rows = run_methods_for_seed(
        task=task, inst_data=inst_data, train_cfg=train_cfg,
        method_configs=method_cfgs, seed=int(seed),
    )
    elapsed = time.time() - t0

    meta = {k: v for k, v in inst_data.meta.items() if k != "feature_cols"}
    for r in rows:
        r["m_instance"] = meta["m"]
        r["n_train"] = meta["n_train"]
        r["n_test"] = meta["n_test"]
        r["instance_sampling"] = meta["instance_sampling"]
        r["fairness_type"] = fairness_type
        r["alpha_fair"] = float(alpha_fair)
        r["budget_frac"] = float(budget_frac)
        r["arch"] = str(arch)
        r["hidden_dim"] = (0 if str(arch).strip().lower() == "linear" else int(hidden_dim))
    return rows, iter_rows, meta, elapsed


def _write_cell(cell_dir: Path, rows, iter_rows, meta, *, fairness_type, alpha_fair, seed,
                methods, lambdas, steps, batch_size, budget_frac, arch, hidden_dim, n_layers,
                elapsed) -> pd.DataFrame:
    cell_dir.mkdir(parents=True, exist_ok=True)
    stage_df = pd.DataFrame(rows)
    if not stage_df.empty:
        stage_df.to_csv(cell_dir / "stage_results.csv", index=False)
    if iter_rows:
        pd.DataFrame(iter_rows).to_csv(cell_dir / "iter_logs.csv", index=False)
    with open(cell_dir / "config.json", "w") as f:
        json.dump({
            "label": f"hc_mi_{fairness_type}_a{alpha_fair}_s{seed}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_sec": float(elapsed),
            "methods": list(methods),
            "lambdas": list(lambdas),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "budget_frac": float(budget_frac),
            "arch": str(arch),
            "hidden_dim": int(hidden_dim),
            "n_layers": int(n_layers),
            "instance_meta": meta,
        }, f, indent=2, default=str)
    return stage_df


def run_cell(
    *, out_root: Path, fairness_type: str, alpha_fair: float, seed: int,
    methods: list[str], lambdas: list[float], steps: int, batch_size: int,
    m: int, n_train: int, n_test: int, instance_sampling: str,
    n_train_max: int | None = None, n_subdir: bool = False, m_subdir: bool = False,
    budget_frac: float = 0.30, hidden_dim: int = 64, n_layers: int = 2, arch: str = "mlp",
    lr: float = 3e-3, optimizer: str = "adam", grad_clip_norm: float = 5.0,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, float]:
    """One cell = (fairness, alpha, seed): draw instances, run all methods (sequential)."""
    cell_dir = _cell_dir_for(out_root, fairness_type=fairness_type, alpha_fair=alpha_fair,
                             seed=seed, m=m, n_train=n_train, m_subdir=m_subdir, n_subdir=n_subdir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    stage_csv = cell_dir / "stage_results.csv"
    if stage_csv.exists() and not overwrite:
        return pd.read_csv(stage_csv), 0.0

    rows, iter_rows, meta, elapsed = _compute_cell_rows(
        fairness_type=fairness_type, alpha_fair=alpha_fair, seed=seed, methods=methods,
        lambdas=lambdas, steps=steps, batch_size=batch_size, m=m, n_train=n_train,
        n_test=n_test, instance_sampling=instance_sampling, n_train_max=n_train_max,
        budget_frac=budget_frac, hidden_dim=hidden_dim, n_layers=n_layers, arch=arch,
        lr=lr, optimizer=optimizer, grad_clip_norm=grad_clip_norm,
    )
    stage_df = _write_cell(cell_dir, rows, iter_rows, meta, fairness_type=fairness_type,
                           alpha_fair=alpha_fair, seed=seed, methods=methods, lambdas=lambdas,
                           steps=steps, batch_size=batch_size, budget_frac=budget_frac,
                           arch=arch, hidden_dim=hidden_dim, n_layers=n_layers, elapsed=elapsed)
    return stage_df, elapsed


def _exec_cell(payload: dict) -> dict:
    df, dt = run_cell(**payload)
    return {"fairness_type": payload["fairness_type"], "alpha": float(payload["alpha_fair"]),
            "seed": int(payload["seed"]), "n_train": int(payload["n_train"]),
            "elapsed_sec": float(dt), "n_rows": int(len(df))}


# ----------------------------------------------------------------------
# Method-level parallel path: (cell x method) units across processes.
# Mirrors run_md_multiinstance.run_grid_method_parallel — methods are
# independent given the shared (deterministic) instances + common init, so this
# is byte-identical to the sequential path, just spread over cores. A cell's CSV
# is written when ALL its methods finish (crash-safe).
# ----------------------------------------------------------------------
# Relative cost for big-first scheduling: decision-aware methods run the alpha-fair
# solve + VJP every step; FPTO is prediction-only, SAA does not train.
_METHOD_WEIGHT = {
    "FDFL": 2.0, "FDFL-0.1": 2.0, "FDFL-0.5": 2.0, "FDFL-Scal": 1.5, "FPLG": 1.5,
    "PCGrad": 2.0, "MGDA": 2.0, "NashMTL": 2.0, "WDRO": 1.0, "FPTO": 0.5, "SAA": 0.1,
}
_BASE_KEYS = ("lambdas", "steps", "batch_size", "n_test", "instance_sampling",
              "n_train_max", "budget_frac", "hidden_dim", "n_layers", "arch", "lr",
              "optimizer", "grad_clip_norm")


def _method_unit_worker(payload: dict) -> tuple:
    rows, iter_rows, meta, _ = _compute_cell_rows(
        fairness_type=payload["fairness_type"], alpha_fair=payload["alpha_fair"],
        seed=payload["seed"], methods=[payload["method"]], m=payload["m"],
        n_train=payload["n_train"], **payload["base"])
    return (payload["key"], payload["method"], rows, iter_rows, meta)


def run_grid_method_parallel(jobs, *, workers, out_root, overwrite, verbose=True):
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor, as_completed

    pending = []
    for j in jobs:
        cell_dir = _cell_dir_for(out_root, fairness_type=j["fairness_type"],
                                 alpha_fair=j["alpha_fair"], seed=j["seed"], m=j["m"],
                                 n_train=j["n_train"], m_subdir=j["m_subdir"], n_subdir=j["n_subdir"])
        if (cell_dir / "stage_results.csv").exists() and not overwrite:
            if verbose:
                print(f"[skip] cell {cell_dir} already done")
            continue
        j = dict(j)
        j["cell_dir"] = cell_dir
        j["key"] = (j["fairness_type"], float(j["alpha_fair"]), int(j["seed"]),
                    int(j["m"]), int(j["n_train"]))
        pending.append(j)
    if not pending:
        print("nothing to do (all cells present).")
        return

    by_key = {j["key"]: j for j in pending}
    units = []
    for j in pending:
        base = {k: j[k] for k in _BASE_KEYS}
        for name in j["methods"]:
            units.append(dict(key=j["key"], method=name, fairness_type=j["fairness_type"],
                              alpha_fair=j["alpha_fair"], seed=j["seed"], m=j["m"],
                              n_train=j["n_train"], base=base))
    units.sort(key=lambda u: -(u["m"] * u["n_train"] * _METHOD_WEIGHT.get(u["method"], 1.0)))
    W = min(int(workers), len(units))
    print(f"{len(pending)} cells x methods = {len(units)} units; {W} workers "
          f"(method-level parallelism)...")

    rows_by = defaultdict(list)
    iters_by = defaultdict(list)
    meta_by: dict = {}
    remaining = {j["key"]: len(j["methods"]) for j in pending}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=W) as pool:
        futs = [pool.submit(_method_unit_worker, u) for u in units]
        for fut in as_completed(futs):
            key, method, rows, iter_rows, meta = fut.result()
            rows_by[key] += rows
            iters_by[key] += iter_rows
            meta_by[key] = meta
            remaining[key] -= 1
            if verbose:
                print(f"  done {method:10s} cell {key}  [{remaining[key]} left]", flush=True)
            if remaining[key] == 0:                       # cell complete -> write now
                j = by_key[key]
                _write_cell(j["cell_dir"], rows_by[key], iters_by[key], meta_by[key],
                            fairness_type=j["fairness_type"], alpha_fair=j["alpha_fair"],
                            seed=j["seed"], methods=j["methods"], lambdas=j["lambdas"],
                            steps=j["steps"], batch_size=j["batch_size"],
                            budget_frac=j["budget_frac"], arch=j["arch"],
                            hidden_dim=j["hidden_dim"], n_layers=j["n_layers"],
                            elapsed=time.time() - t0)
                print(f"[cell done] {key}: {len(rows_by[key])} rows", flush=True)
                rows_by.pop(key)
                iters_by.pop(key)
                meta_by.pop(key)


def build_grand_summary(out_root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(out_root.rglob("stage_results.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="Gating MOO pilot (§10).")
    mode.add_argument("--run-a", action="store_true", help="Run A headline (§5).")

    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    # Overridable knobs
    ap.add_argument("--m", type=int, nargs="+", default=None,
                    help="instance size; a list sweeps m (adds an m<M>/ subdir level).")
    ap.add_argument("--n-train", type=int, nargs="+", default=None,
                    help="Run A: a list sweeps N as a data-efficiency axis "
                         "(default 10 20 50; headline = largest). Pilot: single value.")
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=-1, help="instances/step (<=0 full)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    ap.add_argument("--fairness", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--budget-frac", type=float, default=None,
                    help="per-instance budget fraction Q=frac*sum(c) (default 0.30).")
    ap.add_argument("--hidden-dim", type=int, default=None,
                    help="predictor MLP hidden width (default 64; lower => weaker).")
    ap.add_argument("--n-layers", type=int, default=None, help="predictor MLP depth (default 2).")
    ap.add_argument("--arch", type=str, default=None,
                    choices=["mlp", "linear", "log_linear", "ols"],
                    help="predictor arch (default mlp; 'log_linear' = GLM log-link simple model "
                         "with closed-form baselines (§0.6); 'ols' = identity raw-linear; "
                         "'linear' = published softplus-link linear). Non-mlp ignores hidden-dim.")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (default 3e-3 for adam).")
    ap.add_argument("--lambdas", type=float, nargs="+", default=None,
                    help="override lambda sweep (default 0 0.5 1 2).")
    ap.add_argument("--optimizer", type=str, default="adam",
                    choices=["adam", "adamw", "sgd", "sgd_momentum"],
                    help="optimizer (default adam; use 'sgd' to reproduce the legacy runs).")
    ap.add_argument("--grad-clip-norm", type=float, default=5.0,
                    help="global grad-norm clip (default 5.0; 0 disables).")
    ap.add_argument("--instance-sampling", type=str, default="bootstrap",
                    choices=["bootstrap", "disjoint"],
                    help="cross-instance regime (default bootstrap; 'disjoint' = i.i.d. "
                         "non-overlapping draws for the HC-6 finite-sample bound curve).")
    ap.add_argument("--parallel-mode", type=str, default="method", choices=["method", "cell"],
                    help="with --max-workers>1: 'method' spreads (cell x method) units "
                         "(default, mirrors MD); 'cell' spreads whole cells.")
    args = ap.parse_args()

    # Fail fast with a clear message if the cohort CSV is missing (e.g. wrong cwd).
    require_file(DATA_CSV, "healthcare cohort CSV")

    budget_frac = 0.30 if args.budget_frac is None else float(args.budget_frac)
    hidden_dim = 64 if args.hidden_dim is None else int(args.hidden_dim)
    n_layers = 2 if args.n_layers is None else int(args.n_layers)
    arch = "mlp" if args.arch is None else str(args.arch)
    lr = 3e-3 if args.lr is None else float(args.lr)
    optimizer = str(args.optimizer)
    grad_clip_norm = float(args.grad_clip_norm)

    if args.pilot:
        tag = "pilot"
        m_list = args.m or [4000]
        n_train_list = args.n_train or [20]
        alphas = args.alphas or [2.0]
        fairness = args.fairness or ["mad"]
        seeds = args.seeds or [11, 22, 33]
        methods = args.methods or PILOT_METHODS
        lambdas = args.lambdas or [0.0, 0.5, 1.0, 2.0]
        out_root = args.out_root / "pilot"
        n_subdir = False  # single N -> no n<N> level
    else:  # run-a
        tag = "run_a"
        m_list = args.m or [5000]
        # N sweep folded into Run A as the data-efficiency axis (no separate
        # Run B bound curve). Nested: largest N is the headline; smaller N's are
        # prefixes of it (same pool split + test set) so the curve isolates N.
        n_train_list = args.n_train or [10, 20, 50]
        alphas = args.alphas or [0.5, 2.0]
        fairness = args.fairness or ["mad"]
        seeds = args.seeds or [11, 22, 33]
        methods = args.methods or RUN_A_METHODS
        lambdas = args.lambdas or [0.0, 0.5, 1.0, 2.0]
        out_root = args.out_root / "variant_a"
        n_subdir = len(n_train_list) > 1  # n<N>/ level only when sweeping

    n_train_max = max(n_train_list)
    m_subdir = len(m_list) > 1  # m<M>/ level only when sweeping m
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        dict(out_root=out_root, fairness_type=ft, alpha_fair=a, seed=s,
             methods=methods, lambdas=lambdas, steps=args.steps, batch_size=args.batch_size,
             m=mv, n_train=n, n_train_max=n_train_max, n_subdir=n_subdir, m_subdir=m_subdir,
             budget_frac=budget_frac, hidden_dim=hidden_dim, n_layers=n_layers, arch=arch, lr=lr,
             optimizer=optimizer, grad_clip_norm=grad_clip_norm,
             n_test=args.n_test, instance_sampling=args.instance_sampling, overwrite=args.overwrite)
        for ft in fairness for a in alphas for s in seeds for mv in m_list for n in n_train_list
    ]

    print(f"=== HC multi-instance [{tag}] ===")
    print(f"  out:        {out_root}")
    print(f"  m={m_list}  n_train={n_train_list} (headline N={n_train_max})  n_test={args.n_test}  "
          f"steps={args.steps}  batch_size={args.batch_size}")
    print(f"  budget_frac={budget_frac}  arch={arch}  hidden_dim={hidden_dim}  n_layers={n_layers}")
    print(f"  optimizer={optimizer}  lr={lr}  grad_clip_norm={grad_clip_norm}")
    print(f"  fairness={fairness}  alphas={alphas}  seeds={seeds}")
    print(f"  methods={methods}")
    print(f"  cells={len(jobs)}  max_workers={args.max_workers}  parallel_mode={args.parallel_mode}\n")

    summary = []
    t_all = time.time()
    if args.max_workers <= 1:
        for j in jobs:
            row = _exec_cell(j)
            summary.append(row)
            print(f"  [{row['fairness_type']} a={row['alpha']} s={row['seed']} N={row['n_train']}] "
                  f"{row['n_rows']} rows in {row['elapsed_sec']:.1f}s", flush=True)
    elif args.parallel_mode == "method":
        # (cell x method) units across processes; cell CSVs written as methods finish.
        run_grid_method_parallel(jobs, workers=args.max_workers, out_root=out_root,
                                 overwrite=args.overwrite)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            for i, row in enumerate(pool.map(_exec_cell, jobs), 1):
                summary.append(row)
                print(f"  [{i}/{len(jobs)}] {row['fairness_type']} a={row['alpha']} "
                      f"s={row['seed']} N={row['n_train']}: {row['n_rows']} rows in "
                      f"{row['elapsed_sec']:.1f}s", flush=True)

    total = time.time() - t_all
    grand = build_grand_summary(out_root)
    if not grand.empty:
        grand.to_csv(out_root / "grand_summary.csv", index=False)
    with open(out_root / "grid_summary.json", "w") as f:
        json.dump({"tag": tag, "m": m_list, "n_train": n_train_list, "n_test": args.n_test,
                   "steps": args.steps, "batch_size": args.batch_size,
                   "budget_frac": budget_frac, "hidden_dim": hidden_dim, "n_layers": n_layers,
                   "methods": methods, "fairness": fairness, "alphas": alphas, "seeds": seeds,
                   "summary": summary, "grand_total_sec": float(total), "n_cells": len(summary)},
                  f, indent=2)
    print(f"\n=== done: {len(summary)} cells, {len(grand)} rows, {total:.1f}s "
          f"({total/60:.1f} min) ===")
    print(f"    grand_summary -> {out_root / 'grand_summary.csv'}")


if __name__ == "__main__":
    main()
