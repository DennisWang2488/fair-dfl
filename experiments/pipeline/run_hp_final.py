"""Final-run driver — Phase 3 of the protocol (new_experiment_design.md §0.5).

Reads the tuned hyperparameters from ``best_hp.csv`` (produced by ``run_hp_tuning``),
then for each method runs the **lambda-sweep x 3 seeds {11,22,33}** with that method's
selected lr (and kappa, for FPLG) and reports **test** regret / MAD / MSE.

**Per-(method, seed) + crash-safe** (the finest Colab unit): each (method, seed) writes
``<out>/stage__<method>__s<seed>.csv`` and is skipped if it already exists, so a Colab
timeout loses at most the running unit. The aggregators glob ``stage__*.csv``.

  HC: closed-form, early-stop on val (restore best-val), report test. lambda {0,0.5,1,2}.
  MD: cvxpylayers, fixed steps + final val. lambda {0,1,2}, m=50, N_train=50, max_steps=100.

Usage (cells normally import ``run_final_method``; CLI shown for reference):
  python -m experiments.pipeline.run_hp_final --hc --methods FDFL,FPLG
  python -m experiments.pipeline.run_hp_final --md --methods FPLG --seeds 11
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

# CPU-only pipeline: hide any GPU BEFORE torch import so torch never initializes CUDA
# (CUDA + fork in the ProcessPool workers crashes on a Colab GPU runtime). No GPU speedup
# for us; use a CPU/High-RAM runtime for cores.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Pin BLAS to 1 thread/process so (method,seed) process workers saturate cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG  # noqa: E402
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
from experiments.pipeline.run_hp_tuning import _alpha_schedule  # noqa: E402
from experiments.paths import HC_GRID, MD_GRID  # noqa: E402

FINAL_SEEDS = [11, 22, 33]
HC_LAMBDAS = [0.0, 0.5, 1.0, 2.0]
MD_LAMBDAS = [0.0, 1.0, 2.0]

HC_HP = HC_GRID / "hp" / "best_hp.csv"
HC_OUT = HC_GRID / "final"
# MD v6 = corrected per-individual (row-sum) model; v5 MD results are the old
# formulation and must not be extended. Per-arch hp lives in hp/<cap>/best_hp.csv
# (mlp64x2 is the anchor); pass --md-hp for other archs.
MD_HP = MD_GRID / "hp" / "mlp64x2" / "best_hp.csv"
MD_OUT = MD_GRID / "final"


def load_best_hp(path) -> dict:
    """best_hp.csv -> {(alpha, method): {"lr":..., "kappa":...}}."""
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        lr = None if pd.isna(r.get("lr")) else float(r["lr"])
        kappa = None if pd.isna(r.get("kappa")) else float(r["kappa"])
        out[(float(r["alpha"]), str(r["method"]))] = {"lr": lr, "kappa": kappa}
    return out


def _hp_for(best_hp: dict, alpha: float, method: str) -> dict:
    hp = best_hp.get((float(alpha), method))
    if hp is None:
        return {"lr": 3e-3, "kappa": 1.0 if method == "FPLG" else None}
    return {"lr": (hp["lr"] if hp["lr"] is not None else 3e-3), "kappa": hp["kappa"]}


def _tag(r, task, fairness_type, alpha, method, seed, hp, elapsed,
         arch=None, hidden_dim=None, n_train=None) -> dict:
    out = dict(r)
    out.update({"task": task, "fairness_type": fairness_type, "alpha": float(alpha),
                "method": method, "seed": int(seed), "lr": hp["lr"], "kappa": hp["kappa"],
                "elapsed_sec": round(float(elapsed), 2)})
    if arch is not None:
        out["arch"] = str(arch)
        out["hidden_dim"] = int(hidden_dim) if hidden_dim is not None else None
    if n_train is not None:
        out["n_train"] = int(n_train)
    return out


def _save_seed(out_dir: Path, method: str, seed: int, rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = out_dir / f"stage__{method.lower()}__s{seed}.csv"
    df.to_csv(path, index=False)
    keep = [c for c in ("alpha", "lambda", "test_regret_normalized", "test_fairness",
                        "test_pred_mse") if c in df.columns]
    print(f"  wrote {path.name} ({len(df)} rows)")
    if keep:
        print(df[keep].to_string(index=False))
    return df


# ----------------------------------------------------------------------
# HC final (closed-form, early-stop on val, report test)
# ----------------------------------------------------------------------
def run_final_method_hc(method, *, best_hp, alphas, fairness_type, seeds=FINAL_SEEDS,
                        out_dir=HC_OUT, quick=False, overwrite=False,
                        arch="mlp", hidden_dim=64, n_layers=2, lambdas=None,
                        n_train=50, n_train_max=None, pred_floor=None) -> pd.DataFrame:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # §0.6 capacity axis: on the (log-)linear model the prediction baselines (FPTO/PTO/SAA/WDRO)
    # are fit in CLOSED FORM (steps ignored, handled in the loop); the DFL family trains by SGD
    # with NO early-stopping. We use the SAME 200 steps as the MLP (the old 600 just burned time
    # and stalled m=5000 — the DFL family does not *converge* on the exp link, so extra steps only
    # add divergence). The rich MLP keeps the §0.5 early-stopping protocol.
    is_loglin = str(arch).strip().lower() in ("log_linear", "loglinear", "linear", "ols")
    m, n_val, n_test = 5000, 30, 30
    max_steps = 200
    lambdas = list(lambdas) if lambdas else HC_LAMBDAS
    if quick:
        m, n_train, n_val, n_test, max_steps = 800, min(int(n_train), 6), 4, 4, 40
        lambdas = [0.0, 1.0]
    # HC-4 data-efficiency axis: draw the LARGEST N once (so the val/test pools + the
    # train prefix are shared across N) and slice to the requested N, isolating N cleanly.
    gen_n_train = int(n_train_max) if (n_train_max and int(n_train_max) > int(n_train)) else int(n_train)

    dfs = []
    for seed in seeds:
        path = out_dir / f"stage__{method.lower()}__s{seed}.csv"
        if path.exists() and not overwrite:
            print(f"[skip] {path.name} exists"); dfs.append(pd.read_csv(path)); continue
        inst_cfg = hc_inst_cfg(m=m, n_train=gen_n_train, n_test=n_test, instance_sampling="bootstrap")
        inst_cfg["n_val"] = n_val
        inst_data = make_hc_instances(cohort=DATA_CSV, seed=int(seed), **inst_cfg)
        if gen_n_train != int(n_train):                       # nested slice to the requested N
            inst_data.train = inst_data.train[:int(n_train)]
            inst_data.meta["n_train"] = int(n_train)
        rows = []
        for alpha in alphas:
            hp = _hp_for(best_hp, alpha, method)
            task = hc_make_task(hc_task_cfg(fairness_type=fairness_type, alpha_fair=float(alpha)))
            tcfg = hc_train_cfg(seed=int(seed), lambdas=lambdas, steps=max_steps,
                                batch_size=-1, lr=hp["lr"],
                                arch=arch, hidden_dim=hidden_dim, n_layers=n_layers)
            tcfg["early_stopping"] = (method != "SAA") and (not is_loglin)
            tcfg["early_stop_eval_every"] = 5 if quick else 10
            tcfg["early_stop_patience"] = 10 if quick else 30
            if pred_floor is not None:
                tcfg["prediction_floor"] = pred_floor
            if method == "FPLG":
                tcfg["alpha_schedule"] = _alpha_schedule(hp["kappa"] or 1.0, max_steps)
            t0 = time.time()
            mrows, _ = run_methods_for_seed(
                task=task, inst_data=inst_data, train_cfg=tcfg,
                method_configs={method: copy.deepcopy(ALL_METHOD_CONFIGS[method])}, seed=int(seed))
            el = time.time() - t0
            for r in mrows:
                rows.append(_tag(r, "hc", fairness_type, alpha, method, seed, hp, el,
                                 arch=arch, hidden_dim=(0 if is_loglin else hidden_dim),
                                 n_train=int(n_train)))
            print(f"[HC] {method:10s} a={alpha} seed={seed} arch={arch} lr={hp['lr']:.0e} "
                  f"k={hp['kappa']} -> {len(mrows)} lambda-rows ({el:.1f}s)")
        dfs.append(_save_seed(out_dir, method, seed, rows))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ----------------------------------------------------------------------
# MD final (cvxpylayers, fixed steps + final val, report test)
# ----------------------------------------------------------------------
def _md_model_cfg(arch, hidden_dim, n_layers):
    """MD predictor: 'linear' (2-D linear; positivity via task.compute's softplus) or MLP.
    No HC-style output_link/closed-form here -- MD positivity is handled inside compute."""
    if str(arch).strip().lower() == "linear":
        return {"arch": "linear", "init_mode": "default"}
    return {"arch": "mlp", "hidden_dim": int(hidden_dim), "n_layers": int(n_layers),
            "activation": "relu", "dropout": 0.0, "batch_norm": False, "init_mode": "default"}


def run_final_method_md(method, *, best_hp, alphas=(2.0,), seeds=FINAL_SEEDS, imbalance=0.6,
                        out_dir=MD_OUT, quick=False, overwrite=False,
                        arch="mlp", hidden_dim=64, n_layers=2, lambdas=None,
                        decision_backend="cvxpylayers", m=200, n_train=50,
                        budget_tightness=0.35, n_groups=2, n_resources=3) -> pd.DataFrame:
    # Anchor scale m=200 / N_train=50 (matches the capacity smokes, so their
    # conclusions transfer); supported range m in [50, 200], N_train in [20, 100].
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    m, n_train = int(m), int(n_train)
    n_val, n_test, max_steps = 30, 30, 100
    lambdas = list(lambdas) if lambdas else MD_LAMBDAS
    if quick:
        m, n_train, n_val, n_test, max_steps = 30, 4, 4, 4, 8
        lambdas = [0.0, 1.0]
    alphas = [float(a) for a in alphas]
    model_cfg = _md_model_cfg(arch, hidden_dim, n_layers)

    dfs = []
    for seed in seeds:
        path = out_dir / f"stage__{method.lower()}__s{seed}.csv"
        if path.exists() and not overwrite:
            print(f"[skip] {path.name} exists"); dfs.append(pd.read_csv(path)); continue
        # alpha is a DECISION axis (not a data axis): draw instances once per seed (the
        # imbalance fixes the data), then evaluate each alpha by rebinding the task.
        # budget_tightness MUST be passed here — budgets are baked into the instance
        # data inside make_md_instances; the task_cfg copy below is bookkeeping only.
        inst = make_md_instances(
            m=m, n_train=n_train, n_val=n_val, n_test=n_test, n_features=5,
            n_resources=int(n_resources), n_groups=int(n_groups),
            alpha_fair=alphas[0], imbalance=imbalance, budget_tightness=float(budget_tightness),
            instance_sampling="bootstrap", seed=int(seed))
        rows = []
        for alpha in alphas:
            task_cfg = {"n_features": 5, "n_resources": int(n_resources), "alpha_fair": float(alpha),
                        "imbalance": imbalance, "fairness_type": "mad", "decision_mode": "group",
                        "budget_tightness": float(budget_tightness), "group_ratio": 0.5,
                        "poly_degree": 2, "snr": 5.0, "n_groups": int(n_groups)}
            hp = _hp_for(best_hp, alpha, method)
            tcfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
            tcfg.update({
                "steps_per_lambda": max_steps, "batch_size": -1, "lr": hp["lr"], "lr_decay": 5e-4,
                "grad_clip_norm": 5.0, "optimizer": "adam", "device": "cpu",
                "lambdas": lambdas, "eval_train": False, "log_every": 10,
                "decision_grad_backend": str(decision_backend),
                "model": copy.deepcopy(model_cfg)})
            if method == "FPLG":
                tcfg["alpha_schedule"] = _alpha_schedule(hp["kappa"] or 1.0, max_steps)
            t0 = time.time()
            mrows, _ = run_methods_for_seed_md(
                task_cfg=task_cfg, inst_data=inst, train_cfg=tcfg,
                method_configs={method: copy.deepcopy(ALL_METHOD_CONFIGS[method])}, seed=int(seed))
            el = time.time() - t0
            for r in mrows:
                row = _tag(r, "md", "mad", alpha, method, seed, hp, el,
                           arch=arch, hidden_dim=(0 if str(arch).lower() == "linear" else hidden_dim))
                row["imbalance"] = float(imbalance)
                row["n_groups"] = int(n_groups)
                row["n_resources"] = int(n_resources)
                rows.append(row)
            print(f"[MD] {method:10s} a={alpha} seed={seed} imb={imbalance} arch={arch} "
                  f"lr={hp['lr']:.0e} k={hp['kappa']} -> {len(mrows)} lambda-rows ({el:.1f}s)")
        dfs.append(_save_seed(out_dir, method, seed, rows))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def run_final_method(exp, method, **kw) -> pd.DataFrame:
    """Dispatch one method's final run (the per-cell entry point for Colab)."""
    if exp == "hc":
        return run_final_method_hc(method, **kw)
    if exp == "md":
        return run_final_method_md(method, **kw)
    raise ValueError(f"exp must be 'hc' or 'md', got {exp!r}")


# ----------------------------------------------------------------------
# (method, seed)-level parallelism (each unit = one process; cores saturated)
# ----------------------------------------------------------------------
_WEIGHT = {"FPLG": 12, "NashMTL": 8, "FDFL": 6, "FDFL-0.1": 6, "FDFL-0.5": 6,
           "PCGrad": 6, "MGDA": 6, "WDRO": 5, "FPTO": 4, "FDFL-Scal": 3, "SAA": 1}


def _final_unit(u: dict):
    """ProcessPool worker: one (method, seed) final run -> stage__<m>__s<seed>.csv."""
    if u["exp"] == "hc":
        run_final_method_hc(u["method"], best_hp=u["best_hp"], alphas=u["alphas"],
                            fairness_type=u["fairness_type"], seeds=[u["seed"]],
                            out_dir=u["out_dir"], quick=u["quick"], overwrite=u["overwrite"],
                            arch=u.get("arch", "mlp"), hidden_dim=u.get("hidden_dim", 64),
                            n_layers=u.get("n_layers", 2), lambdas=u.get("lambdas"),
                            n_train=u.get("n_train", 50), n_train_max=u.get("n_train_max"),
                            pred_floor=u.get("pred_floor"))
    else:
        run_final_method_md(u["method"], best_hp=u["best_hp"], alphas=u.get("alphas") or [2.0],
                            seeds=[u["seed"]], imbalance=u.get("imbalance", 0.6),
                            out_dir=u["out_dir"], quick=u["quick"], overwrite=u["overwrite"],
                            arch=u.get("arch", "mlp"), hidden_dim=u.get("hidden_dim", 64),
                            n_layers=u.get("n_layers", 2), lambdas=u.get("lambdas"),
                            decision_backend=u.get("decision_backend", "cvxpylayers"),
                            m=u.get("md_m", 200), n_train=u.get("md_n_train", 50),
                            n_groups=u.get("n_groups", 2), n_resources=u.get("n_resources", 3))
    return (u["method"], u["seed"])


def final_pool(exp, *, methods, seeds, best_hp, out_dir, workers, alphas=None,
               fairness_type=None, quick=False, overwrite=False,
               arch="mlp", hidden_dim=64, n_layers=2, lambdas=None,
               n_train=50, n_train_max=None, imbalance=0.6, decision_backend="cvxpylayers",
               md_m=200, md_n_train=50, n_groups=2, n_resources=3, pred_floor=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    units = [{"exp": exp, "method": m, "seed": s, "best_hp": best_hp, "alphas": alphas,
              "fairness_type": fairness_type, "out_dir": str(out_dir), "quick": quick,
              "overwrite": overwrite, "arch": arch, "hidden_dim": hidden_dim,
              "n_layers": n_layers, "lambdas": lambdas, "n_train": n_train,
              "n_train_max": n_train_max, "imbalance": imbalance, "pred_floor": pred_floor,
              "decision_backend": decision_backend,
              "md_m": md_m, "md_n_train": md_n_train,
              "n_groups": n_groups, "n_resources": n_resources} for m in methods for s in seeds]
    if not overwrite:
        units = [u for u in units
                 if not (out_dir / f"stage__{u['method'].lower()}__s{u['seed']}.csv").exists()]
    if not units:
        print("[parallel] all (method,seed) files present."); return
    units.sort(key=lambda u: -_WEIGHT.get(u["method"], 1))
    W = max(1, min(int(workers), len(units)))
    print(f"[parallel] {len(units)} (method,seed) units across {W} workers (BLAS pinned)...")
    with ProcessPoolExecutor(max_workers=W) as pool:
        futmap = {pool.submit(_final_unit, u): u for u in units}
        for fut in as_completed(futmap):
            u = futmap[fut]
            try:
                fut.result()
                print(f"[done] {u['method']} seed={u['seed']}", flush=True)
            except Exception as e:   # one (method,seed) failing must not abort the others
                print(f"[FAIL] {u['method']} seed={u['seed']}: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hc", action="store_true")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--methods", type=str, default=None)
    ap.add_argument("--seeds", type=str, default=None, help="Comma-sep subset of 11,22,33.")
    ap.add_argument("--alphas", type=str, default="0.5,2.0")
    ap.add_argument("--fairness-type", type=str, default="atkinson")
    ap.add_argument("--lambdas", type=str, default=None,
                    help="HC lambda sweep (comma-sep; default 0,0.5,1,2). §0.6 main = 0,1.")
    ap.add_argument("--arch", type=str, default="mlp",
                    choices=["mlp", "log_linear", "ols", "linear"],
                    help="HC predictor arch (default mlp; 'log_linear' = §0.6 GLM simple model).")
    ap.add_argument("--hidden", type=int, default=64, help="HC MLP hidden width (capacity axis).")
    ap.add_argument("--n-layers", type=int, default=2, help="HC MLP depth (1 for the depth point).")
    ap.add_argument("--imbalance", type=float, default=0.6,
                    help="MD imbalance level (data axis; anchor 0.6). Sweep via repeated runs.")
    ap.add_argument("--md-backend", type=str, default="cvxpylayers",
                    choices=["cvxpylayers"],
                    help="MD decision-gradient backend (cvxpylayers differentiable conic layer; open SCS/ECOS solvers, no MOSEK).")
    ap.add_argument("--n-train", type=int, default=50, help="HC #train instances (default 50).")
    ap.add_argument("--md-m", type=int, default=200,
                    help="MD stakeholders per instance (anchor 200; supported 50-200).")
    ap.add_argument("--n-groups", type=int, default=2,
                    help="MD #protected groups K (default 2 = main grid; K>2 robustness axis).")
    ap.add_argument("--n-resources", type=int, default=3,
                    help="MD #resources D (default 3 = main grid; D=5 resource axis).")
    ap.add_argument("--md-n-train", type=int, default=50,
                    help="MD #train instances (anchor 50; supported 20-100).")
    ap.add_argument("--n-train-max", type=int, default=None,
                    help="HC-4 data-efficiency: draw this many instances once and slice to "
                         "--n-train (nested, so the N-curve isolates N; pass the largest N).")
    ap.add_argument("--hc-hp", type=Path, default=HC_HP)
    ap.add_argument("--md-hp", type=Path, default=MD_HP)
    ap.add_argument("--hc-out", type=Path, default=HC_OUT)
    ap.add_argument("--md-out", type=Path, default=MD_OUT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-workers", type=int, default=1,
                    help="(method,seed)-level parallel processes (>1 => run across cores).")
    ap.add_argument("--pred-floor", type=str, default=None,
                    help="HC prediction floor: 'auto' (clamp to min observed train benefit) or a "
                         "numeric value. No-op for in-support methods; rescues out-of-support "
                         "predict-then-optimize baselines on the softplus-linear rung. HC only.")
    args = ap.parse_args()
    if not (args.hc or args.md):
        ap.error("pick at least one of --hc / --md")
    methods = [s.strip() for s in args.methods.split(",")] if args.methods else list(RUN_A_METHODS)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else FINAL_SEEDS
    W = int(args.max_workers)

    if args.hc:
        best = load_best_hp(args.hc_hp)
        alphas = [float(x) for x in args.alphas.split(",")]
        lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else None
        if W > 1:
            final_pool("hc", methods=methods, seeds=seeds, best_hp=best, out_dir=args.hc_out,
                       workers=W, alphas=alphas, fairness_type=args.fairness_type,
                       quick=args.quick, overwrite=args.overwrite,
                       arch=args.arch, hidden_dim=args.hidden, n_layers=args.n_layers, lambdas=lambdas,
                       n_train=args.n_train, n_train_max=args.n_train_max, pred_floor=args.pred_floor)
        else:
            for mth in methods:
                run_final_method_hc(mth, best_hp=best, alphas=alphas, fairness_type=args.fairness_type,
                                    seeds=seeds, out_dir=args.hc_out, quick=args.quick,
                                    overwrite=args.overwrite, arch=args.arch, hidden_dim=args.hidden,
                                    n_layers=args.n_layers, lambdas=lambdas,
                                    n_train=args.n_train, n_train_max=args.n_train_max,
                                    pred_floor=args.pred_floor)
    if args.md:
        best = load_best_hp(args.md_hp)
        alphas = [float(x) for x in args.alphas.split(",")]
        lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else None
        if W > 1:
            final_pool("md", methods=methods, seeds=seeds, best_hp=best, out_dir=args.md_out,
                       workers=W, alphas=alphas, quick=args.quick, overwrite=args.overwrite,
                       arch=args.arch, hidden_dim=args.hidden, n_layers=args.n_layers,
                       lambdas=lambdas, imbalance=args.imbalance, decision_backend=args.md_backend,
                       md_m=args.md_m, md_n_train=args.md_n_train,
                       n_groups=args.n_groups, n_resources=args.n_resources)
        else:
            for mth in methods:
                run_final_method_md(mth, best_hp=best, alphas=alphas, seeds=seeds,
                                    imbalance=args.imbalance, out_dir=args.md_out,
                                    quick=args.quick, overwrite=args.overwrite, arch=args.arch,
                                    hidden_dim=args.hidden, n_layers=args.n_layers,
                                    lambdas=lambdas, decision_backend=args.md_backend,
                                    m=args.md_m, n_train=args.md_n_train,
                                    n_groups=args.n_groups, n_resources=args.n_resources)


if __name__ == "__main__":
    main()
