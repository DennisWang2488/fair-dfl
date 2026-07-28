"""Shared experiment configuration — method registry, training defaults, and plot styling."""

from __future__ import annotations

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
ALPHA_VALUES = [0.5, 2.0]
N_SAMPLE_SMALL = 500  # quick verification
N_SAMPLE_FULL = 0     # 0 = use all patients (48,784)

# ---------------------------------------------------------------------------
# Method registry
#
# Each method config declares its objective flags explicitly:
#   use_dec:  use decision regret gradient
#   use_pred: use prediction (MSE) gradient
#   use_fair: use fairness gradient
#
# The "method" key maps to the training loop backend:
#   - Core backends: fpto, dfl, fdfl, moo, fair_moo, saa, var_dro, wdro
#
# MOO methods set "mo_method" to override the default gradient combination.
#
# ---------------------------------------------------------------------------
# Method taxonomy (grouped by training strategy)
# ---------------------------------------------------------------------------
#   1. Predict-then-Optimize (PTO):
#        PTO, FPTO, SAA, WDRO, VarDRO
#      No decision gradient during training; predictor is fit to outcomes
#      (MSE, or a distributionally-robust surrogate) and the solver is
#      applied post-hoc at evaluation time.
#
#   2. Static decision-focused (constant prediction weight):
#        DFL, FDFL, FDFL-0.1, FDFL-0.5, FDFL-Scal
#      The decision-regret gradient is combined with a fixed-weight
#      prediction / fairness term.  FDFL-0.1 / FDFL-0.5 use mu ∈ {0.1, 0.5}
#      as a small prediction-anchor term on top of FDFL's dec+fair; FDFL-Scal
#      uses mu=1 (i.e. standard weighted-sum scalarization).
#
#   3. Dynamic decision-focused (adaptive per-step combination):
#        FPLG, PCGrad, MGDA, NashMTL
#      The gradient combination rule changes every step — either through
#      PLG's alpha_t schedule or through a multi-objective handler that
#      resolves conflicts between per-objective gradients online.
# ---------------------------------------------------------------------------
#
# FDFL loss (new):
#     L_FDFL = L_regret + mu * L_pred + lambda * F
# where mu = pred_weight (static, see pred_weight_mode) and
# lambda = fairness penalty weight (via the lambda sweep).
# ---------------------------------------------------------------------------
ALL_METHOD_CONFIGS = {
    # ================================================================
    # Predict-then-optimize (PTO) — no decision gradient during training
    # ================================================================
    "PTO":    {"method": "fpto",    "use_dec": False, "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "fixed1",
               "lambdas": [0.0], "force_lambda_path_all_methods": False},
    "FPTO":   {"method": "fpto",    "use_dec": False, "use_pred": True,  "use_fair": True,
               "pred_weight_mode": "fixed1"},
    "SAA":    {"method": "saa",     "use_dec": False, "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "fixed1"},
    "WDRO":   {"method": "wdro",    "use_dec": False, "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "fixed1", "wdro_epsilon": 0.1},
    "DFL":       {"method": "dfl",  "use_dec": True,  "use_pred": False, "use_fair": False,
                  "pred_weight_mode": "zero"},
    "FDFL":      {"method": "fdfl", "use_dec": True,  "use_pred": False, "use_fair": True,
                  "pred_weight_mode": "zero"},
    "FDFL-0.1":  {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                  "pred_weight_mode": "0.1", "gradient_merge": "raw"},
    "FDFL-0.5":  {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                  "pred_weight_mode": "0.5", "gradient_merge": "raw"},
    "FDFL-Scal": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                  "pred_weight_mode": "fixed1", "gradient_merge": "raw"},
    "FPLG":   {"method": "fair_moo",  "use_dec": True,  "use_pred": True,  "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": False, "allow_orthogonalization": True},
    "PCGrad": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "fixed1", "continuation": False, "allow_orthogonalization": True,
               "mo_method": "pcgrad", "mo_pcgrad_normalize": False,
               "mo_pcgrad_variant": "original"},
    "MGDA":   {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "fixed1", "continuation": False, "allow_orthogonalization": True,
               "mo_method": "mgda"},
    "NashMTL": {"method": "fair_moo", "use_dec": True, "use_pred": True, "use_fair": True,
                "pred_weight_mode": "fixed1", "continuation": False, "allow_orthogonalization": True,
                "mo_method": "nashmtl",
                "mo_nashmtl_n_iters": 20,
                "mo_nashmtl_normalize": True,
                "mo_nashmtl_eps": 1e-8},
    "alpha_schedule": {"type": "paper_decay", "kappa": 1.0, "c": 100.0, "temperature": 20.0},
    "warmstart_fraction": 0.0,
    # Solver arguments for the differentiable multidimensional-knapsack decision
    # layer (cvxpylayers -> diffcp -> SCS). These MUST be set: the library default
    # is eps=1e-8 with no iteration cap, a tolerance SCS (a first-order method)
    # cannot reach on the alpha=0.5 instances -- it exhausts its iterations and
    # returns an "unbounded (inaccurate)" status, which diffcp raises as a
    # SolverError and which aborts the run. The values below match the tolerance
    # the task itself uses for its exact conic solves (tasks/md_knapsack.py).
    "decision_grad_cvxpylayers_solver_args": {
        "solve_method": "SCS", "eps": 1e-6, "max_iters": 10000,
    },
    "force_lambda_path_all_methods": False,
    "grad_clip_norm": 10000.0,
    "explode_threshold": 1000000.0,
    "fairness_smoothing": 1e-6,
    "log_every": 5,
    "pareto_sweep_mode": True,
    # NOTE: lambda_train is only used when pareto_sweep_mode=False
    # (single-lambda training). In sweep mode (our default), the
    # "lambdas" list is used instead. Kept for backward compatibility.
    "lambda_train": 0.0,
    "model": {
        "arch": "mlp",
        "hidden_dim": 64,
        "n_layers": 2,
        "activation": "relu",
        "dropout": 0.0,
        "batch_norm": False,
        "init_mode": "default",     # "default", "best_practice" (Kaiming He), "legacy_core"
    },
    "device": DEVICE,
}


# ---------------------------------------------------------------------------
# Task config builder
# ---------------------------------------------------------------------------
def make_task_cfg(
    data_csv: str,
    n_sample: int,
    alpha_fair: float,
    fairness_type: str = "mad",
    val_fraction: float = 0.2,
) -> dict:
    return {
        "name": "medical_resource_allocation",
        "data_csv": data_csv,
        "n_sample": n_sample,
        "data_seed": 42,
        "split_seed": 2,
        "test_fraction": 0.5,
        "val_fraction": val_fraction,
        "alpha_fair": alpha_fair,
        "budget": -1,
        "budget_rho": 0.35,
        "decision_mode": "group",
        "fairness_type": fairness_type,
    }


def compute_full_batch_size(data_csv: str, n_sample: int,
                            test_fraction: float = 0.5,
                            val_fraction: float = 0.2) -> int:
    """Compute the full training set size for use as batch_size.

    Full-batch training is required because the allocation solver needs to see
    all patients simultaneously to respect the global budget constraint.
    """
    df = pd.read_csv(data_csv)
    n_total = n_sample if (n_sample > 0 and n_sample < len(df)) else len(df)
    n_test = int(round(test_fraction * n_total))
    n_remaining = n_total - n_test
    n_val = int(round(val_fraction * n_remaining))
    n_train = n_remaining - n_val
    return n_train


# ---------------------------------------------------------------------------
# Plot styling — shared across all plots
# ---------------------------------------------------------------------------
COLOR_MAP = {
    "FPTO": "#1f77b4", "FDFL": "#ff7f0e",
    "FDFL-0.1": "#ffa64d", "FDFL-0.5": "#ff9800", "FDFL-Scal": "#d95f02",
    "FDFL-Scal-mu0.01": "#fdae6b", "FDFL-Scal-mu2": "#a63603",
    "WS-balanced": "#7f7f7f",
    "MGDA": "#bcbd22", "PCGrad": "#17becf",
    "NashMTL": "#d62728",
    "DFL": "#c5b0d5", "FPLG": "#f7b6d2",
    "SAA": "#e6550d", "WDRO": "#393b79",
    "PTO": "#636363", }

MARKER_MAP = {
    "FPTO": "o", "FDFL": "s",
    "FDFL-0.1": "s", "FDFL-0.5": "s", "FDFL-Scal": "s",
    "FDFL-Scal-mu0.01": "s", "FDFL-Scal-mu2": "s",
    "WS-balanced": "p",
    "MGDA": "h", "PCGrad": "*",
    "NashMTL": "X",
    "DFL": "8", "FPLG": "x",
    "SAA": "D", "WDRO": "2",
    "PTO": "o", }
