"""Finite-difference decision-gradient helper.

Extracted from the retired ``core_methods`` legacy trainer (Stage B of
``docs/REFACTOR_PLAN.md``). This is a behavior-preserving move: the function is
byte-for-byte the same code; only its home changed. It is imported lazily by
``training/loop.py`` when the finite-difference decision-gradient backend is selected.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from ..tasks.base import BaseTask
from ..tasks.portfolio_qp_simplex import PortfolioQPSimplexTask


def _finite_diff_decision_grad(
    task: BaseTask,
    raw_pred: np.ndarray,
    true: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, int, float]:
    """Finite-difference gradient of mean decision regret w.r.t. raw_pred.

    Uses per-instance solves (no full-batch re-solves) for separable tasks.
    """
    t0 = perf_counter()
    bsz = int(raw_pred.shape[0])
    dim = int(raw_pred.shape[1])
    grad = np.zeros_like(raw_pred, dtype=float)

    if isinstance(task, PortfolioQPSimplexTask):
        if task._cvx_problem is None:
            raise RuntimeError("PortfolioQPSimplexTask missing CVXPY context; bind_context must run before training.")
        sigma = np.asarray(task._cvx_problem["sigma"], dtype=float)
        mu_pred = np.asarray(raw_pred, dtype=float)
        mu_true = np.asarray(true, dtype=float)

        obj_true = np.zeros(bsz, dtype=float)
        for b in range(bsz):
            w_true = task._solve_single(mu_true[b])
            obj_true[b] = float(task._objective(w_true, mu_true[b], sigma, task.risk_aversion))

        solver_calls = 0
        for b in range(bsz):
            mu_base = mu_pred[b].copy()
            mu_true_b = mu_true[b]
            for j in range(dim):
                mu_plus = mu_base.copy()
                mu_minus = mu_base.copy()
                mu_plus[j] += float(eps)
                mu_minus[j] -= float(eps)
                w_plus = task._solve_single(mu_plus)
                w_minus = task._solve_single(mu_minus)
                obj_pred_plus = float(task._objective(w_plus, mu_true_b, sigma, task.risk_aversion))
                obj_pred_minus = float(task._objective(w_minus, mu_true_b, sigma, task.risk_aversion))
                regret_plus = max(float(obj_true[b] - obj_pred_plus), 0.0)
                regret_minus = max(float(obj_true[b] - obj_pred_minus), 0.0)
                grad[b, j] = (regret_plus - regret_minus) / (2.0 * float(eps) * float(bsz))
                solver_calls += 2

        decision_ms = (perf_counter() - t0) * 1000.0
        return grad, int(solver_calls + bsz), float(decision_ms)

    raise ValueError(f"Finite-difference decision gradient is not implemented for task: {type(task).__name__}")
