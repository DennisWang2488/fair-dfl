"""SPSA (Simultaneous Perturbation Stochastic Approximation) decision gradient.

Uses random direction perturbations to estimate the decision gradient.
Much faster than element-wise finite differences: only 2 solver calls per
sample per perturbation direction, regardless of prediction dimensionality.

Cost per step:  bsz * (1 + 2 * n_dirs)   solver calls
    - bsz oracle solves   (for regret clamping)
    - bsz * 2 * n_dirs    perturbation solves

Example (batch_size=32, n_dirs=1):  96 calls/step
vs element-wise FD with dim=7:     32 * (1 + 14) = 480 calls/step

Reference
---------
Spall, J. C. (1992). "Multivariate Stochastic Approximation Using a
Simultaneous Perturbation Gradient Approximation."
IEEE Transactions on Automatic Control, 37(3), 332-341.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ...tasks.base import BaseTask
from ...tasks.md_knapsack import MultiDimKnapsackTask
from ...tasks.medical_resource_allocation import MedicalResourceAllocationTask
from ...tasks.single_resource_alpha_fair import SingleResourceAlphaFairTask
from ..interface import DecisionGradientStrategy, DecisionResult

_BATCH_TASKS = (MedicalResourceAllocationTask, SingleResourceAlphaFairTask)


def _softplus_np(x: np.ndarray) -> np.ndarray:
    positive = np.maximum(x, 0.0)
    exp_term = np.exp(-np.abs(x))
    return positive + np.log1p(exp_term)


class SPSAStrategy(DecisionGradientStrategy):
    """Decision gradient via SPSA (random simultaneous perturbation).

    Instead of perturbing each prediction dimension separately (2 * dim
    solves per sample), SPSA perturbs ALL dimensions at once with a
    random Rademacher vector and estimates the full gradient from just
    2 solves per sample per direction.

    Parameters
    ----------
    eps : float
        Perturbation magnitude.  Slightly larger than element-wise FD
        because each perturbation touches all dimensions simultaneously.
    n_dirs : int
        Number of independent random directions to average over.
        n_dirs=1 is cheapest; higher values reduce variance.
    rng_seed : int | None
        Seed for the Rademacher generator (reproducibility).
    """

    def __init__(
        self,
        eps: float = 5e-3,
        n_dirs: int = 1,
        rng_seed: int | None = None,
    ) -> None:
        self.eps = eps
        self.n_dirs = n_dirs
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def compute(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        need_grads: bool = True,
        fairness_smoothing: float = 1e-6,
        **ctx: Any,
    ) -> DecisionResult:
        # --- base task output (pred/fair losses) ----------------------
        task_output = ctx.get("task_output")
        if task_output is not None:
            out = task_output
            base_solver_calls = 0
            base_decision_ms = 0.0
        elif isinstance(task, _BATCH_TASKS):
            out = task.compute_batch(
                raw_pred=pred,
                true=true,
                cost=ctx["cost"],
                race=ctx["race"],
                need_grads=False,
                fairness_smoothing=fairness_smoothing,
            )
            base_solver_calls = int(out.get("solver_calls", 0))
            base_decision_ms = float(out.get("decision_ms", 0.0))
        else:
            out = task.compute(
                raw_pred=pred,
                true=true,
                need_grads=False,
                fairness_smoothing=fairness_smoothing,
            )
            base_solver_calls = int(out.get("solver_calls", 0))
            base_decision_ms = float(out.get("decision_ms", 0.0))

        if not need_grads:
            return DecisionResult(
                loss_dec=float(out["loss_dec"]),
                grad_dec=np.zeros_like(pred),
                solver_calls=base_solver_calls,
                decision_ms=base_decision_ms,
                task_output=out,
            )

        # --- SPSA gradient --------------------------------------------
        t0 = perf_counter()

        # MD knapsack: the entire (n, n_resources) prediction matrix is ONE
        # population-level optimization. Perturb the whole matrix at once
        # with a Rademacher direction (2 solves per random direction).
        if isinstance(task, MultiDimKnapsackTask):
            grad, solver_calls = self._spsa_md_knapsack(pred, true, task)
            decision_ms = (perf_counter() - t0) * 1000.0
            return DecisionResult(
                loss_dec=float(out["loss_dec"]),
                grad_dec=grad,
                solver_calls=base_solver_calls + solver_calls,
                decision_ms=base_decision_ms + decision_ms,
                task_output=out,
            )

        if hasattr(task, "solve_decision") and hasattr(task, "evaluate_objective"):
            grad, solver_calls = self._spsa_generic(pred, true, task, **ctx)
            decision_ms = (perf_counter() - t0) * 1000.0
            return DecisionResult(
                loss_dec=float(out["loss_dec"]),
                grad_dec=grad,
                solver_calls=base_solver_calls + solver_calls,
                decision_ms=base_decision_ms + decision_ms,
                task_output=out,
            )

        raise ValueError(
            f"SPSA not implemented for {type(task).__name__}. "
            f"Implement solve_decision() and evaluate_objective() on the task."
        )

    # ------------------------------------------------------------------
    # MD-knapsack-specific SPSA (population-level, not per-row)
    # ------------------------------------------------------------------
    def _spsa_md_knapsack(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        task: MultiDimKnapsackTask,
    ) -> tuple[np.ndarray, int]:
        """SPSA over the whole (n, n_resources) matrix as ONE instance.

        Per direction: 2 LP solves (vs n*nr*2 for full FD).
        """
        nr = int(task.n_resources)
        raw = np.asarray(raw_pred, dtype=float).reshape(-1, nr)
        y = np.asarray(true, dtype=float).reshape(-1, nr)
        n = raw.shape[0]

        if task.scenario == "lp":
            pred_pos = raw.copy()
        else:
            pred_pos = _softplus_np(raw) + 1e-5

        batch = getattr(task, "_active_batch", None)
        if batch is None:
            raise RuntimeError("MultiDimKnapsackTask: bind a batch before SPSA.")
        groups = batch.groups

        # Oracle objective (one solve over the whole population).
        d_true = task._solve(np.clip(y, 1e-8, None))
        obj_true = task._objective(d_true, y, groups)
        solver_calls = 1

        grad_acc = np.zeros_like(pred_pos, dtype=float)
        for _d in range(self.n_dirs):
            delta = self._rng.choice([-1.0, 1.0], size=pred_pos.shape)

            if task.scenario == "lp":
                pred_plus = pred_pos + self.eps * delta
                pred_minus = pred_pos - self.eps * delta
            else:
                # Perturb the raw (pre-softplus) prediction so the gradient
                # we return is wrt the model output, not the post-processed
                # benefit. We then re-softplus before solving.
                raw_plus = raw + self.eps * delta
                raw_minus = raw - self.eps * delta
                pred_plus = _softplus_np(raw_plus) + 1e-5
                pred_minus = _softplus_np(raw_minus) + 1e-5

            d_plus = task._solve(pred_plus)
            d_minus = task._solve(pred_minus)
            obj_plus = task._objective(d_plus, y, groups)
            obj_minus = task._objective(d_minus, y, groups)

            regret_plus = max(float(obj_true - obj_plus), 0.0)
            regret_minus = max(float(obj_true - obj_minus), 0.0)
            diff = regret_plus - regret_minus

            # SPSA estimator: g_ij = diff / (2 * eps * delta_ij)
            grad_acc += diff / (2.0 * self.eps * delta)
            solver_calls += 2

        if self.n_dirs > 1:
            grad_acc /= self.n_dirs

        return grad_acc.reshape(raw_pred.shape), solver_calls

    # ------------------------------------------------------------------
    # Core SPSA logic
    # ------------------------------------------------------------------
    def _spsa_generic(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        **ctx: Any,
    ) -> tuple[np.ndarray, int]:
        """SPSA gradient using task.solve_decision / evaluate_objective."""
        bsz = int(pred.shape[0])
        dim = int(pred.reshape(bsz, -1).shape[1])
        pred_flat = pred.reshape(bsz, dim)
        grad_acc = np.zeros_like(pred_flat, dtype=float)
        solver_calls = 0

        # Oracle objectives (needed for regret = max(obj_true - obj_pred, 0))
        obj_true = np.zeros(bsz, dtype=float)
        oracle_solver = getattr(task, "solve_oracle_decision", task.solve_decision)
        for b in range(bsz):
            d_true = oracle_solver(true[b : b + 1], **ctx)
            obj_true[b] = task.evaluate_objective(d_true, true[b : b + 1], **ctx)
            solver_calls += 1

        for _d in range(self.n_dirs):
            # Rademacher perturbation: each element independently ±1
            delta = self._rng.choice([-1.0, 1.0], size=(bsz, dim))

            for b in range(bsz):
                pert = self.eps * delta[b]
                pred_plus = (pred_flat[b] + pert)[None, :]
                pred_minus = (pred_flat[b] - pert)[None, :]

                d_plus = task.solve_decision(pred_plus, **ctx)
                d_minus = task.solve_decision(pred_minus, **ctx)

                obj_plus = task.evaluate_objective(d_plus, true[b : b + 1], **ctx)
                obj_minus = task.evaluate_objective(d_minus, true[b : b + 1], **ctx)

                regret_plus = max(float(obj_true[b] - obj_plus), 0.0)
                regret_minus = max(float(obj_true[b] - obj_minus), 0.0)

                # SPSA estimator: g_j = (f+ - f-) / (2 * eps * delta_j)
                diff = regret_plus - regret_minus
                grad_acc[b] += diff / (2.0 * self.eps * delta[b] * bsz)
                solver_calls += 2

        # Average over random directions
        if self.n_dirs > 1:
            grad_acc /= self.n_dirs

        return grad_acc.reshape(pred.shape), solver_calls

    # ------------------------------------------------------------------
    def supports_task(self, task: BaseTask) -> bool:
        return hasattr(task, "solve_decision") and hasattr(
            task, "evaluate_objective"
        )

    @property
    def name(self) -> str:
        return f"SPSA(eps={self.eps}, n_dirs={self.n_dirs})"
