"""Finite-difference decision gradient strategy.

Computes decision gradients by perturbing predictions and re-solving
the optimization problem. Generalized to work with any task that
implements solve_decision() and evaluate_objective().
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


class FiniteDiffStrategy(DecisionGradientStrategy):
    """Compute decision gradients via central finite differences.

    For each prediction element, perturbs by +/- eps, re-solves the
    optimization problem, and estimates the gradient from regret differences.

    Works with any task that implements solve_decision() and evaluate_objective(),
    or falls back to task-specific logic for known task types.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps

    def compute(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        need_grads: bool = True,
        fairness_smoothing: float = 1e-6,
        **ctx: Any,
    ) -> DecisionResult:
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

        # Compute finite-diff gradient
        t0 = perf_counter()

        # MD knapsack treats the entire (n, n_resources) pred matrix as ONE
        # population-level optimization, not as bsz independent instances —
        # the generic path's per-row split would mis-shape the cvxpy problem.
        if isinstance(task, MultiDimKnapsackTask):
            grad, solver_calls = self._fd_md_knapsack(pred, true, task)
            decision_ms = (perf_counter() - t0) * 1000.0
            return DecisionResult(
                loss_dec=float(out["loss_dec"]),
                grad_dec=grad,
                solver_calls=base_solver_calls + solver_calls,
                decision_ms=base_decision_ms + decision_ms,
                task_output=out,
            )

        # Try generic interface first
        if hasattr(task, "solve_decision") and hasattr(task, "evaluate_objective"):
            try:
                grad, solver_calls = self._generic_finite_diff(pred, true, task, **ctx)
                decision_ms = (perf_counter() - t0) * 1000.0
                return DecisionResult(
                    loss_dec=float(out["loss_dec"]),
                    grad_dec=grad,
                    solver_calls=base_solver_calls + solver_calls,
                    decision_ms=base_decision_ms + decision_ms,
                    task_output=out,
                )
            except NotImplementedError:
                pass

        raise ValueError(
            f"Finite-difference not implemented for {type(task).__name__}. "
            f"Implement solve_decision() and evaluate_objective() on the task."
        )

        decision_ms = (perf_counter() - t0) * 1000.0
        return DecisionResult(
            loss_dec=float(out["loss_dec"]),
            grad_dec=grad,
            solver_calls=base_solver_calls + solver_calls,
            decision_ms=base_decision_ms + decision_ms,
            task_output=out,
        )

    def _generic_finite_diff(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        **ctx: Any,
    ) -> tuple[np.ndarray, int]:
        """Generic finite-diff using task.solve_decision() and task.evaluate_objective()."""
        bsz = int(pred.shape[0])
        dim = int(pred.reshape(bsz, -1).shape[1])
        pred_flat = pred.reshape(bsz, dim)
        grad = np.zeros_like(pred_flat, dtype=float)
        solver_calls = 0

        # Compute true objective for each sample
        obj_true = np.zeros(bsz, dtype=float)
        oracle_solver = getattr(task, "solve_oracle_decision", task.solve_decision)
        for b in range(bsz):
            d_true = oracle_solver(true[b:b+1], **ctx)
            obj_true[b] = task.evaluate_objective(d_true, true[b:b+1], **ctx)
            solver_calls += 1

        for b in range(bsz):
            for j in range(dim):
                pred_plus = pred_flat[b].copy()
                pred_minus = pred_flat[b].copy()
                pred_plus[j] += self.eps
                pred_minus[j] -= self.eps

                d_plus = task.solve_decision(pred_plus[None, :], **ctx)
                d_minus = task.solve_decision(pred_minus[None, :], **ctx)

                obj_plus = task.evaluate_objective(d_plus, true[b:b+1], **ctx)
                obj_minus = task.evaluate_objective(d_minus, true[b:b+1], **ctx)

                regret_plus = max(float(obj_true[b] - obj_plus), 0.0)
                regret_minus = max(float(obj_true[b] - obj_minus), 0.0)

                grad[b, j] = (regret_plus - regret_minus) / (2.0 * self.eps * bsz)
                solver_calls += 2

        return grad.reshape(pred.shape), solver_calls

    def _fd_md_knapsack(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        task: MultiDimKnapsackTask,
    ) -> tuple[np.ndarray, int]:
        """Finite-diff over the population for the redesigned MD knapsack.

        The whole (n, n_resources) prediction matrix is one optimization
        instance, so we hold the active batch (cost, groups, budgets) fixed
        and perturb each entry of ``raw_pred`` in turn, re-solving the LP
        and re-evaluating the objective on the *true* benefits.
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
            raise RuntimeError("MultiDimKnapsackTask: bind a batch before finite-diff.")
        groups = batch.groups

        # True-side objective (one solve over the whole population).
        d_true = task._solve(np.clip(y, 1e-8, None))
        obj_true = task._objective(d_true, y, groups)
        solver_calls = 1

        grad = np.zeros_like(pred_pos, dtype=float)
        for i in range(n):
            for j in range(nr):
                base = pred_pos[i, j]
                pred_plus = pred_pos.copy()
                pred_minus = pred_pos.copy()
                if task.scenario == "lp":
                    pred_plus[i, j] = base + self.eps
                    pred_minus[i, j] = base - self.eps
                else:
                    plus_raw = raw[i, j] + self.eps
                    minus_raw = raw[i, j] - self.eps
                    pred_plus[i, j] = float(_softplus_np(np.array([plus_raw]))[0]) + 1e-5
                    pred_minus[i, j] = float(_softplus_np(np.array([minus_raw]))[0]) + 1e-5

                d_plus = task._solve(pred_plus)
                d_minus = task._solve(pred_minus)
                obj_plus = task._objective(d_plus, y, groups)
                obj_minus = task._objective(d_minus, y, groups)
                regret_plus = max(float(obj_true - obj_plus), 0.0)
                regret_minus = max(float(obj_true - obj_minus), 0.0)
                grad[i, j] = (regret_plus - regret_minus) / (2.0 * self.eps)
                solver_calls += 2

        return grad.reshape(raw_pred.shape), solver_calls

    def supports_task(self, task: BaseTask) -> bool:
        if isinstance(task, MultiDimKnapsackTask):
            return True
        return hasattr(task, "solve_decision") and hasattr(task, "evaluate_objective")
