"""SPO+ (Smart Predict then Optimize+) decision gradient.

Provides a convex surrogate loss for LP decision regret.  Unlike SPSA which
uses random perturbations, SPO+ exploits LP structure directly: it solves with
a modified cost vector ``2 * r_hat - r`` and uses the resulting decision
difference as a subgradient.

Cost per step:  2 * bsz  solver calls  (oracle + SPO+ per sample)

Reference
---------
Elmachtoub, A. N. & Grigas, P. (2022). "Smart 'Predict, then Optimize'."
Management Science, 68(1), 9-26.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ...tasks.base import BaseTask
from ..interface import DecisionGradientStrategy, DecisionResult


class SPOPlusStrategy(DecisionGradientStrategy):
    """Decision gradient via SPO+ convex surrogate (LP only).

    For a maximization LP ``max r^T d  s.t. d in D``:
      - SPO+ cost vector: c_spo = 2 * r_hat - r
      - SPO+ decision:    d_spo  = argmax_{d in D} c_spo^T d
      - Oracle decision:  d_star = argmax_{d in D} r^T d
      - SPO+ loss:  L = max(c_spo^T d_spo - r^T d_star, 0)
      - Subgradient w.r.t. r_hat:  g = 2 * (d_spo - d_star)
    """

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
        else:
            out = task.compute(
                raw_pred=pred,
                true=true,
                need_grads=False,
                fairness_smoothing=fairness_smoothing,
                skip_regret=True,
            )

        if not need_grads:
            return DecisionResult(
                loss_dec=float(out["loss_dec"]),
                grad_dec=np.zeros_like(pred),
                solver_calls=0,
                decision_ms=0.0,
                task_output=out,
            )

        # --- SPO+ gradient --------------------------------------------
        t0 = perf_counter()
        pred_2d = np.atleast_2d(np.asarray(pred, dtype=float))
        true_2d = np.atleast_2d(np.asarray(true, dtype=float))
        bsz = pred_2d.shape[0]

        grad_acc = np.zeros_like(pred_2d, dtype=float)
        total_loss = 0.0
        solver_calls = 0

        for b in range(bsz):
            r_hat = pred_2d[b]
            r = true_2d[b]

            # SPO+ cost vector (can be negative — task._solve_raw handles it)
            c_spo = 2.0 * r_hat - r

            # Solve LP with SPO+ cost and with true benefits
            d_spo = task._solve_raw(c_spo)
            d_star = task._solve_raw(r)
            solver_calls += 2

            # SPO+ loss (clamped at 0)
            loss_b = float(np.dot(c_spo, d_spo) - np.dot(r, d_star))
            total_loss += max(loss_b, 0.0)

            # SPO+ subgradient w.r.t. r_hat
            grad_acc[b] = 2.0 * (d_spo - d_star)

        avg_loss = total_loss / max(bsz, 1)
        grad_avg = grad_acc / max(bsz, 1)
        decision_ms = (perf_counter() - t0) * 1000.0

        return DecisionResult(
            loss_dec=avg_loss,
            grad_dec=grad_avg.reshape(pred.shape),
            solver_calls=solver_calls,
            decision_ms=decision_ms,
            task_output=out,
        )

    def supports_task(self, task: BaseTask) -> bool:
        from ...tasks.md_knapsack import MultiDimKnapsackTask

        return isinstance(task, MultiDimKnapsackTask) and task.scenario == "lp"

    @property
    def name(self) -> str:
        return "SPO+"
