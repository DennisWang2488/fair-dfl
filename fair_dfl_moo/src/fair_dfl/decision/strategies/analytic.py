"""Analytic decision gradient strategy.

Delegates to the task's built-in compute() or compute_batch() method
which provides closed-form Jacobian-based gradients.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...tasks.base import BaseTask
from ...tasks.medical_resource_allocation import MedicalResourceAllocationTask
from ...tasks.single_resource_alpha_fair import SingleResourceAlphaFairTask
from ..interface import DecisionGradientStrategy, DecisionResult

_BATCH_TASKS = (MedicalResourceAllocationTask, SingleResourceAlphaFairTask)


class AnalyticStrategy(DecisionGradientStrategy):
    """Uses the task's analytic gradient computation.

    This is the default strategy for tasks that implement
    closed-form decision gradients via Jacobian or VJP.
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
        if isinstance(task, _BATCH_TASKS):
            out = task.compute_batch(
                raw_pred=pred,
                true=true,
                cost=ctx["cost"],
                race=ctx["race"],
                need_grads=need_grads,
                fairness_smoothing=fairness_smoothing,
            )
        else:
            out = task.compute(
                raw_pred=pred,
                true=true,
                need_grads=need_grads,
                fairness_smoothing=fairness_smoothing,
            )

        grad_dec = np.asarray(out["grad_dec"], dtype=float).reshape(pred.shape) if need_grads else np.zeros_like(pred)

        return DecisionResult(
            loss_dec=float(out["loss_dec"]),
            grad_dec=grad_dec,
            solver_calls=int(out.get("solver_calls", 0)),
            decision_ms=float(out.get("decision_ms", 0.0)),
            task_output=out,
        )

    def supports_task(self, task: BaseTask) -> bool:
        # Analytic strategy works for any task that implements compute()
        return True
