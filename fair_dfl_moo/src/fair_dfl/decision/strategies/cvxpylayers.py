"""cvxpylayers-based decision gradient strategy (MD knapsack).

Wraps the task's pre-built cvxpy problem in a ``CvxpyLayer`` and backprops
the regret through the KKT system using diffcp. This is the "gold standard"
for smooth convex DFL: gradient is unbiased (modulo solver tolerance) and
exact at the KKT point, in contrast to SPSA (zero-mean noisy) and FOLD-opt
(biased when forward / backward solvers disagree).

Currently implemented for ``MultiDimKnapsackTask`` only. Other tasks can be
added by wiring their cvxpy problem + a torch objective evaluator here.

Requirements
------------
    pip install cvxpylayers diffcp

DPP compliance
--------------
The MD-knapsack alpha-fair group formulation (alpha >= 2) was confirmed
DPP-compliant during the 2026-05-11 environment unblock. If a new problem
formulation breaks DPP, ``CvxpyLayer(...)`` raises ``DPPError`` at build time.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ...tasks.base import BaseTask
from ...tasks.md_knapsack import MultiDimKnapsackTask
from ..interface import DecisionGradientStrategy, DecisionResult


def _objective_torch_md(
    d: torch.Tensor,
    true_benefit: np.ndarray,
    alpha: float,
    n_resources: int,
    decision_mode: str,
    groups: np.ndarray,
) -> torch.Tensor:
    """Torch mirror of ``MultiDimKnapsackTask._objective``.

    Differentiable in ``d`` (the decision tensor returned by CvxpyLayer).
    ``true_benefit`` and ``groups`` are static (numpy).
    """
    eps = 1e-10
    r = torch.as_tensor(true_benefit, dtype=d.dtype, device=d.device)
    # Per-individual utility (row sum over resources), matching _objective.
    utility = torch.clamp((r * d).sum(dim=1), min=eps)            # (n,)
    flat_groups = np.asarray(groups)                             # (n,) per individual

    use_group = (
        decision_mode == "group"
        and groups is not None
        and len(np.unique(groups)) > 1
    )

    if not use_group:
        if abs(alpha - 1.0) < 1e-12:
            return torch.sum(torch.log(utility))
        if alpha < 1.0:
            return torch.sum(utility.pow(1.0 - alpha)) / (1.0 - alpha)
        return -torch.sum(utility.pow(1.0 - alpha)) / (alpha - 1.0)

    # Two-level group alpha-fairness — matches numpy _objective exactly.
    unique_groups = np.unique(groups)
    gk_list = []
    for g in unique_groups:
        mask = torch.as_tensor(flat_groups == g, dtype=torch.bool, device=d.device)
        yk = utility[mask]
        if abs(alpha - 1.0) < 1e-12:
            gk = torch.sum(torch.log(yk))
        elif 0.0 < alpha < 1.0:
            gk = torch.sum(yk.pow(1.0 - alpha)) / (1.0 - alpha)
        elif alpha > 1.0:
            gk = (alpha - 1.0) / torch.clamp(torch.sum(yk.pow(1.0 - alpha)), min=eps)
        else:
            gk = torch.sum(yk)
        gk_list.append(gk)
    gk_arr = torch.clamp(torch.stack(gk_list), min=eps)
    if abs(alpha - 1.0) < 1e-12:
        return torch.sum(torch.log(gk_arr))
    if abs(alpha) < 1e-12:
        return torch.sum(gk_arr)
    return torch.sum(gk_arr.pow(1.0 - alpha) / (1.0 - alpha))


def _softplus_np(x: np.ndarray) -> np.ndarray:
    positive = np.maximum(x, 0.0)
    exp_term = np.exp(-np.abs(x))
    return positive + np.log1p(exp_term)


class CvxpyLayersStrategy(DecisionGradientStrategy):
    """Decision gradient via cvxpylayers + diffcp.

    Caches a ``CvxpyLayer`` keyed by the task's ``_cvx_signature`` so the
    KKT structure is only re-derived when the active batch changes.

    Parameters
    ----------
    solver_args : dict, optional
        Forwarded to ``CvxpyLayer.__call__``. Defaults aim for solver
        robustness on diffcp's SCS backend (cvxpylayers uses SCS under
        the hood for differentiation regardless of forward solver).
    """

    def __init__(self, solver_args: Dict[str, Any] | None = None) -> None:
        # cvxpylayers forwards solver_args to SCS for the differentiable solve.
        # Tighter tolerances reduce gradient noise at the cost of wall-time.
        self.solver_args = solver_args or {"solve_method": "SCS", "eps": 1e-8}
        self._layer = None
        self._layer_signature: tuple | None = None

    # ------------------------------------------------------------------
    def supports_task(self, task: BaseTask) -> bool:
        return isinstance(task, MultiDimKnapsackTask)

    @property
    def name(self) -> str:
        return "CvxpyLayers"

    # ------------------------------------------------------------------
    def _ensure_layer(self, task: MultiDimKnapsackTask):
        """Build (or reuse) a CvxpyLayer for the task's current bound batch."""
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer  # local import for optional dep

        # Force cvxpy problem to exist for this batch.
        if task._cvx_problem is None or task._cvx_signature is None:
            raise RuntimeError(
                "CvxpyLayersStrategy: task has no bound cvxpy problem. "
                "Call task.bind_split(...) or bind_batch(...) first."
            )

        if self._layer is None or self._layer_signature != task._cvx_signature:
            batch = getattr(task, "_active_batch", None)
            if batch is None:
                raise RuntimeError(
                    "CvxpyLayersStrategy: task has no active batch for layer construction."
                )

            n = int(batch.x.shape[0])
            nr = int(task.n_resources)
            cost = np.asarray(batch.cost, dtype=float)
            groups = np.asarray(batch.groups)
            budgets = np.asarray(batch.budgets, dtype=float)

            # Build a cvxpylayers-only copy with an explicit flat decision
            # variable. Some cvxpylayers/cvxpy combinations fail primal
            # recovery for the task's matrix variable even though the problem
            # is DPP-compliant; the flat variable keeps var_id_to_col stable.
            d_flat = cp.Variable(n * nr)
            d_mat = cp.reshape(d_flat, (n, nr), order="C")
            r_param = cp.Parameter((n, nr), nonneg=(task.scenario != "lp"))
            constraints = [d_flat >= 0.0]
            for j in range(nr):
                constraints.append(
                    cp.sum(cp.multiply(cost[:, j], d_mat[:, j])) <= float(budgets[j])
                )

            if task.scenario == "lp":
                constraints.append(d_flat <= 1.0)
                objective = cp.Maximize(cp.sum(cp.multiply(r_param, d_mat)))
            else:
                alpha = float(task.alpha_fair)
                # Per-individual utility (row sum over resources), matching the
                # corrected MultiDimKnapsackTask._build_cvxpy / _objective.
                u = cp.sum(cp.multiply(r_param, d_mat), axis=1)   # (n,) affine
                constraints.append(d_flat >= 1e-6)

                use_group = (
                    task.decision_mode == "group"
                    and groups is not None
                    and len(np.unique(groups)) > 1
                )

                if use_group and alpha < 1.0:
                    group_terms = []
                    for g in np.unique(groups):
                        idx = np.where(groups == g)[0]
                        G_k = cp.sum(cp.power(u[idx], 1.0 - alpha)) / (1.0 - alpha)
                        group_terms.append(G_k)
                    outer_terms = [cp.power(G_k, 1.0 - alpha) for G_k in group_terms]
                    objective = cp.Maximize(cp.sum(cp.hstack(outer_terms)) / (1.0 - alpha))
                elif use_group and alpha >= 2.0 - 1e-12:
                    c_const = (alpha - 1.0) ** (1.0 - alpha) / (1.0 - alpha)
                    S_k_list = []
                    for g in np.unique(groups):
                        idx = np.where(groups == g)[0]
                        S_k = cp.sum(cp.power(u[idx], 1.0 - alpha))
                        S_k_list.append(S_k)
                    outer = cp.hstack([cp.power(S_k, alpha - 1.0) for S_k in S_k_list])
                    objective = cp.Maximize(c_const * cp.sum(outer))
                else:
                    if abs(alpha - 1.0) < 1e-12:
                        objective = cp.Maximize(cp.sum(cp.log(u)))
                    elif alpha < 1.0:
                        objective = cp.Maximize(
                            cp.sum(cp.power(u, 1.0 - alpha)) / (1.0 - alpha)
                        )
                    else:
                        objective = cp.Maximize(
                            -cp.sum(cp.power(u, 1.0 - alpha)) / (alpha - 1.0)
                        )

            problem = cp.Problem(objective, constraints)
            # CvxpyLayer requires the problem to be DPP-compliant; this
            # raises a clear error at build time if the formulation is
            # parameter-affine in the wrong way.
            self._layer = CvxpyLayer(problem, parameters=[r_param], variables=[d_flat])
            self._layer_signature = task._cvx_signature
        return self._layer

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
        if not isinstance(task, MultiDimKnapsackTask):
            raise ValueError(
                f"CvxpyLayersStrategy currently supports MultiDimKnapsackTask only, "
                f"got {type(task).__name__}."
            )

        # ---- base task output (pred/fair losses) ---------------------
        task_output = ctx.get("task_output")
        if task_output is not None:
            out = task_output
            base_solver_calls = 0
            base_decision_ms = 0.0
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

        # ---- cvxpylayers gradient ------------------------------------
        t0 = perf_counter()

        nr = int(task.n_resources)
        raw = np.asarray(pred, dtype=float).reshape(-1, nr)
        y = np.asarray(true, dtype=float).reshape(-1, nr)
        batch = getattr(task, "_active_batch", None)
        if batch is None:
            raise RuntimeError(
                "MultiDimKnapsackTask: bind a batch before cvxpylayers backward."
            )
        groups = batch.groups
        alpha = float(task.alpha_fair)

        layer = self._ensure_layer(task)

        # Forward: raw -> softplus -> benefit_param. We backprop into raw
        # so the gradient returned matches the model's pre-softplus output
        # (consistent with FD and SPSA strategies).
        raw_t = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        if task.scenario == "lp":
            benefit_t = raw_t
        else:
            benefit_t = torch.nn.functional.softplus(raw_t) + 1e-5

        # CvxpyLayer call — this internally solves via SCS and prepares the
        # KKT system for the eventual backward.
        (d_star_flat_t,) = layer(benefit_t, solver_args=self.solver_args)
        d_star_t = d_star_flat_t.reshape(-1, nr)

        # Static oracle objective (no grad through this — true is constant).
        d_true = task._solve(np.clip(y, 1e-8, None))
        obj_true_val = float(task._objective(d_true, y, groups))
        # Two solves: layer forward (counts as 1) + oracle solve (1).
        solver_calls = 2

        # Objective on predicted decision — differentiable in d_star_t.
        obj_pred_t = _objective_torch_md(
            d=d_star_t,
            true_benefit=y,
            alpha=alpha,
            n_resources=nr,
            decision_mode=task.decision_mode,
            groups=groups,
        )

        regret_t = torch.clamp_min(
            torch.as_tensor(obj_true_val, dtype=obj_pred_t.dtype) - obj_pred_t,
            min=0.0,
        )

        # Backward: populates raw_t.grad with d/dpred (regret).
        regret_t.backward()
        grad = (
            raw_t.grad.detach().cpu().numpy().reshape(pred.shape)
            if raw_t.grad is not None
            else np.zeros_like(pred, dtype=float)
        )
        decision_ms = (perf_counter() - t0) * 1000.0

        return DecisionResult(
            loss_dec=float(out["loss_dec"]),
            grad_dec=grad.astype(float, copy=False),
            solver_calls=base_solver_calls + solver_calls,
            decision_ms=base_decision_ms + decision_ms,
            task_output=out,
            extra={"cvxpylayers_obj_true": obj_true_val,
                   "cvxpylayers_obj_pred": float(obj_pred_t.item()),
                   "cvxpylayers_regret_raw": float(regret_t.item())},
        )
