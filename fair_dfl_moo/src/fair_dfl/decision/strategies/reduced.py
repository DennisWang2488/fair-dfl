"""CES-smoothed exact decision-gradient strategy for the MD alpha-fair knapsack.

Corrected model (per-individual / row-sum utility, matching the paper's welfare
over the n stakeholders, eqs. for ``W_alpha``)::

    u_i = sum_j r_ij d_ij ,   max_{d>=0} W_alpha(u)   s.t.  sum_i c_ij d_ij <= B_j.

Because resources are perfect substitutes inside ``u_i``, the exact optimum
``d*`` is non-unique (corner/tie solutions), so ``d*(r)`` is non-smooth and a
generic conic differentiable layer (cvxpylayers/diffcp) returns an ill-defined
gradient. We therefore solve a **CES-smoothed** program with a single
temperature ``tau`` (softmax temperature ``tau``; share exponent ``1+tau``;
smooth-max exponent ``1/tau``): the smooth-max over resources replaces the
hard "pick the best benefit/cost resource", giving a smooth, closed-form KKT
system whose solution converges to the exact optimum as ``tau -> 0`` (bias
``O(tau*log m)``). Exact (unsmoothed) solves remain via cvxpy/MOSEK for oracles
and regret evaluation.

Forward: Newton on the reduced residual ``H(z)=0`` with ``z = (log lambda,
log theta)`` (two-level) or ``z = log lambda`` (item-level), all in log space.
Backward: adjoint implicit-function-theorem VJP through the smooth log-space
maps (one ``(m+K) x (m+K)`` linear solve). The gradient is exact for the
smoothed program (matches finite differences of the forward to ~1e-7).

Supported ``alpha in (0, 5]``; ``alpha=1`` is item-level (log welfare). See
``fair-dfl-experiments/experiments/reduced_md_layer/`` for the derivation and
verification (forward vs MOSEK, backward vs finite differences, regret gradient).
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ...losses import softplus_with_grad
from ...tasks.base import BaseTask
from ...tasks.md_knapsack import MultiDimKnapsackTask
from ..interface import DecisionGradientStrategy, DecisionResult
from .cvxpylayers import _objective_torch_md


# ============================================================================
# Regime + info
# ============================================================================
def regime_for(alpha: float, n_groups: int, decision_mode: str = "group") -> str:
    """Objective regime the corrected task uses (item-level vs two-level group)."""
    use_group = decision_mode == "group" and n_groups > 1
    if use_group and alpha < 1.0:
        return "two_level"
    if use_group and alpha >= 2.0 - 1e-12:
        return "two_level" if alpha > 2.0 + 1e-9 else "item_level"  # a=2 collapses
    return "item_level"


@dataclass
class SolveInfo:
    regime: str
    iters: int
    forward_ms: float
    converged: bool
    resid: float
    reduced_dim: int       # m (item-level) or m+K (two-level)


# ============================================================================
# CES log-space maps + reduced residual
# ============================================================================
def _ces_log_maps(z, logr, logc, masks, alpha, tau, regime, n, m):
    log_lam = z[:m]
    log_beta = logr - log_lam.unsqueeze(0) - logc            # (n,m) bang-per-buck
    log_A = tau * torch.logsumexp(log_beta / tau, dim=1)     # (n,) smooth max
    log_w = torch.log_softmax(log_beta / tau, dim=1)         # (n,m) soft assignment
    if regime == "two_level":
        log_theta = z[m:]
        log_theta_full = torch.zeros(n, dtype=z.dtype)
        for k, mask in enumerate(masks):
            log_theta_full = log_theta_full + mask.to(z.dtype) * log_theta[k]
        log_utilde = (log_theta_full + log_A) / alpha
    else:
        log_utilde = log_A / alpha
    log_d = log_utilde.unsqueeze(1) - logr + (1.0 + tau) * log_w
    return log_d, log_utilde, log_A, log_w


def _ces_residual(z, logr, logc, logB, masks, alpha, tau, regime, n, m):
    log_d, _, log_A, _ = _ces_log_maps(z, logr, logc, masks, alpha, tau, regime, n, m)
    H_budget = torch.logsumexp(logc + log_d, dim=0) - logB    # (m,)
    if regime == "item_level":
        return H_budget
    e = (1.0 - alpha) / alpha
    log_T = torch.stack([torch.logsumexp(e * log_A[mask], dim=0) for mask in masks])
    log_theta = z[m:]
    if alpha < 1.0:
        H_theta = (2.0 - alpha) * log_theta + alpha * log_T - alpha * np.log(1.0 - alpha)
    else:
        H_theta = ((alpha * alpha - 2 * alpha + 2) / alpha) * log_theta \
            - (2.0 - alpha) * np.log(alpha - 1.0) - (alpha - 2.0) * log_T
    return torch.cat([H_budget, H_theta])


# ============================================================================
# Forward (Newton) + backward (adjoint IFT)
# ============================================================================
def ces_forward(r, c, B, groups, alpha, tau, z0=None, tol=1e-12, max_iter=100):
    """Newton solve of the CES reduced system. Returns dict(d, z, info-fields)."""
    t0 = perf_counter()
    a = float(alpha)
    if a <= 0.0 or a > 5.0 + 1e-9:
        raise NotImplementedError(
            f"CES reduced layer supports alpha in (0, 5] (got alpha={a}); the "
            "max-min limit (alpha->inf) needs a lexicographic formulation."
        )
    r = np.asarray(r, float); c = np.asarray(c, float); B = np.asarray(B, float)
    n, m = r.shape
    uniq = np.unique(groups); K = len(uniq)
    regime = regime_for(a, K)
    dim = m + (K if regime == "two_level" else 0)
    logr = torch.tensor(np.log(r)); logc = torch.tensor(np.log(c)); logB = torch.tensor(np.log(B))
    masks = [torch.tensor(groups == g) for g in uniq]
    z = torch.zeros(dim, dtype=torch.float64) if z0 is None else torch.as_tensor(z0).clone()

    def H_fn(zz):
        return _ces_residual(zz, logr, logc, logB, masks, a, tau, regime, n, m)

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        H = H_fn(z)
        res = float(H.abs().max())
        if res < tol:
            converged = True
            break
        J = torch.autograd.functional.jacobian(H_fn, z)
        dz = torch.linalg.solve(J, -H)
        step = 1.0
        for _ls in range(40):
            if float(H_fn(z + step * dz).abs().max()) < res:
                break
            step *= 0.5
        z = z + step * dz
    res = float(H_fn(z).abs().max())
    if not converged and res >= tol:
        raise RuntimeError(
            f"CES Newton did not converge in {max_iter} iters "
            f"(alpha={a}, tau={tau}, resid={res:.2e})."
        )
    log_d, _, _, _ = _ces_log_maps(z, logr, logc, masks, a, tau, regime, n, m)
    d = np.clip(torch.exp(log_d).detach().numpy(), 0.0, None)
    ms = (perf_counter() - t0) * 1000.0
    return dict(d=d, z=z, regime=regime, dim=dim, iters=it, converged=converged,
                resid=res, forward_ms=ms)


def ces_vjp(r, c, B, groups, alpha, tau, w, z_star, regime):
    """w^T d(d*)/d(r) via adjoint IFT through the CES maps (one (m+K) solve)."""
    r = np.asarray(r, float); c = np.asarray(c, float); B = np.asarray(B, float)
    n, m = r.shape
    a = float(alpha)
    uniq = np.unique(groups)
    masks = [torch.tensor(groups == g) for g in uniq]
    logc = torch.tensor(np.log(c)); logB = torch.tensor(np.log(B))
    w_t = torch.tensor(np.asarray(w, float))
    r_t = torch.tensor(r, requires_grad=True)
    z_t = torch.as_tensor(z_star).clone().requires_grad_(True)

    log_d, _, _, _ = _ces_log_maps(z_t, torch.log(r_t), logc, masks, a, tau, regime, n, m)
    d = torch.exp(log_d)
    gz_d, gr_d = torch.autograd.grad(d, (z_t, r_t), grad_outputs=w_t, retain_graph=True)

    logr_fixed = torch.log(torch.tensor(r))
    J = torch.autograd.functional.jacobian(
        lambda zz: _ces_residual(zz, logr_fixed, logc, logB, masks, a, tau, regime, n, m),
        z_t.detach())
    a_vec = torch.linalg.solve(J.transpose(0, 1), (-gz_d).unsqueeze(1)).squeeze(1)

    H2 = _ces_residual(z_t.detach(), torch.log(r_t), logc, logB, masks, a, tau, regime, n, m)
    (gr_H,) = torch.autograd.grad(H2, r_t, grad_outputs=a_vec)
    return (gr_d + gr_H).detach().numpy()


# ============================================================================
# Strategy
# ============================================================================
class ReducedAlphaFairStrategy(DecisionGradientStrategy):
    """Exact, fast decision gradient for the corrected MD alpha-fair knapsack via
    a CES-smoothed reduced solve + adjoint implicit differentiation.

    Computes the same regret gradient as the cvxpylayers backend -- d(regret)/
    d(raw_pred) with raw -> softplus -> benefit -> argmax allocation -> regret
    scored at the true benefit -- but exactly (for the tau-smoothed program) and
    without a conic solver in the hot path. Required, not just faster: under the
    corrected (perfect-substitutes) model the exact decision is non-unique, so a
    raw conic layer's gradient is ill-defined; the smoothing resolves it.
    Supported alpha in (0, 5]; smoothing temperature ``tau`` (default 0.02,
    bias O(tau*log m)).
    """

    def __init__(self, tau: float = 0.02) -> None:
        self.tau = float(tau)

    def supports_task(self, task: BaseTask) -> bool:
        return isinstance(task, MultiDimKnapsackTask) and task.scenario != "lp"

    @property
    def name(self) -> str:
        return "ReducedAlphaFair"

    def compute(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        need_grads: bool = True,
        fairness_smoothing: float = 1e-6,
        **ctx: Any,
    ) -> DecisionResult:
        if not isinstance(task, MultiDimKnapsackTask) or task.scenario == "lp":
            raise ValueError(
                "ReducedAlphaFairStrategy supports the MultiDimKnapsackTask "
                "alpha-fair scenario only."
            )

        out = ctx.get("task_output")
        if out is not None:
            base_solver_calls, base_decision_ms = 0, 0.0
        else:
            out = task.compute(
                raw_pred=pred, true=true, need_grads=False,
                fairness_smoothing=fairness_smoothing, skip_regret=True,
            )
            base_solver_calls = int(out.get("solver_calls", 0))
            base_decision_ms = float(out.get("decision_ms", 0.0))

        if not need_grads:
            return DecisionResult(
                loss_dec=float(out.get("loss_dec", 0.0)),
                grad_dec=np.zeros_like(pred, dtype=float),
                solver_calls=base_solver_calls, decision_ms=base_decision_ms,
                task_output=out,
            )

        t0 = perf_counter()
        nr = int(task.n_resources)
        raw = np.asarray(pred, dtype=float).reshape(-1, nr)
        y = np.asarray(true, dtype=float).reshape(-1, nr)
        batch = getattr(task, "_active_batch", None)
        if batch is None:
            raise RuntimeError(
                "MultiDimKnapsackTask: bind a batch before the reduced backward."
            )
        cost = np.asarray(batch.cost, dtype=float)
        budgets = np.asarray(batch.budgets, dtype=float)
        groups = np.asarray(batch.groups)
        alpha = float(task.alpha_fair)
        mode = task.decision_mode
        regime = regime_for(alpha, len(np.unique(groups)), mode)

        pred_pos, pred_pos_grad = softplus_with_grad(raw)
        benefit = pred_pos + 1e-5

        fwd = ces_forward(benefit, cost, budgets, groups, alpha, self.tau)
        d_star = fwd["d"]
        d_true = ces_forward(np.clip(y, 1e-8, None), cost, budgets, groups,
                             alpha, self.tau)["d"]
        obj_true = float(task._objective(d_true, y, groups))
        obj_pred = float(task._objective(d_star, y, groups))
        regret = max(obj_true - obj_pred, 0.0)
        solver_calls = 2

        if regret <= 0.0:
            grad = np.zeros_like(pred, dtype=float)
        else:
            d_t = torch.tensor(d_star, dtype=torch.float64, requires_grad=True)
            obj_pred_t = _objective_torch_md(d_t, y, alpha, nr, mode, groups)
            (g_obj_d,) = torch.autograd.grad(obj_pred_t, d_t)
            w = -g_obj_d.detach().numpy()
            grad_benefit = ces_vjp(benefit, cost, budgets, groups, alpha, self.tau,
                                   w, fwd["z"], regime)
            grad = (grad_benefit * pred_pos_grad).reshape(pred.shape)

        decision_ms = (perf_counter() - t0) * 1000.0
        return DecisionResult(
            loss_dec=float(out.get("loss_dec", 0.0)),
            grad_dec=grad.astype(float, copy=False),
            solver_calls=base_solver_calls + solver_calls,
            decision_ms=base_decision_ms + decision_ms,
            task_output=out,
            extra={
                "reduced_obj_true": obj_true,
                "reduced_obj_pred": obj_pred,
                "reduced_regret_raw": regret,
                "reduced_iters": int(fwd["iters"]),
                "reduced_regime": regime,
                "reduced_tau": self.tau,
            },
        )
