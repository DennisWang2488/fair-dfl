from __future__ import annotations

"""Simplex-constrained mean-variance portfolio QP solved via CVXPY.

This task exists for smoke tests where the decision layer is a true QP with
simplex constraints (sum(w)=1, w>=0). Analytic decision gradients are not
provided; use finite-difference decision gradients in the trainer.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List

import numpy as np

from ..losses import group_fairness_loss_and_grad, mse_loss_and_grad
from .base import BaseTask, SplitData, TaskData, add_bias_column


@dataclass
class PortfolioQPSimplexTask(BaseTask):
    n_samples_train: int
    n_samples_val: int
    n_samples_test: int
    n_features: int
    n_assets: int
    risk_aversion: float
    group_bias: float
    noise_std: float
    fairness_type: str = "mad"
    fairness_ge_alpha: float = 2.0
    cvxpy_solvers: List[str] | None = None

    def __post_init__(self) -> None:
        self.name = "portfolio_qp_simplex"
        self.n_outputs = self.n_assets
        self._cvx_problem = None

    def generate_data(self, seed: int) -> TaskData:
        rng = np.random.default_rng(seed)
        groups = np.zeros(self.n_assets, dtype=int)
        groups[self.n_assets // 2 :] = 1

        a = rng.normal(size=(self.n_assets, self.n_assets))
        sigma = (a.T @ a) / float(self.n_assets) + 0.2 * np.eye(self.n_assets)
        true_w = rng.normal(loc=0.0, scale=0.5, size=(self.n_features + 1, self.n_assets))
        group_shift = np.where(groups == 0, -self.group_bias, self.group_bias)

        def sample(n_rows: int) -> SplitData:
            x = rng.normal(size=(n_rows, self.n_features))
            raw = add_bias_column(x) @ true_w
            y = raw + group_shift[None, :] + rng.normal(scale=self.noise_std, size=raw.shape)
            return SplitData(x=x, y=y)

        train = sample(n_rows=self.n_samples_train)
        val = sample(n_rows=self.n_samples_val)
        test = sample(n_rows=self.n_samples_test)
        return TaskData(
            train=train,
            val=val,
            test=test,
            groups=groups,
            meta={"sigma": sigma},
        )

    def _prepare_cvxpy(self, sigma: np.ndarray) -> None:
        import cvxpy as cp

        sigma = np.asarray(sigma, dtype=float)
        sigma = 0.5 * (sigma + sigma.T)
        sigma = sigma + 1e-6 * np.eye(sigma.shape[0])

        w = cp.Variable(self.n_assets)
        mu = cp.Parameter(self.n_assets)
        quad = cp.quad_form(w, sigma)
        objective = cp.Minimize(0.5 * float(self.risk_aversion) * quad - mu @ w)
        constraints = [w >= 0, cp.sum(w) == 1]
        problem = cp.Problem(objective, constraints)
        self._cvx_problem = {"problem": problem, "w": w, "mu": mu, "sigma": sigma}

    def _solve_single(self, mu_vec: np.ndarray) -> np.ndarray:
        import cvxpy as cp

        if self._cvx_problem is None:
            raise RuntimeError("CVXPY problem not initialized; call bind_context first.")
        prob = self._cvx_problem["problem"]
        w = self._cvx_problem["w"]
        mu = self._cvx_problem["mu"]

        mu.value = np.asarray(mu_vec, dtype=float).reshape(self.n_assets)

        solvers = self.cvxpy_solvers or ["OSQP", "CLARABEL", "SCS"]
        last_err: Exception | None = None
        for solver_name in solvers:
            try:
                prob.solve(
                    solver=getattr(cp, str(solver_name)),
                    warm_start=True,
                    verbose=False,
                )
                if w.value is None:
                    raise RuntimeError("CVXPY returned no solution.")
                w_np = np.asarray(w.value, dtype=float).reshape(self.n_assets)
                if not np.all(np.isfinite(w_np)):
                    raise RuntimeError("CVXPY solution contains non-finite values.")
                return w_np
            except Exception as e:  # pragma: no cover (solver-dependent)
                last_err = e
                continue
        raise RuntimeError(f"All CVXPY solvers failed. Last error: {last_err}")

    @staticmethod
    def _objective(w: np.ndarray, mu_true: np.ndarray, sigma: np.ndarray, risk_aversion: float) -> float:
        w = np.asarray(w, dtype=float).reshape(-1)
        mu_true = np.asarray(mu_true, dtype=float).reshape(-1)
        ret = float(mu_true @ w)
        quad = float(w @ (sigma @ w))
        return ret - 0.5 * float(risk_aversion) * quad

    def _decision_regret(self, mu_pred: np.ndarray, mu_true: np.ndarray) -> tuple[float, float, float, int]:
        if self._cvx_problem is None:
            raise RuntimeError("CVXPY problem not initialized; call bind_context first.")
        sigma = np.asarray(self._cvx_problem["sigma"], dtype=float)

        batch = int(mu_pred.shape[0])
        regrets = np.zeros(batch, dtype=float)
        obj_true = np.zeros(batch, dtype=float)
        obj_pred = np.zeros(batch, dtype=float)
        for b in range(batch):
            w_true = self._solve_single(mu_true[b])
            w_pred = self._solve_single(mu_pred[b])
            obj_true[b] = self._objective(w_true, mu_true[b], sigma, self.risk_aversion)
            obj_pred[b] = self._objective(w_pred, mu_true[b], sigma, self.risk_aversion)
            # obj_true should dominate, but clip for solver tolerance/numerical drift.
            regrets[b] = max(float(obj_true[b] - obj_pred[b]), 0.0)

        loss_dec = float(np.mean(regrets))
        denom_bounded = np.abs(obj_true) + np.abs(obj_pred) + 1e-8
        loss_dec_normalized = float(np.mean(regrets / denom_bounded))
        loss_dec_normalized_true = float(np.mean(regrets / (np.abs(obj_true) + 1e-8)))
        solver_calls = int(2 * batch)
        return loss_dec, loss_dec_normalized, loss_dec_normalized_true, solver_calls

    def compute(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, np.ndarray | float]:
        if need_grads:
            raise ValueError("PortfolioQPSimplexTask does not provide analytic decision gradients; use finite_diff backend.")

        loss_pred, grad_pred = mse_loss_and_grad(raw_pred, true)
        loss_fair, grad_fair = group_fairness_loss_and_grad(
            raw_pred,
            true,
            self._current_groups,
            fairness_type=self.fairness_type,
            smoothing=fairness_smoothing,
            ge_alpha=self.fairness_ge_alpha,
        )

        t0 = perf_counter()
        loss_dec, loss_dec_normalized, loss_dec_normalized_true, solver_calls = self._decision_regret(
            mu_pred=np.asarray(raw_pred, dtype=float),
            mu_true=np.asarray(true, dtype=float),
        )
        decision_ms = (perf_counter() - t0) * 1000.0

        return {
            "loss_dec": loss_dec,
            "loss_dec_normalized": loss_dec_normalized,
            "loss_dec_normalized_true": loss_dec_normalized_true,
            "loss_pred": loss_pred,
            "loss_fair": loss_fair,
            "grad_dec": np.zeros_like(raw_pred),
            "grad_pred": grad_pred,
            "grad_fair": grad_fair,
            "solver_calls": solver_calls,
            "decision_ms": decision_ms,
        }

    # ------------------------------------------------------------------
    # Generic decision gradient interface
    # ------------------------------------------------------------------

    def solve_decision(self, pred: np.ndarray, **ctx: Any) -> np.ndarray:
        pred_2d = np.atleast_2d(pred)
        batch = pred_2d.shape[0]
        decisions = np.zeros_like(pred_2d)
        for b in range(batch):
            decisions[b] = self._solve_single(pred_2d[b])
        return decisions

    def evaluate_objective(self, decision: np.ndarray, true: np.ndarray, **ctx: Any) -> float:
        if self._cvx_problem is None:
            raise RuntimeError("CVXPY problem not initialized; call bind_context first.")
        sigma = np.asarray(self._cvx_problem["sigma"], dtype=float)
        decision_2d = np.atleast_2d(decision)
        true_2d = np.atleast_2d(true)
        batch = decision_2d.shape[0]
        objs = np.zeros(batch, dtype=float)
        for b in range(batch):
            objs[b] = self._objective(decision_2d[b], true_2d[b], sigma, self.risk_aversion)
        return float(np.mean(objs))

    def supported_gradient_strategies(self) -> List[str]:
        return ["finite_diff"]

    def bind_context(self, groups: np.ndarray, sigma: np.ndarray) -> None:
        self._current_groups = np.asarray(groups, dtype=int)
        self._prepare_cvxpy(sigma=sigma)
