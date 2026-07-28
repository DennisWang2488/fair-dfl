"""Task layer for data-driven optimization experiments used by the 8 active methods."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List

import numpy as np

from ..losses import mse_loss_and_grad
from .base import BaseTask, SplitData, TaskData, add_bias_column


@dataclass
class PortfolioQPMultiConstraintTask(BaseTask):
    n_samples_train: int
    n_samples_val: int
    n_samples_test: int
    n_features: int
    n_assets: int
    n_factors: int
    risk_aversion: float
    group_bias: float
    noise_std: float

    def __post_init__(self) -> None:
        self.name = "portfolio_qp_multi_constraint"
        self.n_outputs = self.n_assets
        if self.n_assets <= self.n_factors + 1:
            raise ValueError("n_assets must be greater than n_factors + 1 for full-rank constraints.")
        if self.n_factors < 1:
            raise ValueError("n_factors must be >= 1.")

    def generate_data(self, seed: int) -> TaskData:
        rng = np.random.default_rng(seed)

        groups = np.arange(self.n_assets, dtype=int) % 3
        group_offsets = np.array([-self.group_bias, 0.0, self.group_bias], dtype=float)
        group_shift = group_offsets[groups]
        group_noise_scales = np.array([0.65, 1.0, 1.65], dtype=float)
        per_asset_noise = group_noise_scales[groups]

        a = rng.normal(size=(self.n_assets, self.n_assets))
        sigma = (a.T @ a) / float(self.n_assets) + 0.25 * np.eye(self.n_assets)

        factor_loadings = rng.normal(size=(self.n_factors, self.n_assets))
        factor_loadings = factor_loadings - factor_loadings.mean(axis=1, keepdims=True)
        factor_loadings = factor_loadings / (factor_loadings.std(axis=1, keepdims=True) + 1e-8)
        constraints = np.vstack([np.ones((1, self.n_assets)), factor_loadings])
        if np.linalg.matrix_rank(constraints) < constraints.shape[0]:
            factor_loadings = factor_loadings + 1e-2 * rng.normal(size=factor_loadings.shape)
            factor_loadings = factor_loadings - factor_loadings.mean(axis=1, keepdims=True)
            constraints = np.vstack([np.ones((1, self.n_assets)), factor_loadings])
        if np.linalg.matrix_rank(constraints) < constraints.shape[0]:
            raise ValueError("Generated constraint matrix is rank-deficient.")

        targets = np.zeros(constraints.shape[0], dtype=float)
        targets[0] = 1.0

        true_w = rng.normal(loc=0.0, scale=0.5, size=(self.n_features + 1, self.n_assets))
        factor_to_feature = rng.normal(loc=0.0, scale=0.25, size=(self.n_features, self.n_factors))

        def sample(n_rows: int) -> SplitData:
            x = rng.normal(size=(n_rows, self.n_features))
            base = add_bias_column(x) @ true_w
            factor_signal = (x @ factor_to_feature) @ factor_loadings
            y = base + 0.2 * factor_signal + group_shift[None, :] + rng.normal(
                scale=self.noise_std * per_asset_noise[None, :], size=base.shape
            )
            return SplitData(x=x, y=y)

        train = sample(n_rows=self.n_samples_train)
        val = sample(n_rows=self.n_samples_val)
        test = sample(n_rows=self.n_samples_test)

        return TaskData(
            train=train,
            val=val,
            test=test,
            groups=groups,
            meta={
                "sigma": sigma,
                "constraints": constraints,
                "targets": targets,
            },
        )

    def _prepare_kkt(self, sigma: np.ndarray, constraints: np.ndarray, targets: np.ndarray) -> None:
        ridge = 1e-4
        q = self.risk_aversion * sigma + ridge * np.eye(sigma.shape[0])
        q_inv = np.linalg.inv(q)
        m = constraints @ q_inv @ constraints.T
        m_inv = np.linalg.inv(m + 1e-8 * np.eye(m.shape[0]))

        qinv_ct = q_inv @ constraints.T
        jac = q_inv - qinv_ct @ m_inv @ constraints @ q_inv
        jac = 0.5 * (jac + jac.T)
        offset = qinv_ct @ m_inv @ targets

        self._kkt_cache = {
            "q": q,
            "q_inv": q_inv,
            "constraints": constraints,
            "targets": targets,
            "m": m,
            "m_inv": m_inv,
            "jac": jac,
            "offset": offset,
        }

    def _solve_weights(self, mu: np.ndarray) -> np.ndarray:
        jac = self._kkt_cache["jac"]
        offset = self._kkt_cache["offset"]
        return mu @ jac.T + offset[None, :]

    def _objective(self, w: np.ndarray, mu_true: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        ret = np.sum(mu_true * w, axis=1)
        quad = np.einsum("bi,ij,bj->b", w, sigma, w)
        return ret - 0.5 * self.risk_aversion * quad

    def _decision_regret_and_grad(
        self,
        mu_pred: np.ndarray,
        mu_true: np.ndarray,
        sigma: np.ndarray,
        need_grads: bool,
    ) -> tuple[float, np.ndarray, int]:
        w_true = self._solve_weights(mu_true)
        w_pred = self._solve_weights(mu_pred)

        obj_true = self._objective(w_true, mu_true, sigma)
        obj_pred = self._objective(w_pred, mu_true, sigma)
        loss_dec = float(np.mean(obj_true - obj_pred))

        grad_mu = np.zeros_like(mu_pred)
        if need_grads:
            jac = self._kkt_cache["jac"]
            grad_w = -(mu_true - self.risk_aversion * (w_pred @ sigma)) / float(mu_pred.shape[0])
            grad_mu = grad_w @ jac.T

        solver_calls = int(2 * mu_pred.shape[0])
        return loss_dec, grad_mu, solver_calls

    def _group_mean_parity_loss_and_grad(
        self,
        pred: np.ndarray,
        groups: np.ndarray,
        smoothing: float,
    ) -> tuple[float, np.ndarray]:
        unique_groups = np.unique(groups)
        group_means = []
        for g in unique_groups:
            mask = groups == g
            group_means.append(pred[:, mask].mean())
        means = np.asarray(group_means, dtype=float)
        mean_of_means = float(means.mean())
        gap = means - mean_of_means
        smooth_abs = np.sqrt(gap * gap + smoothing)
        loss = float(smooth_abs.mean())

        dphi = gap / smooth_abs
        dloss_dmean = (dphi - dphi.mean()) / max(len(unique_groups), 1)
        grad = np.zeros_like(pred)
        for idx, g in enumerate(unique_groups):
            mask = groups == g
            denom = pred.shape[0] * int(mask.sum())
            if denom == 0:
                continue
            grad[:, mask] = dloss_dmean[idx] / float(denom)
        return loss, grad

    def compute(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, np.ndarray | float]:
        sigma = self._current_sigma
        loss_pred, grad_pred = mse_loss_and_grad(raw_pred, true)
        loss_fair, grad_fair = self._group_mean_parity_loss_and_grad(
            raw_pred,
            self._current_groups,
            smoothing=fairness_smoothing,
        )

        t0 = perf_counter()
        loss_dec, grad_dec, solver_calls = self._decision_regret_and_grad(
            mu_pred=raw_pred,
            mu_true=true,
            sigma=sigma,
            need_grads=need_grads,
        )
        decision_ms = (perf_counter() - t0) * 1000.0

        return {
            "loss_dec": loss_dec,
            "loss_pred": loss_pred,
            "loss_fair": loss_fair,
            "grad_dec": grad_dec if need_grads else np.zeros_like(raw_pred),
            "grad_pred": grad_pred if need_grads else np.zeros_like(raw_pred),
            "grad_fair": grad_fair if need_grads else np.zeros_like(raw_pred),
            "solver_calls": solver_calls,
            "decision_ms": decision_ms,
        }

    # ------------------------------------------------------------------
    # Generic decision gradient interface
    # ------------------------------------------------------------------

    def solve_decision(self, pred: np.ndarray, **ctx: Any) -> np.ndarray:
        pred_2d = np.atleast_2d(pred)
        return self._solve_weights(pred_2d)

    def evaluate_objective(self, decision: np.ndarray, true: np.ndarray, **ctx: Any) -> float:
        if not hasattr(self, "_current_sigma"):
            raise RuntimeError("Call bind_context() first.")
        decision_2d = np.atleast_2d(decision)
        true_2d = np.atleast_2d(true)
        return float(np.mean(self._objective(decision_2d, true_2d, self._current_sigma)))

    def supported_gradient_strategies(self) -> List[str]:
        return ["analytic", "finite_diff"]

    def bind_context(
        self,
        groups: np.ndarray,
        sigma: np.ndarray,
        constraints: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self._current_groups = groups
        self._current_sigma = sigma
        self._current_constraints = constraints
        self._current_targets = targets
        self._prepare_kkt(sigma=sigma, constraints=constraints, targets=targets)
