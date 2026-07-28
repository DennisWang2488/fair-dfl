"""LODL — Locally Optimized Decision Loss (Shah et al., NeurIPS 2022).

A surrogate-decision-loss gradient estimator. Instead of differentiating
through the optimization problem, we **fit a quadratic surrogate** of the
true regret-as-a-function-of-prediction landscape and take its gradient.

For population-level decisions (e.g., MD knapsack with decision_mode=
'group'), there is only one "instance" per training step, so the surrogate
is fit per population-state by probing K perturbations of the current
prediction and accumulating (delta, regret) pairs across steps in a
rolling buffer.

Surrogate model
---------------
With ``p0 = current prediction`` and probe ``p = p0 + delta``,

    R(p) ≈ a + b^T (p - p0) + 0.5 * (p - p0)^T diag(h) (p - p0),  h_i >= 0.

We fit ``b ∈ R^d`` and ``h ∈ R_+^d`` by ridge-regularized least squares
over the buffer of probes recentred to the current ``p0``. The gradient
returned at ``p0`` is just ``b``.

Notes
-----
- LODL was designed for per-instance DFL where one surrogate is amortized
  across many training examples. For our population-level setting, the
  "instance" is constant within a run, so we fit one surrogate per step
  using a rolling probe buffer.
- This implementation is intentionally minimal — it exists so we can
  benchmark LODL gradient quality vs SPSA / FOLD-opt / FD on the MD
  knapsack alpha-fair smoke test.

Reference
---------
Shah, K., Wilder, B., Tambe, M. (2022).
"Decision-Focused Learning without Decision-Making: Learning Locally
Optimized Decision Losses." NeurIPS 2022.
"""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any, Deque, Tuple

import numpy as np

from ...tasks.base import BaseTask
from ...tasks.md_knapsack import MultiDimKnapsackTask
from ..interface import DecisionGradientStrategy, DecisionResult


def _softplus_np(x: np.ndarray) -> np.ndarray:
    positive = np.maximum(x, 0.0)
    return positive + np.log1p(np.exp(-np.abs(x)))


class LODLStrategy(DecisionGradientStrategy):
    """Decision gradient via a learned local quadratic regret surrogate.

    Parameters
    ----------
    n_probes_per_step : int
        K — number of fresh perturbations sampled at each step.
    sigma : float
        Probe magnitude (Gaussian std on raw pred).
    buffer_size : int
        Rolling buffer of past (delta, regret) pairs used to fit the
        surrogate. Older entries are recentred to the current p0 before
        the fit, so the buffer accelerates surrogate convergence in the
        early steps without requiring a re-solve.
    ridge : float
        L2 regularizer on the surrogate parameters (b, h).
    rng_seed : int | None
        Seed for the perturbation generator.
    """

    def __init__(
        self,
        n_probes_per_step: int = 8,
        sigma: float = 5e-3,
        buffer_size: int = 200,
        ridge: float = 1e-4,
        rng_seed: int | None = None,
    ) -> None:
        self.n_probes = int(n_probes_per_step)
        self.sigma = float(sigma)
        self.buffer_size = int(buffer_size)
        self.ridge = float(ridge)
        self._rng = np.random.default_rng(rng_seed)
        # Buffer entries: (raw_pred_at_probe_time, regret_at_probe).
        # We store absolute raw_pred (not delta) so that recentring on
        # later steps is exact even though p0 has moved.
        self._buffer: Deque[Tuple[np.ndarray, float]] = deque(maxlen=self.buffer_size)

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def name(self) -> str:
        return f"LODL(K={self.n_probes}, sigma={self.sigma}, buf={self.buffer_size})"

    def supports_task(self, task: BaseTask) -> bool:
        return isinstance(task, MultiDimKnapsackTask)

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
                f"LODLStrategy only supports MultiDimKnapsackTask, got {type(task).__name__}"
            )

        # --- task baseline output (loss_dec, loss_pred, loss_fair) ---
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

        t0 = perf_counter()
        nr = int(task.n_resources)
        raw = np.asarray(pred, dtype=float).reshape(-1, nr).copy()
        y = np.asarray(true, dtype=float).reshape(-1, nr)
        n = raw.shape[0]
        d = raw.size

        batch = getattr(task, "_active_batch", None)
        if batch is None:
            raise RuntimeError("LODL: bind a batch before calling compute().")
        groups = batch.groups

        # Oracle objective (for regret = max(obj_true - obj_pred, 0)).
        d_true = task._solve(np.clip(y, 1e-8, None))
        obj_true = task._objective(d_true, y, groups)
        solver_calls = 1

        # --- Probe K fresh perturbations around the current raw pred ---
        flat0 = raw.reshape(-1)
        for _k in range(self.n_probes):
            delta = self._rng.standard_normal(d) * self.sigma
            probe_flat = flat0 + delta
            probe = probe_flat.reshape(n, nr)
            if task.scenario == "lp":
                probe_pos = probe
            else:
                probe_pos = _softplus_np(probe) + 1e-5
            probe_pos = np.clip(probe_pos, 1e-8, None)
            d_probe = task._solve(probe_pos)
            obj_probe = task._objective(d_probe, y, groups)
            regret_probe = max(float(obj_true - obj_probe), 0.0)
            self._buffer.append((probe_flat.copy(), regret_probe))
            solver_calls += 1

        # --- Fit quadratic surrogate at p0 = flat0 ---
        # Surrogate: R(p) ≈ a + b^T (p - p0) + 0.5 sum_i h_i (p_i - p0_i)^2
        # Stack rows: [1, delta_1, ..., delta_d, 0.5*delta_1^2, ..., 0.5*delta_d^2]
        # Solve via ridge LS for (a, b, h).
        if len(self._buffer) < 2:
            grad = np.zeros_like(raw)
        else:
            P = np.stack([p for p, _ in self._buffer], axis=0)        # (M, d)
            R = np.array([r for _, r in self._buffer], dtype=float)   # (M,)
            D = P - flat0[None, :]                                    # (M, d)
            M = D.shape[0]
            X = np.concatenate(
                [np.ones((M, 1)), D, 0.5 * D**2], axis=1
            )                                                          # (M, 1+2d)
            # Ridge LS:  (X^T X + ridge*I) theta = X^T R
            n_params = 1 + 2 * d
            A = X.T @ X + self.ridge * np.eye(n_params)
            B = X.T @ R
            try:
                theta = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(A, B, rcond=None)[0]
            b_vec = theta[1 : 1 + d]
            grad = b_vec.reshape(n, nr)

        decision_ms = (perf_counter() - t0) * 1000.0
        return DecisionResult(
            loss_dec=float(out["loss_dec"]),
            grad_dec=grad.reshape(pred.shape),
            solver_calls=base_solver_calls + solver_calls,
            decision_ms=base_decision_ms + decision_ms,
            task_output=out,
            extra={"lodl_buffer_size": len(self._buffer)},
        )
