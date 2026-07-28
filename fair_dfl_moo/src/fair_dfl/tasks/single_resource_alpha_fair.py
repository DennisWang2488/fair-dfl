"""Single-resource group-coupled alpha-fairness task with controllable imbalance.

Combines the **closed-form solver / VJP / objective gradients** from
``MedicalResourceAllocationTask`` (which are vectorised and analytic) with the
**synthetic data generator with benefit_group_bias / cost_group_bias knobs**
from ``MultiDimKnapsackTask``. Built specifically to validate the Track 2
imbalance dissociation finding in a setting where SPSA + CVXPY/SCS gradient
blow-ups cannot contaminate the result.

Design notes
------------
- ``n_outputs = 1`` — a single resource per individual, hence the name.
- The decision problem is the convex non-linear program

    max_{d >= 0}  sum_g f_alpha( sum_{i in g} u(b_i, d_i) )
    s.t.          sum_i c_i d_i <= budget

  with closed form ``MedicalResourceAllocationTask._solve_group``. To make the
  problem non-trivially constrained, set a small ``budget_tightness`` (e.g.
  0.20–0.35) — the budget then binds and prediction errors propagate.
- All fairness metrics from HC plus a new ``gini`` option (Gini coefficient on
  per-group MSE; smooth-aboslute aggregator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np

from ..losses import softplus_with_grad
from .base import BaseTask, SplitData, TaskData
from .medical_resource_allocation import (
    MedicalResourceAllocationTask,
    MedicalSplit,
)


@dataclass
class SingleResourceAlphaFairTask(BaseTask):
    """Synthetic single-resource alpha-fair task with bb/cb imbalance knobs."""

    n_train: int
    n_val: int
    n_test: int
    n_features: int = 5
    alpha_fair: float = 2.0
    poly_degree: int = 2
    snr: float = 5.0

    # Imbalance knobs
    benefit_group_bias: float = 0.0
    benefit_noise_ratio: float = 1.0
    cost_group_bias: float = 0.0
    cost_noise_ratio: float = 1.0

    # Cost generation baseline
    cost_mean: float = 1.0
    cost_std: float = 0.2

    # Decision constraint and fairness penalty
    budget_tightness: float = 0.25
    fairness_type: str = "mad"
    group_ratio: float = 0.5
    decision_mode: str = "group"

    _splits: Dict[str, MedicalSplit] = field(default=None, repr=False, init=False)
    _signal_std: np.ndarray = field(default=None, repr=False, init=False)
    _budget: float = field(default=0.0, repr=False, init=False)

    def __post_init__(self) -> None:
        self.name = "single_resource_alpha_fair"
        self.n_outputs = 1
        if self.alpha_fair <= 0.0:
            raise ValueError("alpha_fair must be positive.")
        if self.decision_mode != "group":
            raise ValueError("Only decision_mode='group' is supported here.")
        ft = str(self.fairness_type).strip().lower()
        if ft == "demographic_parity":
            ft = "dp"
        if ft == "bias_parity":
            ft = "bp"
        if ft not in {"gap", "mad", "atkinson", "dp", "bp", "gini"}:
            raise ValueError(f"fairness_type must be one of {{gap, mad, atkinson, dp, bp, gini}}, got {self.fairness_type!r}")
        self.fairness_type = ft

    # ------------------------------------------------------------------
    # Data generation (synthetic, with bb/cb knobs)
    # ------------------------------------------------------------------
    def _polynomial_signal(self, x: np.ndarray, W_list: List[np.ndarray]) -> np.ndarray:
        out = np.zeros((x.shape[0], 1), dtype=float)
        for degree, W in enumerate(W_list, start=1):
            out += np.power(x, degree) @ W
        return out

    def generate_data(self, seed: int) -> TaskData:
        rng_w = np.random.default_rng(seed + 1000)
        W_list = [rng_w.normal(scale=1.0 / float(d), size=(self.n_features, 1))
                  for d in range(1, self.poly_degree + 1)]

        # Pilot for signal-std calibration so SNR is decoupled from group bias.
        rng_pilot = np.random.default_rng(seed + 9876)
        n_pilot = max(2048, 8 * self.n_train)
        x_pilot = rng_pilot.normal(size=(n_pilot, self.n_features))
        signal_pilot = self._polynomial_signal(x_pilot, W_list)
        signal_std = float(signal_pilot.std(ddof=1))
        signal_std = signal_std if signal_std >= 1e-8 else 1.0
        self._signal_std = np.array([signal_std], dtype=float)

        noise_std_baseline = signal_std / np.sqrt(float(self.snr))

        def sample_split(n_rows: int, split_seed: int) -> MedicalSplit:
            rs = np.random.default_rng(split_seed)
            x = rs.normal(size=(n_rows, self.n_features))
            n0 = max(1, min(n_rows - 1, int(round(self.group_ratio * n_rows))))
            groups = np.zeros(n_rows, dtype=int)
            groups[n0:] = 1
            groups = groups[rs.permutation(n_rows)]

            benefit_shift = np.where(groups == 0, +self.benefit_group_bias, -self.benefit_group_bias)
            cost_shift = np.where(groups == 0, +self.cost_group_bias, -self.cost_group_bias)
            benefit_noise_scale = np.where(groups == 0, 1.0, float(self.benefit_noise_ratio))
            cost_noise_scale = np.where(groups == 0, 1.0, float(self.cost_noise_ratio))

            signal = self._polynomial_signal(x, W_list).reshape(-1)
            benefit_noise = rs.normal(size=signal.shape) * noise_std_baseline * benefit_noise_scale
            benefit_raw = signal + benefit_shift + benefit_noise
            benefit_pos, _ = softplus_with_grad(benefit_raw)
            benefit = benefit_pos + 0.05

            cost_noise = rs.normal(size=signal.shape) * float(self.cost_std)
            cost_raw = float(self.cost_mean) + cost_shift + cost_noise * cost_noise_scale
            cost = np.clip(cost_raw, 1e-3, None)

            return MedicalSplit(
                x=x.astype(np.float64, copy=False),
                y=benefit.astype(np.float64, copy=False),
                cost=cost.astype(np.float64, copy=False),
                race=groups.astype(np.int64, copy=False),
            )

        train = sample_split(self.n_train, seed + 1)
        val = sample_split(self.n_val, seed + 2)
        test = sample_split(self.n_test, seed + 3)

        # Set the budget on the train split — same convention as Track 2's
        # ``budget_tightness * sum(cost)``.
        self._budget = float(self.budget_tightness) * float(train.cost.sum())

        self._splits = {"train": train, "val": val, "test": test}

        return TaskData(
            train=SplitData(x=train.x, y=train.y),
            val=SplitData(x=val.x, y=val.y),
            test=SplitData(x=test.x, y=test.y),
            groups=train.race,
            meta={
                "n_train": np.asarray([train.x.shape[0]], dtype=np.int64),
                "n_val": np.asarray([val.x.shape[0]], dtype=np.int64),
                "n_test": np.asarray([test.x.shape[0]], dtype=np.int64),
                "signal_std": self._signal_std,
                "budget": np.asarray([self._budget], dtype=float),
            },
        )

    @property
    def budget(self) -> float:
        return self._budget

    # ------------------------------------------------------------------
    # Decision regret + grad — closed-form, reusing HC static methods
    # ------------------------------------------------------------------
    def _decision_regret_and_grad(
        self,
        pred_r: np.ndarray,
        true_r: np.ndarray,
        cost: np.ndarray,
        race: np.ndarray,
        need_grad: bool,
    ) -> Tuple[float, float, float, np.ndarray, int, float, np.ndarray, np.ndarray]:
        from time import perf_counter
        t0 = perf_counter()
        d_true = MedicalResourceAllocationTask._solve_group(true_r, cost, race, budget=self._budget, alpha=self.alpha_fair)
        d_hat = MedicalResourceAllocationTask._solve_group(pred_r, cost, race, budget=self._budget, alpha=self.alpha_fair)
        obj_true = MedicalResourceAllocationTask._group_objective(d_true, true_r, race, alpha=self.alpha_fair)
        obj_hat = MedicalResourceAllocationTask._group_objective(d_hat, true_r, race, alpha=self.alpha_fair)

        regret = obj_true - obj_hat
        loss_dec = float(max(0.0, regret))
        loss_dec_normalized = MedicalResourceAllocationTask._normalized_regret(loss_dec, obj_true, eps=1e-7)
        loss_dec_normalized_pred_obj = MedicalResourceAllocationTask._normalized_regret(loss_dec, obj_hat, eps=1e-7)

        grad_pred = np.zeros_like(pred_r)
        if need_grad and loss_dec > 0.0:
            grad_obj_d = MedicalResourceAllocationTask._group_grad_wrt_decision(d_hat, true_r, race, alpha=self.alpha_fair)
            grad_pred = -MedicalResourceAllocationTask._solve_group_vjp(
                grad_obj_d, pred_r, cost, race, budget=self._budget, alpha=self.alpha_fair
            )
        decision_ms = (perf_counter() - t0) * 1000.0
        return loss_dec, loss_dec_normalized, loss_dec_normalized_pred_obj, grad_pred, 2, decision_ms, d_true, d_hat

    # ------------------------------------------------------------------
    # Fairness — reuses HC's static methods, plus Gini from ..losses
    # ------------------------------------------------------------------
    def _compute_fairness(self, pred: np.ndarray, true: np.ndarray, race: np.ndarray, smoothing: float) -> Tuple[float, np.ndarray]:
        if self.fairness_type == "gini":
            from ..losses import group_mse_gini_loss_and_grad
            # losses module expects (B, N) — wrap to 2D and unwrap.
            loss, grad = group_mse_gini_loss_and_grad(
                pred=pred.reshape(1, -1), true=true.reshape(1, -1), groups=race, smoothing=smoothing
            )
            return float(loss), grad.reshape(-1)
        if self.fairness_type == "gap":
            return MedicalResourceAllocationTask._fair_loss_and_grad_gap(pred, true, race, smoothing)
        if self.fairness_type == "mad":
            return MedicalResourceAllocationTask._fair_loss_and_grad_mad(pred, true, race, smoothing)
        if self.fairness_type == "atkinson":
            return MedicalResourceAllocationTask._fair_loss_and_grad_atkinson(pred, true, race, smoothing)
        if self.fairness_type == "dp":
            return MedicalResourceAllocationTask._fair_loss_and_grad_dp(pred, true, race, smoothing)
        if self.fairness_type == "bp":
            return MedicalResourceAllocationTask._fair_loss_and_grad_bias_parity(pred, true, race, smoothing)
        raise ValueError(f"Unknown fairness_type: {self.fairness_type}")

    @staticmethod
    def _pred_loss_and_grad(pred: np.ndarray, true: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = pred - true
        return float(np.mean(diff * diff)), 2.0 * diff / float(max(len(pred), 1))

    # ------------------------------------------------------------------
    # Compute / evaluate — mirrors HC interface so runner.py reuses logic
    # ------------------------------------------------------------------
    def compute_batch(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        cost: np.ndarray,
        race: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, np.ndarray | float]:
        pred = np.clip(np.asarray(raw_pred, dtype=float).reshape(-1), 1e-6, None)
        y = np.asarray(true, dtype=float).reshape(-1)
        c = np.asarray(cost, dtype=float).reshape(-1)
        r = np.asarray(race, dtype=int).reshape(-1)

        loss_dec, loss_dec_normalized, loss_dec_normalized_pred_obj, grad_dec, solver_calls, decision_ms, d_true, d_hat = self._decision_regret_and_grad(
            pred_r=pred, true_r=y, cost=c, race=r, need_grad=need_grads,
        )
        loss_pred, grad_pred = self._pred_loss_and_grad(pred=pred, true=y)
        loss_fair, grad_fair = self._compute_fairness(pred=pred, true=y, race=r, smoothing=float(fairness_smoothing))
        if not need_grads:
            grad_dec = np.zeros_like(pred)
        return {
            "loss_dec": float(loss_dec),
            "loss_dec_normalized": float(loss_dec_normalized),
            "loss_dec_normalized_pred_obj": float(loss_dec_normalized_pred_obj),
            "loss_pred": float(loss_pred),
            "loss_fair": float(loss_fair),
            "grad_dec": grad_dec,
            "grad_pred": grad_pred,
            "grad_fair": grad_fair,
            "solver_calls": int(solver_calls),
            "decision_ms": float(decision_ms),
            "decision_true": d_true,
            "decision_pred": d_hat,
        }

    def evaluate_split(self, split: str, pred: np.ndarray, fairness_smoothing: float = 1e-6) -> Dict[str, float]:
        s = self._splits[split]
        out = self.compute_batch(
            raw_pred=np.asarray(pred, dtype=float).reshape(-1),
            true=s.y, cost=s.cost, race=s.race,
            need_grads=False, fairness_smoothing=fairness_smoothing,
        )
        return {
            "regret": float(out["loss_dec"]),
            "regret_normalized": float(out["loss_dec_normalized"]),
            "regret_normalized_pred_obj": float(out["loss_dec_normalized_pred_obj"]),
            "pred_mse": float(out["loss_pred"]),
            "fairness": float(out["loss_fair"]),
            "solver_calls_eval": float(out["solver_calls"]),
            "decision_ms_eval": float(out["decision_ms"]),
        }

    def sample_batch(self, split: str, batch_size: int, rng: np.random.Generator) -> MedicalSplit:
        s = self._splits[split]
        n = s.x.shape[0]
        if batch_size <= 0 or batch_size >= n:
            return s
        idx = rng.choice(n, size=batch_size, replace=False)
        return MedicalSplit(x=s.x[idx], y=s.y[idx], cost=s.cost[idx], race=s.race[idx])

    def solve_decision(self, pred: np.ndarray, **ctx: Any) -> np.ndarray:
        pred_r = np.clip(np.asarray(pred, dtype=float).reshape(-1), 1e-6, None)
        cost = np.asarray(ctx["cost"], dtype=float).reshape(-1)
        race = np.asarray(ctx["race"], dtype=int).reshape(-1)
        return MedicalResourceAllocationTask._solve_group(pred_r, cost, race, budget=self._budget, alpha=self.alpha_fair)

    def evaluate_objective(self, decision: np.ndarray, true: np.ndarray, **ctx: Any) -> float:
        return MedicalResourceAllocationTask._group_objective(
            np.asarray(decision, dtype=float).reshape(-1),
            np.asarray(true, dtype=float).reshape(-1),
            np.asarray(ctx["race"], dtype=int).reshape(-1),
            alpha=self.alpha_fair,
        )

    def supported_gradient_strategies(self) -> List[str]:
        return ["analytic"]

    def compute(self, raw_pred, true, need_grads, fairness_smoothing: float = 1e-6):
        raise NotImplementedError("Use compute_batch for single_resource_alpha_fair.")
