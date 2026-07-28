"""Synthetic multi-dimensional knapsack task.

Per-individual, multi-resource design (advisor redesign, 2026-04).

Population
----------
A population of ``n`` individuals (the training/val/test samples), each with
a feature vector ``x_i in R^d`` and a discrete group label ``g_i in {0, 1}``.
A fixed set of ``n_resources`` resources is shared across the population
(default ``n_resources = 2``, e.g. "basic tutor" and "premium tutor").

Per-individual data
-------------------
For each individual ``i`` and resource ``j``:

* ``benefit_{i,j}`` — the marginal benefit of allocating one unit of
  resource ``j`` to individual ``i``. **This is the prediction target**:
  the model predicts ``benefit_hat`` from ``x_i``.
* ``cost_{i,j}``   — how much of resource ``j``'s budget individual ``i``
  consumes per unit allocation. **Cost is known up front** — it never enters
  the prediction, only the constraint set.

Decision and constraints
------------------------
Decision variable ``d in R^{n x n_resources}``, ``d_{i,j} >= 0``. For each
resource ``j`` there is one budget ``B_j`` and the constraint
``sum_i cost_{i,j} d_{i,j} <= B_j``. In the LP scenario we additionally
require ``d_{i,j} <= 1`` (bounded knapsack). This is a genuine
**multi-dimensional knapsack** because there are multiple budget rows.

Objective
---------
* ``scenario="lp"`` — ``max sum_{i,j} benefit_{i,j} * d_{i,j}``.
* ``scenario="alpha_fair"`` — two-level alpha-fair:

      G_g(d) = aggregate of benefit_{i,j} * d_{i,j} over (i in g, j) using
                alpha-fair welfare
      Phi(d) = aggregate of G_g across groups using alpha-fair welfare

  Mirrors the healthcare ``_group_objective`` aggregator.

SNR parameterisation
--------------------
The data generator decouples the *signal-to-noise ratio* from the group bias.
The raw polynomial signal has an empirical std ``sigma_signal`` (estimated
from a pilot sample); per-resource benefit noise std is fixed as

      noise_std = sigma_signal / sqrt(snr).

Group-imbalance knobs (``benefit_group_bias``, ``benefit_noise_ratio``,
``cost_group_bias``, ``cost_noise_ratio``) modulate the per-group means and
spreads *on top of* this baseline so that toggling unfairness does not
silently change the overall SNR. The advisor's hypothesis is that **benefit
imbalance affects the prediction task** (and cascades to decisions) while
**cost imbalance affects only the decision** — both are recorded per group
in the stage-level CSV via :meth:`group_diagnostics` so the effect is
visible empirically even before any plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Tuple
import warnings

import cvxpy as cp
import numpy as np

from ..losses import group_fairness_loss_and_grad, mse_loss_and_grad, softplus_with_grad
from .base import BaseTask, SplitData, TaskData

# Solver preference: MOSEK (most accurate, requires license) -> CLARABEL
# (bundled with CVXPY 1.4+) -> SCS (last resort, may be inaccurate).
_SOLVER_CHAIN = [
    (cp.MOSEK,    {}),
    (cp.CLARABEL, {}),
    (cp.SCS,      {"eps": 1e-6, "max_iters": 10000}),
]
_SCS_WARNED = False


@dataclass
class KnapsackSplit:
    """Per-split materialised data for the multi-dimensional knapsack task."""

    x: np.ndarray         # (n, n_features)
    y: np.ndarray         # (n, n_resources) — true benefits (prediction target)
    cost: np.ndarray      # (n, n_resources) — known costs
    groups: np.ndarray    # (n,) — group label per individual
    budgets: np.ndarray   # (n_resources,) — per-resource budgets


@dataclass
class MultiDimKnapsackTask(BaseTask):
    """Per-individual, multi-resource knapsack with LP and alpha-fair objectives."""

    n_samples_train: int
    n_samples_val: int
    n_samples_test: int
    n_features: int
    n_resources: int = 2
    scenario: str = "alpha_fair"
    alpha_fair: float = 2.0
    poly_degree: int = 2
    snr: float = 5.0

    # --- Group-imbalance knobs ---
    # Benefit imbalance: shifts per-resource benefit mean by +/- bias for the
    # two groups. Per-group noise ratio = group-1 std / group-0 std (baseline).
    benefit_group_bias: float = 0.3
    benefit_noise_ratio: float = 1.0
    # Cost imbalance: same, but applied to the *known* cost. Cost imbalance
    # affects the decision only — not the prediction task.
    cost_group_bias: float = 0.0
    cost_noise_ratio: float = 1.0

    # --- Cost generation baseline (independent of features) ---
    cost_mean: float = 1.0
    cost_std: float = 0.2

    # --- Constraint and decision ---
    budget_tightness: float = 0.5
    fairness_type: str = "mad"
    fairness_ge_alpha: float = 2.0
    group_ratio: float = 0.5
    n_groups: int = 2          # K>2: equal-size groups, biases ramp +bias..-bias
    decision_mode: str = "group"

    # --- Internal state (rebuilt per active split) ---
    _splits: Dict[str, KnapsackSplit] = field(default=None, repr=False, init=False)
    _active_split: str = field(default="", repr=False, init=False)
    _signal_std: np.ndarray = field(default=None, repr=False, init=False)
    _cvx_problem: cp.Problem = field(default=None, repr=False, init=False)
    _cvx_r_param: cp.Parameter = field(default=None, repr=False, init=False)
    _cvx_d_var: cp.Variable = field(default=None, repr=False, init=False)
    _cvx_signature: tuple = field(default=None, repr=False, init=False)

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.scenario not in {"lp", "alpha_fair"}:
            raise ValueError(f"scenario must be 'lp' or 'alpha_fair', got {self.scenario!r}")
        if self.alpha_fair <= 0.0:
            raise ValueError("alpha_fair must be positive.")
        if not (0.0 < self.group_ratio <= 1.0):
            raise ValueError(f"group_ratio must be in (0, 1], got {self.group_ratio}")
        if int(self.n_groups) < 2:
            raise ValueError(f"n_groups must be >= 2, got {self.n_groups}")
        if self.decision_mode not in {"group", "item"}:
            raise ValueError(f"decision_mode must be 'group' or 'item', got {self.decision_mode!r}")
        if int(self.n_resources) < 1:
            raise ValueError(f"n_resources must be >= 1, got {self.n_resources}")
        if float(self.snr) <= 0.0:
            raise ValueError(f"snr must be > 0, got {self.snr}")
        self.name = "md_knapsack"
        self.n_outputs = int(self.n_resources)

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    def _polynomial_signal(self, x: np.ndarray, W_list: List[np.ndarray]) -> np.ndarray:
        """Return f(x) where f is sum_d (x^d) @ W_d, shape (n, n_resources)."""
        out = np.zeros((x.shape[0], self.n_resources), dtype=float)
        for degree, W in enumerate(W_list, start=1):
            out += np.power(x, degree) @ W
        return out

    def generate_data(self, seed: int) -> TaskData:
        rng = np.random.default_rng(seed)

        # Polynomial weights for the benefit signal (one set per resource is
        # already encoded by the second axis of each W matrix).
        rng_w = np.random.default_rng(seed + 1000)
        W_list: List[np.ndarray] = []
        for degree in range(1, self.poly_degree + 1):
            W_list.append(
                rng_w.normal(scale=1.0 / float(degree),
                             size=(self.n_features, self.n_resources))
            )

        # --- Empirical signal-std pilot (decouples SNR from group bias) ---
        rng_pilot = np.random.default_rng(seed + 9876)
        n_pilot = max(2048, 8 * max(self.n_samples_train, 1))
        x_pilot = rng_pilot.normal(size=(n_pilot, self.n_features))
        signal_pilot = self._polynomial_signal(x_pilot, W_list)
        signal_std = signal_pilot.std(axis=0, ddof=1)            # (n_resources,)
        signal_std = np.where(signal_std < 1e-8, 1.0, signal_std)
        self._signal_std = signal_std

        noise_std_baseline = signal_std / np.sqrt(float(self.snr))   # (n_resources,)

        def _sample_split(n_rows: int, split_seed: int) -> KnapsackSplit:
            rs = np.random.default_rng(split_seed)
            # Features.
            x = rs.normal(size=(n_rows, self.n_features))
            # Group assignment: first floor(group_ratio * n) individuals are
            # group 0, rest are group 1; then shuffle so groups are random
            # rather than block-structured.
            K = int(self.n_groups)
            if K == 2:
                n0 = max(1, min(n_rows - 1, int(round(self.group_ratio * n_rows))))
                groups = np.zeros(n_rows, dtype=int)
                groups[n0:] = 1
            else:
                # K equal-size groups (group_ratio only applies to the K=2 case).
                groups = np.repeat(np.arange(K), int(np.ceil(n_rows / K)))[:n_rows]
            perm = rs.permutation(n_rows)
            groups = groups[perm]
            # Linear ramp across groups: group 0 -> +1, group K-1 -> -1 (K=2: exactly +/-1,
            # matching the historical two-group behaviour).
            ramp = (1.0 - 2.0 * groups / max(K - 1, 1))[:, None]    # (n, 1)
            tfrac = (groups / max(K - 1, 1))[:, None]               # (n, 1) in [0, 1]

            # Per-group mean shifts (additive) for benefit and cost.
            # group 0 receives +bias, group 1 receives -bias  (per resource).
            benefit_shift = ramp * self.benefit_group_bias      # (n, 1) -> broadcast
            cost_shift = ramp * self.cost_group_bias

            # Per-group noise scaling (group 0 has baseline std, group 1 is
            # scaled by *_noise_ratio). The advisor wants this to be *separate*
            # from the bias knob.
            benefit_noise_scale = 1.0 + (float(self.benefit_noise_ratio) - 1.0) * tfrac
            cost_noise_scale = 1.0 + (float(self.cost_noise_ratio) - 1.0) * tfrac

            # --- Benefit (the prediction target) ---
            signal = self._polynomial_signal(x, W_list)             # (n, n_resources)
            benefit_noise = (
                rs.normal(size=signal.shape)
                * noise_std_baseline[None, :]
                * benefit_noise_scale
            )
            benefit_raw = signal + benefit_shift + benefit_noise
            # Softplus to keep benefit positive for alpha-fair power objectives;
            # LP can handle raw real benefit but the natural interpretation of
            # "benefit" is non-negative anyway, so we apply it uniformly.
            benefit_pos, _ = softplus_with_grad(benefit_raw)
            benefit = benefit_pos + 0.05

            # --- Cost (KNOWN — never predicted) ---
            cost_baseline_noise = rs.normal(size=signal.shape) * float(self.cost_std)
            cost_raw = (
                float(self.cost_mean)
                + cost_shift
                + cost_baseline_noise * cost_noise_scale
            )
            cost = np.clip(cost_raw, 1e-3, None)

            # --- Per-resource budget B_j = budget_tightness * sum_i cost[i,j] ---
            budgets = float(self.budget_tightness) * cost.sum(axis=0)

            return KnapsackSplit(
                x=x.astype(np.float64, copy=False),
                y=benefit.astype(np.float64, copy=False),
                cost=cost.astype(np.float64, copy=False),
                groups=groups.astype(np.int64, copy=False),
                budgets=budgets.astype(np.float64, copy=False),
            )

        self._splits = {
            "train": _sample_split(self.n_samples_train, seed + 1),
            "val":   _sample_split(self.n_samples_val,   seed + 2),
            "test":  _sample_split(self.n_samples_test,  seed + 3),
        }

        # Default the active split to train so that bare compute() calls
        # (e.g. tests using full-batch training) work without explicit binding.
        self.bind_split("train")

        train_split = self._splits["train"]
        return TaskData(
            train=SplitData(x=train_split.x, y=train_split.y),
            val=SplitData(x=self._splits["val"].x, y=self._splits["val"].y),
            test=SplitData(x=self._splits["test"].x, y=self._splits["test"].y),
            groups=train_split.groups,    # advisory only — per-split groups are canonical
            meta={
                "n_resources": np.asarray([self.n_resources], dtype=np.int64),
                "snr": np.asarray([self.snr], dtype=float),
                "signal_std": signal_std.astype(float, copy=False),
            },
        )

    # ------------------------------------------------------------------
    # CVXPY problem cache
    # ------------------------------------------------------------------
    def _build_cvxpy(
        self,
        n: int,
        cost: np.ndarray,
        groups: np.ndarray,
        budgets: np.ndarray,
    ) -> None:
        """(Re)build the cvxpy problem for a population of size ``n``.

        ``cost`` and ``budgets`` are baked into constants — callers that change
        the population (different split or batch) must call this again.
        """
        nr = int(self.n_resources)
        d = cp.Variable((n, nr), nonneg=True)
        # LP allows negative benefits (needed for SPO+); alpha-fair must be
        # non-negative for the power-function objectives.
        r = cp.Parameter((n, nr), nonneg=(self.scenario != "lp"))
        constraints: List[cp.constraints.constraint.Constraint] = []
        # Per-resource budget constraints (one per resource = "multi-budget").
        for j in range(nr):
            constraints.append(cp.sum(cp.multiply(cost[:, j], d[:, j])) <= float(budgets[j]))

        if self.scenario == "lp":
            constraints.append(d <= 1)
            objective = cp.Maximize(cp.sum(cp.multiply(r, d)))
        else:
            alpha = float(self.alpha_fair)
            # Per-individual utility (row sum over resources): u_i = sum_j r_ij d_ij.
            # Welfare is over the n stakeholders, NOT over n*nr cells.
            u = cp.sum(cp.multiply(r, d), axis=1)                  # (n,) affine
            constraints.append(d >= 1e-6)

            use_group = (
                self.decision_mode == "group"
                and groups is not None
                and len(np.unique(groups)) > 1
            )

            if use_group and alpha < 1.0:
                # Two-level group alpha-fairness, 0 < alpha < 1.
                group_terms = []
                for g in np.unique(groups):
                    idx = np.where(groups == g)[0]
                    G_k = cp.sum(cp.power(u[idx], 1.0 - alpha)) / (1.0 - alpha)
                    group_terms.append(G_k)
                outer_terms = [cp.power(G_k, 1.0 - alpha) for G_k in group_terms]
                objective = cp.Maximize(cp.sum(cp.hstack(outer_terms)) / (1.0 - alpha))

            elif use_group and alpha >= 2.0 - 1e-12:
                # Two-level group alpha-fairness, alpha >= 2 — algebraic
                # rewrite that is DCP-compatible.  Phi = c * sum_k S_k^{a-1}
                # where S_k = sum_{i in g} u_i^{1-a}, c = (a-1)^{1-a}/(1-a)<0.
                c_const = (alpha - 1.0) ** (1.0 - alpha) / (1.0 - alpha)
                S_k_list = []
                for g in np.unique(groups):
                    idx = np.where(groups == g)[0]
                    S_k = cp.sum(cp.power(u[idx], 1.0 - alpha))
                    S_k_list.append(S_k)
                outer = cp.hstack([cp.power(S_k, alpha - 1.0) for S_k in S_k_list])
                objective = cp.Maximize(c_const * cp.sum(outer))

            else:
                # Item-level alpha-fairness (single group, alpha=1, or
                # 1 < alpha < 2 where two-level is not DCP).
                if abs(alpha - 1.0) < 1e-12:
                    objective = cp.Maximize(cp.sum(cp.log(u)))
                elif alpha < 1.0:
                    objective = cp.Maximize(cp.sum(cp.power(u, 1.0 - alpha)) / (1.0 - alpha))
                else:
                    objective = cp.Maximize(-cp.sum(cp.power(u, 1.0 - alpha)) / (alpha - 1.0))

        self._cvx_problem = cp.Problem(objective, constraints)
        self._cvx_r_param = r
        self._cvx_d_var = d

    def bind_split(self, split_name: str) -> None:
        """Make ``split_name`` the active split (rebuild cvxpy if needed)."""
        if self._splits is None or split_name not in self._splits:
            raise RuntimeError(f"Split {split_name!r} unknown — call generate_data first.")
        split = self._splits[split_name]
        self.bind_batch(split)
        self._active_split = split_name

    def bind_batch(self, batch: KnapsackSplit) -> None:
        """Bind a (possibly subset) batch as the active population."""
        signature = (
            batch.x.shape[0],
            batch.cost.tobytes(),
            batch.groups.tobytes(),
            batch.budgets.tobytes(),
            self.scenario,
            float(self.alpha_fair),
            self.decision_mode,
        )
        if signature != self._cvx_signature or self._cvx_problem is None:
            self._build_cvxpy(
                n=batch.x.shape[0],
                cost=batch.cost,
                groups=batch.groups,
                budgets=batch.budgets,
            )
            self._cvx_signature = signature
        self._active_batch = batch

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    def _run_solver(self) -> np.ndarray:
        global _SCS_WARNED
        nr = int(self.n_resources)
        for solver, kwargs in _SOLVER_CHAIN:
            try:
                self._cvx_problem.solve(solver=solver, **kwargs)
                if self._cvx_d_var.value is not None:
                    if solver == cp.SCS and not _SCS_WARNED:
                        warnings.warn(
                            "Knapsack solver fell back to SCS (MOSEK and CLARABEL "
                            "unavailable or failed). Results may be inaccurate.",
                            stacklevel=2,
                        )
                        _SCS_WARNED = True
                    return np.clip(np.asarray(self._cvx_d_var.value, dtype=float), 0.0, None)
            except Exception:
                # Fall through to the next solver on ANY failure, not just
                # cp.SolverError: an installed-but-unlicensed MOSEK raises
                # mosek.Error (a missing-license error), which must not abort the
                # MOSEK -> CLARABEL -> SCS fallback chain.
                continue
        n = self._cvx_r_param.shape[0] if hasattr(self._cvx_r_param.shape, '__len__') else 0
        return np.zeros((n, nr), dtype=float)

    def _solve(self, benefit: np.ndarray) -> np.ndarray:
        """Solve with non-negative benefit (alpha-fair safe)."""
        b = np.asarray(benefit, dtype=float)
        if b.shape != (self._cvx_r_param.shape[0], self._cvx_r_param.shape[1]):
            raise ValueError(
                f"benefit shape {b.shape} does not match active cvxpy "
                f"shape {tuple(self._cvx_r_param.shape)}"
            )
        self._cvx_r_param.value = np.clip(b, 1e-8, None)
        return self._run_solver()

    def _solve_raw(self, benefit: np.ndarray) -> np.ndarray:
        """Solve LP with raw benefit (may contain negatives). For SPO+."""
        if self.scenario != "lp":
            raise ValueError("_solve_raw is only valid for the LP scenario.")
        b = np.asarray(benefit, dtype=float)
        self._cvx_r_param.value = b
        return self._run_solver()

    # ------------------------------------------------------------------
    # Objective and regret
    # ------------------------------------------------------------------
    def _objective(self, decision: np.ndarray, true_benefit: np.ndarray, groups: np.ndarray) -> float:
        d = np.asarray(decision, dtype=float)
        r = np.asarray(true_benefit, dtype=float)
        if self.scenario == "lp":
            return float(np.sum(r * d))

        eps = 1e-10
        alpha = float(self.alpha_fair)
        # Per-individual utility (row sum over resources): u_i = sum_j r_ij d_ij.
        # Welfare is over the n stakeholders (matching the paper's W over n
        # individuals), NOT over n*nr (individual, resource) cells.
        utility = np.clip(np.sum(r * d, axis=1), eps, None)        # (n,)
        flat_groups = np.asarray(groups)                           # (n,) per individual

        use_group = (
            self.decision_mode == "group"
            and groups is not None
            and len(np.unique(groups)) > 1
        )

        if not use_group:
            if abs(alpha - 1.0) < 1e-12:
                return float(np.sum(np.log(utility)))
            if alpha < 1.0:
                return float(np.sum(utility ** (1.0 - alpha)) / (1.0 - alpha))
            return float(-np.sum(utility ** (1.0 - alpha)) / (alpha - 1.0))

        # Two-level group alpha-fairness (matches healthcare _group_objective).
        unique_groups = np.unique(groups)
        gk_vals: List[float] = []
        for g in unique_groups:
            mask = flat_groups == g
            yk = utility[mask]
            if abs(alpha - 1.0) < 1e-12:
                gk = float(np.sum(np.log(yk)))
            elif 0.0 < alpha < 1.0:
                gk = float(np.sum(yk ** (1.0 - alpha)) / (1.0 - alpha))
            elif alpha > 1.0:
                gk = float((alpha - 1.0) / max(np.sum(yk ** (1.0 - alpha)), eps))
            else:
                gk = float(np.sum(yk))
            gk_vals.append(gk)
        gk_arr = np.clip(np.asarray(gk_vals, dtype=float), eps, None)
        if abs(alpha - 1.0) < 1e-12:
            return float(np.sum(np.log(gk_arr)))
        if abs(alpha) < 1e-12:
            return float(np.sum(gk_arr))
        return float(np.sum(gk_arr ** (1.0 - alpha) / (1.0 - alpha)))

    def _decision_regret(
        self,
        pred_pos: np.ndarray,
        true: np.ndarray,
        groups: np.ndarray,
    ) -> Tuple[float, float, float, int, float, np.ndarray, np.ndarray]:
        t0 = perf_counter()
        d_true = self._solve(true)
        d_pred = self._solve(pred_pos)
        solver_calls = 2
        decision_ms = (perf_counter() - t0) * 1000.0
        obj_true = self._objective(d_true, true, groups)
        obj_pred = self._objective(d_pred, true, groups)
        regret = max(obj_true - obj_pred, 0.0)
        denom = abs(obj_true) + abs(obj_pred) + 1e-8
        loss_dec_norm = float(regret / denom)
        loss_dec_norm_true = float(regret / max(abs(obj_true), 1e-8))
        return float(regret), loss_dec_norm, loss_dec_norm_true, solver_calls, decision_ms, d_true, d_pred

    # ------------------------------------------------------------------
    # compute() — called from training loop / eval
    # ------------------------------------------------------------------
    def compute(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
        skip_regret: bool = False,
    ) -> Dict[str, Any]:
        if need_grads:
            raise ValueError(
                "MultiDimKnapsackTask does not provide analytic decision gradients. "
                "Use training.decision_grad_backend='finite_diff' or 'spsa'."
            )

        nr = int(self.n_resources)
        raw = np.asarray(raw_pred, dtype=float).reshape(-1, nr)
        y = np.asarray(true, dtype=float).reshape(-1, nr)
        if y.shape[0] != raw.shape[0]:
            raise ValueError(
                f"pred and true have mismatched #rows: {raw.shape} vs {y.shape}"
            )

        # Active batch (set via bind_split / bind_batch).
        batch = getattr(self, "_active_batch", None)
        if batch is None:
            raise RuntimeError("MultiDimKnapsackTask: no active batch — call bind_split first.")
        if batch.x.shape[0] != raw.shape[0]:
            raise RuntimeError(
                f"Active batch has {batch.x.shape[0]} rows but pred has {raw.shape[0]}."
            )
        groups = batch.groups

        # LP allows negative benefits (raw); alpha-fair softplus-positives them.
        if self.scenario == "lp":
            pred_pos = raw
            pred_pos_grad = np.ones_like(raw)
        else:
            pred_pos, pred_pos_grad = softplus_with_grad(raw)
            pred_pos = pred_pos + 1e-5

        # Prediction MSE and analytic gradient.
        loss_pred, grad_pred_pos = mse_loss_and_grad(pred_pos, y)
        # Fairness on per-row predictions; group_fairness expects (B, N) so we
        # treat the rows as the "items" axis (B=1 view).
        # Use the existing 2D formulation over (1, n*n_resources) by flattening.
        flat_pred = pred_pos.reshape(1, -1)
        flat_true = y.reshape(1, -1)
        flat_groups = np.repeat(groups, nr)
        loss_fair, grad_fair_flat = group_fairness_loss_and_grad(
            flat_pred,
            flat_true,
            flat_groups,
            fairness_type=self.fairness_type,
            smoothing=fairness_smoothing,
            ge_alpha=self.fairness_ge_alpha,
        )
        grad_fair_pos = grad_fair_flat.reshape(pred_pos.shape)

        if skip_regret:
            loss_dec = 0.0
            loss_dec_norm = 0.0
            loss_dec_norm_true = 0.0
            solver_calls = 0
            decision_ms = 0.0
            dec_fair: Dict[str, float] = {}
        else:
            loss_dec, loss_dec_norm, loss_dec_norm_true, solver_calls, decision_ms, _d_true, d_pred = (
                self._decision_regret(pred_pos, y, groups)
            )
            dec_fair = self._decision_fairness_metrics(d_pred, y, groups)

        result: Dict[str, Any] = {
            "loss_dec": loss_dec,
            "loss_dec_normalized": loss_dec_norm,
            "loss_dec_normalized_true": loss_dec_norm_true,
            "loss_pred": loss_pred,
            "loss_fair": loss_fair,
            "grad_dec": np.zeros_like(raw),
            "grad_pred": grad_pred_pos * pred_pos_grad,
            "grad_fair": grad_fair_pos * pred_pos_grad,
            "solver_calls": solver_calls,
            "decision_ms": decision_ms,
        }
        result.update(dec_fair)
        # Per-group descriptive statistics for diagnostics (used by stage CSV).
        result.update(self.group_diagnostics(batch=batch, decision=None if skip_regret else d_pred))
        return result

    # ------------------------------------------------------------------
    # Helpers for the training loop and eval
    # ------------------------------------------------------------------
    def sample_batch(self, split_name: str, batch_size: int, rng: np.random.Generator) -> KnapsackSplit:
        if self._splits is None:
            raise RuntimeError("Call generate_data() before sample_batch().")
        s = self._splits[split_name]
        n = s.x.shape[0]
        if batch_size <= 0 or batch_size >= n:
            self.bind_batch(s)
            return s
        idx = rng.choice(n, size=batch_size, replace=False)
        sub_cost = s.cost[idx]
        # Re-scale per-resource budgets so that the sub-population sees a
        # comparable budget tightness as the full split.
        sub_budgets = float(self.budget_tightness) * sub_cost.sum(axis=0)
        sub = KnapsackSplit(
            x=s.x[idx], y=s.y[idx], cost=sub_cost, groups=s.groups[idx], budgets=sub_budgets,
        )
        self.bind_batch(sub)
        return sub

    def evaluate_split(
        self,
        split: str,
        pred: np.ndarray,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, float]:
        s = self._splits[split]
        self.bind_batch(s)
        out = self.compute(
            raw_pred=np.asarray(pred, dtype=float),
            true=s.y,
            need_grads=False,
            fairness_smoothing=fairness_smoothing,
        )
        metrics: Dict[str, float] = {
            "regret": float(out["loss_dec"]),
            "regret_normalized": float(out.get("loss_dec_normalized", 0.0)),
            "regret_normalized_true": float(out.get("loss_dec_normalized_true", 0.0)),
            "pred_mse": float(out["loss_pred"]),
            "fairness": float(out["loss_fair"]),
            "solver_calls_eval": float(out["solver_calls"]),
            "decision_ms_eval": float(out["decision_ms"]),
        }
        for key, val in out.items():
            if key.startswith("decision_") or key.startswith("group_"):
                metrics[key] = float(val)
        return metrics

    def solve_decision(self, pred: np.ndarray, **ctx: Any) -> np.ndarray:
        nr = int(self.n_resources)
        raw = np.asarray(pred, dtype=float).reshape(-1, nr)
        if self.scenario == "lp":
            return self._solve_raw(raw) if raw.min() < 0 else self._solve(raw)
        pred_pos, _ = softplus_with_grad(raw)
        pred_pos = pred_pos + 1e-5
        return self._solve(pred_pos)

    def solve_oracle_decision(self, true: np.ndarray, **ctx: Any) -> np.ndarray:
        nr = int(self.n_resources)
        true_2d = np.asarray(true, dtype=float).reshape(-1, nr)
        return self._solve(np.clip(true_2d, 1e-8, None))

    def evaluate_objective(self, decision: np.ndarray, true: np.ndarray, **ctx: Any) -> float:
        nr = int(self.n_resources)
        decision_2d = np.asarray(decision, dtype=float).reshape(-1, nr)
        true_2d = np.asarray(true, dtype=float).reshape(-1, nr)
        groups = ctx.get("groups", None)
        if groups is None:
            batch = getattr(self, "_active_batch", None)
            if batch is None:
                raise ValueError("groups must be provided via ctx or via bind_batch.")
            groups = batch.groups
        return self._objective(decision_2d, true_2d, np.asarray(groups, dtype=int))

    def supported_gradient_strategies(self) -> List[str]:
        if self.scenario == "lp":
            return ["finite_diff", "spsa", "spo_plus"]
        # alpha-fair: finite_diff / spsa (model-free), cvxpylayers (generic conic
        # differentiable layer), and reduced (exact water-filling + implicit diff,
        # the fast/robust default for alpha in (0, 5]).
        return ["finite_diff", "spsa", "cvxpylayers", "reduced"]

    # ------------------------------------------------------------------
    # Diagnostics — recorded into stage CSV
    # ------------------------------------------------------------------
    def _decision_fairness_metrics(
        self,
        decision: np.ndarray,
        true_benefit: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, float]:
        """Decision-level group fairness summaries (evaluation only)."""
        d = np.asarray(decision, dtype=float)
        t = np.asarray(true_benefit, dtype=float)
        g0 = groups == 0
        g1 = groups == 1
        if g0.sum() == 0 or g1.sum() == 0:
            return {
                "decision_alloc_gap": 0.0,
                "decision_selection_gap": 0.0,
                "decision_welfare_gap": 0.0,
            }

        alloc_gap = abs(float(d[g0].mean()) - float(d[g1].mean()))
        sel = (d > 0.5).astype(float)
        sel_gap = abs(float(sel[g0].mean()) - float(sel[g1].mean()))
        welfare = t * d
        welfare_gap = abs(float(welfare[g0].mean()) - float(welfare[g1].mean()))
        return {
            "decision_alloc_gap": alloc_gap,
            "decision_selection_gap": sel_gap,
            "decision_welfare_gap": welfare_gap,
        }

    def group_diagnostics(
        self,
        batch: KnapsackSplit,
        decision: np.ndarray | None,
    ) -> Dict[str, float]:
        """Per-group descriptive statistics for benefit, cost, and decision.

        These are recorded in ``compute()`` results so the per-stage CSV row
        carries enough information to inspect the **benefit imbalance** vs
        **cost imbalance** effect predicted by the advisor without having to
        re-run the data generator. Only the statistics we want in the table
        are emitted — keep the column count manageable.
        """
        out: Dict[str, float] = {}
        groups = batch.groups
        nr = int(self.n_resources)
        for g in (0, 1):
            mask = groups == g
            if mask.sum() == 0:
                continue
            for j in range(nr):
                # Benefit and cost summaries.
                out[f"group_{g}_benefit_mean_r{j}"] = float(batch.y[mask, j].mean())
                out[f"group_{g}_benefit_std_r{j}"]  = float(batch.y[mask, j].std(ddof=0))
                out[f"group_{g}_cost_mean_r{j}"]    = float(batch.cost[mask, j].mean())
                out[f"group_{g}_cost_std_r{j}"]     = float(batch.cost[mask, j].std(ddof=0))
            if decision is not None:
                d_arr = np.asarray(decision, dtype=float).reshape(-1, nr)
                for j in range(nr):
                    out[f"group_{g}_decision_mean_r{j}"] = float(d_arr[mask, j].mean())
        return out
