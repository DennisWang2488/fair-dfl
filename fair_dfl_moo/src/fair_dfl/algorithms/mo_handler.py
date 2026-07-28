"""Multi-objective gradient handler implementations for Decision-Focused Learning.

Provides abstract base class and eight concrete strategies:
  - WeightedSumHandler: normalized weighted sum of objective gradients
  - PCGradHandler: projecting away conflicting gradient components (Yu et al. 2020)
  - MGDAHandler: minimum-norm point in the convex hull (Sener & Koltun 2018)
  - PLGHandler3Obj: prediction-loss-guided 3-objective extension for DFL
  - NashMTLHandler: Nash bargaining solution for MTL (Navon et al. NeurIPS 2022)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from ..metrics import cosine, l2_norm, project_orthogonal


class MultiObjectiveGradientHandler(ABC):
    """Abstract base for multi-objective gradient combination strategies."""

    @abstractmethod
    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        """Return combined gradient direction. Same shape as each input grad (1D, flattened params)."""

    @abstractmethod
    def extra_logs(self) -> Dict[str, float]:
        """Diagnostics from last call.

        Must include:
        - mo_grad_norm_{name}: per-objective gradient norms
        - mo_cos_{name1}_{name2}: pairwise cosine similarities
        - direction_alignment_with_dec_regret: cosine(output, g_dec)
        - stationarity_proxy: min_{lambda in Delta} ||sum lambda_i g_i||
        """

    # ------------------------------------------------------------------
    # Shared diagnostic helpers
    # ------------------------------------------------------------------

    def _compute_common_diagnostics(
        self,
        grads: Dict[str, np.ndarray],
        direction: np.ndarray,
    ) -> Dict[str, float]:
        """Compute standard diagnostics shared across all handlers."""
        diag: Dict[str, float] = {}
        names = sorted(grads.keys())

        # Per-objective gradient norms.
        for name in names:
            diag[f"mo_grad_norm_{name}"] = l2_norm(grads[name])

        # Pairwise cosine similarities.
        for n1, n2 in combinations(names, 2):
            diag[f"mo_cos_{n1}_{n2}"] = cosine(grads[n1], grads[n2])

        # Alignment of output direction with decision regret gradient.
        if "decision_regret" in grads:
            diag["direction_alignment_with_dec_regret"] = cosine(direction, grads["decision_regret"])
        else:
            diag["direction_alignment_with_dec_regret"] = 0.0

        # Stationarity proxy: min over simplex of ||sum lambda_i g_i||.
        diag["stationarity_proxy"] = _stationarity_proxy(list(grads.values()))

        return diag


# ======================================================================
# Utility: stationarity proxy via grid search over the simplex
# ======================================================================

def _simplex_grid(m: int, n_per_dim: int = 200) -> np.ndarray:
    """Generate ~n_per_dim uniformly spaced points on the (m-1)-simplex."""
    if m == 1:
        return np.array([[1.0]])
    if m == 2:
        t = np.linspace(0.0, 1.0, n_per_dim)
        return np.column_stack([t, 1.0 - t])
    if m == 3:
        pts: List[np.ndarray] = []
        n = max(int(np.sqrt(n_per_dim)), 10)
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                pts.append(np.array([i / n, j / n, k / n]))
        return np.array(pts)
    # General case: random sampling.
    rng = np.random.default_rng(0)
    raw = rng.exponential(size=(n_per_dim, m))
    return raw / raw.sum(axis=1, keepdims=True)


def _stationarity_proxy(grad_list: List[np.ndarray]) -> float:
    """min_{lambda in simplex} ||sum lambda_i g_i||, grid search."""
    m = len(grad_list)
    if m == 0:
        return 0.0
    G = np.stack([g.ravel() for g in grad_list], axis=0)  # (m, d)
    lambdas = _simplex_grid(m, n_per_dim=200)  # (K, m)
    # Vectorized: combined = lambdas @ G  -> (K, d)
    combined = lambdas @ G
    norms = np.sqrt(np.sum(combined * combined, axis=1) + 1e-12)
    return float(np.min(norms))


# ======================================================================
# 1. WeightedSumHandler
# ======================================================================

class WeightedSumHandler(MultiObjectiveGradientHandler):
    """Normalized weighted sum of per-objective gradients."""

    def __init__(self, weights: Dict[str, float]) -> None:
        self._weights = dict(weights) if weights else {}
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        # Resolve weights: default to equal if not specified.
        raw_w = {n: self._weights.get(n, 1.0) for n in names}
        total_w = sum(raw_w.values())
        if total_w < 1e-12:
            total_w = 1.0
        norm_w = {n: raw_w[n] / total_w for n in names}

        dim = next(iter(grads.values())).shape[0]
        direction = np.zeros(dim, dtype=float)
        for n in names:
            direction += norm_w[n] * grads[n].ravel()

        self._last_diag = self._compute_common_diagnostics(grads, direction)
        for n in names:
            self._last_diag[f"mo_ws_weight_{n}"] = norm_w[n]
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


# ======================================================================
# 2. PCGradHandler
# ======================================================================

class PCGradHandler(MultiObjectiveGradientHandler):
    """PCGrad: project away conflicting components (Yu et al. 2020).

    Parameters
    ----------
    normalize : bool, default False
        If True, L2-normalize each per-objective gradient to unit length
        before the pairwise conflict projection, then rescale the summed
        direction by the mean of the original norms. This avoids a large
        objective (e.g. decision regret under SPSA, whose scale can dwarf
        prediction / fairness gradients by 3+ orders of magnitude)
        dominating the projection geometry while preserving an
        objective-scale step size.
    variant : str, default "deterministic"
        "deterministic": fixed sorted task order, conflict test on the
        ORIGINAL gradient (a deprecated legacy variant, not used by the
        paper). "original": faithful Yu et al. 2020 Algorithm 1 --
        the variant used by all reported PCGrad results --
        fresh random task order per step and conflict test on the RUNNING
        projected gradient. For a faithful original run also set
        normalize=False.
    rng_seed : int | None
        Seed for the shuffle RNG of the "original" variant (ignored by
        "deterministic"). Pass a per-(seed, stage) value for reproducibility.
    """

    def __init__(self, normalize: bool = False, variant: str = "deterministic",
                 rng_seed: int | None = None) -> None:
        self._last_diag: Dict[str, float] = {}
        self._normalize = bool(normalize)
        self._variant = str(variant).strip().lower()
        if self._variant not in {"deterministic", "original"}:
            raise ValueError(f"unknown PCGrad variant: {variant!r}")
        self._rng = np.random.default_rng(rng_seed)

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        m = len(names)
        flat_grads = {n: grads[n].ravel().copy() for n in names}

        orig_norms: Dict[str, float] = {n: float(np.linalg.norm(flat_grads[n])) for n in names}
        if self._normalize:
            for n in names:
                gn = orig_norms[n]
                if gn > 1e-12:
                    flat_grads[n] = flat_grads[n] / gn

        n_projections = 0
        n_pairs = 0

        projected = {}
        if self._variant == "original":
            # Faithful Yu et al. 2020 Algorithm 1: for each g_i, traverse the
            # other tasks in a fresh RANDOM order and test conflict on the
            # RUNNING projected gradient.
            for ni in names:
                gi_running = flat_grads[ni].copy()
                others = [nj for nj in names if nj != ni]
                for idx in self._rng.permutation(len(others)):
                    nj = others[int(idx)]
                    gj = flat_grads[nj]
                    n_pairs += 1
                    dot_ij = float(np.dot(gi_running, gj))
                    if dot_ij < 0.0:
                        dot_jj = float(np.dot(gj, gj)) + 1e-12
                        gi_running = gi_running - (dot_ij / dot_jj) * gj
                        n_projections += 1
                projected[ni] = gi_running
        else:
            # Deterministic variant (all paper runs up to 2026-07): fixed
            # sorted order; the conflict check uses the ORIGINAL g_i (NOT the
            # running-projected state as in Yu et al. 2020 Algorithm 1), while
            # the projection is applied to the running state.  The original-
            # gradient check makes the projection decisions independent of
            # earlier projections within the same i-loop.
            for i, ni in enumerate(names):
                gi_orig = flat_grads[ni]          # original gradient, read-only
                gi_running = flat_grads[ni].copy()  # running state for sequential projection
                for j, nj in enumerate(names):
                    if i == j:
                        continue
                    n_pairs += 1
                    gj = flat_grads[nj]
                    # Conflict check on ORIGINAL g_i (not gi_running).
                    cos_ij = cosine(gi_orig, gj)
                    if cos_ij < 0.0:
                        # Project the running state onto the normal plane of g_j.
                        dot_ij = float(np.dot(gi_running, gj))
                        dot_jj = float(np.dot(gj, gj)) + 1e-12
                        gi_running = gi_running - (dot_ij / dot_jj) * gj
                        n_projections += 1
                projected[ni] = gi_running

        # Sum projected gradients.
        dim = next(iter(grads.values())).shape[0]
        direction = np.zeros(dim, dtype=float)
        for n in names:
            direction += projected[n]

        if self._normalize:
            # Restore an objective-scale step size: mean of original norms.
            mean_norm = float(np.mean([orig_norms[n] for n in names]))
            direction = direction * mean_norm

        conflict_fraction = float(n_projections) / max(n_pairs, 1)

        self._last_diag = self._compute_common_diagnostics(grads, direction)
        self._last_diag["mo_pcgrad_n_projections"] = float(n_projections)
        self._last_diag["mo_pcgrad_conflict_fraction"] = conflict_fraction
        self._last_diag["mo_pcgrad_normalize"] = float(self._normalize)
        self._last_diag["mo_pcgrad_variant_original"] = float(self._variant == "original")
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


class MGDAHandler(MultiObjectiveGradientHandler):
    """MGDA: minimum-norm point in convex hull via SLSQP."""

    def __init__(self) -> None:
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        m = len(names)
        G = np.stack([grads[n].ravel() for n in names], axis=0)  # (m, d)

        # Gram matrix for the QP: min_{lambda} lambda^T M lambda
        # where M_ij = g_i . g_j
        M = G @ G.T  # (m, m)

        lambdas = _solve_mgda_qp(M, m)

        direction = (lambdas @ G).ravel()

        self._last_diag = self._compute_common_diagnostics(grads, direction)
        for i, n in enumerate(names):
            self._last_diag[f"mo_mgda_lambda_{n}"] = float(lambdas[i])
        self._last_diag["mo_mgda_min_norm"] = l2_norm(direction)
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


def _solve_mgda_qp(M: np.ndarray, m: int) -> np.ndarray:
    """Solve min_{lambda in simplex} lambda^T M lambda via SLSQP.

    Falls back to equal weights if optimization fails.
    """
    def objective(lam: np.ndarray) -> float:
        return float(lam @ M @ lam)

    def grad_objective(lam: np.ndarray) -> np.ndarray:
        return 2.0 * (M @ lam)

    x0 = np.ones(m) / m
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    bounds = [(0.0, 1.0)] * m

    try:
        result = minimize(
            objective,
            x0,
            jac=grad_objective,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if result.success:
            lam = np.clip(result.x, 0.0, 1.0)
            lam /= lam.sum() + 1e-12
            return lam
    except Exception:
        pass

    # Fallback: equal weights.
    return np.ones(m) / m


class PLGHandler3Obj(MultiObjectiveGradientHandler):
    """Prediction-Loss-Guided 3-objective handler for DFL.

    Step 1: MGDA on primary objectives (decision_regret, pred_fairness) -> d_primary
    Step 2: Add orthogonal guiding component from pred_loss gradient.
    Step 3: Decay kappa over training steps.
    Falls back to full 3-obj MGDA if ||d_primary|| < epsilon.
    """

    def __init__(
        self,
        kappa_0: float = 1.0,
        kappa_decay: float = 0.01,
        primary_objectives: Tuple[str, ...] = ("decision_regret", "pred_fairness"),
        guiding_objectives: Tuple[str, ...] = ("pred_loss",),
    ) -> None:
        self._kappa_0 = kappa_0
        self._kappa_decay = kappa_decay
        self._primary = primary_objectives
        self._guiding = guiding_objectives
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        kappa_t = self._kappa_0 / (1.0 + self._kappa_decay * step)
        fallback_used = 0.0

        # Sentinel values — overwritten on the relevant code paths.
        dim = next(iter(grads.values())).shape[0]
        d_primary: np.ndarray = np.zeros(dim, dtype=float)
        d_primary_norm: float = 0.0
        guiding_component_norm: float = 0.0

        # Identify primary and guiding gradients.
        primary_names = [n for n in self._primary if n in grads]
        guiding_names = [n for n in self._guiding if n in grads]

        if len(primary_names) < 1:
            # No primary objectives found — fall back to equal-weight sum.
            direction = np.zeros(dim, dtype=float)
            for n in names:
                direction += grads[n].ravel()
            direction /= max(len(names), 1)
            fallback_used = 1.0
        else:
            # Step 1: MGDA on primary objectives.
            m_p = len(primary_names)
            G_primary = np.stack([grads[n].ravel() for n in primary_names], axis=0)
            M_primary = G_primary @ G_primary.T
            lambdas_primary = _solve_mgda_qp(M_primary, m_p)
            d_primary = (lambdas_primary @ G_primary).ravel()
            d_primary_norm = l2_norm(d_primary)

            if d_primary_norm < epsilon:
                # Fallback: full m-objective MGDA over all gradients.
                m_all = len(names)
                G_all = np.stack([grads[n].ravel() for n in names], axis=0)
                M_all = G_all @ G_all.T
                lambdas_all = _solve_mgda_qp(M_all, m_all)
                direction = (lambdas_all @ G_all).ravel()
                fallback_used = 1.0
            else:
                # Step 2: Add orthogonal guiding component.
                if guiding_names:
                    g_guide = np.zeros(dim, dtype=float)
                    for gn in guiding_names:
                        g_guide += grads[gn].ravel()
                    g_guide /= max(len(guiding_names), 1)

                    # Orthogonal component: g_guide - proj_{d_primary}(g_guide)
                    g_orth = project_orthogonal(g_guide, d_primary)
                    guiding_component_norm = l2_norm(kappa_t * g_orth)
                    direction = d_primary + kappa_t * g_orth
                else:
                    direction = d_primary

        self._last_diag = self._compute_common_diagnostics(grads, direction)
        self._last_diag["mo_plg_kappa_t"] = kappa_t
        self._last_diag["mo_plg_primary_norm"] = d_primary_norm
        self._last_diag["mo_plg_guiding_component_norm"] = guiding_component_norm
        self._last_diag["mo_plg_fallback_used"] = fallback_used
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    x_shifted = x - np.max(x)
    e = np.exp(x_shifted)
    return e / (e.sum() + 1e-12)


def _softmax_jacobian(z: np.ndarray) -> np.ndarray:
    """Jacobian of softmax: J_ij = z_i * (delta_ij - z_j).

    Parameters
    ----------
    z : (k,) softmax output

    Returns
    -------
    J : (k, k) Jacobian matrix
    """
    return np.diag(z) - np.outer(z, z)


class NashMTLHandler(MultiObjectiveGradientHandler):
    """Nash-MTL: Multi-Task Learning as a Bargaining Game (Navon et al. NeurIPS 2022).

    At each step, finds per-objective weights w* that satisfy the Nash
    bargaining solution by maximising the product of per-objective gradient
    improvements.  Concretely, solves:

        max_{w in Delta_K}  sum_i log( (G w)_i )

    where G[i,j] = g_i . g_j is the Gram matrix of per-objective gradients,
    via Frank-Wolfe iterations.  The update direction is then:

        d = sum_i w*_i * g_i

    Lambda/mu context (from ``set_step_context``) is used to pre-scale the
    fairness and prediction gradients before the Nash bargaining, so the
    user's Pareto-sweep preferences are respected while Nash further adjusts
    weights to ensure simultaneous progress on all objectives.

    Parameters
    ----------
    n_iters : int
        Frank-Wolfe iterations per step (20 is the paper default).
    normalize : bool
        If True, L2-normalise each (context-scaled) gradient before computing
        the Gram matrix and rescale the output direction by the mean norm.
    eps : float
        Numerical floor inside the Frank-Wolfe log / division.
    """

    _DEC = "decision_regret"
    _PRED = "pred_loss"
    _FAIR = "pred_fairness"

    def __init__(
        self,
        n_iters: int = 20,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        self._n_iters = int(n_iters)
        self._normalize = bool(normalize)
        self._eps = float(eps)
        self._mu_context: float = 1.0
        self._lam_context: float = 1.0
        self._last_diag: Dict[str, float] = {}

    def set_step_context(self, *, mu: float = 1.0, lam: float = 1.0) -> None:
        """Receive the current prediction/fairness weights for this step.

        Called by the training loop before ``compute_direction`` so that Nash
        bargaining operates on lambda-scaled gradients, making the fairness
        Pareto sweep effective.
        """
        self._mu_context = float(mu)
        self._lam_context = float(lam)

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        m = len(names)

        # Pre-scale by context weights so lambda / mu preferences are honoured.
        context_scale = {
            self._DEC: 1.0,
            self._PRED: max(self._mu_context, 0.0),
            self._FAIR: max(self._lam_context, 0.0),
        }
        scaled: Dict[str, np.ndarray] = {
            n: grads[n].ravel() * context_scale.get(n, 1.0) for n in names
        }

        flat = np.stack([scaled[n] for n in names], axis=0)  # (m, d)
        orig_norms = np.array([float(np.linalg.norm(flat[i])) for i in range(m)])

        if self._normalize:
            normed = np.zeros_like(flat)
            for i in range(m):
                if orig_norms[i] > self._eps:
                    normed[i] = flat[i] / orig_norms[i]
            G = normed @ normed.T  # cosine-similarity Gram matrix
            w = _nash_frank_wolfe(G, n_iters=self._n_iters, eps=self._eps)
            # Direction: weighted sum of NORMALIZED grads, rescaled to objective norms.
            direction = (w @ normed).ravel()
            mean_norm = float(np.mean(orig_norms))
            if mean_norm > self._eps:
                direction = direction * mean_norm
        else:
            G = flat @ flat.T  # raw Gram matrix
            w = _nash_frank_wolfe(G, n_iters=self._n_iters, eps=self._eps)
            direction = (w @ flat).ravel()

        self._last_diag = self._compute_common_diagnostics(grads, direction)
        for i, n in enumerate(names):
            self._last_diag[f"mo_nash_w_{n}"] = float(w[i])
        self._last_diag["mo_nash_w_entropy"] = float(
            -np.sum(w * np.log(np.maximum(w, self._eps)))
        )
        self._last_diag["mo_nash_normalize"] = float(self._normalize)
        self._last_diag["mo_nash_mu"] = float(self._mu_context)
        self._last_diag["mo_nash_lambda"] = float(self._lam_context)
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


