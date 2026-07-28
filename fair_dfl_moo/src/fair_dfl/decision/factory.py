"""Factory for building DecisionGradientComputer instances."""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..tasks.base import BaseTask
from .interface import DecisionGradientStrategy, DecisionResult
from .strategies.analytic import AnalyticStrategy
from .strategies.finite_diff import FiniteDiffStrategy


class DecisionGradientComputer:
    """Facade wrapping a DecisionGradientStrategy.

    Provides the primary interface for the training loop to compute
    decision gradients regardless of the backend strategy.
    """

    def __init__(self, strategy: DecisionGradientStrategy) -> None:
        self.strategy = strategy

    def compute(self, pred, true, task, **ctx) -> DecisionResult:
        return self.strategy.compute(pred=pred, true=true, task=task, **ctx)

    def reset(self) -> None:
        self.strategy.reset()

    @property
    def name(self) -> str:
        return self.strategy.name


def build_decision_gradient(
    train_cfg: Dict[str, Any],
    task: BaseTask,
    device: torch.device | None = None,
) -> DecisionGradientComputer:
    """Build a DecisionGradientComputer from training config.

    Reads 'decision_grad_backend' from train_cfg:
        "analytic"    -> AnalyticStrategy (default)
        "finite_diff" -> FiniteDiffStrategy
        "spsa"        -> SPSAStrategy
        "spo_plus"    -> SPOPlusStrategy
        "lodl"        -> LODLStrategy
        "cvxpylayers" -> CvxpyLayersStrategy
        "reduced"     -> ReducedAlphaFairStrategy (MD alpha-fair; exact + fast)
        "autograd"    -> TorchAutogradStrategy
    """
    backend = str(train_cfg.get("decision_grad_backend", "analytic")).strip().lower()

    if backend == "analytic":
        strategy: DecisionGradientStrategy = AnalyticStrategy()

    elif backend == "finite_diff":
        eps = float(train_cfg.get("decision_grad_fd_eps", 1e-3))
        strategy = FiniteDiffStrategy(eps=eps)

    elif backend == "spsa":
        from .strategies.spsa import SPSAStrategy
        eps = float(train_cfg.get("decision_grad_spsa_eps", 5e-3))
        n_dirs = int(train_cfg.get("decision_grad_spsa_n_dirs", 1))
        strategy = SPSAStrategy(eps=eps, n_dirs=n_dirs)

    elif backend == "spo_plus":
        from .strategies.spo_plus import SPOPlusStrategy
        strategy = SPOPlusStrategy()

    elif backend == "lodl":
        from .strategies.lodl import LODLStrategy
        n_probes = int(train_cfg.get("decision_grad_lodl_probes", 8))
        sigma = float(train_cfg.get("decision_grad_lodl_sigma", 5e-3))
        buffer_size = int(train_cfg.get("decision_grad_lodl_buffer", 200))
        ridge = float(train_cfg.get("decision_grad_lodl_ridge", 1e-4))
        strategy = LODLStrategy(
            n_probes_per_step=n_probes,
            sigma=sigma,
            buffer_size=buffer_size,
            ridge=ridge,
        )

    elif backend == "cvxpylayers":
        from .strategies.cvxpylayers import CvxpyLayersStrategy
        solver_args = train_cfg.get("decision_grad_cvxpylayers_solver_args")
        strategy = CvxpyLayersStrategy(solver_args=solver_args)

    elif backend == "reduced":
        # CES-smoothed exact gradient for the corrected (per-individual) MD
        # alpha-fair knapsack: reduced Newton solve + adjoint implicit
        # differentiation. Required under perfect-substitute utilities (the exact
        # decision is non-unique, so a raw conic layer's gradient is ill-defined).
        # Supported alpha in (0, 5]; smoothing temperature tau (default 0.02).
        from .strategies.reduced import ReducedAlphaFairStrategy
        tau = float(train_cfg.get("decision_grad_reduced_tau", 0.02))
        strategy = ReducedAlphaFairStrategy(tau=tau)

    else:
        raise ValueError(
            f"Unknown decision_grad_backend: {backend!r}. Options: analytic, "
            f"finite_diff, spsa, spo_plus, lodl, cvxpylayers, reduced"
        )

    if not strategy.supports_task(task):
        raise ValueError(
            f"Decision gradient strategy {strategy.name!r} does not support "
            f"task {type(task).__name__!r}."
        )

    return DecisionGradientComputer(strategy)
