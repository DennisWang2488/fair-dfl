"""Core abstractions for decision gradient computation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from ..tasks.base import BaseTask


@dataclass
class DecisionResult:
    """Result of decision gradient computation.

    Contains the decision loss, its gradient w.r.t. predictions,
    and associated metadata for logging.
    """
    loss_dec: float
    grad_dec: np.ndarray
    solver_calls: int = 0
    decision_ms: float = 0.0

    # Full task output dict (includes loss_pred, loss_fair, grad_pred, grad_fair, etc.)
    task_output: Dict[str, Any] = field(default_factory=dict)

    # Strategy-specific diagnostics
    extra: Dict[str, Any] = field(default_factory=dict)


class DecisionGradientStrategy(ABC):
    """Abstract base for decision gradient computation strategies.

    Each strategy implements a different approach to computing
    the gradient of the decision loss w.r.t. model predictions:
    - Analytic: task provides closed-form Jacobian
    - FiniteDiff: numerical perturbation (element-wise)
    - SPSA: simultaneous perturbation stochastic approximation
    - SPO+: convex surrogate for LP problems
    - CvxpyLayers: cvxpylayers backend
    - TorchAutograd: end-to-end differentiable solver
    """

    @abstractmethod
    def compute(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        need_grads: bool = True,
        fairness_smoothing: float = 1e-6,
        **ctx: Any,
    ) -> DecisionResult:
        """Compute decision loss and gradient.

        Args:
            pred: Predicted parameters, shape (batch,) or (batch, dim).
            true: True parameters, same shape as pred.
            task: The optimization task.
            need_grads: Whether to compute grad_dec.
            fairness_smoothing: Smoothing for fairness loss.
            **ctx: Task-specific context (e.g. cost, race for medical).

        Returns:
            DecisionResult with loss and gradient.
        """

    def supports_task(self, task: BaseTask) -> bool:
        """Check if this strategy can handle the given task."""
        return True

    def reset(self) -> None:
        """Reset any internal state (called between lambda stages)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
