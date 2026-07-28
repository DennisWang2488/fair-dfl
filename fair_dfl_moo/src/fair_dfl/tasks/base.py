"""Task layer for data-driven optimization experiments used by the 8 active methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray


@dataclass
class TaskData:
    train: SplitData
    val: SplitData
    test: SplitData
    groups: np.ndarray
    meta: Dict[str, np.ndarray | float]


class BaseTask:
    name: str
    n_outputs: int

    def generate_data(self, seed: int) -> TaskData:
        raise NotImplementedError

    def compute(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, np.ndarray | float]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Decision gradient interface (for generic finite-diff and other strategies)
    # ------------------------------------------------------------------

    def solve_decision(self, pred: np.ndarray, **ctx: Any) -> np.ndarray:
        """Solve the optimization problem given predictions.

        Args:
            pred: Predicted parameters (single sample or batch).
            **ctx: Task-specific context (costs, constraints, etc.).

        Returns:
            Decision variables, same batch structure as pred.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement solve_decision(). "
            f"Required for generic finite-diff or custom gradient strategies."
        )

    def evaluate_objective(self, decision: np.ndarray, true: np.ndarray, **ctx: Any) -> float:
        """Evaluate the objective function given decisions and true parameters.

        Args:
            decision: Decision variables from solve_decision().
            true: True (oracle) parameters.
            **ctx: Task-specific context.

        Returns:
            Scalar objective value (higher is better for maximization tasks).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement evaluate_objective(). "
            f"Required for generic finite-diff or custom gradient strategies."
        )

    def supported_gradient_strategies(self) -> List[str]:
        """Return list of supported decision gradient strategy names."""
        return ["analytic"]


def add_bias_column(x: np.ndarray) -> np.ndarray:
    """Append a bias column of ones to a 2D feature matrix."""
    if x.ndim != 2:
        raise ValueError("add_bias_column expects a 2D array.")
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.concatenate([x, ones], axis=1)
