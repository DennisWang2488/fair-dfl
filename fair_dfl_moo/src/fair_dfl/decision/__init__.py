"""Decision gradient computation abstraction.

Provides a strategy pattern for computing gradients of decision loss
w.r.t. predicted parameters, supporting multiple differentiation backends.
"""

from .interface import DecisionGradientStrategy, DecisionResult
from .factory import DecisionGradientComputer, build_decision_gradient

__all__ = [
    "DecisionGradientStrategy",
    "DecisionResult",
    "DecisionGradientComputer",
    "build_decision_gradient",
]
