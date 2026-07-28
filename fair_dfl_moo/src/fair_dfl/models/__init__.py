"""Unified predictor model system.

Public API:
    build_predictor   - Create a PredictorHandle from config dict
    register_predictor - Register a custom nn.Module class
    PredictorHandle   - Wrapper around nn.Module with post-processing
    PostProcessor     - Output activation (softplus, relu, exp, none)
"""

from .registry import PredictorHandle, build_predictor, register_predictor
from .postprocessing import PostProcessor

__all__ = [
    "build_predictor",
    "register_predictor",
    "PredictorHandle",
    "PostProcessor",
]
