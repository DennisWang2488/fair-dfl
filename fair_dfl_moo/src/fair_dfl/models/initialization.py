"""Weight initialization strategies for predictor models."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def init_weights(module: nn.Module, mode: str = "default", seed: int = 0) -> None:
    """Apply weight initialization to a module.

    Modes:
        "default"       - PyTorch defaults (no-op).
        "best_practice" - Kaiming He for ReLU/GELU, Xavier for others, zero bias.
        "legacy_core"   - Reproduces the original normal(0, 0.15) init for
                          backward compat with the old core_methods.py linear predictor.
    """
    mode = mode.strip().lower()
    if mode == "default":
        return  # PyTorch defaults are already applied at construction
    if mode == "legacy_core":
        _init_legacy_core(module, seed)
        return
    if mode == "best_practice":
        _init_best_practice(module)
        return
    raise ValueError(f"Unknown init_mode: {mode!r}")


def _init_best_practice(module: nn.Module) -> None:
    """Kaiming He normal for ReLU/GELU layers, Xavier for others."""
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            # Determine the activation that follows this linear layer.
            # Default to Kaiming He (assumes ReLU-family).
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)


def _init_legacy_core(module: nn.Module, seed: int) -> None:
    """Reproduce old core_methods.py normal(0, 0.15) linear init."""
    rng = np.random.default_rng(31_337 + seed * 17)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            in_f, out_f = m.in_features, m.out_features
            w_np = rng.normal(loc=0.0, scale=0.15, size=(in_f + 1, out_f))
            with torch.no_grad():
                m.weight.copy_(torch.as_tensor(w_np[:-1, :].T, dtype=m.weight.dtype))
                m.bias.copy_(torch.as_tensor(w_np[-1, :], dtype=m.bias.dtype))
