"""Output post-processing transforms applied after model forward pass."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PostProcessor:
    """Applies an element-wise activation to raw model output.

    Configured per-task to ensure predictions satisfy domain constraints
    (e.g. positivity for resource allocation).
    """

    def __init__(self, transform: str = "none", eps: float = 1e-6) -> None:
        self.transform = transform.strip().lower()
        self.eps = eps
        if self.transform not in ("none", "softplus", "exp", "relu"):
            raise ValueError(f"Unknown post-transform: {transform!r}")

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        if self.transform == "none":
            return raw
        if self.transform == "softplus":
            return F.softplus(raw) + self.eps
        if self.transform == "exp":
            return torch.exp(raw) + self.eps
        if self.transform == "relu":
            return F.relu(raw) + self.eps
        raise ValueError(f"Unknown post-transform: {self.transform!r}")

    def __repr__(self) -> str:
        return f"PostProcessor(transform={self.transform!r}, eps={self.eps})"
