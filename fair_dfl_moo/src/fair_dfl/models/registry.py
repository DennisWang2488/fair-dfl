"""Predictor registry, factory, and PredictorHandle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Type

import numpy as np
import torch
from torch import nn

from .architectures import MLP, FTTransformer, Poly2, ResNetTabular
from .initialization import init_weights
from .postprocessing import PostProcessor

# ======================================================================
# Registry
# ======================================================================

_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_predictor(name: str, cls: Type[nn.Module]) -> None:
    """Register a custom nn.Module class as a predictor architecture.

    The class must accept (input_dim: int, output_dim: int, **kwargs) in __init__.
    """
    _REGISTRY[name.lower().strip()] = cls


def list_predictors() -> list[str]:
    """Return sorted list of registered architecture names."""
    return sorted(_REGISTRY.keys())


# Register built-in architectures
register_predictor("linear", nn.Linear)  # nn.Linear(input_dim, output_dim)
register_predictor("poly2", Poly2)
register_predictor("mlp", MLP)
register_predictor("resnet_tabular", ResNetTabular)
register_predictor("ft_transformer", FTTransformer)


# ======================================================================
# Legacy config mapping
# ======================================================================

_LEGACY_FAMILY_MAP: Dict[str, Dict[str, Any]] = {
    "linear": {"arch": "linear"},
    "mlp_2x64_relu": {
        "arch": "mlp",
        "hidden_dim": 64,
        "n_layers": 2,
        "activation": "relu",
    },
    # Keep old name as alias for backward compatibility
    "mlp_2x64_softplus": {
        "arch": "mlp",
        "hidden_dim": 64,
        "n_layers": 2,
        "activation": "relu",
    },
    "mlp": {"arch": "mlp", "hidden_dim": 64, "n_layers": 2},
}


def _resolve_model_config(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve model config from either new 'model' dict or legacy 'predictor_family' string."""
    if "model" in train_cfg and isinstance(train_cfg["model"], dict):
        return dict(train_cfg["model"])
    # Legacy path: convert predictor_family string to config dict
    family = str(train_cfg.get("predictor_family", "linear")).strip().lower()
    if family in _LEGACY_FAMILY_MAP:
        cfg = dict(_LEGACY_FAMILY_MAP[family])
    else:
        # Treat unknown family as arch name directly
        cfg = {"arch": family}
    # Pass through any legacy mlp_* keys
    if "mlp_hidden_dim" in train_cfg:
        cfg.setdefault("hidden_dim", int(train_cfg["mlp_hidden_dim"]))
    if "mlp_layers" in train_cfg:
        cfg.setdefault("n_layers", int(train_cfg["mlp_layers"]))
    return cfg


# ======================================================================
# PredictorHandle
# ======================================================================

@dataclass
class PredictorHandle:
    """Unified wrapper around a predictor nn.Module.

    Provides:
        - predict_numpy(x) for eval without autograd
        - predict_processed(x_tensor) for model + post-processing in one call
        - parameters(), train(), eval() delegation
    """
    module: nn.Module
    arch: str
    device: torch.device
    dtype: torch.dtype
    post_processor: PostProcessor = field(default_factory=PostProcessor)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """Run model + post-processor on numpy input, return numpy output."""
        self.module.eval()
        with torch.no_grad():
            xb = torch.as_tensor(x, dtype=self.dtype, device=self.device)
            raw = self.module(xb)
            out = self.post_processor(raw)
        self.module.train()
        return out.detach().cpu().numpy()

    def predict_raw_numpy(self, x: np.ndarray) -> np.ndarray:
        """Run model only (no post-processing) on numpy input."""
        self.module.eval()
        with torch.no_grad():
            xb = torch.as_tensor(x, dtype=self.dtype, device=self.device)
            raw = self.module(xb)
        self.module.train()
        return raw.detach().cpu().numpy()

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.module.parameters()

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state)


# ======================================================================
# Factory
# ======================================================================

def build_predictor(
    config: Dict[str, Any],
    input_dim: int,
    output_dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    post_transform: str = "none",
    post_eps: float = 1e-6,
) -> PredictorHandle:
    """Build a predictor from a config dict.

    Args:
        config: Model config with at least "arch" key. Additional keys are
            passed as **kwargs to the architecture constructor.
        input_dim: Number of input features.
        output_dim: Number of output dimensions.
        seed: Random seed for reproducibility.
        device: Torch device.
        dtype: Torch dtype (float32 or float64).
        post_transform: Post-processing activation ("none", "softplus", "exp", "relu").
        post_eps: Epsilon for post-processing numerical stability.

    Returns:
        PredictorHandle wrapping the constructed model.
    """
    config = dict(config)
    arch = config.pop("arch", "linear").strip().lower()
    init_mode = config.pop("init_mode", "default")

    if arch not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture: {arch!r}. "
            f"Registered: {sorted(_REGISTRY)}. "
            f"Use register_predictor() to add custom architectures."
        )

    cls = _REGISTRY[arch]

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Construct module
    if arch == "linear":
        # nn.Linear has a different signature: (in_features, out_features)
        module: nn.Module = cls(input_dim, output_dim)
    else:
        module = cls(input_dim=input_dim, output_dim=output_dim, **config)

    # Apply initialization
    init_weights(module, mode=init_mode, seed=seed)

    # Move to device and dtype
    module.to(device=device, dtype=dtype)

    post = PostProcessor(transform=post_transform, eps=post_eps)
    return PredictorHandle(
        module=module,
        arch=arch,
        device=device,
        dtype=dtype,
        post_processor=post,
    )
