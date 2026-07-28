"""Built-in predictor architectures for tabular prediction tasks."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn


# ======================================================================
# 1. MLP — configurable multi-layer perceptron
# ======================================================================

class MLP(nn.Module):
    """Configurable MLP for tabular data.

    Config keys:
        hidden_dim  (int)   : width of hidden layers, default 64
        n_layers    (int)   : number of hidden layers, default 2
        activation  (str)   : relu, gelu, silu, default "relu"
        dropout     (float) : dropout rate, default 0.0
        batch_norm  (bool)  : whether to use BatchNorm1d, default False
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs: Any) -> None:
        super().__init__()
        hidden_dim = int(kwargs.get("hidden_dim", 64))
        n_layers = int(kwargs.get("n_layers", 2))
        act_name = str(kwargs.get("activation", "relu")).lower()
        dropout = float(kwargs.get("dropout", 0.0))
        use_bn = bool(kwargs.get("batch_norm", False))

        act_fn = _get_activation(act_name)
        layers: list[nn.Module] = []
        d_in = input_dim
        for _ in range(max(n_layers, 1)):
            layers.append(nn.Linear(d_in, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# 1b. Poly2 — degree-2 polynomial features + linear head
# ======================================================================

class Poly2(nn.Module):
    """Linear model on elementwise polynomial features: [x, x^2, ..., x^degree(, x_i*x_j)].

    A capacity point between ``linear`` and a small MLP. With
    ``interactions=False`` (default) the features are elementwise powers —
    exactly the well-specified class for generators whose signal is
    ``sum_d x**d @ W_d`` (e.g. the MD knapsack benefit with matching poly_degree).

    Config keys:
        degree (int)        : highest elementwise power, default 2 (the
                              historical Poly2 behaviour).
        interactions (bool) : also include pairwise products x_i*x_j (i<j),
                              default False.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs: Any) -> None:
        super().__init__()
        self.degree = int(kwargs.get("degree", 2))
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1, got {self.degree}")
        self.interactions = bool(kwargs.get("interactions", False))
        n_feat = self.degree * input_dim
        if self.interactions:
            n_feat += input_dim * (input_dim - 1) // 2
        self.head = nn.Linear(n_feat, output_dim)
        if self.interactions:
            iu = torch.triu_indices(input_dim, input_dim, offset=1)
            self.register_buffer("_iu", iu, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x ** d for d in range(1, self.degree + 1)]
        if self.interactions:
            feats.append(x[:, self._iu[0]] * x[:, self._iu[1]])
        return self.head(torch.cat(feats, dim=1))


# ======================================================================
# 2. ResNetTabular — residual MLP blocks
# ======================================================================

class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, activation: str) -> None:
        super().__init__()
        act_fn = _get_activation(activation)
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.bn_out = nn.BatchNorm1d(dim)
        self.act_out = act_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_out(self.bn_out(x + self.block(x)))


class ResNetTabular(nn.Module):
    """Residual MLP for tabular data (Gorishniy et al. 2021 style).

    Config keys:
        hidden_dim  (int)   : block width, default 128
        n_blocks    (int)   : number of residual blocks, default 3
        dropout     (float) : dropout rate, default 0.1
        activation  (str)   : relu, gelu, silu, default "relu"
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs: Any) -> None:
        super().__init__()
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        n_blocks = int(kwargs.get("n_blocks", 3))
        dropout = float(kwargs.get("dropout", 0.1))
        activation = str(kwargs.get("activation", "relu")).lower()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.input_act = _get_activation(activation)()

        self.blocks = nn.Sequential(
            *[_ResBlock(hidden_dim, dropout, activation) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_act(self.input_bn(self.input_proj(x)))
        h = self.blocks(h)
        return self.head(h)


# ======================================================================
# 3. FTTransformer — Feature Tokenizer + Transformer
# ======================================================================

class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data (Gorishniy et al. 2021).

    Each input feature is linearly projected to a d_token-dimensional embedding.
    A learnable [CLS] token is prepended, and standard transformer encoder blocks
    process the token sequence. The [CLS] output feeds an MLP head.

    Config keys:
        d_token     (int)   : token embedding dimension, default 64
        n_heads     (int)   : number of attention heads, default 4
        n_layers    (int)   : number of transformer blocks, default 2
        dropout     (float) : dropout rate, default 0.1
        head_hidden (int)   : MLP head hidden dim, default 64
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs: Any) -> None:
        super().__init__()
        d_token = int(kwargs.get("d_token", 64))
        n_heads = int(kwargs.get("n_heads", 4))
        n_layers = int(kwargs.get("n_layers", 2))
        dropout = float(kwargs.get("dropout", 0.1))
        head_hidden = int(kwargs.get("head_hidden", 64))

        # Per-feature linear embeddings: each scalar -> d_token vector
        self.feature_embeddings = nn.Linear(input_dim, input_dim * d_token)
        self.n_features = input_dim
        self.d_token = d_token

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.ln_final = nn.LayerNorm(d_token)

        # MLP head on [CLS] output
        self.head = nn.Sequential(
            nn.Linear(d_token, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Embed each feature: (batch, input_dim) -> (batch, n_features, d_token)
        tokens = self.feature_embeddings(x).view(bsz, self.n_features, self.d_token)
        # Prepend [CLS] token
        cls = self.cls_token.expand(bsz, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, 1+n_features, d_token)
        # Transformer
        tokens = self.transformer(tokens)
        tokens = self.ln_final(tokens)
        # Extract [CLS] output and classify
        cls_out = tokens[:, 0]  # (batch, d_token)
        return self.head(cls_out)


# ======================================================================
# Activation helper
# ======================================================================

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def _get_activation(name: str):
    name = name.strip().lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name!r}. Options: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]
