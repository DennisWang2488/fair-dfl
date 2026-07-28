"""Shared torch gradient utilities used by core and advanced method trainers."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from torch import nn

from ..metrics import l2_norm


def resolve_device_or_warn(requested_device: str | None) -> torch.device:
    requested = str(requested_device or "cuda").strip()
    lowered = requested.lower()
    if lowered.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            (
                f"CUDA requested via training.device='{requested}' but CUDA is unavailable. "
                "Falling back to CPU."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    try:
        return torch.device(requested)
    except Exception as exc:
        raise ValueError(f"Unsupported torch device: {requested}") from exc


def to_torch(
    x: Any,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def flatten_param_grads(module: nn.Module) -> np.ndarray:
    flat_grads = []
    for p in module.parameters():
        if p.grad is None:
            flat_grads.append(torch.zeros_like(p).reshape(-1))
        else:
            flat_grads.append(p.grad.detach().reshape(-1))
    if not flat_grads:
        return np.zeros(1, dtype=float)
    return torch.cat(flat_grads).detach().cpu().numpy()


def parameter_l2_norm(module: nn.Module, eps: float = 1e-12) -> float:
    total = 0.0
    with torch.no_grad():
        for p in module.parameters():
            total += float((p.detach() * p.detach()).sum().item())
    return float(np.sqrt(total + eps))


def backward_param_grad_from_output_grad(
    module: nn.Module,
    output: torch.Tensor,
    grad_out: np.ndarray,
    retain_graph: bool,
    device: torch.device,
) -> np.ndarray:
    module.zero_grad(set_to_none=True)
    grad_arr = np.asarray(grad_out, dtype=float)
    if grad_arr.shape != tuple(output.shape):
        if int(np.prod(grad_arr.shape)) != int(output.numel()):
            raise ValueError(
                f"Gradient shape mismatch. grad_out={grad_arr.shape}, output={tuple(output.shape)}"
            )
        grad_arr = grad_arr.reshape(tuple(output.shape))
    grad_t = to_torch(grad_arr, device=device, dtype=output.dtype)
    output.backward(grad_t, retain_graph=retain_graph)
    return flatten_param_grads(module)


def merge_guided_dec_pred_gradient(
    g_dec: np.ndarray,
    g_pred: np.ndarray,
    alpha_t: float,
    scale_mode: str = "geom",
    norm_floor: float = 1e-3,
    eps: float = 1e-12,
    return_diag: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    dec = np.asarray(g_dec, dtype=float)
    pred = np.asarray(g_pred, dtype=float)
    dec_norm = l2_norm(dec, eps=eps)
    pred_norm = l2_norm(pred, eps=eps)
    ratio_pred_over_dec = float(pred_norm / max(dec_norm, eps))

    if dec_norm <= eps and pred_norm <= eps:
        out = np.zeros_like(dec)
        if return_diag:
            return out, {
                "norm_dec": float(dec_norm),
                "norm_pred": float(pred_norm),
                "guided_scale": 0.0,
                "ratio_pred_over_dec": ratio_pred_over_dec,
                "dir_norm": 0.0,
            }
        return out
    if dec_norm <= eps:
        out = float(alpha_t) * pred
        if return_diag:
            return out, {
                "norm_dec": float(dec_norm),
                "norm_pred": float(pred_norm),
                "guided_scale": float(abs(alpha_t) * pred_norm),
                "ratio_pred_over_dec": ratio_pred_over_dec,
                "dir_norm": float(pred_norm),
            }
        return out
    if pred_norm <= eps:
        out = dec
        if return_diag:
            return out, {
                "norm_dec": float(dec_norm),
                "norm_pred": float(pred_norm),
                "guided_scale": float(dec_norm),
                "ratio_pred_over_dec": ratio_pred_over_dec,
                "dir_norm": float(dec_norm),
            }
        return out

    u_dec = dec / max(dec_norm, eps)
    u_pred = pred / max(pred_norm, eps)
    direction = float(alpha_t) * u_pred + u_dec
    direction_norm = l2_norm(direction, eps=eps)
    if direction_norm <= eps:
        out = np.zeros_like(dec)
        if return_diag:
            return out, {
                "norm_dec": float(dec_norm),
                "norm_pred": float(pred_norm),
                "guided_scale": 0.0,
                "ratio_pred_over_dec": ratio_pred_over_dec,
                "dir_norm": 0.0,
            }
        return out

    mode = str(scale_mode).strip().lower()
    if mode == "geom":
        scale = float(np.sqrt(max(dec_norm * pred_norm, 0.0)))
    elif mode == "dec":
        scale = float(dec_norm)
    elif mode == "sum":
        scale = float(dec_norm + float(alpha_t) * pred_norm)
    elif mode == "geom_clip":
        dec_clip = max(dec_norm, float(norm_floor))
        pred_clip = max(pred_norm, float(norm_floor))
        scale = float(np.sqrt(dec_clip * pred_clip))
    else:
        raise ValueError(f"Unknown guided merge scale_mode: {scale_mode}")

    out = scale * (direction / max(direction_norm, eps))
    if return_diag:
        return out, {
            "norm_dec": float(dec_norm),
            "norm_pred": float(pred_norm),
            "guided_scale": float(scale),
            "ratio_pred_over_dec": ratio_pred_over_dec,
            "dir_norm": float(direction_norm),
        }
    return out
