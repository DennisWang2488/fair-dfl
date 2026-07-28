"""Gradient-space metrics and projections used by method optimization loops."""

import numpy as np


def flatten(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1)


def l2_norm(arr: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum(arr * arr) + eps))


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    fa = flatten(a)
    fb = flatten(b)
    na = l2_norm(fa, eps=eps)
    nb = l2_norm(fb, eps=eps)
    if na <= eps or nb <= eps:
        return 0.0
    return float(np.dot(fa, fb) / (na * nb + eps))


def project_orthogonal(v: np.ndarray, u: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    fu = flatten(u)
    denom = float(np.dot(fu, fu) + eps)
    coef = float(np.dot(flatten(v), fu) / denom)
    return v - coef * u
