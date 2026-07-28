"""MethodSpec dataclass and resolution from inline config flags or legacy method names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MethodSpec:
    use_dec: bool
    use_pred: bool
    use_fair: bool
    pred_weight_mode: str
    continuation: bool
    allow_orthogonalization: bool
    # Combination strategy for the use_dec+use_pred case:
    #   "guided" — geometric-mean renormalized merge (historical FDFL-Scal/FPLG behavior)
    #   "raw"    — true weighted sum g_dec + mu*g_pred (the corrected FDFL-Scal scalarization)
    # Methods with an mo_method handler ignore this (they combine in parameter space).
    gradient_merge: str = "guided"


# Legacy mapping from method name -> MethodSpec (kept for backward compat)
_LEGACY_METHOD_SPECS: Dict[str, MethodSpec] = {
    "fair_moo": MethodSpec(
        use_dec=True, use_pred=True, use_fair=True,
        pred_weight_mode="schedule", continuation=True, allow_orthogonalization=True,
    ),
    "grid_restart": MethodSpec(
        use_dec=True, use_pred=True, use_fair=True,
        pred_weight_mode="schedule", continuation=False, allow_orthogonalization=True,
    ),
    "fdfl": MethodSpec(
        use_dec=True, use_pred=False, use_fair=True,
        pred_weight_mode="zero", continuation=False, allow_orthogonalization=False,
    ),
    "moo": MethodSpec(
        use_dec=True, use_pred=True, use_fair=False,
        pred_weight_mode="schedule", continuation=False, allow_orthogonalization=False,
    ),
    "fpto": MethodSpec(
        use_dec=False, use_pred=True, use_fair=True,
        pred_weight_mode="fixed1", continuation=False, allow_orthogonalization=False,
    ),
    "dfl": MethodSpec(
        use_dec=True, use_pred=False, use_fair=False,
        pred_weight_mode="zero", continuation=False, allow_orthogonalization=False,
    ),
    "saa": MethodSpec(
        use_dec=False, use_pred=True, use_fair=False,
        pred_weight_mode="fixed1", continuation=False, allow_orthogonalization=False,
    ),
    "var_dro": MethodSpec(
        use_dec=False, use_pred=True, use_fair=False,
        pred_weight_mode="fixed1", continuation=False, allow_orthogonalization=False,
    ),
    "wdro": MethodSpec(
        use_dec=False, use_pred=True, use_fair=False,
        pred_weight_mode="fixed1", continuation=False, allow_orthogonalization=False,
    ),
}


def resolve_method_spec(method_cfg: Dict[str, Any]) -> MethodSpec:
    """Resolve a MethodSpec from either inline flags or legacy 'method' key.

    Inline format (preferred):
        {"use_dec": True, "use_pred": True, "use_fair": True, ...}

    Legacy format (backward compat):
        {"method": "fair_moo", ...}
    """
    if "use_dec" in method_cfg:
        return MethodSpec(
            use_dec=bool(method_cfg["use_dec"]),
            use_pred=bool(method_cfg["use_pred"]),
            use_fair=bool(method_cfg.get("use_fair", False)),
            pred_weight_mode=str(method_cfg.get("pred_weight_mode", "schedule")),
            continuation=bool(method_cfg.get("continuation", False)),
            allow_orthogonalization=bool(method_cfg.get("allow_orthogonalization", True)),
            gradient_merge=str(method_cfg.get("gradient_merge", "guided")),
        )
    # Legacy fallback
    method_name = str(method_cfg.get("method", "")).strip().lower()
    if method_name in _LEGACY_METHOD_SPECS:
        return _LEGACY_METHOD_SPECS[method_name]
    raise ValueError(f"Cannot resolve MethodSpec: no inline flags and unknown method '{method_name}'")


def resolve_method_backend(method_cfg: Dict[str, Any]) -> str:
    """Return the training backend name for dispatch.

    Returns the 'method' key value (e.g., 'fair_moo', 'fpto', 'dfl', 'saa', 'wdro').
    """
    return str(method_cfg.get("method", "fair_moo")).strip().lower()
