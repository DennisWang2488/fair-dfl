"""Unified training loop for all DFL methods.

Handles all method types (FPTO, DFL, FDFL, PLG, FPLG, SAA, VarDRO, WDRO),
MOO handlers (PCGrad, MGDA, NashMTL), and pluggable decision
gradient backends (finite-diff, SPSA, SPO+, etc.) through a single
train_single_stage() function.
"""

from __future__ import annotations

import copy
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..algorithms.mo_handler import (
    MultiObjectiveGradientHandler,
    WeightedSumHandler,
    PCGradHandler,
    MGDAHandler,
    PLGHandler3Obj,
    NashMTLHandler,
)
from ..algorithms.torch_utils import (
    backward_param_grad_from_output_grad,
    flatten_param_grads,
    merge_guided_dec_pred_gradient,
    parameter_l2_norm,
    resolve_device_or_warn,
    to_torch,
)
from ..decision import build_decision_gradient, DecisionGradientComputer
from ..metrics import cosine, l2_norm, project_orthogonal
from ..models import build_predictor, PredictorHandle
from ..models.registry import _resolve_model_config
from ..schedules import alpha_value, lr_value
from ..tasks.base import BaseTask, SplitData, TaskData
from ..tasks.md_knapsack import MultiDimKnapsackTask
from ..tasks.medical_resource_allocation import MedicalResourceAllocationTask
from ..tasks.single_resource_alpha_fair import SingleResourceAlphaFairTask

from .eval import eval_split_medical, evaluate_model

# Tasks that share HC's compute_batch / _splits interface.
_MED_LIKE = (MedicalResourceAllocationTask, SingleResourceAlphaFairTask)
from .method_spec import MethodSpec


# ======================================================================
# Helpers
# ======================================================================

def _safe_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _metric_or_nan(metrics: Dict[str, float], key: str) -> float:
    return float(metrics.get(key, float("nan")))


def _sample_batch(
    split: SplitData, batch_size: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = split.x.shape[0]
    if batch_size <= 0 or batch_size >= n:
        return split.x, split.y
    idx = rng.choice(n, size=batch_size, replace=False)
    return split.x[idx], split.y[idx]


def _pred_weight(mode: str, t: int, alpha_schedule_cfg: Dict[str, Any]) -> float:
    if mode == "zero":
        return 0.0
    if mode == "fixed1":
        return 1.0
    if mode == "schedule":
        return alpha_value(t=t, schedule_cfg=alpha_schedule_cfg)
    # Any numeric string is treated as a fixed prediction weight (mu).
    try:
        return float(mode)
    except (TypeError, ValueError):
        raise ValueError(f"Unknown pred_weight_mode: {mode}")


def _active_spec(base_spec: MethodSpec, iter_idx: int, warmstart_steps: int) -> MethodSpec:
    """During warmstart phase, run as FPTO (pred+fair only, no decision)."""
    if iter_idx < warmstart_steps:
        return MethodSpec(
            use_dec=False, use_pred=True, use_fair=base_spec.use_fair,
            pred_weight_mode="fixed1", continuation=base_spec.continuation,
            allow_orthogonalization=False,
        )
    return base_spec


def _method_uses_fpto_warmstart(method_name: str, train_cfg: Dict[str, Any]) -> bool:
    warmstart_methods = {str(x).strip().lower() for x in train_cfg.get("warmstart_methods", ["fair_moo", "moo"])}
    return method_name.lower() in warmstart_methods


def _build_mo_handler(
    train_cfg: Dict[str, Any],
    rng_seed: int | None = None,
) -> MultiObjectiveGradientHandler | None:
    mo_method = train_cfg.get("mo_method", None)
    if mo_method is None:
        return None
    if mo_method == "weighted_sum":
        return WeightedSumHandler(weights=train_cfg.get("mo_weights", {}))
    if mo_method == "pcgrad":
        return PCGradHandler(
            normalize=bool(train_cfg.get("mo_pcgrad_normalize", False)),
            variant=str(train_cfg.get("mo_pcgrad_variant", "deterministic")),
            rng_seed=rng_seed,
        )
    if mo_method == "mgda":
        return MGDAHandler()
    if mo_method == "plg3":
        return PLGHandler3Obj(
            kappa_0=float(train_cfg.get("mo_plg_kappa_0", 1.0)),
            kappa_decay=float(train_cfg.get("mo_plg_kappa_decay", 0.01)),
        )
    if mo_method == "nashmtl":
        return NashMTLHandler(
            n_iters=int(train_cfg.get("mo_nashmtl_n_iters", 20)),
            normalize=bool(train_cfg.get("mo_nashmtl_normalize", True)),
            eps=float(train_cfg.get("mo_nashmtl_eps", 1e-8)),
        )
    raise ValueError(f"Unknown mo_method: {mo_method}")


def _build_active_moo_payload(
    *,
    iter_spec: MethodSpec,
    out: Dict[str, Any],
    g_dec_param: np.ndarray,
    g_pred_param: np.ndarray,
    g_fair_param: np.ndarray,
    mo_handler: MultiObjectiveGradientHandler,
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    grads: Dict[str, np.ndarray] = {}
    losses: Dict[str, float] = {}
    if iter_spec.use_pred:
        grads["pred_loss"] = g_pred_param
        losses["pred_loss"] = float(out["loss_pred"])
    if iter_spec.use_dec:
        grads["decision_regret"] = g_dec_param
        losses["decision_regret"] = float(out["loss_dec"])
    if iter_spec.use_fair:
        grads["pred_fairness"] = g_fair_param
        losses["pred_fairness"] = float(out["loss_fair"])
    return grads, losses


def _combine_prediction_gradients(
    *,
    gradient_merge: str,
    iter_spec: MethodSpec,
    g_dec_pred: np.ndarray,
    g_pred_pred: np.ndarray,
    g_fair_pred: np.ndarray,
    alpha_t: float,
    beta_t: float,
    guided_scale_mode: str,
    guided_norm_floor: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Combine prediction-space gradients into one descent direction.

    When both the decision and prediction gradients are active, the dec/pred
    combination is chosen explicitly by ``gradient_merge``:
      - "guided": geometric-mean renormalized merge, then add the fairness term
        ->  merge(g_dec, g_pred; mu) + lambda*g_fair
      - "raw":    true weighted-sum scalarization
        ->  g_dec + mu*g_pred + lambda*g_fair
    Degenerate (dec-only / pred-only) cases ignore ``gradient_merge``.
    """
    diag: Dict[str, float] = {
        "guided_norm_dec": float("nan"),
        "guided_norm_pred": float("nan"),
        "guided_scale": float("nan"),
        "guided_ratio_pred_over_dec": float("nan"),
        "guided_dir_norm": float("nan"),
    }
    if iter_spec.use_dec and iter_spec.use_pred:
        if gradient_merge == "raw":
            g_raw = g_dec_pred + alpha_t * g_pred_pred + beta_t * g_fair_pred
            return g_raw, diag
        if gradient_merge != "guided":
            raise ValueError(f"Unknown gradient_merge: {gradient_merge!r}")
        g_guided, guided_diag = merge_guided_dec_pred_gradient(
            g_dec=g_dec_pred,
            g_pred=g_pred_pred,
            alpha_t=alpha_t,
            scale_mode=guided_scale_mode,
            norm_floor=guided_norm_floor,
            return_diag=True,
        )
        diag = {
            "guided_norm_dec": float(guided_diag["norm_dec"]),
            "guided_norm_pred": float(guided_diag["norm_pred"]),
            "guided_scale": float(guided_diag["guided_scale"]),
            "guided_ratio_pred_over_dec": float(guided_diag["ratio_pred_over_dec"]),
            "guided_dir_norm": float(guided_diag["dir_norm"]),
        }
        return g_guided + beta_t * g_fair_pred, diag

    if iter_spec.use_dec:
        return g_dec_pred + beta_t * g_fair_pred, diag
    if iter_spec.use_pred:
        return alpha_t * g_pred_pred + beta_t * g_fair_pred, diag
    return beta_t * g_fair_pred, diag


# ======================================================================
# Unified training loop
# ======================================================================

def train_single_stage(
    task: BaseTask,
    data: TaskData,
    predictor: PredictorHandle,
    base_spec: MethodSpec,
    train_cfg: Dict[str, Any],
    lambda_value: float,
    seed: int,
    method_name: str,
    stage_idx: int,
    dec_grad_computer: DecisionGradientComputer | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Train one lambda stage for any method.

    Handles all method types: base (FPTO, DFL, FDFL, PLG, FPLG), SAA, VarDRO, WDRO,
    MOO handlers, and pluggable decision gradient backends (finite-diff, SPSA, SPO+).
    """
    device = predictor.device
    dtype = predictor.dtype
    rng = np.random.default_rng(seed * 10_000 + stage_idx * 113 + 7)

    # --- Training hyperparameters ---
    steps = int(train_cfg["steps_per_lambda"])
    batch_size = int(train_cfg["batch_size"])
    lr0 = float(train_cfg["lr"])
    lr_decay = float(train_cfg.get("lr_decay", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    explode_threshold = float(train_cfg.get("explode_threshold", 1e12))
    fairness_smoothing = float(train_cfg.get("fairness_smoothing", 1e-6))
    log_every = int(train_cfg.get("log_every", 1))
    beta_mode = str(train_cfg.get("beta_mode", "penalty"))
    fair_tau = float(train_cfg.get("fair_tau", 0.0))
    dual_lr = float(train_cfg.get("dual_lr", 0.01))
    warmstart_fraction = float(train_cfg.get("warmstart_fraction", 0.25))

    # --- Optimizer ---
    optimizer_name = str(train_cfg.get("optimizer", "sgd")).strip().lower()
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    momentum = float(train_cfg.get("momentum", 0.9))
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr0, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(predictor.parameters(), lr=lr0, weight_decay=weight_decay)
    elif optimizer_name == "sgd_momentum":
        optimizer = torch.optim.SGD(predictor.parameters(), lr=lr0, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = torch.optim.SGD(predictor.parameters(), lr=lr0, weight_decay=weight_decay)

    # --- LR warmup + schedule shape ---
    lr_warmup_steps = int(train_cfg.get("lr_warmup_steps", 0))
    # "reciprocal" (default, lr0/(1+lr_decay*t)) | "cosine" (to lr_min) | "constant"
    lr_schedule = str(train_cfg.get("lr_schedule", "reciprocal")).strip().lower()
    lr_min = float(train_cfg.get("lr_min", 0.0))

    # --- Orthogonalization ---
    orth_cfg = train_cfg.get("orthogonalization", {})
    orth_enabled = bool(orth_cfg.get("enabled", False)) and base_spec.allow_orthogonalization
    orth_ref = str(orth_cfg.get("reference", "pred"))
    orth_conflict_threshold = float(orth_cfg.get("conflict_threshold", -0.1))

    # --- Warmstart ---
    warmstart_steps = 0
    if _method_uses_fpto_warmstart(method_name=method_name, train_cfg=train_cfg):
        warmstart_steps = int(np.ceil(max(0.0, warmstart_fraction) * float(steps)))
        warmstart_steps = min(max(warmstart_steps, 0), steps)

    # --- Dual lambda ---
    dual_lambda = float(lambda_value if beta_mode == "penalty" else 0.0)

    # --- Decision gradient backend ---
    decision_backend = str(train_cfg.get("decision_grad_backend", "analytic")).strip().lower()

    fd_methods = {str(x).strip() for x in train_cfg.get("decision_grad_fd_methods", []) if str(x).strip()}
    fd_eps = float(train_cfg.get("decision_grad_fd_eps", 1e-3))
    guided_scale_mode = str(train_cfg.get("guided_merge_scale_mode", "geom")).strip().lower()
    guided_norm_floor = float(train_cfg.get("guided_merge_norm_floor", 1e-3))

    # --- MOO handler ---
    mo_handler = _build_mo_handler(train_cfg, rng_seed=seed * 10_000 + stage_idx * 113 + 7)

    # --- Tracking ---
    nan_or_inf_steps = 0
    exploding_steps = 0
    solver_calls_total = 0
    decision_ms_total = 0.0
    cos_dec_pred_list: List[float] = []
    cos_dec_fair_list: List[float] = []
    cos_pred_fair_list: List[float] = []
    norm_combined_list: List[float] = []
    iter_logs: List[Dict[str, Any]] = []

    # --- Early stopping (val-based, per-stage scope) ---
    # When eval_val_every_k_steps > 0 and a val split is present, evaluate val
    # metrics every K steps inside the inner loop. Keep a snapshot of the best
    # (predictor, optimizer, dual_lambda) state as measured by
    # ``early_stop_metric`` (default: val regret). Restore at stage end.
    #
    # Scope is per-lambda-stage: the next stage starts from the restored best
    # state of the previous stage, which is the desired behavior for
    # ``continuation=True`` methods like FPLG.
    eval_val_every_k = int(train_cfg.get("eval_val_every_k_steps", 0))
    early_stop_metric = str(train_cfg.get("early_stop_metric", "val_regret"))
    early_stop_min_steps = int(train_cfg.get("early_stop_min_steps", 0))
    _val_y = None
    if data.val is not None and getattr(data.val, "y", None) is not None:
        _val_y = data.val.y
    val_available = _val_y is not None and (
        _val_y.size if hasattr(_val_y, "size") else len(_val_y)
    ) > 0
    early_stop_enabled = eval_val_every_k > 0 and val_available
    if early_stop_enabled and not isinstance(task, _MED_LIKE):
        raise ValueError(
            "eval_val_every_k_steps currently only supports "
            "MedicalResourceAllocationTask (uses eval_split_medical)."
        )
    best_val_metric = float("inf")
    best_snapshot: Dict[str, Any] | None = None
    best_step = -1

    # --- SAA: skip training, use mean ---
    saa_mean = 0.0
    if method_name == "saa":
        if isinstance(task, _MED_LIKE):
            saa_mean = float(np.mean(task._splits["train"].y))
        elif isinstance(task, MultiDimKnapsackTask):
            saa_mean = float(np.mean(task._splits["train"].y))
        else:
            saa_mean = float(np.mean(data.train.y))
        steps = 0

    # === MAIN TRAINING LOOP ===
    stage_start = perf_counter()
    predictor.train()

    for t in range(steps):
        do_log = bool(t % max(log_every, 1) == 0)
        iter_spec = _active_spec(base_spec=base_spec, iter_idx=t, warmstart_steps=warmstart_steps)

        # --- Sample batch ---
        if isinstance(task, _MED_LIKE):
            batch = task.sample_batch("train", batch_size=batch_size, rng=rng)
            xb_t = to_torch(batch.x, device=device, dtype=dtype)
            raw_out = predictor.module(xb_t).reshape(-1)
            pred_t = predictor.post_processor(raw_out)
            pred_np = pred_t.detach().cpu().numpy().reshape(-1)
            yb = batch.y
            batch_ctx = {"cost": batch.cost, "race": batch.race}
        elif isinstance(task, MultiDimKnapsackTask):
            # MD knapsack: sample_batch also (re)binds the task's cvxpy
            # problem to the active sub-population so compute() solves the
            # right LP. Cost/groups are stored on the task — no batch_ctx.
            batch = task.sample_batch("train", batch_size=batch_size, rng=rng)
            xb_t = to_torch(batch.x, device=device, dtype=dtype)
            pred_t = predictor.module(xb_t)
            if predictor.post_processor.transform != "none":
                pred_t = predictor.post_processor(pred_t)
            pred_np = pred_t.detach().cpu().numpy()
            yb = batch.y
            batch_ctx = {}
        else:
            xb, yb = _sample_batch(data.train, batch_size=batch_size, rng=rng)
            xb_t = to_torch(xb, device=device, dtype=dtype)
            pred_t = predictor.module(xb_t)
            if predictor.post_processor.transform != "none":
                pred_t = predictor.post_processor(pred_t)
            pred_np = pred_t.detach().cpu().numpy()
            batch_ctx = {}

        # --- Compute task losses and analytic gradients ---
        use_fd = bool(
            decision_backend == "finite_diff"
            and iter_spec.use_dec
            and method_name in fd_methods
        )
        need_dec_grads = bool(iter_spec.use_dec and (not use_fd) and dec_grad_computer is None)

        # Skip expensive decision-regret solver calls during training.
        # The regret in compute() is only for logging — decision gradients
        # come from the finite_diff backend separately.  Regret is always
        # computed during the evaluation phase at the end of each stage.
        _skip_regret = True

        if isinstance(task, _MED_LIKE):
            out = task.compute_batch(
                raw_pred=pred_np, true=yb, cost=batch.cost, race=batch.race,
                need_grads=need_dec_grads, fairness_smoothing=fairness_smoothing,
            )
        else:
            compute_kwargs = dict(
                raw_pred=pred_np, true=yb,
                need_grads=need_dec_grads, fairness_smoothing=fairness_smoothing,
            )
            # Pass skip_regret if the task supports it (md_knapsack does)
            if hasattr(task, 'compute') and 'skip_regret' in task.compute.__code__.co_varnames:
                compute_kwargs['skip_regret'] = _skip_regret
            try:
                out = task.compute(**compute_kwargs)
            except ValueError as exc:
                if iter_spec.use_dec and need_dec_grads and "analytic" in str(exc).lower():
                    compute_kwargs['need_grads'] = False
                    out = task.compute(**compute_kwargs)
                    use_fd = True
                else:
                    raise

        # --- Decision gradient ---
        solver_calls_iter = int(out.get("solver_calls", 0))
        decision_ms_iter = float(out.get("decision_ms", 0.0))

        if dec_grad_computer is not None and iter_spec.use_dec:
            # Use DecisionGradientComputer (finite-diff, SPSA, SPO+, etc.)
            dec_result = dec_grad_computer.compute(
                pred=pred_np, true=yb, task=task,
                need_grads=True, fairness_smoothing=fairness_smoothing,
                step=t, task_output=out, **batch_ctx,
            )
            g_dec_pred = dec_result.grad_dec.reshape(pred_np.shape)
            solver_calls_iter += dec_result.solver_calls
            decision_ms_iter += dec_result.decision_ms
        elif use_fd:
            from ..algorithms.finite_diff_grad import _finite_diff_decision_grad
            g_dec_pred, fd_calls, fd_ms = _finite_diff_decision_grad(
                task=task, raw_pred=pred_np, true=yb, eps=fd_eps,
            )
            solver_calls_iter += fd_calls
            decision_ms_iter += fd_ms
        else:
            g_dec_pred = (
                np.asarray(out["grad_dec"], dtype=float).reshape(pred_np.shape)
                if iter_spec.use_dec else np.zeros_like(pred_np)
            )

        g_pred_pred = (
            np.asarray(out["grad_pred"], dtype=float).reshape(pred_np.shape)
            if iter_spec.use_pred else np.zeros_like(pred_np)
        )
        g_fair_pred = (
            np.asarray(out["grad_fair"], dtype=float).reshape(pred_np.shape)
            if iter_spec.use_fair else np.zeros_like(pred_np)
        )

        # --- VarDRO: f-divergence variance-regularized prediction gradient ---
        # Implements E[loss] + eps * std(loss), following Duchi & Namkoong (2017).
        if method_name == "var_dro":
            dro_eps = float(train_cfg.get("dro_epsilon", 0.1))
            true_np = yb if not isinstance(task, _MED_LIKE) else batch.y
            per_sample_loss = ((pred_np - true_np) ** 2).reshape(pred_np.shape[0], -1).mean(axis=-1)
            mean_loss = per_sample_loss.mean()
            std_loss = per_sample_loss.std()
            if std_loss > 1e-12:
                dro_weights = 1.0 + dro_eps * (per_sample_loss - mean_loss) / std_loss
                dro_weights = np.maximum(dro_weights, 0.0)
            else:
                dro_weights = np.ones_like(per_sample_loss)
            dro_weights = dro_weights.reshape(-1, *([1] * (g_pred_pred.ndim - 1)))
            g_pred_pred = g_pred_pred * dro_weights
            out["loss_pred"] = float(mean_loss + dro_eps * std_loss)

        # --- WassDRO: Wasserstein DRO via input gradient penalty ---
        # Implements E[loss] + eps * E[||nabla_x loss||], following
        # Gao, Chen & Kleywegt (2024) "Wasserstein DRO and Variation Regularization".
        # The penalty on the Lipschitz constant of the loss w.r.t. inputs
        # provides robustness to Wasserstein perturbations in the data distribution.
        wdro_penalty_val = 0.0
        if method_name == "wdro":
            wdro_eps = float(train_cfg.get("wdro_epsilon", 0.1))
            # Re-forward with input gradients enabled for the penalty term
            xb_wdro = xb_t.detach().clone().requires_grad_(True)
            if isinstance(task, _MED_LIKE):
                raw_wdro = predictor.module(xb_wdro).reshape(-1)
                pred_wdro = predictor.post_processor(raw_wdro)
                yb_wdro = to_torch(batch.y, device=device, dtype=dtype)
                per_sample_mse = (pred_wdro - yb_wdro) ** 2
            else:
                pred_wdro = predictor.module(xb_wdro)
                if predictor.post_processor.transform != "none":
                    pred_wdro = predictor.post_processor(pred_wdro)
                yb_wdro = to_torch(yb, device=device, dtype=dtype)
                per_sample_mse = ((pred_wdro - yb_wdro) ** 2).reshape(
                    pred_wdro.shape[0], -1
                ).mean(dim=-1)
            # Gradient of per-sample losses w.r.t. inputs
            grad_x = torch.autograd.grad(
                per_sample_mse.sum(), xb_wdro, create_graph=True,
            )[0]  # (batch, n_features)
            # Per-sample L2 norm of input gradients (Lipschitz proxy)
            grad_norms = (grad_x ** 2).sum(dim=-1).sqrt()  # (batch,)
            penalty = wdro_eps * grad_norms.mean()
            wdro_penalty_val = float(penalty.item())
            # Backprop penalty to get its parameter gradient contribution
            predictor.module.zero_grad(set_to_none=True)
            penalty.backward()
            wdro_penalty_param_grad = flatten_param_grads(predictor.module)
            out["wdro_grad_penalty"] = wdro_penalty_val
            out["loss_pred"] = float(per_sample_mse.mean().item()) + wdro_penalty_val

        # --- Alpha/beta weights ---
        alpha_t = _pred_weight(iter_spec.pred_weight_mode, t=t, alpha_schedule_cfg=train_cfg["alpha_schedule"])
        if not iter_spec.use_fair:
            beta_t = 0.0
        elif beta_mode == "penalty":
            beta_t = float(lambda_value)
        elif beta_mode == "constraint":
            beta_t = float(dual_lambda)
        else:
            raise ValueError(f"Unknown beta_mode: {beta_mode}")

        # --- Orthogonalization ---
        if orth_enabled and iter_spec.use_fair and mo_handler is None:
            ref_grad = g_pred_pred if orth_ref == "pred" else g_dec_pred
            fair_ref_cos = cosine(g_fair_pred, ref_grad)
            if fair_ref_cos < orth_conflict_threshold:
                g_fair_pred = project_orthogonal(g_fair_pred, ref_grad)

        # --- Combine gradients ---
        g_comb_pred, guided_diag = _combine_prediction_gradients(
            gradient_merge=iter_spec.gradient_merge,
            iter_spec=iter_spec,
            g_dec_pred=g_dec_pred,
            g_pred_pred=g_pred_pred,
            g_fair_pred=g_fair_pred,
            alpha_t=alpha_t,
            beta_t=beta_t,
            guided_scale_mode=guided_scale_mode,
            guided_norm_floor=guided_norm_floor,
        )

        # --- Backprop to parameter space ---
        g_dec_param = backward_param_grad_from_output_grad(
            module=predictor.module, output=pred_t, grad_out=g_dec_pred,
            retain_graph=True, device=device,
        )
        g_pred_param = backward_param_grad_from_output_grad(
            module=predictor.module, output=pred_t, grad_out=g_pred_pred,
            retain_graph=True, device=device,
        )
        g_fair_param = backward_param_grad_from_output_grad(
            module=predictor.module, output=pred_t, grad_out=g_fair_pred,
            retain_graph=True, device=device,
        )

        # --- WassDRO: add gradient penalty contribution to pred param grad ---
        if method_name == "wdro":
            g_pred_param = g_pred_param + wdro_penalty_param_grad

        # --- MOO handler or standard backward ---
        if mo_handler is not None:
            mo_grads, mo_losses_dict = _build_active_moo_payload(
                iter_spec=iter_spec,
                out=out,
                g_dec_param=g_dec_param,
                g_pred_param=g_pred_param,
                g_fair_param=g_fair_param,
                mo_handler=mo_handler,
            )
            if hasattr(mo_handler, "set_step_context"):
                mo_handler.set_step_context(mu=float(alpha_t), lam=float(beta_t))
            g_comb_param = mo_handler.compute_direction(mo_grads, mo_losses_dict, step=t, epsilon=1e-4)

            predictor.module.zero_grad(set_to_none=True)
            offset = 0
            with torch.no_grad():
                for p in predictor.module.parameters():
                    numel = p.numel()
                    p.grad = torch.as_tensor(
                        g_comb_param[offset:offset + numel],
                        dtype=p.dtype, device=p.device,
                    ).reshape(p.shape)
                    offset += numel
        else:
            predictor.module.zero_grad(set_to_none=True)
            if method_name == "wdro":
                # The Wasserstein penalty is a parameter-space regularizer, so
                # use the parameter gradient assembled above instead of only
                # backpropagating the prediction-space MSE gradient.
                g_comb_param = g_pred_param
                offset = 0
                with torch.no_grad():
                    for p in predictor.module.parameters():
                        numel = p.numel()
                        p.grad = torch.as_tensor(
                            g_comb_param[offset:offset + numel],
                            dtype=p.dtype, device=p.device,
                        ).reshape(p.shape)
                        offset += numel
            else:
                pred_t.backward(to_torch(g_comb_pred, device=device, dtype=pred_t.dtype), retain_graph=False)
                g_comb_param = flatten_param_grads(predictor.module)

        # --- Gradient diagnostics ---
        cos_dec_pred = cosine(g_dec_param, g_pred_param)
        cos_dec_fair = cosine(g_dec_param, g_fair_param)
        cos_pred_fair = cosine(g_pred_param, g_fair_param)
        cos_dec_pred_list.append(cos_dec_pred)
        cos_dec_fair_list.append(cos_dec_fair)
        cos_pred_fair_list.append(cos_pred_fair)

        grad_norm = l2_norm(g_comb_param)
        norm_combined_list.append(grad_norm)

        nan_or_inf_flag = bool(
            np.isnan(g_comb_param).any()
            or np.isinf(g_comb_param).any()
            or any(np.isnan(float(out[k])) or np.isinf(float(out[k]))
                   for k in ["loss_dec", "loss_pred", "loss_fair"])
        )

        # --- LR schedule (with optional warmup) ---
        if t < lr_warmup_steps:
            lr_t = lr0 * (t + 1) / lr_warmup_steps  # linear warmup
        elif lr_schedule == "cosine":
            # cosine decay from lr0 to lr_min over the post-warmup steps
            prog = (t - lr_warmup_steps) / max(1, steps - lr_warmup_steps)
            lr_t = lr_min + 0.5 * (lr0 - lr_min) * (1.0 + np.cos(np.pi * min(prog, 1.0)))
        elif lr_schedule == "constant":
            lr_t = lr0
        else:  # "reciprocal" (default): lr0 / (1 + lr_decay * t)
            lr_t = lr_value(t=t - lr_warmup_steps, lr=lr0, lr_decay=lr_decay)
        for group in optimizer.param_groups:
            group["lr"] = lr_t

        # --- Step or skip ---
        if nan_or_inf_flag:
            nan_or_inf_steps += 1
            predictor.module.zero_grad(set_to_none=True)
            delta_theta_l2 = 0.0
        else:
            params_before = None
            if do_log:
                with torch.no_grad():
                    params_before = [p.detach().clone() for p in predictor.module.parameters()]

            if grad_norm > explode_threshold:
                exploding_steps += 1
            if grad_clip_norm > 0.0 and grad_norm > grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(predictor.module.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            # Dual lambda update
            if beta_mode == "constraint" and iter_spec.use_fair:
                dual_lambda = max(0.0, dual_lambda + dual_lr * (float(out["loss_fair"]) - fair_tau))

            delta_theta_l2 = 0.0
            if params_before is not None:
                with torch.no_grad():
                    total = sum(
                        float(((p.detach() - pb).reshape(-1) ** 2).sum().item())
                        for p, pb in zip(predictor.module.parameters(), params_before)
                    )
                    delta_theta_l2 = float(np.sqrt(total))

        solver_calls_total += solver_calls_iter
        decision_ms_total += decision_ms_iter

        # --- Logging ---
        if do_log:
            iter_row: Dict[str, Any] = {
                "task": task.name,
                "method": method_name,
                "seed": seed,
                "stage_idx": stage_idx,
                "lambda": lambda_value,
                "iter": t,
                "stage_type": "train",
                "alpha_t": alpha_t,
                "beta_t": beta_t,
                "lr_t": lr_t,
                "loss_dec": float(out["loss_dec"]),
                "loss_pred": float(out["loss_pred"]),
                "loss_fair": float(out["loss_fair"]),
                "grad_norm_dec": l2_norm(g_dec_param),
                "grad_norm_pred": l2_norm(g_pred_param),
                "grad_norm_fair": l2_norm(g_fair_param),
                "grad_norm_combined": grad_norm,
                "delta_theta_l2": float(delta_theta_l2),
                "cos_dec_pred": cos_dec_pred,
                "cos_dec_fair": cos_dec_fair,
                "cos_pred_fair": cos_pred_fair,
                "solver_calls": solver_calls_iter,
                "decision_ms": decision_ms_iter,
                "nan_or_inf_flag": int(nan_or_inf_flag),
                "nan_or_inf_steps_so_far": nan_or_inf_steps,
                "exploding_steps_so_far": exploding_steps,
                "device": str(device),
            }
            if mo_handler is None:
                iter_row.update({
                    "guided_norm_dec": guided_diag["guided_norm_dec"],
                    "guided_norm_pred": guided_diag["guided_norm_pred"],
                    "guided_scale": guided_diag["guided_scale"],
                    "guided_ratio_pred_over_dec": guided_diag["guided_ratio_pred_over_dec"],
                    "guided_dir_norm": guided_diag["guided_dir_norm"],
                })
            for key in ["loss_dec_normalized", "loss_dec_normalized_true", "loss_dec_normalized_pred_obj"]:
                if key in out:
                    iter_row[key] = float(out[key])
            if mo_handler is not None:
                iter_row.update(mo_handler.extra_logs())
                iter_row["mo_method"] = str(train_cfg.get("mo_method", ""))
            iter_logs.append(iter_row)

        # --- Early-stop val check (optional, per K steps) ---
        # Evaluates val metrics every ``eval_val_every_k`` steps and tracks
        # the best snapshot. Gated on (val split present) AND (t+1 past
        # ``early_stop_min_steps``). Appends a ``stage_type="val_check"`` row
        # to iter_logs for post-hoc diagnostics.
        if (
            early_stop_enabled
            and ((t + 1) % eval_val_every_k == 0)
            and ((t + 1) >= early_stop_min_steps)
        ):
            predictor.module.eval()
            with torch.no_grad():
                val_check = eval_split_medical(
                    task=task,
                    predictor=predictor,
                    split_name="val",
                    fairness_smoothing=fairness_smoothing,
                )
            predictor.module.train()

            # Selection metric: val_regret | val_regret_normalized |
            # val_fairness | val_loss_total (regret + lam * fair)
            esm = early_stop_metric.lower().strip()
            if esm in ("val_regret", "regret"):
                current = float(val_check.get("regret", float("inf")))
            elif esm in ("val_regret_normalized", "regret_normalized"):
                current = float(val_check.get("regret_normalized", float("inf")))
            elif esm in ("val_fairness", "fairness"):
                current = float(val_check.get("fairness", float("inf")))
            elif esm in ("val_loss_total", "val_total", "total"):
                current = float(val_check.get("regret", 0.0)) + float(
                    lambda_value
                ) * float(val_check.get("fairness", 0.0))
            else:
                current = float(val_check.get("regret", float("inf")))

            is_best = current < best_val_metric
            if is_best:
                best_val_metric = current
                best_step = t + 1
                best_snapshot = {
                    "predictor": copy.deepcopy(predictor.module.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "dual_lambda": float(dual_lambda),
                }

            iter_logs.append(
                {
                    "task": task.name,
                    "method": method_name,
                    "seed": seed,
                    "stage_idx": stage_idx,
                    "lambda": lambda_value,
                    "iter": t + 1,
                    "stage_type": "val_check",
                    "val_check_regret": float(val_check.get("regret", float("nan"))),
                    "val_check_regret_normalized": float(
                        val_check.get("regret_normalized", float("nan"))
                    ),
                    "val_check_fairness": float(
                        val_check.get("fairness", float("nan"))
                    ),
                    "val_check_pred_mse": float(
                        val_check.get("pred_mse", float("nan"))
                    ),
                    "val_check_metric_used": early_stop_metric,
                    "val_check_selection_value": float(current),
                    "val_check_is_best": int(bool(is_best)),
                    "device": str(device),
                }
            )

    stage_wallclock = float(perf_counter() - stage_start)

    # --- Restore best-val snapshot if early stopping engaged ---
    # Scope: per-stage. Methods with ``continuation=True`` will start the next
    # stage from this restored state, which is the intended behavior (each
    # stage converges to its best val point before passing to the next).
    early_stop_applied = False
    if early_stop_enabled and best_snapshot is not None:
        predictor.module.load_state_dict(best_snapshot["predictor"])
        try:
            optimizer.load_state_dict(best_snapshot["optimizer"])
        except Exception:
            # Stateless SGD load is a no-op; swallow any transient mismatch.
            pass
        dual_lambda = best_snapshot["dual_lambda"]
        early_stop_applied = True

    # === EVALUATION ===
    # Train evaluation runs once per lambda stage (not per iteration) — it
    # involves solver calls for decision regret and is too expensive to do
    # inline. Toggle with train_cfg["eval_train"] (default: True so that
    # stage-level CSVs always carry train_* columns for the Pareto plots).
    eval_train = bool(train_cfg.get("eval_train", True))
    saa_override_val = None
    saa_override_test = None
    saa_override_train = None
    if method_name == "saa":
        if isinstance(task, _MED_LIKE):
            saa_override_val = np.full(task._splits["val"].y.shape[0], saa_mean)
            saa_override_test = np.full(task._splits["test"].y.shape[0], saa_mean)
            if eval_train:
                saa_override_train = np.full(task._splits["train"].y.shape[0], saa_mean)
        elif isinstance(task, MultiDimKnapsackTask):
            saa_override_val = np.full(task._splits["val"].y.shape, saa_mean)
            saa_override_test = np.full(task._splits["test"].y.shape, saa_mean)
            if eval_train:
                saa_override_train = np.full(task._splits["train"].y.shape, saa_mean)
        else:
            saa_override_val = np.full(data.val.y.shape, saa_mean)
            saa_override_test = np.full(data.test.y.shape, saa_mean)
            if eval_train:
                saa_override_train = np.full(data.train.y.shape, saa_mean)

    train_metrics, val_metrics, test_metrics = evaluate_model(
        task=task, predictor=predictor, data=data,
        fairness_smoothing=fairness_smoothing,
        saa_override_val=saa_override_val,
        saa_override_test=saa_override_test,
        saa_override_train=saa_override_train,
        eval_train=eval_train,
    )

    # === BUILD STAGE ROW ===
    grad_min = float(np.min(norm_combined_list)) if norm_combined_list else 0.0
    grad_median = float(np.median(norm_combined_list)) if norm_combined_list else 0.0
    grad_max = float(np.max(norm_combined_list)) if norm_combined_list else 0.0

    stage_row: Dict[str, Any] = {
        "task": task.name,
        "method": method_name,
        "seed": seed,
        "stage_idx": stage_idx,
        "lambda": lambda_value,
        "early_stop_enabled": int(bool(early_stop_enabled)),
        "early_stop_applied": int(bool(early_stop_applied)) if early_stop_enabled else 0,
        "early_stop_step": int(best_step) if (early_stop_enabled and best_step > 0) else int(steps),
        "early_stop_metric": early_stop_metric if early_stop_enabled else "",
        "train_regret": _metric_or_nan(train_metrics, "regret"),
        "train_fairness": _metric_or_nan(train_metrics, "fairness"),
        "train_pred_mse": _metric_or_nan(train_metrics, "pred_mse"),
        "val_regret": _metric_or_nan(val_metrics, "regret"),
        "val_fairness": _metric_or_nan(val_metrics, "fairness"),
        "val_pred_mse": _metric_or_nan(val_metrics, "pred_mse"),
        "test_regret": _metric_or_nan(test_metrics, "regret"),
        "test_fairness": _metric_or_nan(test_metrics, "fairness"),
        "test_pred_mse": _metric_or_nan(test_metrics, "pred_mse"),
        "stage_wallclock_sec": stage_wallclock,
        "solver_calls_train": solver_calls_total,
        "solver_calls_eval": (
            val_metrics.get("solver_calls_eval", 0)
            + test_metrics.get("solver_calls_eval", 0)
            + train_metrics.get("solver_calls_eval", 0)
        ),
        "decision_ms_train": decision_ms_total,
        "decision_ms_eval": (
            val_metrics.get("decision_ms_eval", 0)
            + test_metrics.get("decision_ms_eval", 0)
            + train_metrics.get("decision_ms_eval", 0)
        ),
        "nan_or_inf_steps": nan_or_inf_steps,
        "exploding_steps": exploding_steps,
        "avg_grad_norm_combined": _safe_mean(norm_combined_list),
        "grad_norm_min": grad_min,
        "grad_norm_median": grad_median,
        "grad_norm_max": grad_max,
        "avg_cos_dec_pred": _safe_mean(cos_dec_pred_list),
        "avg_cos_dec_fair": _safe_mean(cos_dec_fair_list),
        "avg_cos_pred_fair": _safe_mean(cos_pred_fair_list),
        "weight_norm": parameter_l2_norm(predictor.module),
        "device": str(device),
    }
    for key, metric in [
        ("regret_normalized", "val_regret_normalized"),
        ("regret_normalized_true", "val_regret_normalized_true"),
        ("regret_normalized_pred_obj", "val_regret_normalized_pred_obj"),
    ]:
        if key in val_metrics or key in test_metrics or key in train_metrics:
            stage_row[metric] = _metric_or_nan(val_metrics, key)
            stage_row[metric.replace("val_", "test_")] = _metric_or_nan(test_metrics, key)
            stage_row[metric.replace("val_", "train_")] = _metric_or_nan(train_metrics, key)

    # Forward decision-level fairness metrics (decision_alloc_gap, etc.) and
    # per-group descriptive statistics (group_*_benefit_mean_r0, etc.) so the
    # stage CSV carries enough info to inspect benefit/cost imbalance effects.
    for prefix, m_dict in [("train_", train_metrics), ("val_", val_metrics), ("test_", test_metrics)]:
        for key, value in m_dict.items():
            if key.startswith("decision_") or key.startswith("group_"):
                stage_row[prefix + key] = value

    return stage_row, iter_logs


# ======================================================================
# Run a full method x seed combination
# ======================================================================

def run_method_seed(
    task: BaseTask,
    data: TaskData,
    train_cfg: Dict[str, Any],
    seed: int,
    method_name: str,
    base_spec: MethodSpec,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run all lambda stages for a single (method, seed) combination."""
    lambdas = [float(x) for x in train_cfg["lambdas"]]
    force_lambda_path = bool(train_cfg.get("force_lambda_path_all_methods", False))
    if (not base_spec.use_fair) and (not force_lambda_path):
        lambdas = [0.0]
    # MOO methods determine objective weighting internally; lambda has no effect
    if train_cfg.get("mo_method") is not None and (not force_lambda_path):
        lambdas = [0.0]

    # Build predictor
    device = resolve_device_or_warn(str(train_cfg.get("device", "cuda")))
    dtype_str = str(train_cfg.get("dtype", "float64")).strip().lower()
    dtype = torch.float32 if dtype_str == "float32" else torch.float64

    model_cfg = _resolve_model_config(train_cfg)
    build_seed = 13_579 + seed * 101 + 1
    decision_backend = str(train_cfg.get("decision_grad_backend", "analytic")).strip().lower()

    # Determine post-transform from task
    post_transform = "none"
    if isinstance(task, _MED_LIKE):
        post_transform = "softplus"

    predictor = build_predictor(
        config=model_cfg,
        input_dim=data.train.x.shape[1],
        output_dim=data.train.y.shape[1] if data.train.y.ndim > 1 else 1,
        seed=build_seed,
        device=device,
        dtype=dtype,
        post_transform=post_transform,
    )
    dec_grad_computer: DecisionGradientComputer | None = None
    if decision_backend not in {"analytic"}:
        dec_grad_computer = build_decision_gradient(train_cfg, task, device)
        warm_start = getattr(dec_grad_computer.strategy, "warm_start", None)
        if callable(warm_start):
            xb_all = to_torch(data.train.x, device=device, dtype=dtype)
            yb_all = to_torch(data.train.y.reshape(data.train.y.shape[0], -1), device=device, dtype=dtype)
            warm_start(predictor.module, xb_all, yb_all)

    initial_state = predictor.state_dict()
    initial_state = {k: v.detach().clone() for k, v in initial_state.items()}

    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []
    cumulative_wallclock = 0.0

    for stage_idx, lam in enumerate(lambdas):
        restart_per_lambda = bool(train_cfg.get("restart_per_lambda", False))
        if stage_idx > 0 and (restart_per_lambda or (not base_spec.continuation)):
            predictor.load_state_dict(initial_state)
            if dec_grad_computer is not None:
                dec_grad_computer.reset()

        stage_row, iter_logs = train_single_stage(
            task=task,
            data=data,
            predictor=predictor,
            base_spec=base_spec,
            train_cfg=train_cfg,
            lambda_value=lam,
            seed=seed,
            method_name=method_name,
            stage_idx=stage_idx,
            dec_grad_computer=dec_grad_computer,
        )
        cumulative_wallclock += float(stage_row["stage_wallclock_sec"])
        stage_row["cumulative_wallclock_sec"] = cumulative_wallclock
        stage_rows.append(stage_row)
        iter_rows.extend(iter_logs)

    return stage_rows, iter_rows


def run_methods(
    task: BaseTask,
    data: TaskData,
    train_cfg: Dict[str, Any],
    method_configs: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run all selected methods across all seeds.

    Args:
        task: The task to train on.
        data: Task data (train/val/test splits).
        train_cfg: Training configuration.
        method_configs: Dict of {method_name: method_config} from configs.py.
    """
    from .method_spec import resolve_method_spec

    seeds = [int(s) for s in train_cfg["seeds"]]
    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []

    for name, method_cfg in method_configs.items():
        spec = resolve_method_spec(method_cfg)

        # Apply method-specific config overrides
        merged_cfg = dict(train_cfg)
        for k, v in method_cfg.items():
            if k not in {"method", "use_dec", "use_pred", "use_fair",
                         "pred_weight_mode", "continuation", "allow_orthogonalization"}:
                merged_cfg[k] = v

        for seed in seeds:
            rows, logs = run_method_seed(
                task=task,
                data=data,
                train_cfg=merged_cfg,
                seed=seed,
                method_name=name.lower(),
                base_spec=spec,
            )
            stage_rows.extend(rows)
            iter_rows.extend(logs)

    return stage_rows, iter_rows
