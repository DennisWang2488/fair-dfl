"""SGD-over-instances training loop for the multi-instance Healthcare run.

Implements the §7 protocol of ``new_experiment_design.md``:

    initialize theta ONCE                          # common init across methods
    for step t in 1..steps:
      for mini-batch of B instances:               # batch axis = instances
        for each instance z_k in the batch:        # solve each one fully
          rhat_k = f_theta(X_k)
          d_k    = argmax_d W_alpha(d; rhat_k) s.t. sum c_i d_i <= Q_k
          accumulate per-objective grads g_dec/g_pred/g_fair  (param space)
        g_* <- mean over the B instances
        theta <- step( combine(g_dec, g_pred, g_fair) )   # scalarized OR MOO — UNCHANGED

The per-instance allocation math (closed-form alpha-fair solve, decision-regret
VJP, prediction MSE, fairness loss/grad) is delegated to
``MedicalResourceAllocationTask.compute_batch`` so the numbers are identical to
the published single-cohort code. The gradient combination
(``_combine_prediction_gradients``) and MOO handlers (``_build_mo_handler``) are
reused **unchanged** from ``fair_dfl.training.loop`` — they are agnostic to
whether the gradient came from one cohort or a batch of instances, so the only
new logic here is the instance loop and the per-instance evaluation aggregator.

Key protocol choices (locked by the spec):
- One predictor; common init keyed only by ``seed`` (paired across methods).
- **No warm-start.** Early stopping is OPT-IN (``train_cfg["early_stopping"]``):
  ON for the HP-tuned runs (val-based, restore best checkpoint, section 0.5); OFF
  by default => fixed-steps ERM estimator (the Run-B bound curve).
- ``batch_size`` now counts **instances**, not patients (``<=0`` => full-batch
  over all N_train instances, the default).
"""

from __future__ import annotations

import copy
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from fair_dfl.algorithms.torch_utils import (
    backward_param_grad_from_output_grad,
    flatten_param_grads,
    parameter_l2_norm,
    resolve_device_or_warn,
    to_torch,
)
from fair_dfl.metrics import cosine, l2_norm
from fair_dfl.models import build_predictor
from fair_dfl.models.registry import _resolve_model_config
from fair_dfl.schedules import alpha_value, lr_value
from fair_dfl.tasks.medical_resource_allocation import MedicalResourceAllocationTask
from fair_dfl.training.loop import (
    _build_active_moo_payload,
    _build_mo_handler,
    _combine_prediction_gradients,
    _pred_weight,
)
from fair_dfl.training.method_spec import MethodSpec, resolve_method_spec

from .hc_instances import HCInstance, HCInstanceData


# ======================================================================
# Math engine: one task object reused as a per-instance solver
# ======================================================================

def _make_task(task_cfg: Dict[str, Any]) -> MedicalResourceAllocationTask:
    """A bare task object used only as a per-instance compute engine.

    ``generate_data`` is intentionally NOT called: ``compute_batch`` /
    ``evaluate`` only read ``alpha_fair`` / ``fairness_type`` / ``decision_mode``
    / ``budget`` (which we set per-instance), never ``_splits``.
    """
    task = MedicalResourceAllocationTask(
        data_csv=str(task_cfg.get("data_csv", "data/data_processed.csv")),
        n_sample=int(task_cfg.get("n_sample", 0)),
        data_seed=int(task_cfg.get("data_seed", 42)),
        split_seed=int(task_cfg.get("split_seed", 2)),
        test_fraction=float(task_cfg.get("test_fraction", 0.20)),
        val_fraction=float(task_cfg.get("val_fraction", 0.0)),
        alpha_fair=float(task_cfg.get("alpha_fair", 2.0)),
        budget=float(task_cfg.get("budget", 1.0)),  # overwritten per instance
        decision_mode=str(task_cfg.get("decision_mode", "group")),
        fairness_type=str(task_cfg.get("fairness_type", "mad")),
        budget_rho=float(task_cfg.get("budget_rho", 0.30)),
    )
    return task


# ======================================================================
# Closed-form OLS / log-OLS fit for (log-)linear prediction baselines (§0.6)
# ======================================================================

def _linear_submodule(module: torch.nn.Module) -> "torch.nn.Linear | None":
    """The single ``nn.Linear`` of a (log-)linear predictor, or None if the
    predictor is not closed-form-fittable (e.g. an MLP)."""
    if isinstance(module, torch.nn.Linear):
        return module
    linears = [m for m in module.modules() if isinstance(m, torch.nn.Linear)]
    return linears[0] if len(linears) == 1 else None


def fit_linear_closed_form(predictor, instances: List[HCInstance], *, link: str) -> bool:
    """Fit a (log-)linear predictor to its analytic optimum on the pooled patients.

    The §0.6 simple model. Avoids the SGD-fragility of a linear predictor — the
    "52% false gap" was the FPTO-linear *under-training* artifact (prior doc F2/⚠),
    not a real effect. Because the HC benefit ``y = max(benefit*100,1)+1 >= 2`` is
    strictly positive, the GLM log-link reduces to a single closed-form least-squares
    on ``log y`` (log-OLS); the identity link is plain OLS:

        link="exp"  -> minimize ||X beta - log y||^2 ; predict exp(X beta)  (post="exp")
        link="none" -> minimize ||X beta -     y||^2 ; predict     X beta   (post="none")

    Writes the fitted weights into the ``nn.Linear`` module so the (identical)
    train/eval forward path uses the analytic optimum. Returns False (no-op) if the
    predictor is not a single linear layer, so the caller falls back to SGD.
    """
    lin = _linear_submodule(predictor.module)
    if lin is None or lin.bias is None:
        return False
    X = np.concatenate([inst.x for inst in instances], axis=0)
    y = np.concatenate([inst.y for inst in instances], axis=0)
    target = np.log(y) if link == "exp" else y
    Xa = np.hstack([X, np.ones((X.shape[0], 1), dtype=float)])
    beta, *_ = np.linalg.lstsq(Xa, target, rcond=None)  # features standardized => well-conditioned
    w, b = beta[:-1], float(beta[-1])
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(w, dtype=lin.weight.dtype,
                                         device=lin.weight.device).reshape(lin.weight.shape))
        lin.bias.copy_(torch.as_tensor([b], dtype=lin.bias.dtype,
                                       device=lin.bias.device).reshape(lin.bias.shape))
    return True


# ======================================================================
# Per-instance evaluation aggregator (§7: per-instance, then average)
# ======================================================================

def _se(values: np.ndarray) -> float:
    n = values.size
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


_EVAL_METRICS = ["regret", "regret_normalized", "regret_normalized_pred_obj", "pred_mse", "fairness"]


def evaluate_instances(
    task: MedicalResourceAllocationTask,
    predictor,
    instances: List[HCInstance],
    fairness_smoothing: float,
    saa_mean: float | None = None,
    pred_floor: float | None = None,
) -> Dict[str, float]:
    """Evaluate per test/train instance, then aggregate mean +/- SE (§7).

    ``pred_floor`` (optional): post-hoc clamp of predictions to a lower bound
    (the observed benefit support) before the decision solve. Used for the
    trained-linear capacity rung where the MSE-optimal fit violates support and
    the alpha=2 welfare then starves the worst-off. No-op for in-support methods.

    Returns a flat dict: for each metric ``<m>`` -> ``<m>`` (mean) and
    ``<m>_se`` (standard error over instances), plus ``n_instances``.
    """
    if not instances:
        return {f"{m}": float("nan") for m in _EVAL_METRICS}
    per: Dict[str, List[float]] = {m: [] for m in _EVAL_METRICS}
    for inst in instances:
        task.budget = float(inst.budget)
        if saa_mean is not None:
            pred = np.full(inst.y.shape[0], float(saa_mean), dtype=float)
        else:
            pred = predictor.predict_numpy(inst.x).reshape(-1)
        if pred_floor is not None:
            pred = np.maximum(pred, float(pred_floor))
        out = task.compute_batch(
            raw_pred=pred, true=inst.y, cost=inst.cost, race=inst.race,
            need_grads=False, fairness_smoothing=fairness_smoothing,
        )
        per["regret"].append(float(out["loss_dec"]))
        per["regret_normalized"].append(float(out["loss_dec_normalized"]))
        per["regret_normalized_pred_obj"].append(float(out["loss_dec_normalized_pred_obj"]))
        per["pred_mse"].append(float(out["loss_pred"]))
        per["fairness"].append(float(out["loss_fair"]))
    agg: Dict[str, float] = {"n_instances": float(len(instances))}
    for m in _EVAL_METRICS:
        arr = np.asarray(per[m], dtype=float)
        agg[m] = float(np.mean(arr))
        agg[f"{m}_se"] = _se(arr)
    return agg


# ======================================================================
# Per-instance gradient computation
# ======================================================================

def _instance_objective_grads(
    *,
    task: MedicalResourceAllocationTask,
    predictor,
    inst: HCInstance,
    spec: MethodSpec,
    method_name: str,
    train_cfg: Dict[str, Any],
    fairness_smoothing: float,
    device,
    dtype,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Forward one instance, solve it fully, return batch-mean-ready param grads.

    Returns (losses_dict, g_dec_param, g_pred_param, g_fair_param) where each
    g_*_param is the prediction-space objective gradient back-propagated to
    parameter space for THIS instance (zeros if the objective is inactive).
    """
    task.budget = float(inst.budget)
    xb_t = to_torch(inst.x, device=device, dtype=dtype)
    raw_out = predictor.module(xb_t).reshape(-1)
    pred_t = predictor.post_processor(raw_out)
    pred_np = pred_t.detach().cpu().numpy().reshape(-1)

    need_dec_grads = bool(spec.use_dec)
    out = task.compute_batch(
        raw_pred=pred_np, true=inst.y, cost=inst.cost, race=inst.race,
        need_grads=need_dec_grads, fairness_smoothing=fairness_smoothing,
    )

    g_dec_pred = np.asarray(out["grad_dec"], dtype=float).reshape(-1) if spec.use_dec else None
    g_pred_pred = np.asarray(out["grad_pred"], dtype=float).reshape(-1) if spec.use_pred else None
    g_fair_pred = np.asarray(out["grad_fair"], dtype=float).reshape(-1) if spec.use_fair else None

    # --- VarDRO: variance-regularized prediction gradient (Duchi & Namkoong) ---
    if method_name == "var_dro" and g_pred_pred is not None:
        dro_eps = float(train_cfg.get("dro_epsilon", 0.1))
        per_sample_loss = (pred_np - inst.y) ** 2
        mean_loss = float(per_sample_loss.mean())
        std_loss = float(per_sample_loss.std())
        if std_loss > 1e-12:
            w = 1.0 + dro_eps * (per_sample_loss - mean_loss) / std_loss
            w = np.maximum(w, 0.0)
        else:
            w = np.ones_like(per_sample_loss)
        g_pred_pred = g_pred_pred * w
        out["loss_pred"] = float(mean_loss + dro_eps * std_loss)

    # --- WDRO: Wasserstein DRO via input-gradient penalty (Gao et al. 2024) ---
    wdro_param_grad = None
    if method_name == "wdro":
        wdro_eps = float(train_cfg.get("wdro_epsilon", 0.1))
        xb_wdro = xb_t.detach().clone().requires_grad_(True)
        raw_wdro = predictor.module(xb_wdro).reshape(-1)
        pred_wdro = predictor.post_processor(raw_wdro)
        yb_wdro = to_torch(inst.y, device=device, dtype=dtype)
        per_sample_mse = (pred_wdro - yb_wdro) ** 2
        grad_x = torch.autograd.grad(per_sample_mse.sum(), xb_wdro, create_graph=True)[0]
        grad_norms = (grad_x ** 2).sum(dim=-1).sqrt()
        penalty = wdro_eps * grad_norms.mean()
        predictor.module.zero_grad(set_to_none=True)
        penalty.backward()
        wdro_param_grad = flatten_param_grads(predictor.module)
        out["loss_pred"] = float(per_sample_mse.mean().item()) + float(penalty.item())

    # --- Back-prop each active objective to parameter space ---
    def to_param(g_pred: np.ndarray | None) -> np.ndarray:
        if g_pred is None or not np.any(g_pred):
            return None  # caller treats None as zero
        return backward_param_grad_from_output_grad(
            module=predictor.module, output=pred_t, grad_out=g_pred,
            retain_graph=True, device=device,
        )

    g_dec_param = to_param(g_dec_pred)
    g_pred_param = to_param(g_pred_pred)
    g_fair_param = to_param(g_fair_pred)
    if wdro_param_grad is not None:
        g_pred_param = wdro_param_grad if g_pred_param is None else (g_pred_param + wdro_param_grad)

    losses = {
        "loss_dec": float(out["loss_dec"]),
        "loss_pred": float(out["loss_pred"]),
        "loss_fair": float(out["loss_fair"]),
        "loss_dec_normalized": float(out.get("loss_dec_normalized", 0.0)),
        "loss_dec_normalized_pred_obj": float(out.get("loss_dec_normalized_pred_obj", 0.0)),
    }
    return losses, g_dec_param, g_pred_param, g_fair_param


# ======================================================================
# One lambda stage
# ======================================================================

def train_single_stage_multiinstance(
    *,
    task: MedicalResourceAllocationTask,
    inst_data: HCInstanceData,
    predictor,
    base_spec: MethodSpec,
    train_cfg: Dict[str, Any],
    lambda_value: float,
    seed: int,
    method_name: str,
    stage_idx: int,
    force_zero_steps: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = predictor.device
    dtype = predictor.dtype
    rng = np.random.default_rng(seed * 10_000 + stage_idx * 113 + 7)

    # force_zero_steps: the predictor was fit in closed form (a pure-MSE baseline on
    # a (log-)linear model) and is already at its optimum -> no SGD, just evaluate.
    steps = 0 if force_zero_steps else int(train_cfg["steps_per_lambda"])
    batch_size = int(train_cfg.get("batch_size", -1))  # instances per step; <=0 => full
    lr0 = float(train_cfg["lr"])
    lr_decay = float(train_cfg.get("lr_decay", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    explode_threshold = float(train_cfg.get("explode_threshold", 1e12))
    fairness_smoothing = float(train_cfg.get("fairness_smoothing", 1e-6))
    log_every = int(train_cfg.get("log_every", 5))
    n_instances = len(inst_data.train)

    # Optional post-hoc prediction floor: clamp eval-time predictions to the
    # observed benefit support (min train benefit). Default off; enable via
    # train_cfg["prediction_floor"] ("auto" -> min train benefit, or a float)
    # for the trained-linear capacity rung whose MSE-optimal fit violates
    # support. No-op for in-support methods (e.g. FDFL).
    pred_floor = None
    _pf_cfg = train_cfg.get("prediction_floor", None)
    if _pf_cfg is not None:
        if isinstance(_pf_cfg, str) and _pf_cfg.strip().lower() == "auto":
            pred_floor = float(min(float(inst.y.min()) for inst in inst_data.train))
        else:
            pred_floor = float(_pf_cfg)
    n_param = sum(p.numel() for p in predictor.module.parameters())

    # --- Early stopping on val normalized regret (HP protocol, section 0.5) ---
    # OFF by default => fixed-steps ERM (the Run-B bound curve). When ON, eval the
    # held-out val instances every E steps, keep the best-val checkpoint, stop after
    # `patience` steps without improvement, and restore the best checkpoint before
    # the final test eval (so reported numbers reflect the selected model).
    early_stopping = (
        bool(train_cfg.get("early_stopping", False))
        and len(getattr(inst_data, "val", []) or []) > 0
        and method_name != "saa"
    )
    es_eval_every = max(int(train_cfg.get("early_stop_eval_every", 10)), 1)
    es_patience = int(train_cfg.get("early_stop_patience", 30))
    es_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))
    es_best_val = float("inf")
    es_best_step = -1
    es_best_state = None
    es_history: List[Tuple[int, float]] = []
    es_stopped_early = False

    # --- Optimizer (mirror the single-cohort loop) ---
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
    lr_warmup_steps = int(train_cfg.get("lr_warmup_steps", 0))

    beta_mode = str(train_cfg.get("beta_mode", "penalty"))
    guided_scale_mode = str(train_cfg.get("guided_merge_scale_mode", "geom")).strip().lower()
    guided_norm_floor = float(train_cfg.get("guided_merge_norm_floor", 1e-3))

    mo_handler = _build_mo_handler(train_cfg)

    # --- SAA: no training, predict the train-pool benefit mean ---
    saa_mean = None
    if method_name == "saa":
        saa_mean = float(np.mean(np.concatenate([inst.y for inst in inst_data.train])))
        steps = 0

    # --- Tracking ---
    nan_or_inf_steps = 0
    exploding_steps = 0
    cos_dec_pred_list: List[float] = []
    cos_dec_fair_list: List[float] = []
    cos_pred_fair_list: List[float] = []
    norm_combined_list: List[float] = []
    iter_logs: List[Dict[str, Any]] = []
    diverged_steps = 0

    stage_start = perf_counter()
    predictor.train()

    t = -1
    for t in range(steps):
        do_log = bool(t % max(log_every, 1) == 0)

        # --- Sample a mini-batch of INSTANCES ---
        if batch_size <= 0 or batch_size >= n_instances:
            batch_ids = np.arange(n_instances)
        else:
            batch_ids = rng.choice(n_instances, size=batch_size, replace=False)
        nB = len(batch_ids)

        # --- Accumulate per-objective param grads over the instance batch ---
        g_dec_param = np.zeros(n_param, dtype=float)
        g_pred_param = np.zeros(n_param, dtype=float)
        g_fair_param = np.zeros(n_param, dtype=float)
        sum_loss_dec = sum_loss_pred = sum_loss_fair = 0.0
        sum_dec_norm = sum_dec_norm_pred = 0.0

        for k in batch_ids:
            losses_k, gd, gp, gf = _instance_objective_grads(
                task=task, predictor=predictor, inst=inst_data.train[k],
                spec=base_spec, method_name=method_name, train_cfg=train_cfg,
                fairness_smoothing=fairness_smoothing, device=device, dtype=dtype,
            )
            if gd is not None:
                g_dec_param += gd
            if gp is not None:
                g_pred_param += gp
            if gf is not None:
                g_fair_param += gf
            sum_loss_dec += losses_k["loss_dec"]
            sum_loss_pred += losses_k["loss_pred"]
            sum_loss_fair += losses_k["loss_fair"]
            sum_dec_norm += losses_k["loss_dec_normalized"]
            sum_dec_norm_pred += losses_k["loss_dec_normalized_pred_obj"]

        inv = 1.0 / float(nB)
        g_dec_param *= inv
        g_pred_param *= inv
        g_fair_param *= inv
        out_mean = {
            "loss_dec": sum_loss_dec * inv,
            "loss_pred": sum_loss_pred * inv,
            "loss_fair": sum_loss_fair * inv,
            "loss_dec_normalized": sum_dec_norm * inv,
            "loss_dec_normalized_pred_obj": sum_dec_norm_pred * inv,
        }

        # --- Weights ---
        alpha_t = _pred_weight(base_spec.pred_weight_mode, t=t, alpha_schedule_cfg=train_cfg["alpha_schedule"])
        if not base_spec.use_fair:
            beta_t = 0.0
        else:
            beta_t = float(lambda_value)  # penalty mode

        # --- Combine: scalarized (param space) OR MOO handler ---
        if mo_handler is not None:
            mo_grads, mo_losses = _build_active_moo_payload(
                iter_spec=base_spec, out=out_mean,
                g_dec_param=g_dec_param, g_pred_param=g_pred_param, g_fair_param=g_fair_param,
                mo_handler=mo_handler,
            )
            if hasattr(mo_handler, "set_step_context"):
                mo_handler.set_step_context(mu=float(alpha_t), lam=float(beta_t))
            g_comb_param = mo_handler.compute_direction(mo_grads, mo_losses, step=t, epsilon=1e-4)
            guided_diag = None
        else:
            g_comb_param, guided_diag = _combine_prediction_gradients(
                gradient_merge=base_spec.gradient_merge, iter_spec=base_spec,
                g_dec_pred=g_dec_param, g_pred_pred=g_pred_param, g_fair_pred=g_fair_param,
                alpha_t=alpha_t, beta_t=beta_t,
                guided_scale_mode=guided_scale_mode, guided_norm_floor=guided_norm_floor,
            )
        g_comb_param = np.asarray(g_comb_param, dtype=float).reshape(-1)

        # --- Diagnostics ---
        cos_dec_pred = cosine(g_dec_param, g_pred_param)
        cos_dec_fair = cosine(g_dec_param, g_fair_param)
        cos_pred_fair = cosine(g_pred_param, g_fair_param)
        cos_dec_pred_list.append(cos_dec_pred)
        cos_dec_fair_list.append(cos_dec_fair)
        cos_pred_fair_list.append(cos_pred_fair)
        grad_norm = l2_norm(g_comb_param)
        norm_combined_list.append(grad_norm)

        nan_or_inf_flag = bool(
            np.isnan(g_comb_param).any() or np.isinf(g_comb_param).any()
            or any(np.isnan(v) or np.isinf(v) for v in
                   [out_mean["loss_dec"], out_mean["loss_pred"], out_mean["loss_fair"]])
        )

        # --- LR schedule ---
        if t < lr_warmup_steps:
            lr_t = lr0 * (t + 1) / max(lr_warmup_steps, 1)
        else:
            lr_t = lr_value(t=t - lr_warmup_steps, lr=lr0, lr_decay=lr_decay)
        for group in optimizer.param_groups:
            group["lr"] = lr_t

        # --- Step (set p.grad from the combined param-space direction) ---
        if nan_or_inf_flag:
            nan_or_inf_steps += 1
            diverged_steps += 1
            predictor.module.zero_grad(set_to_none=True)
        else:
            if grad_norm > explode_threshold:
                exploding_steps += 1
                diverged_steps += 1
            predictor.module.zero_grad(set_to_none=True)
            offset = 0
            with torch.no_grad():
                for p in predictor.module.parameters():
                    numel = p.numel()
                    p.grad = torch.as_tensor(
                        g_comb_param[offset:offset + numel], dtype=p.dtype, device=p.device,
                    ).reshape(p.shape)
                    offset += numel
            if grad_clip_norm > 0.0 and grad_norm > grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(predictor.module.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        if do_log:
            row: Dict[str, Any] = {
                "task": "medical_resource_allocation_multiinstance",
                "method": method_name, "seed": seed, "stage_idx": stage_idx,
                "lambda": lambda_value, "iter": t, "stage_type": "train",
                "alpha_t": alpha_t, "beta_t": beta_t, "lr_t": lr_t,
                "batch_instances": nB,
                "loss_dec": out_mean["loss_dec"], "loss_pred": out_mean["loss_pred"],
                "loss_fair": out_mean["loss_fair"],
                "grad_norm_dec": l2_norm(g_dec_param), "grad_norm_pred": l2_norm(g_pred_param),
                "grad_norm_fair": l2_norm(g_fair_param), "grad_norm_combined": grad_norm,
                "cos_dec_pred": cos_dec_pred, "cos_dec_fair": cos_dec_fair,
                "cos_pred_fair": cos_pred_fair,
                "nan_or_inf_flag": int(nan_or_inf_flag),
                "device": str(device),
            }
            if mo_handler is not None:
                row.update(mo_handler.extra_logs())
                row["mo_method"] = str(train_cfg.get("mo_method", ""))
            iter_logs.append(row)

        # --- Early-stop check on held-out val (every E steps) ---
        if early_stopping and ((t + 1) % es_eval_every == 0):
            predictor.eval()
            val_metrics = evaluate_instances(
                task, predictor, inst_data.val, fairness_smoothing, saa_mean=saa_mean,
                pred_floor=pred_floor,
            )
            predictor.train()
            val_regret = float(val_metrics.get("regret_normalized", float("nan")))
            es_history.append((t, val_regret))
            if np.isfinite(val_regret) and val_regret < es_best_val - es_min_delta:
                es_best_val = val_regret
                es_best_step = t
                es_best_state = {
                    k: v.detach().clone() for k, v in predictor.state_dict().items()
                }
            if es_best_step >= 0 and (t - es_best_step) >= es_patience:
                es_stopped_early = True
                break

    # Restore the best-val checkpoint so the test/train eval below reflects the
    # early-stopping selection, not the last (possibly worse) step.
    if early_stopping and es_best_state is not None:
        predictor.load_state_dict(es_best_state)

    stage_wallclock = float(perf_counter() - stage_start)
    steps_run = t + 1

    # === EVALUATION (per-instance, then average +/- SE) ===
    eval_train = bool(train_cfg.get("eval_train", True))
    predictor.eval()
    test_metrics = evaluate_instances(
        task, predictor, inst_data.test, fairness_smoothing, saa_mean=saa_mean,
        pred_floor=pred_floor,
    )
    # Val is the HP selector (one eval on the restored best-val checkpoint).
    val_metrics = evaluate_instances(
        task, predictor, inst_data.val, fairness_smoothing, saa_mean=saa_mean,
        pred_floor=pred_floor,
    )
    train_metrics = (
        evaluate_instances(task, predictor, inst_data.train, fairness_smoothing, saa_mean=saa_mean,
                           pred_floor=pred_floor)
        if eval_train else {}
    )
    predictor.train()

    grad_min = float(np.min(norm_combined_list)) if norm_combined_list else 0.0
    grad_median = float(np.median(norm_combined_list)) if norm_combined_list else 0.0
    grad_max = float(np.max(norm_combined_list)) if norm_combined_list else 0.0

    stage_row: Dict[str, Any] = {
        "task": "medical_resource_allocation_multiinstance",
        "method": method_name, "seed": seed, "stage_idx": stage_idx, "lambda": lambda_value,
        # test (mean over N_test instances) + standard error
        "test_regret": test_metrics.get("regret", float("nan")),
        "test_regret_se": test_metrics.get("regret_se", float("nan")),
        "test_regret_normalized": test_metrics.get("regret_normalized", float("nan")),
        "test_regret_normalized_se": test_metrics.get("regret_normalized_se", float("nan")),
        "test_regret_normalized_pred_obj": test_metrics.get("regret_normalized_pred_obj", float("nan")),
        "test_fairness": test_metrics.get("fairness", float("nan")),
        "test_fairness_se": test_metrics.get("fairness_se", float("nan")),
        "test_pred_mse": test_metrics.get("pred_mse", float("nan")),
        "test_pred_mse_se": test_metrics.get("pred_mse_se", float("nan")),
        "n_test_instances": test_metrics.get("n_instances", float("nan")),
        # val (HP selector — mean over N_val instances)
        "val_regret_normalized": val_metrics.get("regret_normalized", float("nan")),
        "val_regret_normalized_se": val_metrics.get("regret_normalized_se", float("nan")),
        "val_regret": val_metrics.get("regret", float("nan")),
        "val_fairness": val_metrics.get("fairness", float("nan")),
        "val_pred_mse": val_metrics.get("pred_mse", float("nan")),
        "n_val_instances": val_metrics.get("n_instances", float("nan")),
        # train (mean over N_train instances)
        "train_regret": train_metrics.get("regret", float("nan")),
        "train_regret_se": train_metrics.get("regret_se", float("nan")),
        "train_regret_normalized": train_metrics.get("regret_normalized", float("nan")),
        "train_fairness": train_metrics.get("fairness", float("nan")),
        "train_pred_mse": train_metrics.get("pred_mse", float("nan")),
        "n_train_instances": train_metrics.get("n_instances", float("nan")),
        # diagnostics
        "stage_wallclock_sec": stage_wallclock,
        "nan_or_inf_steps": nan_or_inf_steps,
        "exploding_steps": exploding_steps,
        "diverged_steps": diverged_steps,
        "avg_grad_norm_combined": float(np.mean(norm_combined_list)) if norm_combined_list else 0.0,
        "grad_norm_min": grad_min, "grad_norm_median": grad_median, "grad_norm_max": grad_max,
        "avg_cos_dec_pred": float(np.mean(cos_dec_pred_list)) if cos_dec_pred_list else 0.0,
        "avg_cos_dec_fair": float(np.mean(cos_dec_fair_list)) if cos_dec_fair_list else 0.0,
        "avg_cos_pred_fair": float(np.mean(cos_pred_fair_list)) if cos_pred_fair_list else 0.0,
        "std_cos_dec_fair": float(np.std(cos_dec_fair_list)) if cos_dec_fair_list else 0.0,
        "weight_norm": parameter_l2_norm(predictor.module),
        # early-stopping diagnostics (HP protocol)
        "early_stopping": int(early_stopping),
        "early_stopped": int(es_stopped_early),
        "steps_run": int(steps_run),
        "early_stop_best_step": int(es_best_step),
        "early_stop_best_val_regret_normalized": (
            float(es_best_val) if np.isfinite(es_best_val) else float("nan")
        ),
        "device": str(device),
    }
    return stage_row, iter_logs


# ======================================================================
# Method x seed, and the public entry point
# ======================================================================

def run_method_seed_multiinstance(
    *,
    task: MedicalResourceAllocationTask,
    inst_data: HCInstanceData,
    train_cfg: Dict[str, Any],
    seed: int,
    method_name: str,
    base_spec: MethodSpec,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lambdas = [float(x) for x in train_cfg["lambdas"]]
    force_lambda_path = bool(train_cfg.get("force_lambda_path_all_methods", False))
    if (not base_spec.use_fair) and (not force_lambda_path):
        lambdas = [0.0]
    if train_cfg.get("mo_method") is not None and (not force_lambda_path):
        lambdas = [0.0]

    device = resolve_device_or_warn(str(train_cfg.get("device", "cpu")))
    dtype_str = str(train_cfg.get("dtype", "float64")).strip().lower()
    dtype = torch.float32 if dtype_str == "float32" else torch.float64

    model_cfg = _resolve_model_config(train_cfg)
    build_seed = 13_579 + seed * 101 + 1  # common init across methods (paired)

    # Output link: softplus (default, MLP/raw-linear) keeps the published behavior;
    # the §0.6 log-linear GLM predictor uses "exp" (log-link), the raw-OLS point "none".
    output_link = str(train_cfg.get("output_link", "softplus")).strip().lower()

    predictor = build_predictor(
        config=model_cfg,
        input_dim=int(inst_data.meta["feature_dim"]),
        output_dim=1,
        seed=build_seed,
        device=device,
        dtype=dtype,
        post_transform=output_link,
    )

    # --- Closed-form fit for the pure predict-then-optimize baseline (§0.6 simple model) ---
    # On a (log-)linear predictor the PURE-MSE baseline (PTO) has an exact closed-form fit and
    # is used DIRECTLY -- no SGD, NO warm-start (steps=0): log-OLS for the log link (lstsq on
    # log y, predict exp(X beta) -> conventional positive-target regression; positive like the
    # MLP's softplus), OLS for the identity link. SAA is the featureless train-pool mean
    # (handled in-loop). The fair / robust predict-then-optimize variants (FPTO/WDRO/VarDRO)
    # carry an extra non-quadratic term (fairness / robustness) with no closed form, so they
    # train conventionally -- Adam from the common random init, to convergence, NO warm-start --
    # exactly like the DFL family, just with a prediction(+fairness/robustness) loss.
    # On the (log-)linear simple model, ALL predict-then-optimize baselines are fit in CLOSED
    # FORM (log-OLS / OLS, steps=0) -- not just pure PTO. Verified rationale: the exp link makes
    # SGD explode (grad spikes ~1e16) and a *fitted* linear predictor at alpha=2 is decision-
    # pathological (regret blows up as MSE improves), so we do NOT SGD-train FPTO/WDRO here -- we
    # use the analytic log-OLS fit and skip tuning (lr is inert, steps=0). The fairness/robustness
    # terms are dropped on the linear model: these become the conventional log-OLS predict-then-
    # optimize baseline. (On the MLP, output_link is softplus, so this does NOT fire and the
    # baselines train normally.) SAA = train-pool mean. The DFL family (use_dec) is NOT run on the
    # (log-)linear model (it would explode) -- it lives only on the stable MLP capacities.
    force_zero_steps = False
    is_pred_baseline = (not base_spec.use_dec) and method_name != "saa"
    if output_link in {"exp", "none"} and is_pred_baseline:
        if fit_linear_closed_form(predictor, inst_data.train, link=output_link):
            force_zero_steps = True

    initial_state = {k: v.detach().clone() for k, v in predictor.state_dict().items()}

    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []
    cumulative_wallclock = 0.0

    for stage_idx, lam in enumerate(lambdas):
        restart_per_lambda = bool(train_cfg.get("restart_per_lambda", False))
        if stage_idx > 0 and (restart_per_lambda or (not base_spec.continuation)):
            predictor.load_state_dict(initial_state)

        stage_row, iter_logs = train_single_stage_multiinstance(
            task=task, inst_data=inst_data, predictor=predictor, base_spec=base_spec,
            train_cfg=train_cfg, lambda_value=lam, seed=seed,
            method_name=method_name, stage_idx=stage_idx,
            force_zero_steps=force_zero_steps,
        )
        cumulative_wallclock += float(stage_row["stage_wallclock_sec"])
        stage_row["cumulative_wallclock_sec"] = cumulative_wallclock
        stage_rows.append(stage_row)
        iter_rows.extend(iter_logs)

    return stage_rows, iter_rows


def run_methods_for_seed(
    *,
    task: MedicalResourceAllocationTask,
    inst_data: HCInstanceData,
    train_cfg: Dict[str, Any],
    method_configs: Dict[str, Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run all selected methods at ONE seed on a fixed instance set.

    The paired design (§5): every method sees the *same* instances and the
    *same* common init (keyed by ``seed``) so per-instance composition noise
    cancels in method-vs-method comparisons.
    """
    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []
    for name, method_cfg in method_configs.items():
        spec = resolve_method_spec(method_cfg)
        merged_cfg = dict(train_cfg)
        for k, v in method_cfg.items():
            if k not in {"method", "use_dec", "use_pred", "use_fair",
                         "pred_weight_mode", "continuation", "allow_orthogonalization"}:
                merged_cfg[k] = v
        rows, logs = run_method_seed_multiinstance(
            task=task, inst_data=inst_data, train_cfg=merged_cfg,
            seed=seed, method_name=name.lower(), base_spec=spec,
        )
        stage_rows.extend(rows)
        iter_rows.extend(logs)
    return stage_rows, iter_rows


def run_hc_multiinstance(
    *,
    task_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    method_configs: Dict[str, Dict[str, Any]],
    instance_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Top-level orchestrator: per seed, (re)draw instances then run all methods.

    Per the spec, each experiment ``seed`` controls the pool split + instance
    draws + predictor init; methods are paired within a seed.

    Parameters
    ----------
    task_cfg : dict
        Used only to construct the per-instance compute engine (alpha_fair,
        fairness_type, decision_mode) and to locate the cohort CSV.
    train_cfg : dict
        Training hyper-parameters; ``train_cfg["seeds"]`` drives the seed loop.
    method_configs : dict
        {method_name: ALL_METHOD_CONFIGS[name]} to run.
    instance_cfg : dict
        kwargs for ``make_hc_instances`` (m, n_train, n_test, test_fraction,
        budget_frac, instance_sampling, stratify_by). ``cohort`` and ``seed``
        are supplied by the orchestrator.

    Returns
    -------
    (stage_rows, iter_rows, meta_by_seed)
    """
    from .hc_instances import make_hc_instances

    task = _make_task(task_cfg)
    cohort = task_cfg.get("data_csv", "data/data_processed.csv")
    seeds = [int(s) for s in train_cfg["seeds"]]

    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []
    meta_by_seed: Dict[str, Any] = {}

    for seed in seeds:
        inst_data = make_hc_instances(cohort=cohort, seed=seed, **instance_cfg)
        meta = {k: v for k, v in inst_data.meta.items() if k != "feature_cols"}
        meta_by_seed[str(seed)] = meta
        rows, logs = run_methods_for_seed(
            task=task, inst_data=inst_data, train_cfg=train_cfg,
            method_configs=method_configs, seed=seed,
        )
        # Tag each row with the instance design for self-describing CSVs.
        for r in rows:
            r["m_instance"] = meta["m"]
            r["n_train"] = meta["n_train"]
            r["n_test"] = meta["n_test"]
            r["instance_sampling"] = meta["instance_sampling"]
        stage_rows.extend(rows)
        iter_rows.extend(logs)

    return stage_rows, iter_rows, meta_by_seed
