"""SGD-over-instances training loop for the multi-instance MD-knapsack run.

The MD analogue of ``loop.py`` (Healthcare). Same §7 protocol — one predictor,
common init, no warm-start, ``batch_size`` counts instances. Unlike HC, MD keeps
**fixed steps (no periodic early stopping) by design** — cvxpylayers makes
per-step val solves too expensive (decided 2026-06-08); instead a single
end-of-training val eval (``val_regret_normalized``) is the HP selector.
The per-instance decision gradient comes from the **cvxpylayers** backend
(``build_decision_gradient`` -> ``DecisionGradientComputer.compute``) rather than
a closed form, and predictions are 2D ``(m, n_resources)``.

Per-instance step (validated against the unified loop's MD path):
    task.bind_batch(inst)                       # rebinds the cvxpy problem
    pred = predictor.module(x)                  # RAW (m, nr); softplus is inside compute
    out  = task.compute(raw_pred, true, need_grads=False, skip_regret=True)
                                                # -> grad_pred, grad_fair, losses
    grad_dec = dec_grad_computer.compute(...)   # cvxpylayers VJP, (m, nr)
    backprop each (m, nr) objective grad to params; combine (scalarized OR MOO)

The gradient combination and MOO handlers are reused **unchanged** from
``fair_dfl.training.loop`` (they are agnostic to the grad source / dimension).
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
from fair_dfl.decision import build_decision_gradient
from fair_dfl.metrics import cosine, l2_norm
from fair_dfl.models import build_predictor
from fair_dfl.models.registry import _resolve_model_config
from fair_dfl.schedules import lr_value
from fair_dfl.tasks.md_knapsack import MultiDimKnapsackTask, KnapsackSplit
from fair_dfl.training.loop import (
    _build_active_moo_payload,
    _build_mo_handler,
    _combine_prediction_gradients,
    _pred_weight,
)
from fair_dfl.training.method_spec import MethodSpec, resolve_method_spec

from .md_instances import MDInstanceData, imbalance_params


# ======================================================================
# Compute-engine task (config only; data is bound per instance)
# ======================================================================

def make_md_task(task_cfg: Dict[str, Any]) -> MultiDimKnapsackTask:
    """A bare MD task used only as a per-instance compute engine. ``bind_batch``
    sets the active population + rebuilds the cvxpy problem; ``generate_data`` is
    never called (instances come from ``make_md_instances``)."""
    return MultiDimKnapsackTask(
        n_samples_train=1, n_samples_val=1, n_samples_test=1,
        n_features=int(task_cfg["n_features"]),
        n_resources=int(task_cfg["n_resources"]),
        scenario="alpha_fair",
        alpha_fair=float(task_cfg.get("alpha_fair", 2.0)),
        poly_degree=int(task_cfg.get("poly_degree", 2)),
        snr=float(task_cfg.get("snr", 5.0)),
        cost_mean=float(task_cfg.get("cost_mean", 1.0)),
        cost_std=float(task_cfg.get("cost_std", 0.2)),
        budget_tightness=float(task_cfg.get("budget_tightness", 0.35)),
        fairness_type=str(task_cfg.get("fairness_type", "mad")),
        group_ratio=float(task_cfg.get("group_ratio", 0.5)),
        decision_mode=str(task_cfg.get("decision_mode", "group")),
        **imbalance_params(float(task_cfg.get("imbalance", 0.4))),
    )


def _se(values: np.ndarray) -> float:
    n = values.size
    return 0.0 if n <= 1 else float(np.std(values, ddof=1) / np.sqrt(n))


_EVAL_METRICS = ["regret", "regret_normalized", "pred_mse", "fairness"]


def evaluate_instances_md(
    task: MultiDimKnapsackTask, predictor, instances: List[KnapsackSplit],
    fairness_smoothing: float, saa_raw: np.ndarray | None = None,
) -> Dict[str, float]:
    """Per-instance eval (skip_regret=False => real oracle+pred solve), then
    mean +/- SE over instances (§7).

    ``saa_raw`` (shape ``(n_resources,)``) overrides the predictor with a constant
    per-resource raw prediction (the SAA baseline); softplus inside ``compute``
    maps it back to the per-resource training-benefit mean, broadcast to every
    stakeholder. ``predictor`` may be ``None`` in that case.
    """
    if not instances:
        return {m: float("nan") for m in _EVAL_METRICS}
    per: Dict[str, List[float]] = {m: [] for m in _EVAL_METRICS}
    for inst in instances:
        task.bind_batch(inst)
        if saa_raw is not None:
            pred_np = np.broadcast_to(saa_raw.reshape(1, -1),
                                      (inst.x.shape[0], saa_raw.shape[0])).copy()
        else:
            pred = predictor.module(to_torch(inst.x, device=predictor.device, dtype=predictor.dtype))
            pred_np = pred.detach().cpu().numpy()
        out = task.compute(raw_pred=pred_np, true=inst.y, need_grads=False,
                            skip_regret=False, fairness_smoothing=fairness_smoothing)
        per["regret"].append(float(out["loss_dec"]))
        per["regret_normalized"].append(float(out.get("loss_dec_normalized", 0.0)))
        per["pred_mse"].append(float(out["loss_pred"]))
        per["fairness"].append(float(out["loss_fair"]))
    agg: Dict[str, float] = {"n_instances": float(len(instances))}
    for m in _EVAL_METRICS:
        arr = np.asarray(per[m], dtype=float)
        agg[m] = float(np.mean(arr))
        agg[f"{m}_se"] = _se(arr)
    return agg


# ======================================================================
# Per-instance objective gradients (prediction space -> parameter space)
# ======================================================================

def _instance_objective_grads_md(
    *, task: MultiDimKnapsackTask, predictor, inst: KnapsackSplit,
    spec: MethodSpec, dec_grad_computer, fairness_smoothing: float, step: int,
    device, dtype, method_name: str = "", wdro_eps: float = 0.1,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    task.bind_batch(inst)
    xb_t = to_torch(inst.x, device=device, dtype=dtype)
    pred_t = predictor.module(xb_t)                 # RAW (m, nr); compute() softplus-positives
    pred_np = pred_t.detach().cpu().numpy()

    # Prediction-MSE + fairness grads (analytic, w.r.t. raw); regret solve skipped.
    out = task.compute(raw_pred=pred_np, true=inst.y, need_grads=False,
                        skip_regret=True, fairness_smoothing=fairness_smoothing)

    # Decision gradient via cvxpylayers (only if the method uses it).
    if spec.use_dec:
        dec = dec_grad_computer.compute(
            pred=pred_np, true=inst.y, task=task, need_grads=True,
            fairness_smoothing=fairness_smoothing, step=step, task_output=out,
        )
        g_dec_pred = np.asarray(dec.grad_dec, dtype=float).reshape(pred_np.shape)
    else:
        g_dec_pred = None
    g_pred_pred = np.asarray(out["grad_pred"], dtype=float).reshape(pred_np.shape) if spec.use_pred else None
    g_fair_pred = np.asarray(out["grad_fair"], dtype=float).reshape(pred_np.shape) if spec.use_fair else None

    loss_pred = float(out["loss_pred"])

    # WDRO: Wasserstein-DRO input-gradient penalty (Gao et al. 2024). MSE is taken
    # in benefit space (softplus(raw), to match task.compute's loss_pred) and summed
    # over resources; its param-grad is added to the prediction grad below. Computed
    # before to_param so its zero_grad/backward doesn't disturb the analytic grads.
    wdro_pg = None
    if method_name == "wdro":
        xb_w = xb_t.detach().clone().requires_grad_(True)
        ben_w = torch.nn.functional.softplus(predictor.module(xb_w))
        yb = to_torch(inst.y, device=device, dtype=dtype)
        psm = ((ben_w - yb) ** 2).sum(dim=-1)                 # per-stakeholder MSE over resources
        gx = torch.autograd.grad(psm.sum(), xb_w, create_graph=True)[0]
        gn = (gx ** 2).sum(dim=-1).clamp_min(1e-12).sqrt()
        penalty = wdro_eps * gn.mean()
        predictor.module.zero_grad(set_to_none=True)
        penalty.backward()
        wdro_pg = flatten_param_grads(predictor.module)
        loss_pred += float(penalty.item())

    def to_param(g: np.ndarray | None) -> np.ndarray | None:
        if g is None or not np.any(g):
            return None
        return backward_param_grad_from_output_grad(
            module=predictor.module, output=pred_t, grad_out=g,
            retain_graph=True, device=device,
        )

    g_dec_param = to_param(g_dec_pred)
    g_pred_param = to_param(g_pred_pred)
    g_fair_param = to_param(g_fair_pred)
    if wdro_pg is not None:
        g_pred_param = wdro_pg if g_pred_param is None else (g_pred_param + wdro_pg)

    losses = {
        "loss_dec": float(out.get("loss_dec", 0.0)),
        "loss_pred": loss_pred,
        "loss_fair": float(out["loss_fair"]),
        "loss_dec_normalized": float(out.get("loss_dec_normalized", 0.0)),
    }
    return losses, g_dec_param, g_pred_param, g_fair_param


# ======================================================================
# One lambda stage
# ======================================================================

def train_single_stage_md(
    *, task: MultiDimKnapsackTask, inst_data: MDInstanceData, predictor,
    base_spec: MethodSpec, train_cfg: Dict[str, Any], dec_grad_computer,
    lambda_value: float, seed: int, method_name: str, stage_idx: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device, dtype = predictor.device, predictor.dtype
    rng = np.random.default_rng(seed * 10_000 + stage_idx * 113 + 7)
    steps = int(train_cfg["steps_per_lambda"])
    batch_size = int(train_cfg.get("batch_size", -1))      # instances/step; <=0 => full
    lr0 = float(train_cfg["lr"]); lr_decay = float(train_cfg.get("lr_decay", 0.0))
    fairness_smoothing = float(train_cfg.get("fairness_smoothing", 1e-6))
    wdro_eps = float(train_cfg.get("wdro_epsilon", 0.1))
    log_every = int(train_cfg.get("log_every", 10))
    n_instances = len(inst_data.train)
    n_param = sum(p.numel() for p in predictor.module.parameters())

    opt_name = str(train_cfg.get("optimizer", "sgd")).strip().lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr0)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(predictor.parameters(), lr=lr0)
    else:
        optimizer = torch.optim.SGD(predictor.parameters(), lr=lr0)

    mo_handler = _build_mo_handler(train_cfg)
    guided_scale_mode = str(train_cfg.get("guided_merge_scale_mode", "geom")).strip().lower()
    guided_norm_floor = float(train_cfg.get("guided_merge_norm_floor", 1e-3))

    nan_or_inf_steps = exploding_steps = diverged_steps = 0
    cos_dp: List[float] = []; cos_df: List[float] = []; cos_pf: List[float] = []
    norm_comb: List[float] = []; iter_logs: List[Dict[str, Any]] = []
    explode_threshold = float(train_cfg.get("explode_threshold", 1e12))

    stage_start = perf_counter()
    predictor.train()
    for t in range(steps):
        if batch_size <= 0 or batch_size >= n_instances:
            batch_ids = np.arange(n_instances)
        else:
            batch_ids = rng.choice(n_instances, size=batch_size, replace=False)
        nB = len(batch_ids)

        g_dec = np.zeros(n_param); g_pred = np.zeros(n_param); g_fair = np.zeros(n_param)
        sld = slp = slf = sdn = 0.0
        for k in batch_ids:
            losses_k, gd, gp, gf = _instance_objective_grads_md(
                task=task, predictor=predictor, inst=inst_data.train[k], spec=base_spec,
                dec_grad_computer=dec_grad_computer, fairness_smoothing=fairness_smoothing,
                step=t, device=device, dtype=dtype, method_name=method_name, wdro_eps=wdro_eps,
            )
            if gd is not None: g_dec += gd
            if gp is not None: g_pred += gp
            if gf is not None: g_fair += gf
            sld += losses_k["loss_dec"]; slp += losses_k["loss_pred"]
            slf += losses_k["loss_fair"]; sdn += losses_k["loss_dec_normalized"]
        inv = 1.0 / float(nB)
        g_dec *= inv; g_pred *= inv; g_fair *= inv
        out_mean = {"loss_dec": sld * inv, "loss_pred": slp * inv,
                    "loss_fair": slf * inv, "loss_dec_normalized": sdn * inv}

        alpha_t = _pred_weight(base_spec.pred_weight_mode, t=t,
                               alpha_schedule_cfg=train_cfg["alpha_schedule"])
        beta_t = float(lambda_value) if base_spec.use_fair else 0.0

        if mo_handler is not None:
            mo_grads, mo_losses = _build_active_moo_payload(
                iter_spec=base_spec, out=out_mean,
                g_dec_param=g_dec, g_pred_param=g_pred, g_fair_param=g_fair, mo_handler=mo_handler)
            if hasattr(mo_handler, "set_step_context"):
                mo_handler.set_step_context(mu=float(alpha_t), lam=float(beta_t))
            g_comb = mo_handler.compute_direction(mo_grads, mo_losses, step=t, epsilon=1e-4)
        else:
            g_comb, _ = _combine_prediction_gradients(
                gradient_merge=base_spec.gradient_merge, iter_spec=base_spec,
                g_dec_pred=g_dec, g_pred_pred=g_pred, g_fair_pred=g_fair,
                alpha_t=alpha_t, beta_t=beta_t,
                guided_scale_mode=guided_scale_mode, guided_norm_floor=guided_norm_floor)
        g_comb = np.asarray(g_comb, dtype=float).reshape(-1)

        cdp, cdf, cpf = cosine(g_dec, g_pred), cosine(g_dec, g_fair), cosine(g_pred, g_fair)
        cos_dp.append(cdp); cos_df.append(cdf); cos_pf.append(cpf)
        grad_norm = l2_norm(g_comb); norm_comb.append(grad_norm)
        bad = bool(np.isnan(g_comb).any() or np.isinf(g_comb).any()
                   or not np.isfinite(out_mean["loss_pred"]) or not np.isfinite(out_mean["loss_fair"]))

        # Gradient clipping (essential for the cvxpylayers decision gradient: the
        # alpha-fair 1/u^2 sensitivity makes the raw decision grad spike and run
        # away). Clip the combined param-space direction to a fixed norm.
        grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
        if (not bad) and grad_clip > 0.0 and grad_norm > grad_clip:
            g_comb = g_comb * (grad_clip / grad_norm)

        lr_t = lr_value(t=t, lr=lr0, lr_decay=lr_decay)
        for grp in optimizer.param_groups:
            grp["lr"] = lr_t

        if bad:
            nan_or_inf_steps += 1; diverged_steps += 1
            predictor.module.zero_grad(set_to_none=True)
            break   # stop a diverged stage (else SCS hits max-iter every solve -> hours)
        else:
            if grad_norm > explode_threshold:
                exploding_steps += 1; diverged_steps += 1
            predictor.module.zero_grad(set_to_none=True)
            offset = 0
            with torch.no_grad():
                for p in predictor.module.parameters():
                    nel = p.numel()
                    p.grad = torch.as_tensor(g_comb[offset:offset + nel], dtype=p.dtype,
                                             device=p.device).reshape(p.shape)
                    offset += nel
            optimizer.step()
        if t % max(log_every, 1) == 0:
            iter_logs.append({
                "task": "md_knapsack_multiinstance", "method": method_name, "seed": seed,
                "stage_idx": stage_idx, "lambda": lambda_value, "iter": t,
                "alpha_t": alpha_t, "beta_t": beta_t, "lr_t": lr_t, "batch_instances": nB,
                "loss_dec": out_mean["loss_dec"], "loss_pred": out_mean["loss_pred"],
                "loss_fair": out_mean["loss_fair"], "grad_norm_combined": grad_norm,
                "cos_dec_pred": cdp, "cos_dec_fair": cdf, "cos_pred_fair": cpf,
                "nan_or_inf_flag": int(bad), "device": str(device),
            })

    stage_wall = float(perf_counter() - stage_start)
    predictor.eval()
    test_m = evaluate_instances_md(task, predictor, inst_data.test, fairness_smoothing)
    # Val is the HP selector. MD does NOT do periodic early stopping (cvxpylayers
    # cost); a single end-of-training val eval is the cheap selector (decided
    # 2026-06-08). N_val solves once vs N_val * (steps/E) for periodic early-stop.
    val_m = evaluate_instances_md(task, predictor, inst_data.val, fairness_smoothing)
    train_m = (evaluate_instances_md(task, predictor, inst_data.train, fairness_smoothing)
               if bool(train_cfg.get("eval_train", True)) else {})
    predictor.train()

    stage_row = {
        "task": "md_knapsack_multiinstance", "method": method_name, "seed": seed,
        "stage_idx": stage_idx, "lambda": lambda_value,
        "test_regret": test_m.get("regret", float("nan")),
        "test_regret_se": test_m.get("regret_se", float("nan")),
        "test_regret_normalized": test_m.get("regret_normalized", float("nan")),
        "test_regret_normalized_se": test_m.get("regret_normalized_se", float("nan")),
        "test_fairness": test_m.get("fairness", float("nan")),
        "test_fairness_se": test_m.get("fairness_se", float("nan")),
        "test_pred_mse": test_m.get("pred_mse", float("nan")),
        "test_pred_mse_se": test_m.get("pred_mse_se", float("nan")),
        "n_test_instances": test_m.get("n_instances", float("nan")),
        "val_regret_normalized": val_m.get("regret_normalized", float("nan")),
        "val_regret_normalized_se": val_m.get("regret_normalized_se", float("nan")),
        "val_regret": val_m.get("regret", float("nan")),
        "val_fairness": val_m.get("fairness", float("nan")),
        "val_pred_mse": val_m.get("pred_mse", float("nan")),
        "n_val_instances": val_m.get("n_instances", float("nan")),
        "train_regret_normalized": train_m.get("regret_normalized", float("nan")),
        "train_fairness": train_m.get("fairness", float("nan")),
        "train_pred_mse": train_m.get("pred_mse", float("nan")),
        "n_train_instances": float(n_instances),
        "stage_wallclock_sec": stage_wall,
        "nan_or_inf_steps": nan_or_inf_steps, "exploding_steps": exploding_steps,
        "diverged_steps": diverged_steps,
        "avg_grad_norm_combined": float(np.mean(norm_comb)) if norm_comb else 0.0,
        "grad_norm_max": float(np.max(norm_comb)) if norm_comb else 0.0,
        "avg_cos_dec_pred": float(np.mean(cos_dp)) if cos_dp else 0.0,
        "avg_cos_dec_fair": float(np.mean(cos_df)) if cos_df else 0.0,
        "avg_cos_pred_fair": float(np.mean(cos_pf)) if cos_pf else 0.0,
        "weight_norm": parameter_l2_norm(predictor.module), "device": str(device),
    }
    return stage_row, iter_logs


# ======================================================================
# Method x seed, and the per-cell entry point
# ======================================================================

def _run_saa_md(
    *, task: MultiDimKnapsackTask, inst_data: MDInstanceData, train_cfg: Dict[str, Any], seed: int,
) -> List[Dict[str, Any]]:
    """SAA baseline: predict the **per-resource** training-benefit mean (literature
    convention), no training. The mean (benefit space) is mapped to a raw prediction
    via inverse-softplus so task.compute's softplus recovers it. Returns one stage_row
    mirroring train_single_stage_md's schema (diverged/steps = 0)."""
    fairness_smoothing = float(train_cfg.get("fairness_smoothing", 1e-6))
    eval_train = bool(train_cfg.get("eval_train", True))
    ys = np.concatenate([inst.y for inst in inst_data.train], axis=0)   # (sum_m, n_resources)
    mean_vec = np.clip(ys.mean(axis=0), 1e-6, None)                     # per-resource mean (nr,)
    saa_raw = np.log(np.expm1(mean_vec))                               # inverse softplus -> raw
    t0 = perf_counter()
    test_m = evaluate_instances_md(task, None, inst_data.test, fairness_smoothing, saa_raw=saa_raw)
    val_m = evaluate_instances_md(task, None, inst_data.val, fairness_smoothing, saa_raw=saa_raw)
    train_m = (evaluate_instances_md(task, None, inst_data.train, fairness_smoothing, saa_raw=saa_raw)
               if eval_train else {})
    wall = float(perf_counter() - t0)
    return [{
        "task": "md_knapsack_multiinstance", "method": "saa", "seed": seed,
        "stage_idx": 0, "lambda": 0.0,
        "test_regret": test_m.get("regret", float("nan")),
        "test_regret_se": test_m.get("regret_se", float("nan")),
        "test_regret_normalized": test_m.get("regret_normalized", float("nan")),
        "test_regret_normalized_se": test_m.get("regret_normalized_se", float("nan")),
        "test_fairness": test_m.get("fairness", float("nan")),
        "test_fairness_se": test_m.get("fairness_se", float("nan")),
        "test_pred_mse": test_m.get("pred_mse", float("nan")),
        "test_pred_mse_se": test_m.get("pred_mse_se", float("nan")),
        "n_test_instances": test_m.get("n_instances", float("nan")),
        "val_regret_normalized": val_m.get("regret_normalized", float("nan")),
        "val_regret_normalized_se": val_m.get("regret_normalized_se", float("nan")),
        "val_regret": val_m.get("regret", float("nan")),
        "val_fairness": val_m.get("fairness", float("nan")),
        "val_pred_mse": val_m.get("pred_mse", float("nan")),
        "n_val_instances": val_m.get("n_instances", float("nan")),
        "train_regret_normalized": train_m.get("regret_normalized", float("nan")),
        "train_fairness": train_m.get("fairness", float("nan")),
        "train_pred_mse": train_m.get("pred_mse", float("nan")),
        "n_train_instances": float(len(inst_data.train)),
        "stage_wallclock_sec": wall, "cumulative_wallclock_sec": wall,
        "nan_or_inf_steps": 0, "exploding_steps": 0, "diverged_steps": 0,
        "avg_grad_norm_combined": 0.0, "grad_norm_max": 0.0,
        "avg_cos_dec_pred": 0.0, "avg_cos_dec_fair": 0.0, "avg_cos_pred_fair": 0.0,
        "weight_norm": 0.0, "device": str(train_cfg.get("device", "cpu")),
    }]


def run_method_seed_md(
    *, task: MultiDimKnapsackTask, inst_data: MDInstanceData, train_cfg: Dict[str, Any],
    dec_grad_computer, seed: int, method_name: str, base_spec: MethodSpec,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if method_name == "saa":                       # no training; per-resource mean predictor
        return _run_saa_md(task=task, inst_data=inst_data, train_cfg=train_cfg, seed=seed), []
    lambdas = [float(x) for x in train_cfg["lambdas"]]
    force = bool(train_cfg.get("force_lambda_path_all_methods", False))
    if (not base_spec.use_fair) and (not force):
        lambdas = [0.0]
    if train_cfg.get("mo_method") is not None and (not force):
        lambdas = [0.0]

    device = resolve_device_or_warn(str(train_cfg.get("device", "cpu")))
    dtype = torch.float64
    model_cfg = _resolve_model_config(train_cfg)
    build_seed = 13_579 + seed * 101 + 1               # common init across methods (paired)
    predictor = build_predictor(
        config=model_cfg, input_dim=int(inst_data.meta["feature_dim"]),
        output_dim=int(inst_data.meta["n_resources"]), seed=build_seed,
        device=device, dtype=dtype, post_transform="none",   # softplus is inside task.compute
    )
    initial_state = {k: v.detach().clone() for k, v in predictor.state_dict().items()}

    stage_rows: List[Dict[str, Any]] = []; iter_rows: List[Dict[str, Any]] = []
    cumulative = 0.0
    for stage_idx, lam in enumerate(lambdas):
        if stage_idx > 0 and (bool(train_cfg.get("restart_per_lambda", False)) or (not base_spec.continuation)):
            predictor.load_state_dict(initial_state)
        stage_row, logs = train_single_stage_md(
            task=task, inst_data=inst_data, predictor=predictor, base_spec=base_spec,
            train_cfg=train_cfg, dec_grad_computer=dec_grad_computer, lambda_value=lam,
            seed=seed, method_name=method_name, stage_idx=stage_idx)
        cumulative += float(stage_row["stage_wallclock_sec"])
        stage_row["cumulative_wallclock_sec"] = cumulative
        stage_rows.append(stage_row); iter_rows.extend(logs)
    return stage_rows, iter_rows


def run_methods_for_seed_md(
    *, task_cfg: Dict[str, Any], inst_data: MDInstanceData, train_cfg: Dict[str, Any],
    method_configs: Dict[str, Dict[str, Any]], seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run all selected methods at ONE seed on a fixed MD instance set (paired)."""
    task = make_md_task(task_cfg)
    device = resolve_device_or_warn(str(train_cfg.get("device", "cpu")))
    # Decision-gradient computer (shared; reads the task's bound batch). The MD
    # decisions are differentiated with the "cvxpylayers" conic layer (open
    # SCS/ECOS solvers, no MOSEK required).
    dec_cfg = dict(train_cfg)
    dec_cfg["decision_grad_backend"] = str(train_cfg.get("decision_grad_backend", "cvxpylayers"))
    dec_grad_computer = build_decision_gradient(dec_cfg, task, device)

    stage_rows: List[Dict[str, Any]] = []; iter_rows: List[Dict[str, Any]] = []
    for name, method_cfg in method_configs.items():
        spec = resolve_method_spec(method_cfg)
        merged = dict(train_cfg)
        for k, v in method_cfg.items():
            if k not in {"method", "use_dec", "use_pred", "use_fair", "pred_weight_mode",
                         "continuation", "allow_orthogonalization"}:
                merged[k] = v
        rows, logs = run_method_seed_md(
            task=task, inst_data=inst_data, train_cfg=merged,
            dec_grad_computer=dec_grad_computer, seed=seed,
            method_name=name.lower(), base_spec=spec)
        stage_rows.extend(rows); iter_rows.extend(logs)
    return stage_rows, iter_rows
