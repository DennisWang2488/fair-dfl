"""Unified evaluation functions for all methods and task types."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..models import PredictorHandle
from ..tasks.base import BaseTask, SplitData
from ..tasks.md_knapsack import MultiDimKnapsackTask
from ..tasks.medical_resource_allocation import MedicalResourceAllocationTask
from ..tasks.single_resource_alpha_fair import SingleResourceAlphaFairTask

_MED_LIKE = (MedicalResourceAllocationTask, SingleResourceAlphaFairTask)


def eval_split(
    task: BaseTask,
    predictor: PredictorHandle,
    split: SplitData,
    fairness_smoothing: float,
    override_pred: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate a predictor on a non-medical task split."""
    if override_pred is not None:
        raw_pred = override_pred
    else:
        raw_pred = predictor.predict_numpy(split.x)
    out = task.compute(
        raw_pred=raw_pred,
        true=split.y,
        need_grads=False,
        fairness_smoothing=fairness_smoothing,
    )
    metrics = {
        "regret": float(out["loss_dec"]),
        "pred_mse": float(out["loss_pred"]),
        "fairness": float(out["loss_fair"]),
        "solver_calls_eval": float(out.get("solver_calls", 0)),
        "decision_ms_eval": float(out.get("decision_ms", 0.0)),
    }
    for key, metric_name in [
        ("loss_dec_normalized", "regret_normalized"),
        ("loss_dec_normalized_true", "regret_normalized_true"),
        ("loss_dec_normalized_pred_obj", "regret_normalized_pred_obj"),
    ]:
        if key in out:
            metrics[metric_name] = float(out[key])
    # Forward decision-level fairness metrics (e.g. decision_alloc_gap)
    for key in out:
        if key.startswith("decision_"):
            metrics[key] = float(out[key])
    return metrics


def eval_split_medical(
    task: MedicalResourceAllocationTask,
    predictor: PredictorHandle,
    split_name: str,
    fairness_smoothing: float,
    override_pred: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate a predictor on a medical task split."""
    if override_pred is not None:
        return task.evaluate_split(
            split=split_name, pred=override_pred, fairness_smoothing=fairness_smoothing,
        )
    split = task._splits[split_name]
    pred = predictor.predict_numpy(split.x).reshape(-1)
    return task.evaluate_split(
        split=split_name, pred=pred, fairness_smoothing=fairness_smoothing,
    )


def eval_split_md_knapsack(
    task: MultiDimKnapsackTask,
    predictor: PredictorHandle,
    split_name: str,
    fairness_smoothing: float,
    override_pred: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate a predictor on a multi-dim knapsack split.

    Binds the cvxpy problem to the requested split before solving so that
    val/test evaluation uses the right population (different ``n``).
    """
    split = task._splits[split_name]
    if override_pred is not None:
        pred = np.asarray(override_pred, dtype=float).reshape(-1, task.n_resources)
    else:
        pred = predictor.predict_numpy(split.x).reshape(-1, task.n_resources)
    return task.evaluate_split(
        split=split_name, pred=pred, fairness_smoothing=fairness_smoothing,
    )


_EMPTY_METRICS: Dict[str, float] = {
    "regret": float("nan"),
    "pred_mse": float("nan"),
    "fairness": float("nan"),
    "regret_normalized": float("nan"),
    "regret_normalized_true": float("nan"),
    "regret_normalized_pred_obj": float("nan"),
    "solver_calls_eval": 0.0,
    "decision_ms_eval": 0.0,
}


def evaluate_model(
    task: BaseTask,
    predictor: PredictorHandle,
    data: Any,
    fairness_smoothing: float,
    saa_override_val: np.ndarray | None = None,
    saa_override_test: np.ndarray | None = None,
    saa_override_train: np.ndarray | None = None,
    eval_train: bool = False,
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Evaluate on train (optional), val, and test splits.

    Train evaluation is **expensive** for tasks where decision regret requires
    solver calls — it is gated behind ``eval_train`` and intended to be called
    once per lambda stage (not per training iteration). Val/test are always
    computed.

    If the val split is empty (val_fraction=0.0), val_metrics returns NaNs
    for score columns so downstream analysis can distinguish "missing split"
    from a genuine zero-valued metric. Train metrics return NaN sentinels when
    ``eval_train=False``.

    Returns
    -------
    (train_metrics, val_metrics, test_metrics) tuple.
    """
    if isinstance(task, _MED_LIKE):
        val_split = task._splits.get("val")
        if val_split is not None and val_split.x.shape[0] > 0:
            val_metrics = eval_split_medical(
                task=task, predictor=predictor, split_name="val",
                fairness_smoothing=fairness_smoothing, override_pred=saa_override_val,
            )
        else:
            val_metrics = dict(_EMPTY_METRICS)
        test_metrics = eval_split_medical(
            task=task, predictor=predictor, split_name="test",
            fairness_smoothing=fairness_smoothing, override_pred=saa_override_test,
        )
        if eval_train:
            train_split = task._splits.get("train")
            if train_split is not None and train_split.x.shape[0] > 0:
                train_metrics = eval_split_medical(
                    task=task, predictor=predictor, split_name="train",
                    fairness_smoothing=fairness_smoothing, override_pred=saa_override_train,
                )
            else:
                train_metrics = dict(_EMPTY_METRICS)
        else:
            train_metrics = dict(_EMPTY_METRICS)
    elif isinstance(task, MultiDimKnapsackTask):
        # MD knapsack tasks own their split data and need explicit binding
        # before each evaluation so the cvxpy problem is sized to the right
        # population (train vs val vs test may all differ).
        val_split = task._splits.get("val")
        if val_split is not None and val_split.x.shape[0] > 0:
            val_metrics = eval_split_md_knapsack(
                task=task, predictor=predictor, split_name="val",
                fairness_smoothing=fairness_smoothing, override_pred=saa_override_val,
            )
        else:
            val_metrics = dict(_EMPTY_METRICS)
        test_metrics = eval_split_md_knapsack(
            task=task, predictor=predictor, split_name="test",
            fairness_smoothing=fairness_smoothing, override_pred=saa_override_test,
        )
        if eval_train:
            train_split = task._splits.get("train")
            if train_split is not None and train_split.x.shape[0] > 0:
                train_metrics = eval_split_md_knapsack(
                    task=task, predictor=predictor, split_name="train",
                    fairness_smoothing=fairness_smoothing, override_pred=saa_override_train,
                )
            else:
                train_metrics = dict(_EMPTY_METRICS)
        else:
            train_metrics = dict(_EMPTY_METRICS)
        # Restore the active split to "train" so subsequent training iterations
        # see the right cvxpy problem on the next call.
        task.bind_split("train")
    else:
        if data.val is not None and data.val.x.shape[0] > 0:
            val_metrics = eval_split(
                task=task, predictor=predictor, split=data.val,
                fairness_smoothing=fairness_smoothing, override_pred=saa_override_val,
            )
        else:
            val_metrics = dict(_EMPTY_METRICS)
        test_metrics = eval_split(
            task=task, predictor=predictor, split=data.test,
            fairness_smoothing=fairness_smoothing, override_pred=saa_override_test,
        )
        if eval_train:
            if data.train is not None and data.train.x.shape[0] > 0:
                train_metrics = eval_split(
                    task=task, predictor=predictor, split=data.train,
                    fairness_smoothing=fairness_smoothing, override_pred=saa_override_train,
                )
            else:
                train_metrics = dict(_EMPTY_METRICS)
        else:
            train_metrics = dict(_EMPTY_METRICS)
    return train_metrics, val_metrics, test_metrics
