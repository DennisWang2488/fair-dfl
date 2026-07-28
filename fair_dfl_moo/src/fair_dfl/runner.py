"""Top-level experiment runner and public method registry dispatch."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .training import run_methods as _run_methods_unified, resolve_method_spec
from .tasks import (
    MedicalResourceAllocationTask,
    MultiDimKnapsackTask,
    SingleResourceAlphaFairTask,
)
from .tasks.base import BaseTask, SplitData, TaskData


def _apply_subset_fraction(
    task: BaseTask,
    data: TaskData,
    train_subset_fraction: float,
    subset_seed: int,
) -> TaskData:
    frac = float(train_subset_fraction)
    if frac >= 1.0:
        return data
    if not (0.0 < frac <= 1.0):
        raise ValueError("training.train_subset_fraction must be in (0, 1].")

    rng = np.random.default_rng(int(subset_seed) * 1009 + 17)

    def choose_idx(n_rows: int) -> np.ndarray:
        n_keep = max(1, int(round(float(n_rows) * frac)))
        n_keep = min(n_keep, n_rows)
        if n_keep == n_rows:
            return np.arange(n_rows, dtype=int)
        return np.sort(rng.choice(n_rows, size=n_keep, replace=False))

    train_idx = choose_idx(data.train.x.shape[0])
    val_idx = choose_idx(data.val.x.shape[0])
    test_idx = np.arange(data.test.x.shape[0], dtype=int)

    train = SplitData(x=data.train.x[train_idx], y=data.train.y[train_idx])
    val = SplitData(x=data.val.x[val_idx], y=data.val.y[val_idx])
    test = SplitData(x=data.test.x[test_idx], y=data.test.y[test_idx])

    if isinstance(task, (MedicalResourceAllocationTask, SingleResourceAlphaFairTask)):
        for split_name, idx in [("train", train_idx), ("val", val_idx)]:
            med_split = task._splits[split_name]
            task._splits[split_name] = type(med_split)(
                x=med_split.x[idx],
                y=med_split.y[idx],
                cost=med_split.cost[idx],
                race=med_split.race[idx],
            )

    if isinstance(task, MultiDimKnapsackTask):
        for split_name, idx in [("train", train_idx), ("val", val_idx)]:
            sub = task._splits[split_name]
            sub_cost = sub.cost[idx]
            sub_budgets = float(task.budget_tightness) * sub_cost.sum(axis=0)
            task._splits[split_name] = type(sub)(
                x=sub.x[idx],
                y=sub.y[idx],
                cost=sub_cost,
                groups=sub.groups[idx],
                budgets=sub_budgets,
            )
        task.bind_split("train")

    meta = dict(data.meta)
    if "n_train" in meta:
        meta["n_train"] = np.asarray([train.x.shape[0]], dtype=np.int64)
    if "n_val" in meta:
        meta["n_val"] = np.asarray([val.x.shape[0]], dtype=np.int64)
    if "n_test" in meta:
        meta["n_test"] = np.asarray([test.x.shape[0]], dtype=np.int64)
    if "n_total" in meta:
        meta["n_total"] = np.asarray([train.x.shape[0] + val.x.shape[0] + test.x.shape[0]], dtype=np.int64)

    return TaskData(
        train=train,
        val=val,
        test=test,
        groups=data.groups,
        meta=meta,
    )


def _build_task(task_cfg: Dict[str, Any]) -> Tuple[BaseTask, TaskData]:
    name = str(task_cfg["name"])
    fairness_type = str(task_cfg.get("fairness_type", "mad"))
    fairness_ge_alpha = float(task_cfg.get("fairness_ge_alpha", 2.0))
    if name == "md_knapsack":
        task = MultiDimKnapsackTask(
            n_samples_train=int(task_cfg["n_samples_train"]),
            n_samples_val=int(task_cfg["n_samples_val"]),
            n_samples_test=int(task_cfg["n_samples_test"]),
            n_features=int(task_cfg["n_features"]),
            n_resources=int(task_cfg.get("n_resources", 2)),
            scenario=str(task_cfg.get("scenario", "alpha_fair")),
            alpha_fair=float(task_cfg.get("alpha_fair", 2.0)),
            poly_degree=int(task_cfg.get("poly_degree", 2)),
            snr=float(task_cfg.get("snr", 5.0)),
            benefit_group_bias=float(task_cfg.get("benefit_group_bias", task_cfg.get("group_bias", 0.3))),
            benefit_noise_ratio=float(task_cfg.get("benefit_noise_ratio", 1.0)),
            cost_group_bias=float(task_cfg.get("cost_group_bias", 0.0)),
            cost_noise_ratio=float(task_cfg.get("cost_noise_ratio", 1.0)),
            cost_mean=float(task_cfg.get("cost_mean", 1.0)),
            cost_std=float(task_cfg.get("cost_std", 0.2)),
            budget_tightness=float(task_cfg.get("budget_tightness", 0.5)),
            fairness_type=fairness_type,
            fairness_ge_alpha=fairness_ge_alpha,
            group_ratio=float(task_cfg.get("group_ratio", 0.5)),
            decision_mode=str(task_cfg.get("decision_mode", "group")),
        )
        data_seed = int(task_cfg.get("data_seed", 42))
        data = task.generate_data(seed=data_seed)
        task.bind_split("train")
        return task, data

    if name == "single_resource_alpha_fair":
        task = SingleResourceAlphaFairTask(
            n_train=int(task_cfg.get("n_train", task_cfg.get("n_samples_train", 200))),
            n_val=int(task_cfg.get("n_val", task_cfg.get("n_samples_val", 50))),
            n_test=int(task_cfg.get("n_test", task_cfg.get("n_samples_test", 100))),
            n_features=int(task_cfg.get("n_features", 5)),
            alpha_fair=float(task_cfg.get("alpha_fair", 2.0)),
            poly_degree=int(task_cfg.get("poly_degree", 2)),
            snr=float(task_cfg.get("snr", 5.0)),
            benefit_group_bias=float(task_cfg.get("benefit_group_bias", 0.0)),
            benefit_noise_ratio=float(task_cfg.get("benefit_noise_ratio", 1.0)),
            cost_group_bias=float(task_cfg.get("cost_group_bias", 0.0)),
            cost_noise_ratio=float(task_cfg.get("cost_noise_ratio", 1.0)),
            cost_mean=float(task_cfg.get("cost_mean", 1.0)),
            cost_std=float(task_cfg.get("cost_std", 0.2)),
            budget_tightness=float(task_cfg.get("budget_tightness", 0.25)),
            fairness_type=fairness_type,
            group_ratio=float(task_cfg.get("group_ratio", 0.5)),
            decision_mode=str(task_cfg.get("decision_mode", "group")),
        )
        data = task.generate_data(seed=int(task_cfg.get("data_seed", 42)))
        return task, data

    if name == "medical_resource_allocation":
        task = MedicalResourceAllocationTask(
            data_csv=str(task_cfg["data_csv"]),
            n_sample=int(task_cfg.get("n_sample", 5000)),
            data_seed=int(task_cfg.get("data_seed", 42)),
            split_seed=int(task_cfg.get("split_seed", 2)),
            test_fraction=float(task_cfg.get("test_fraction", 0.5)),
            val_fraction=float(task_cfg.get("val_fraction", 0.2)),
            alpha_fair=float(task_cfg.get("alpha_fair", 2.0)),
            budget=float(task_cfg.get("budget", 2500.0)),
            decision_mode=str(task_cfg.get("decision_mode", "group")),
            fairness_type=str(task_cfg.get("fairness_type", "mad")),
            budget_rho=float(task_cfg.get("budget_rho", 0.35)),
        )
        data = task.generate_data(seed=int(task_cfg.get("data_seed", 42)))
        return task, data

    raise ValueError(f"Unknown task name: {name}")


def run_experiment_unified(
    cfg: Dict[str, Any],
    method_configs: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run experiments using the unified training loop.

    Accepts the full method_configs dict from configs.py (with inline
    use_dec/use_pred/use_fair flags) and uses the unified training loop
    for all methods.

    Args:
        cfg: Experiment config with "task" and "training" keys.
        method_configs: Dict of {method_name: method_config} from ALL_METHOD_CONFIGS.

    Returns:
        (stage_results_df, iter_logs_df)
    """
    task, data = _build_task(cfg["task"])
    train_cfg_raw = cfg.get("training", {})
    train_cfg = dict(train_cfg_raw)
    pareto_sweep_mode = bool(train_cfg.get("pareto_sweep_mode", train_cfg.get("frontier_mode", False)))
    if pareto_sweep_mode:
        lambdas = [float(v) for v in train_cfg.get("lambdas", [0.0])]
        train_cfg["lambdas"] = lambdas if lambdas else [0.0]
    else:
        train_cfg["lambdas"] = [float(train_cfg.get("lambda_train", 0.0))]

    subset_fraction = float(train_cfg.get("train_subset_fraction", 1.0))
    subset_seed = int(cfg.get("task", {}).get("data_seed", 42))
    data = _apply_subset_fraction(
        task=task, data=data,
        train_subset_fraction=subset_fraction,
        subset_seed=subset_seed,
    )

    stage_rows, iter_rows = _run_methods_unified(
        task=task,
        data=data,
        train_cfg=train_cfg,
        method_configs=method_configs,
    )
    return pd.DataFrame(stage_rows), pd.DataFrame(iter_rows)
