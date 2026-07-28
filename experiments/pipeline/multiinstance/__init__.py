"""Multi-instance Healthcare experiment (redesign per new_experiment_design.md).

A training sample is **one optimization instance** = one cohort with its own
budget-coupled allocation problem. One predictor ``f_theta`` is trained across
``N_train`` instances (stratified bootstrap draws of ``m`` patients) and
evaluated on ``N_test`` held-out instances, averaged per-instance.

Public API:
    make_hc_instances(...)              -> HCInstanceData   (hc_instances.py)
    run_methods_multiinstance(...)      -> (stage_df, iter_df)  (loop.py)
"""

from .hc_instances import HCInstance, HCInstanceData, make_hc_instances
from .loop import (
    evaluate_instances,
    run_hc_multiinstance,
    run_methods_for_seed,
)

__all__ = [
    "HCInstance",
    "HCInstanceData",
    "make_hc_instances",
    "evaluate_instances",
    "run_methods_for_seed",
    "run_hc_multiinstance",
]
