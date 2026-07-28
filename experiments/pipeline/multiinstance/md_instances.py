"""Multi-instance sampler for the MD knapsack task (``make_md_instances``).

Mirrors ``make_hc_instances`` but for the synthetic multi-dimensional knapsack:
one *instance* = a population of ``m`` stakeholders that all share the SAME
underlying benefit mapping (the polynomial weights ``W``), differing only in the
sampled features / noise. This is the standard DFL multi-instance setup (SPO /
PyEPO), where a single predictor is learned across many optimization instances.

CRITICAL design point
----------------------
All instances must share one underlying ``x -> benefit`` mapping, otherwise a
single predictor cannot fit them. We therefore generate **one large pool** with
a fixed seed (=> fixed ``W``) and sub-sample ``m``-stakeholder instances from it
(bootstrap or disjoint), recomputing each instance's per-resource budget as
``budget_tightness * sum_i cost_i`` (the task's own ``sample_batch`` convention).
This is the synthetic analogue of drawing patient cohorts from one fixed cohort
in ``make_hc_instances``.

The instances are returned as ``KnapsackSplit`` objects so the training loop can
``task.bind_batch(inst)`` directly before each per-instance compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from fair_dfl.tasks.md_knapsack import MultiDimKnapsackTask, KnapsackSplit


@dataclass
class MDInstanceData:
    train: List[KnapsackSplit]
    test: List[KnapsackSplit]
    val: List[KnapsackSplit] = field(default_factory=list)   # held-out for HP early-stop (§0.5)
    meta: Dict[str, Any] = field(default_factory=dict)


def imbalance_params(level: float, mode: str = "both") -> dict:
    """Single-knob imbalance — mirrors ``run_md_main_v2.imbalance_params`` and
    ``bench_md_timing.imbalance_params`` so the generative process matches the
    published MD experiment."""
    out = {
        "benefit_group_bias": level * 0.9,
        "benefit_noise_ratio": 1.0 + level * 1.0,
        "cost_group_bias": level * 0.9,
        "cost_noise_ratio": 1.0 + level * 1.0,
    }
    if mode == "cost":      # advisor ask: imbalance only on cost, not benefit
        out["benefit_group_bias"], out["benefit_noise_ratio"] = 0.0, 1.0
    elif mode == "benefit":
        out["cost_group_bias"], out["cost_noise_ratio"] = 0.0, 1.0
    elif mode != "both":
        raise ValueError(f"imbalance_mode must be both|cost|benefit, got {mode!r}")
    return out


def _instance_from_indices(pool: KnapsackSplit, idx: np.ndarray,
                           budget_tightness: float) -> KnapsackSplit:
    """Sub-sample ``m`` stakeholders and recompute the per-resource budget for
    the sub-population (same convention as ``MultiDimKnapsackTask.sample_batch``)."""
    sub_cost = pool.cost[idx].astype(np.float64, copy=True)
    sub_budgets = float(budget_tightness) * sub_cost.sum(axis=0)   # (n_resources,)
    return KnapsackSplit(
        x=pool.x[idx].astype(np.float64, copy=True),
        y=pool.y[idx].astype(np.float64, copy=True),
        cost=sub_cost,
        groups=pool.groups[idx].astype(np.int64, copy=True),
        budgets=sub_budgets,
    )


def _draw_bootstrap(pool_n: int, m: int, n_instances: int,
                    rng: np.random.Generator) -> List[np.ndarray]:
    """Independent draws of ``m`` distinct stakeholders; overlap allowed across
    instances (lets ``n_instances`` exceed pool / m)."""
    if m > pool_n:
        raise ValueError(f"m={m} exceeds pool slice of {pool_n}.")
    return [rng.choice(pool_n, size=m, replace=False) for _ in range(n_instances)]


def _draw_disjoint(pool_n: int, m: int, n_instances: int,
                   rng: np.random.Generator) -> List[np.ndarray]:
    """Disjoint partition: no stakeholder reused across instances (genuinely
    i.i.d. instances). Requires ``n_instances * m <= pool_n``."""
    if n_instances * m > pool_n:
        raise ValueError(
            f"disjoint partition needs {n_instances * m} stakeholders but the "
            f"pool slice has {pool_n}; raise pool_multiple or use bootstrap."
        )
    perm = rng.permutation(pool_n)
    return [perm[k * m:(k + 1) * m] for k in range(n_instances)]


def make_md_instances(
    *,
    m: int = 100,
    n_train: int = 50,
    n_val: int = 30,
    n_test: int = 30,
    n_features: int = 5,
    n_resources: int = 3,
    alpha_fair: float = 2.0,
    imbalance: float = 0.4,
    imbalance_mode: str = "both",
    n_groups: int = 2,
    poly_degree: int = 2,
    snr: float = 5.0,
    cost_mean: float = 1.0,
    cost_std: float = 0.2,
    budget_tightness: float = 0.35,
    group_ratio: float = 0.5,
    fairness_type: str = "mad",
    decision_mode: str = "group",
    instance_sampling: str = "bootstrap",
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    pool_multiple: int = 12,
    seed: int = 0,
) -> MDInstanceData:
    """Build train + test MD optimization instances sharing one benefit mapping.

    Parameters mirror ``MultiDimKnapsackTask`` plus the multi-instance knobs
    (``m``, ``n_train``, ``n_test``, ``instance_sampling``). One pool is generated
    with ``seed`` (fixing ``W``); the pool is split stakeholder-disjoint into
    train / val / test slices (``val_fraction`` / ``test_fraction``); instances are
    drawn from each slice. The val instances feed the HP-tuning early-stopping
    signal (§0.5) and are unused by the fixed-steps bound curve.
    """
    inst_sampling = str(instance_sampling).strip().lower()
    if inst_sampling not in {"bootstrap", "disjoint"}:
        raise ValueError("instance_sampling must be 'bootstrap' or 'disjoint'.")

    # Pool large enough that each disjoint slice can supply its instances; for
    # bootstrap a generous pool keeps cross-instance overlap modest. Generating
    # the pool is cheap (sampling only — no conic solve).
    train_frac = 1.0 - val_fraction - test_fraction
    train_need = n_train * m if inst_sampling == "disjoint" else m
    val_need = n_val * m if inst_sampling == "disjoint" else m
    test_need = n_test * m if inst_sampling == "disjoint" else m
    pool_n = int(max(pool_multiple * m, 4000,
                     int(np.ceil(train_need / train_frac)),
                     int(np.ceil(val_need / val_fraction)),
                     int(np.ceil(test_need / test_fraction))))

    gen_task = MultiDimKnapsackTask(
        n_samples_train=pool_n, n_samples_val=1, n_samples_test=1,
        n_features=int(n_features), n_resources=int(n_resources),
        scenario="alpha_fair", alpha_fair=float(alpha_fair),
        poly_degree=int(poly_degree), snr=float(snr),
        cost_mean=float(cost_mean), cost_std=float(cost_std),
        budget_tightness=float(budget_tightness), fairness_type=str(fairness_type),
        group_ratio=float(group_ratio), decision_mode=str(decision_mode),
        n_groups=int(n_groups),
        **imbalance_params(float(imbalance), str(imbalance_mode)),
    )
    gen_task.generate_data(int(seed))
    pool: KnapsackSplit = gen_task._splits["train"]   # size pool_n, fixed W

    # Stakeholder-disjoint train/test slice of the pool (so test instances are
    # NEW stakeholders under the SAME mapping — generalization, not memorization).
    rng = np.random.default_rng(int(seed) * 100_003 + 7)
    perm = rng.permutation(pool_n)
    n_test_pool = int(round(test_fraction * pool_n))
    n_val_pool = int(round(val_fraction * pool_n))
    test_slice = perm[:n_test_pool]
    val_slice = perm[n_test_pool:n_test_pool + n_val_pool]
    train_slice = perm[n_test_pool + n_val_pool:]

    draw = _draw_disjoint if inst_sampling == "disjoint" else _draw_bootstrap
    train_local = draw(len(train_slice), m, n_train, rng)
    val_local = draw(len(val_slice), m, n_val, rng)
    test_local = draw(len(test_slice), m, n_test, rng)

    train_instances = [
        _instance_from_indices(pool, train_slice[ix], budget_tightness) for ix in train_local
    ]
    val_instances = [
        _instance_from_indices(pool, val_slice[ix], budget_tightness) for ix in val_local
    ]
    test_instances = [
        _instance_from_indices(pool, test_slice[ix], budget_tightness) for ix in test_local
    ]

    meta: Dict[str, Any] = {
        "m": int(m),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "feature_dim": int(n_features),
        "n_resources": int(n_resources),
        "alpha_fair": float(alpha_fair),
        "imbalance": float(imbalance),
        "imbalance_mode": str(imbalance_mode),
        "n_groups": int(n_groups),
        "poly_degree": int(poly_degree),
        "snr": float(snr),
        "budget_tightness": float(budget_tightness),
        "group_ratio": float(group_ratio),
        "instance_sampling": inst_sampling,
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "pool_n": int(pool_n),
        "n_train_pool": int(len(train_slice)),
        "n_val_pool": int(len(val_slice)),
        "n_test_pool": int(len(test_slice)),
        "group1_share_pool": float(np.mean(pool.groups == 1)),
        "seed": int(seed),
    }
    return MDInstanceData(
        train=train_instances, val=val_instances, test=test_instances, meta=meta,
    )
