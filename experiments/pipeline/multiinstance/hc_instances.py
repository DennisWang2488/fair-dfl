"""Healthcare multi-instance sampler (``make_hc_instances``).

Implements the §4 sampling design of ``new_experiment_design.md``:

- Patient-disjoint **80/20** split of the M-patient cohort into stratified
  train/test pools (stratified on the protected attribute ``race`` only —
  never on the prediction target; see §4.1).
- Each *instance* is a draw of ``m`` patients with a race quota
  ``ceil(pi * m)`` from the minority stratum (proportionate stratified
  sampling, §4.3), drawn **without replacement within an instance**.
- Per-instance budget ``Q_k = budget_frac * sum_i c_i`` (§4.2).

Two cross-instance sampling regimes:
- ``"bootstrap"`` (Run A): independent draws, overlap allowed, lets
  ``n_train`` exceed pool/``m``.
- ``"disjoint"`` (Run B): a disjoint partition of the pools, no patient reuse,
  so the instances are genuinely i.i.d. for the finite-N bound curve (§4.4).

Feature/label transforms reuse ``MedicalResourceAllocationTask`` so the
per-instance allocation math is identical to the published single-cohort code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from fair_dfl.tasks.medical_resource_allocation import MedicalResourceAllocationTask


@dataclass
class HCInstance:
    """One optimization instance (a cohort with its own budget)."""

    x: np.ndarray            # (m, d) standardized features
    y: np.ndarray            # (m,)   true benefit r_i
    cost: np.ndarray         # (m,)   per-patient cost c_i
    race: np.ndarray         # (m,)   protected group (0/1)
    budget: float            # Q_k = budget_frac * sum_i c_i
    patient_ids: np.ndarray  # (m,)   row indices into the source cohort (provenance)


@dataclass
class HCInstanceData:
    train: List[HCInstance]
    test: List[HCInstance]
    val: List[HCInstance] = field(default_factory=list)   # held-out for HP early-stop (§0.5)
    meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Cohort loading (mirrors MedicalResourceAllocationTask.generate_data)
# ----------------------------------------------------------------------

def _load_cohort_arrays(cohort: "str | Path | pd.DataFrame") -> Dict[str, np.ndarray]:
    """Return raw (unscaled) feature/label/cost/race arrays for the full cohort.

    Replicates the exact transforms in
    ``MedicalResourceAllocationTask.generate_data`` so that per-instance
    regret/fairness numbers are directly comparable to the single-cohort runs:
        r_i = max(benefit * 100, 1) + 1
        c_i = max(cost_t_capped * 10, 1)
    """
    if isinstance(cohort, pd.DataFrame):
        df = cohort
    else:
        path = Path(cohort)
        if not path.exists():
            # Allow repo-relative paths (matches the task's _resolve_data_csv).
            repo_root = Path(__file__).resolve().parents[3]
            alt = (repo_root / cohort).resolve()
            if not alt.exists():
                raise FileNotFoundError(f"Cohort CSV not found: {cohort}")
            path = alt
        df = pd.read_csv(path)

    feature_cols = MedicalResourceAllocationTask._get_all_features(df)
    if not feature_cols:
        raise ValueError("No medical features selected from cohort.")

    x_all = df[feature_cols].to_numpy(dtype=float)
    true_benefit = np.maximum(df["benefit"].to_numpy(dtype=float) * 100.0, 1.0) + 1.0
    cost = np.maximum(df["cost_t_capped"].to_numpy(dtype=float) * 10.0, 1.0)
    race = df["race"].to_numpy(dtype=int)

    return {
        "x": x_all,
        "y": true_benefit,
        "cost": cost,
        "race": race,
        "feature_cols": np.asarray(feature_cols, dtype=object),
    }


def _stratified_pool_split(
    race: np.ndarray, val_fraction: float, test_fraction: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Patient-disjoint train / val / test split, stratified on race (§0.5).

    Splits each race stratum independently into test / val / train blocks so all
    three pools preserve the population minority share pi. The val pool feeds the
    HP-tuning early-stopping signal; carving it out of the patient pool (rather
    than re-drawing from train) keeps val patient-disjoint from train even under
    the bootstrap regime.
    """
    train_idx: List[np.ndarray] = []
    val_idx: List[np.ndarray] = []
    test_idx: List[np.ndarray] = []
    for g in np.unique(race):
        g_idx = np.flatnonzero(race == g)
        rng.shuffle(g_idx)
        n = len(g_idx)
        n_test = min(max(int(round(test_fraction * n)), 1), n - 2)
        n_val = min(max(int(round(val_fraction * n)), 1), n - n_test - 1)
        test_idx.append(g_idx[:n_test])
        val_idx.append(g_idx[n_test:n_test + n_val])
        train_idx.append(g_idx[n_test + n_val:])
    train = np.sort(np.concatenate(train_idx))
    val = np.sort(np.concatenate(val_idx))
    test = np.sort(np.concatenate(test_idx))
    return {"train": train, "val": val, "test": test}


def _draw_instances_bootstrap(
    *,
    pool_idx: np.ndarray,
    race: np.ndarray,
    n_instances: int,
    n_min: int,
    n_maj: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Bootstrap draws: each instance samples without replacement *within*
    itself, but instances are independent (overlap allowed across instances)."""
    minority = pool_idx[race[pool_idx] == 1]
    majority = pool_idx[race[pool_idx] == 0]
    if n_min > len(minority) or n_maj > len(majority):
        raise ValueError(
            f"Instance size exceeds pool stratum: need {n_min} minority / {n_maj} "
            f"majority but pool has {len(minority)} / {len(majority)}."
        )
    out: List[np.ndarray] = []
    for _ in range(n_instances):
        s_min = rng.choice(minority, size=n_min, replace=False)
        s_maj = rng.choice(majority, size=n_maj, replace=False)
        out.append(np.concatenate([s_min, s_maj]))
    return out


def _draw_instances_disjoint(
    *,
    pool_idx: np.ndarray,
    race: np.ndarray,
    n_instances: int,
    n_min: int,
    n_maj: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Disjoint partition: shuffle each stratum once and hand out consecutive,
    non-overlapping blocks. No patient is reused across instances (§4.4)."""
    minority = pool_idx[race[pool_idx] == 1].copy()
    majority = pool_idx[race[pool_idx] == 0].copy()
    rng.shuffle(minority)
    rng.shuffle(majority)
    need_min = n_instances * n_min
    need_maj = n_instances * n_maj
    if need_min > len(minority) or need_maj > len(majority):
        raise ValueError(
            f"Disjoint partition infeasible: need {need_min} minority / {need_maj} "
            f"majority for {n_instances} instances but pool has "
            f"{len(minority)} / {len(majority)}. Reduce n_train, m, or use bootstrap."
        )
    out: List[np.ndarray] = []
    for k in range(n_instances):
        s_min = minority[k * n_min:(k + 1) * n_min]
        s_maj = majority[k * n_maj:(k + 1) * n_maj]
        out.append(np.concatenate([s_min, s_maj]))
    return out


def make_hc_instances(
    cohort: "str | Path | pd.DataFrame",
    *,
    m: int = 5000,
    n_train: int = 50,
    n_val: int = 30,
    n_test: int = 30,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    budget_frac: float = 0.30,
    stratify_by: str = "race",
    instance_sampling: str = "bootstrap",
    seed: int = 0,
) -> HCInstanceData:
    """Build train + test optimization instances for the multi-instance HC run.

    Parameters
    ----------
    cohort : path / DataFrame
        The M-patient cohort (``data/data_processed.csv``).
    m : int
        Patients per instance.
    n_train, n_val, n_test : int
        Number of train / val / test instances. The val instances feed the
        HP-tuning early-stopping signal (§0.5); they are unused by the fixed-steps
        Run-B bound curve.
    val_fraction, test_fraction : float
        Fractions of patients reserved (patient-disjoint) for the val / test pools.
    budget_frac : float
        Per-instance budget fraction Q_k = budget_frac * sum_i c_i.
    stratify_by : str
        Stratification attribute (only ``"race"`` is supported / safe — see §4.1).
    instance_sampling : {"bootstrap", "disjoint"}
        Cross-instance regime. Test instances always bootstrap from the (already
        disjoint) test pool.
    seed : int
        Controls the pool split and the instance draws.

    Returns
    -------
    HCInstanceData with ``.train`` / ``.test`` instance lists and ``.meta``.
    """
    if stratify_by != "race":
        raise ValueError(
            "Only stratify_by='race' is supported: stratifying on a function of "
            "the target (e.g. cost_avoidable) biases the learned predictor (§4.1)."
        )
    instance_sampling = str(instance_sampling).strip().lower()
    if instance_sampling not in {"bootstrap", "disjoint"}:
        raise ValueError("instance_sampling must be 'bootstrap' or 'disjoint'.")

    raw = _load_cohort_arrays(cohort)
    x_all, y_all, cost_all, race_all = raw["x"], raw["y"], raw["cost"], raw["race"]
    n_total = x_all.shape[0]

    pi_population = float(np.mean(race_all == 1))
    n_min = int(math.ceil(pi_population * m))
    n_maj = int(m - n_min)
    if n_maj <= 0:
        raise ValueError(f"m={m} too small for minority quota {n_min}.")

    rng = np.random.default_rng(int(seed) * 100_003 + 7)
    pools = _stratified_pool_split(
        race_all, val_fraction=val_fraction, test_fraction=test_fraction, rng=rng,
    )
    train_pool, val_pool, test_pool = pools["train"], pools["val"], pools["test"]

    # Standardize features using the TRAIN POOL statistics only (no leakage).
    mean = x_all[train_pool].mean(axis=0, keepdims=True)
    std = x_all[train_pool].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    x_scaled = (x_all - mean) / std

    draw = _draw_instances_disjoint if instance_sampling == "disjoint" else _draw_instances_bootstrap
    train_idx_lists = draw(
        pool_idx=train_pool, race=race_all, n_instances=n_train,
        n_min=n_min, n_maj=n_maj, rng=rng,
    )
    # Val instances: same cross-instance regime as train, drawn from the
    # patient-disjoint val pool so the early-stop signal never overlaps train.
    val_idx_lists = draw(
        pool_idx=val_pool, race=race_all, n_instances=n_val,
        n_min=n_min, n_maj=n_maj, rng=rng,
    )
    # Test instances are always bootstrap draws from the held-out test pool
    # (the pool is already patient-disjoint from train/val by construction).
    test_idx_lists = _draw_instances_bootstrap(
        pool_idx=test_pool, race=race_all, n_instances=n_test,
        n_min=n_min, n_maj=n_maj, rng=rng,
    )

    def build(idx: np.ndarray) -> HCInstance:
        c = cost_all[idx].astype(np.float64, copy=True)
        return HCInstance(
            x=x_scaled[idx].astype(np.float64, copy=True),
            y=y_all[idx].astype(np.float64, copy=True),
            cost=c,
            race=race_all[idx].astype(np.int64, copy=True),
            budget=float(budget_frac * c.sum()),
            patient_ids=idx.astype(np.int64, copy=True),
        )

    train_instances = [build(ix) for ix in train_idx_lists]
    val_instances = [build(ix) for ix in val_idx_lists]
    test_instances = [build(ix) for ix in test_idx_lists]

    meta: Dict[str, Any] = {
        "m": int(m),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "feature_dim": int(x_scaled.shape[1]),
        "n_total": int(n_total),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "budget_frac": float(budget_frac),
        "instance_sampling": instance_sampling,
        "stratify_by": stratify_by,
        "pi_population": pi_population,
        "pi_train_pool": float(np.mean(race_all[train_pool] == 1)),
        "pi_val_pool": float(np.mean(race_all[val_pool] == 1)),
        "pi_test_pool": float(np.mean(race_all[test_pool] == 1)),
        "n_minority_quota": int(n_min),
        "n_majority_quota": int(n_maj),
        "n_train_pool": int(len(train_pool)),
        "n_val_pool": int(len(val_pool)),
        "n_test_pool": int(len(test_pool)),
        "n_train_pool_minority": int(np.sum(race_all[train_pool] == 1)),
        "n_val_pool_minority": int(np.sum(race_all[val_pool] == 1)),
        "n_test_pool_minority": int(np.sum(race_all[test_pool] == 1)),
        "seed": int(seed),
        "feature_cols": raw["feature_cols"],
    }
    return HCInstanceData(
        train=train_instances, val=val_instances, test=test_instances, meta=meta,
    )
