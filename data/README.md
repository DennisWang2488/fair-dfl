# Healthcare data

The healthcare experiment uses the publicly released, de-identified patient
sample from:

> Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019).
> *Dissecting racial bias in an algorithm used to manage the health of
> populations.* **Science**, 366(6464), 447–453.
> https://doi.org/10.1126/science.aax2342

Public data release (file `data/data_new.csv`, 48,784 patients):
**https://gitlab.com/labsysmed/dissecting-bias**

We do **not** redistribute the raw sample. Instead, [`prepare_data.py`](prepare_data.py)
downloads it and reconstructs the processed file used by the experiments.

## How to build `data_processed.csv`

```bash
python data/prepare_data.py
```

This downloads `data_new.csv` from the public repository and writes
`data/data_processed.csv` next to it. To additionally verify the reconstruction
against a reference copy:

```bash
python data/prepare_data.py --check /path/to/reference/data_processed.csv
```

All derived columns reproduce to machine precision (max abs error ~1e-16).

## Derived columns

The decision task ([`fair_dfl.tasks.medical_resource_allocation`]) consumes the
public feature columns plus three derived columns, all built deterministically
from public columns:

| Derived column | Formula |
|---|---|
| `race` | `1` if public `race == "black"` else `0` |
| `benefit` | `0.5 · minmax(gagne_sum_t) + 0.5 · minmax(log(1 + cost_avoidable_t))` |
| `cost_t_capped` | `min(cost_t, Q99(cost_t)) / Q99(cost_t)`, where `Q99(cost_t) = $78,585` |

The ground-truth benefit score `r_i` enters the task rescaled to `[2, 101]` via
`max(benefit · 100, 1) + 1` inside the task loader, matching the paper's
description.

## Note on the benefit construction (un-imputed avoidable cost)

The conference-paper text describes imputing avoidable cost savings for patients
without an observed value via nearest-neighbor matching. In the data actually
used for all reported results, the `benefit` score uses the **raw, released
`cost_avoidable_t` column without imputation** — for the ~73.6% of patients whose
released `cost_avoidable_t` is `0`, the cost-savings component `s_i` is `0` and
their benefit is driven by the chronic-illness component alone. This package
reproduces the experiments exactly as run, so `prepare_data.py` uses the raw
column. (The nearest-neighbor-imputed values exist in the original working file
as a separate, unused `avoidable_cost_mapped` column; they do not affect any
reported result.)
