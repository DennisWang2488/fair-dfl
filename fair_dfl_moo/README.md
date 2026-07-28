# `fair_dfl` — the FDFL algorithm package

This is the reusable algorithm library behind the paper, shipped **as source** so
the method itself can be read and checked. Installing the reproduction package
(`pip install -r requirements.txt` at the repository root) installs this directory.

Everything the paper's method section describes lives here — the reproduction
pipeline in [`../experiments/`](../experiments/) only orchestrates runs over it.

## Where the paper's pieces are

| Paper concept | Module |
|---|---|
| Prediction, decision, and fairness losses | `src/fair_dfl/losses.py` |
| Method definition (`use_dec`/`use_pred`/`use_fair`, μ, λ-continuation, gradient merge) | `src/fair_dfl/training/method_spec.py` |
| Unified training loop | `src/fair_dfl/training/loop.py` |
| Multi-objective gradient handlers (PCGrad, MGDA, NashMTL) | `src/fair_dfl/algorithms/mo_handler.py` |
| FPLG guidance-weight schedule (Jeon et al. 2025, eq. 4) | `src/fair_dfl/schedules.py` |
| Healthcare α-fair allocation task | `src/fair_dfl/tasks/medical_resource_allocation.py` |
| Multidimensional-knapsack task | `src/fair_dfl/tasks/md_knapsack.py` |
| Decision-gradient backends | `src/fair_dfl/decision/strategies/` |
| Experiment entry point | `src/fair_dfl/runner.py` |

## Decision-gradient backends

`src/fair_dfl/decision/strategies/` holds several backends. The two the paper's
experiments use are:

- **`analytic.py`** — closed-form decision gradients for the healthcare α-fair
  allocation. Used for all healthcare results.
- **`cvxpylayers.py`** — differentiable conic layer (cvxpylayers → diffcp → SCS).
  Used for all multidimensional-knapsack results.

The remaining strategy modules are part of the library but are **not**
selected by the reproduction pipeline: `--md-backend` accepts only
`cvxpylayers`.

## Solver note

The knapsack layer differentiates through SCS via diffcp. The reproduction
pipeline passes `{"eps": 1e-6, "max_iters": 10000}` (see
`../experiments/configs.py`) rather than the library default `eps=1e-8`, which SCS
cannot reach on the α=0.5 instances. The healthcare path is closed-form and needs
no solver.

## License

MIT — see [`LICENSE`](LICENSE).
