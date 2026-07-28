# End-to-End Fairness Optimization with Fair Decision-Focused Learning


**Fair Decision-Focused Learning (FDFL)** trains predictors end-to-end through an
α-fair welfare allocation program, so that prediction error, prediction disparity,
and decision regret are optimized jointly. The reusable algorithm is the
`fair_dfl` package, shipped as source under [`fair_dfl_moo/`](fair_dfl_moo/);
everything under [`experiments/`](experiments/) is the pipeline that reproduces
the paper's two experiments — a single-resource healthcare allocation with
closed-form decision gradients, and a synthetic multi-resource allocation
differentiated with `cvxpylayers`.

Citation metadata is in [`CITATION.cff`](CITATION.cff).

## Layout

```
fair_dfl_moo/src/fair_dfl/   the algorithm (losses, training loop, decision
                             backends, MTL gradient handlers, tasks, models)
experiments/configs.py       method registry + training defaults
experiments/pipeline/        runners, aggregators, table/figure builders
data/prepare_data.py         builds data/data_processed.csv
paper_figures/               reference figures + aggregated metrics to diff against
reproduce_all.ipynb          annotated end-to-end reproduction
reproduce.sh                 CLI equivalent: smoke | full | tables
```

## Install

Python ≥ 3.10. Open-source solvers (SCS/ECOS) suffice; MOSEK/Gurobi are not
required.

```
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt        # builds ./fair_dfl_moo + pinned deps
```

## Data

The healthcare experiment uses the publicly released Obermeyer et al. sample. No
raw data is redistributed; build the processed file with

```
python data/prepare_data.py        # writes data/data_processed.csv
```

Source, license, and the derived-column formulas are in
[`data/README.md`](data/README.md). The multi-resource experiment is fully
synthetic and generated in code.

## Reproduce

```
bash reproduce.sh smoke    # fast end-to-end check on a reduced grid (minutes)
bash reproduce.sh full     # full reproduction from scratch (many CPU-hours)
bash reproduce.sh tables   # rebuild tables + figures from an existing results/
```

`reproduce_all.ipynb` runs the same steps cell by cell; its `QUICK` flag toggles
between the smoke grid and the full paper grid. The `full` mode is a faithful
transcription, in order, of the commands that produced the reported grids.

Outputs are produced, not shipped (all git-ignored): per-run and aggregated CSVs
to `results/`, main-text and appendix tables to `tables/`, figures to `figures/`.

Results reproduce within seed variation; a bit-for-bit match is not expected,
since the conic solvers' iterates depend on the solver build, BLAS, and thread
count. The qualitative orderings and the effect sizes the paper draws conclusions
from are what should reproduce.

## License

MIT — see [`LICENSE`](LICENSE).
