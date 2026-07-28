#!/usr/bin/env bash
# Reproduce the FDFL paper tables and figures from scratch.
#
#   bash reproduce.sh smoke   # fast end-to-end check (reduced grid, ~minutes) — verifies the toolchain
#   bash reproduce.sh full    # full reproduction of the paper grid (long-running, many CPU-hours)
#   bash reproduce.sh tables  # (re)build tables + figures from an existing results/ dir
#
# No precomputed results are shipped; `smoke`/`full` generate everything under results/.
# The annotated, cell-by-cell version of `full` lives in reproduce_all.ipynb.
set -euo pipefail
MODE="${1:-smoke}"
PY="${PYTHON:-python}"
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

HC_FINAL="results/healthcare/main_v5_multiinstance/grid/final"
HC_TABLES="results/healthcare/main_v5_multiinstance/grid/tables"
MD_FINAL="results/md_knapsack/main_v6_rowsum/grid/final"
MD_TABLES="results/md_knapsack/main_v6_rowsum/grid/tables"

aggregate () {
  "$PY" -m experiments.pipeline.aggregators.slice_grid_tables --task hc --grid "$HC_FINAL" --out "$HC_TABLES"
  "$PY" -m experiments.pipeline.aggregators.slice_grid_tables --task md --grid "$MD_FINAL" --out "$MD_TABLES"
}
build_tables_and_figures () {
  # Main-text assets (the manuscript's Sec 5.3 tables/figures)
  "$PY" -m experiments.pipeline.analyzers.build_main_tables
  "$PY" -m experiments.pipeline.analyzers.build_appendix_tables
  "$PY" -m experiments.pipeline.plotters.plot_main_figures
  # Online-supplement assets (complete result tables + figure set)
  "$PY" -m experiments.pipeline.analyzers.build_supplement_tables
  "$PY" -m experiments.pipeline.plotters.plot_paper_findings
  "$PY" -m experiments.pipeline.plotters.plot_healthcare
  "$PY" -m experiments.pipeline.plotters.plot_knapsack
}

case "$MODE" in
  smoke)
    echo "=== smoke: tiny end-to-end (FDFL, FPTO) ==="
    [ -f data/data_processed.csv ] || "$PY" data/prepare_data.py
    "$PY" -m experiments.pipeline.run_hp_tuning --hc --md --methods FDFL,FPTO --quick --overwrite
    "$PY" -m experiments.pipeline.run_hp_final  --hc --md --methods FDFL,FPTO --quick --overwrite \
      --hc-hp results/healthcare/main_v5_multiinstance/hp/m5000/best_hp.csv \
      --md-hp results/md_knapsack/main_v6_rowsum/hp/mlp64x2/best_hp.csv
    echo "smoke ok — see results/*/grid/{hp,final}/"
    ;;
  full)
    echo "=== full reproduction (long-running) ==="
    [ -f data/data_processed.csv ] || "$PY" data/prepare_data.py
    MAX_WORKERS="${MAX_WORKERS:-$(getconf _NPROCESSORS_ONLN)}"
    # ------------------------------------------------------------------
    # The commands below are a faithful transcription of the Colab runs
    # that produced the published grids (scripts/colab/capacity_alpha/
    # grid_hc, hc6_ncurve, grid_md, md_groups), in order.
    # The MD decisions are differentiated with the `cvxpylayers` conic
    # layer (open SCS/ECOS solvers, no MOSEK); evaluation uses exact
    # conic solves.
    # ------------------------------------------------------------------
    # --- Healthcare grid (capacity ladder x alpha x fairness measures) ---
  "$PY" experiments/pipeline/run_hp_tuning.py --hc --alphas 0.5,2.0 --fairness-type mad --arch log_linear --methods FPTO,SAA --hc-out results/healthcare/main_v5_multiinstance/grid/hp/log_linear_mad --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 0.5,2.0 --fairness-type mad --arch log_linear --methods FPTO,SAA --seeds 11,22,33,44,55 --lambdas 0,1 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/log_linear_mad/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/log_linear_mad_n50 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --hc --alphas 0.5,2.0 --fairness-type mad --arch mlp --hidden 16 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --hc-out results/healthcare/main_v5_multiinstance/grid/hp/mlp16_mad --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 0.5,2.0 --fairness-type mad --arch mlp --hidden 16 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/mlp16_mad/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/mlp16_mad_n50 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --hc --alphas 2.0 --fairness-type mad --arch mlp --hidden 32 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --hc-out results/healthcare/main_v5_multiinstance/grid/hp/mlp32_mad --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 2.0 --fairness-type mad --arch mlp --hidden 32 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/mlp32_mad/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/mlp32_mad_n50 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --hc --alphas 0.5,1.5,2.0,4.0 --fairness-type mad --arch mlp --hidden 64 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --hc-out results/healthcare/main_v5_multiinstance/grid/hp/mlp64_mad --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 0.5,1.5,2.0,4.0 --fairness-type mad --arch mlp --hidden 64 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/mlp64_mad/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/mlp64_mad_n50 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 2.0 --fairness-type mad --arch mlp --hidden 64 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --n-train 10 --n-train-max 100 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/mlp64_mad/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/mlp64_mad_n10 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 2.0 --fairness-type mad --arch mlp --hidden 64 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --n-train 20 --n-train-max 100 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/mlp64_mad/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/mlp64_mad_n20 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 2.0 --fairness-type mad --arch mlp --hidden 64 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/mlp64_mad/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/mlp64_mad_n100 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --hc --alphas 2.0 --fairness-type mad --arch mlp --hidden 128 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --hc-out results/healthcare/main_v5_multiinstance/grid/hp/mlp128_mad --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 2.0 --fairness-type mad --arch mlp --hidden 128 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/mlp128_mad/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/mlp128_mad_n50 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --hc --alphas 2.0 --fairness-type dp --arch mlp --hidden 64 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --hc-out results/healthcare/main_v5_multiinstance/grid/hp/mlp64_dp --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 2.0 --fairness-type dp --arch mlp --hidden 64 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/mlp64_dp/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/mlp64_dp_n50 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --hc --alphas 2.0 --fairness-type w2_dp --arch mlp --hidden 64 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --hc-out results/healthcare/main_v5_multiinstance/grid/hp/mlp64_w2_dp --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --hc --alphas 2.0 --fairness-type w2_dp --arch mlp --hidden 64 --n-layers 2 --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --hc-hp results/healthcare/main_v5_multiinstance/grid/hp/mlp64_w2_dp/m5000/best_hp.csv --hc-out results/healthcare/main_v5_multiinstance/grid/final/mlp64_w2_dp_n50 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/aggregators/slice_grid_tables.py --task hc --grid results/healthcare/main_v5_multiinstance/grid/final --out results/healthcare/main_v5_multiinstance/grid/tables
    # --- Healthcare training-set-size curve (N axis) ---
  "$PY" experiments/pipeline/run_hc_multiinstance.py --run-a --instance-sampling disjoint --m 1200 --n-train 4 8 16 24 32 --n-test 30 --alphas 2.0 --fairness mad --methods FDFL-Scal --seeds 11 22 33 44 55 --lambdas 1.0 --steps 200 --arch mlp --hidden-dim 64 --lr 3e-3 --out-root results/healthcare/main_v5_multiinstance/grid/hc6 --max-workers "$MAX_WORKERS"
    # --- MD knapsack grid (capacity ladder x imbalance) ---
  "$PY" experiments/pipeline/run_hp_tuning.py --md --alphas 0.5,2.0 --arch linear --imbalance 0.6 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --md-out results/md_knapsack/main_v6_rowsum/grid/hp/linear_imb0.6 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 0.5,2.0 --arch linear --imbalance 0.6 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/linear_imb0.6/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/linear_imb0.6 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --md --alphas 2.0 --arch mlp --hidden 16 --n-layers 1 --imbalance 0.6 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --md-out results/md_knapsack/main_v6_rowsum/grid/hp/mlp16_imb0.6 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 2.0 --arch mlp --hidden 16 --n-layers 1 --imbalance 0.6 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/mlp16_imb0.6/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/mlp16_imb0.6 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --md --alphas 0.5,1.5,2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.6 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --md-out results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.6 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 0.5,1.5,2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.6 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.6/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/mlp64_imb0.6 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --md --alphas 2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.0 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --md-out results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.0 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.0 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.0/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/mlp64_imb0.0 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --md --alphas 2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.2 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --md-out results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.2 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.2 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.2/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/mlp64_imb0.2 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --md --alphas 2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.4 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --md-out results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.4 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.4 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.4/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/mlp64_imb0.4 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_tuning.py --md --alphas 2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.8 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --md-out results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.8 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.8 --md-backend cvxpylayers --methods FPTO,SAA,WDRO,FDFL,FDFL-0.1,FDFL-0.5,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.8/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/mlp64_imb0.8 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/aggregators/slice_grid_tables.py --task md --grid results/md_knapsack/main_v6_rowsum/grid/final --out results/md_knapsack/main_v6_rowsum/grid/tables
    # --- MD K=4 groups ---
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 2.0 --arch linear --imbalance 0.6 --n-groups 4 --methods FPTO,SAA,WDRO,FDFL,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-backend cvxpylayers --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/linear_imb0.6/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/linear_imb0.6_K4 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 2.0 --arch mlp --hidden 16 --n-layers 1 --imbalance 0.6 --n-groups 4 --methods FPTO,SAA,WDRO,FDFL,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-backend cvxpylayers --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/mlp16_imb0.6/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/mlp16_imb0.6_K4 --max-workers "$MAX_WORKERS"
  "$PY" experiments/pipeline/run_hp_final.py --md --alphas 2.0 --arch mlp --hidden 64 --n-layers 2 --imbalance 0.6 --n-groups 4 --methods FPTO,SAA,WDRO,FDFL,FDFL-Scal,FPLG,PCGrad,MGDA,NashMTL --seeds 11,22,33,44,55 --lambdas 0,1 --md-m 200 --md-n-train 50 --md-backend cvxpylayers --md-hp results/md_knapsack/main_v6_rowsum/grid/hp/mlp64_imb0.6/best_hp.csv --md-out results/md_knapsack/main_v6_rowsum/grid/final/mlp64_imb0.6_K4 --max-workers "$MAX_WORKERS"
    # --- NOT reproduced here: the preliminary MD predictor-class ladder ---
    # The supplement's `tab:supp-md-misspec` comes from a separate per-config
    # HP-search ablation (FPTO/FDFL only, 3 seeds) that is out of scope for this
    # package; it is labeled PRELIMINARY in the supplement. The table is skipped
    # automatically when its inputs are absent -- every other table and figure is
    # regenerated in full.
    aggregate
    build_tables_and_figures
    ;;
  tables)
    echo "=== build tables + figures from existing results/ ==="
    build_tables_and_figures
    ;;
  *)
    echo "usage: bash reproduce.sh [smoke|full|tables]"; exit 1;;
esac
echo "=== done (mode: $MODE) ==="
