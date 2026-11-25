# Anderson recalibration and CPU reference v2

The revised comparison tunes Anderson-m on the same five seeds per frozen-adjoint cell, over memories {3,5,10,20,50}, mixing {0.5,0.8,1.0}, and regularization {1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2}. All six selected settings have positive regularization 1e-14, the *lower endpoint* of the tested search; no global optimum is claimed. The original reference_results/implicit/anderson_best_config.csv and earlier solver_cost_summary.csv are historical and superseded for Anderson.

- anderson_best_config_extended.csv: revised per-cell settings and action counts.
- solver_cost_v2_per_seed.csv: single-environment tanh rho=0.95 timing and exact-error outputs (5 seeds, 9 repetitions each, one BLAS thread).
- solver_cost_v2_summary.csv: medians across seeds and active core-array estimates for the revised Supplementary Table S5.

The reference GMRES function preallocates an n x (n+1) Arnoldi basis and (n+1) x n Hessenberg array (326.25 KiB at n=144), while only n x (k+1) plus (k+1) x k entries are needed at observed action count k=24 (32.8125 KiB). The revised *core-state* column counts the latter; it does **not** describe all allocated NumPy arrays. For Anderson m=20, core stored history is three n x m arrays (67.50 KiB), excluding temporary column stacks, the KKT matrix and BLAS workspace. Core-array figures are neither peak RAM nor wall-clock memory measurements.

Reference actions for the tanh rho=0.95 cell: AR=160, CAT=39, GMRES=24, Broyden=29 and Anderson=25. CPU median times in milliseconds: 1.911, 0.474, 1.867, 1.476 and 1.627 respectively (seed-level medians, then median across five seeds). They are specific to one CPU environment; compare methods within that environment only.

Run `python experiments/run_solver_cost_audit.py --anderson-config revision_results/anderson_best_config_extended.csv --out results/solver_cost_v2_reproduced --repeats 9 --blas-threads 1` to reproduce the *protocol*; absolute CPU times depend on hardware. Run `python experiments/run_nonlocal_solver_reference.py` to repeat the expanded parameter search. The old first-order and second-order CAT/AR comparisons are not affected.
