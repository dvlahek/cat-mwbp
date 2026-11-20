# Additional CAT adjoint controls

These scripts supplement the reference CAT 1.0.0 experiments for *Local Second-Order Adjoint Dynamics for Implicit Neural Networks*. They are not part of the original v1.0.0 release. All commands are run **from the repository root** with the packages in `requirements.txt` installed. Source scripts live in `experiments/`.

## 1. Trained parameter-count scaling

```bash
python experiments/run_revision_scaling.py --study width --datasets breast_cancer,digits,synthetic_large --widths 32,64,128,256,512 --seeds 3 --epochs 20 --out results/revision_width
```

Output: `width_runs.csv`, `width_pairs.csv` and exploratory per-dataset figures. To reproduce the caption-free two-panel article figure:

```bash
python experiments/make_cat_scaling_compact.py --input results/revision_width/width_pairs.csv --output results/revision_width/cat_parameter_scaling_compact
```

The experiment changes the **number of learned parameters** and the implicit hidden-state dimension of a compact structured recurrent classifier. It reports AR/CAT local transpose-Jacobian action ratios and held-out accuracy. The empirical results do **not** establish large-DEQ scalability or isolate parameter count from the changing Jacobian spectrum. `elapsed_run_s_including_audit` includes dataset loading, training, and post-training eigenvalue auditing; it is not isolated solver runtime.

## 2. Matched finite-tolerance analysis (manuscript sensitivity figure)

```bash
python experiments/cat_tolerance_study.py --rhos 0.8,0.95,0.99 --tolerances 1e-3,1e-4,1e-5,1e-6,1e-7 --seeds 5 --side 12 --out results/cat_tolerance
python experiments/make_cat_tolerance_figure.py --input results/cat_tolerance/tolerance_pairs.csv --output results/cat_tolerance/cat_tolerance_sensitivity
```

The script matches the deterministic right-hand-side and symmetric, non-bipartite graph construction in `experiments/run_implicit_spectral_suite.py`. Outputs are `tolerance_runs.csv`, `tolerance_pairs.csv` and `tolerance_summary.csv`, plus PDF/PNG figures. Both **oracle** and **conservative-bound** spectral calibration are reported. The exact direct solve is used solely as an offline error reference and stopping criterion, not as a practical distributed termination mechanism. The observed finite-tolerance ratio depends on initial conditions, the eigenvalue distribution and requested accuracy; the measured/theoretical factor near 0.82 is not universal.

## 3. Preliminary tolerance script (different diagnostic)

`run_revision_scaling.py --study tolerance` is an earlier complementary sensitivity control with independently generated right-hand sides and a requirement on both adjoint and gradient errors. It is **not** the source of the article's matched finite-tolerance figure. Prefer `cat_tolerance_study.py` when reproducing that figure.

## 4. Scope and archival notes

Each source file is independent of the original reference results. Generated results belong under `results/` and should not overwrite the original v1.0.0 outputs. Captions are authored in the LaTeX manuscript; the article figure scripts do not embed caption text or a revision label. A final journal submission must archive the exact code and numerical result version in a DOI-assigning repository, such as Zenodo, and cite the resulting DOI in Code Availability. The next unresolved empirical requirement is evaluation on a more representative implicit architecture with a clearly specified local-computation and spectral-calibration regime.
