# Causal Adjoint Transport (CAT)

Research code for *Local Second-Order Adjoint Dynamics for Implicit Neural Networks*.

CAT applies a fixed-coefficient, second-degree stationary iteration to the implicit adjoint system `(I - J^T) lambda = b`. The experiments compare local backward action counts, predictive outcomes and solver costs in structured contractive implicit networks.

## Installation and verification

Python with NumPy, SciPy, pandas, scikit-learn and Matplotlib:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/verify_revision_results.py
python experiments/verify_archived_audits.py
```

The graph and image experiments additionally require PyTorch, `ogb` (graph), `scikit-image` and `Pillow` (images). CPU timing and memory controls require `threadpoolctl` and `psutil`. Install matching versions appropriate for the machine, and document them when reproducing timings. Public datasets are downloaded separately. The three retrained spatial checkpoints are included in the audit archive.

## Reproduce the reported experiments

Run commands from the repository root. Paths below are output folders and are ignored by Git.

| Reported evidence | Code |
|---|---|
| Feed-forward CAT/AR comparison | `run_npl_suite.py`, `experiments/run_journal_suite.py`, `src/metric_wave/` |
| Symmetric implicit spectral benchmark | `experiments/run_implicit_spectral_suite.py` |
| Scheduled Chebyshev comparison | `experiments/run_chebyshev_corrected_audit.py` |
| Fixed-recurrent datasets and MNIST/CIFAR-10 | `experiments/run_implicit_dataset_suite.py` |
| Trained recurrent edges | `experiments/run_trainable_recurrent_control.py` |
| GMRES, Broyden, Anderson comparison and CPU cost | `experiments/run_nonlocal_solver_reference.py`, `experiments/run_solver_cost_audit.py` |
| Directed normal/nonnormal spectral controls | `experiments/run_directed_spectrum_control.py`, `experiments/run_complex_spectrum_oracle.py` |
| Learned-parameter scaling | `experiments/run_revision_scaling.py` |
| Finite-tolerance sensitivity | `experiments/cat_tolerance_study.py` |
| Full-graph `ogbn-arxiv` controlled IGNN-style layer | `experiments/run_ignn_ogbn_arxiv.py`, `experiments/run_ignn_postcheck.py` |
| Spatial implicit denoising, three training seeds | `experiments/run_spatial_implicit_denoising.py`, `experiments/summarize_spatial_denoising.py` |
| Manuscript-data figure generators | `experiments/make_cat_scaling_compact.py`, `experiments/make_cat_tolerance_figure.py` |

### Fast method sanity checks

```bash
python experiments/run_chebyshev_corrected_audit.py --quick --out results/chebyshev_smoke
python experiments/run_implicit_spectral_suite.py --profile quick --output results/implicit_smoke
python experiments/run_implicit_dataset_suite.py --profile quick --group core --output results/recurrent_smoke
```

For the full procedures use `--profile standard` or `--profile full` as appropriate. Feed-forward:

```bash
python run_npl_suite.py --profile full --jobs 2 --output results/feedforward
```

Spectral, scaling and solver-cost experiments:

```bash
python experiments/run_chebyshev_corrected_audit.py --out results/chebyshev_corrected
python experiments/cat_tolerance_study.py --rhos 0.8,0.95,0.99 --tolerances 1e-3,1e-4,1e-5,1e-6,1e-7 --seeds 5 --side 12 --out results/tolerance
python experiments/run_revision_scaling.py --study width --datasets breast_cancer,digits,synthetic_large --widths 32,64,128,256,512 --seeds 3 --epochs 20 --out results/revision_width
python experiments/run_solver_cost_audit.py --anderson-config revision_results/anderson_best_config_extended.csv --seeds 5 --repeats 9 --blas-threads 1 --out results/costs
```

Graph model (download `ogbn-arxiv` only from the official OGB dataset provider; official split):

```bash
python experiments/run_ignn_ogbn_arxiv.py --smoke --epochs 2 --width 16 --out results/ignn_smoke
python experiments/run_ignn_ogbn_arxiv.py --device cpu --epochs 20 --width 32 --rho 0.95 --seed 0 --data-root data/ogb --out results/ignn_s0
```

Repeat for seeds 1–2. To add the explicitly charged terminal residual evaluation at frozen checkpoints, run `experiments/run_ignn_postcheck.py` with `--checkpoint`, `--data-root` and `--out`. The opt-in `--trust-ogb-cache` workaround must only be used for locally verified OGB cache files.

Spatial denoising (repeat for seeds 0, 1 and 2):

```bash
python experiments/run_spatial_implicit_denoising.py --seed 0 --steps 300 --side 80 --patch 48 --eval-side 64 --audit-side 48 --width 24 --rho 0.95 --center-weight 0.35 --sigma 0.12 --lr 0.002 --threads 2 --repeats 3 --out results/spatial_denoising_s0
python experiments/summarize_spatial_denoising.py --root results --prefix spatial_denoising_s --out results/spatial_denoising_summary
```

## Results and archived data

Across 240 matched spectral cases, Chebyshev uses fewer Jacobian-transpose actions than CAT in 164 cases, ties in 42 and uses more in 34 (median Chebyshev/CAT ratio 0.913). CAT uses fixed coefficients; Chebyshev uses scheduled coefficients without extra messages in the synchronous implementation. GMRES uses fewer actions in the large applications but requires global Krylov algebra.

- [reference_results/](reference_results/) contains the feed-forward results, recurrent classification summaries, spectral diagnostics and solver comparisons.
- [revision_results/](revision_results/) contains the Chebyshev comparisons, Anderson configurations, CPU cost measurements, width-scaling data and graph and spatial audits.
- [Solver configurations and CPU measurements](revision_results/README_anderson_v2.md) documents Table 3 and Supplementary Table S5.
- [MNIST and CIFAR-10 adjoint comparisons](revision_results/README_vision_chebyshev.md) documents the frozen-model oracle audit.

### Frozen-model audits
- `experiments/recheck_fine_budget_certification.py` reloads a **frozen** `ogbn-arxiv` or spatial-denoising checkpoint and checks the contraction certificate at every iteration for AR, CAT and Chebyshev. This is an offline analysis; the fixed-budget results remain the comparison in Figure 4. Run `python experiments/recheck_fine_budget_certification.py --selftest` and see `experiments/README_fine_budget.md` for frozen-checkpoint commands. The reported offline first-pass counts are given in `revision_results/fine_grid_first_pass.csv`. The spatial audit uses independently retrained checkpoints, not the original image models.
- `experiments/make_cat_scaling_absolute.py` plots absolute AR and CAT action counts alongside accuracy from `revision_results/width_pairs.csv` (Supplementary Figure S3).

### Reproducibility limits
The archive includes the full graph and spatial audit trajectories and three retrained spatial checkpoints. Original spatial checkpoints and graph checkpoint binaries are not included. Figure 4 and Table S6 report first passes on predefined budget grids. The separate per-iteration audits count the first passing iteration plus one equivalent terminal action; neither protocol measures continuously monitored distributed stopping. Individual width-training solve convergence flags were not retained. CPU times are implementation- and hardware-specific.

### Archived audits and figures

The audit archive is in [revision_results/audit/](revision_results/audit/). It contains 2,700 trajectory records, first-pass summaries, manifests and three retrained spatial checkpoints. The SHA-256 checks are performed by `verify_archived_audits.py`.

```bash
python experiments/make_cat_scaling_compact.py --input revision_results/width_pairs.csv --output results/figure3
python experiments/make_cat_scaling_absolute.py --input revision_results/width_pairs.csv --output results/figureS3
```

## Versioned archive

The software version is recorded in `VERSION` and `CITATION.cff`.

## Citation and license

See `CITATION.cff` and `LICENSE` (MIT).
