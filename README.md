# Causal Adjoint Transport (CAT)

Research code and numerical results for *Local Second-Order Adjoint Dynamics for Implicit Neural Networks*.

CAT applies a fixed-coefficient second-degree stationary iteration to the implicit adjoint system `(I - J^T) lambda = b`. Each step uses one Jacobian-transpose action and two local adjoint states. The experiments examine convergence and backward-solver costs in structured implicit networks.

## Installation

Use Python 3.13 and run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/verify_revision_results.py
python experiments/verify_archived_audits.py
```

The graph and spatial experiments require PyTorch and the following additional packages:

```bash
python -m pip install ogb scikit-image pillow
python -m pip install threadpoolctl psutil
```

Install PyTorch for the target device using the [official installation instructions](https://pytorch.org/get-started/locally/). The spatial replication used PyTorch 2.9.1+cpu and scikit-image 0.26.0. Package versions, model configurations and checkpoint hashes are recorded in the archived manifests. The base requirements describe the original NumPy-based environment.

## Results

| Manuscript | Numerical data |
|---|---|
| Table 2, Section 6.1 | [Spectral benchmark](reference_results/implicit/) |
| Table 3 | [Solver comparison](reference_results/implicit/nonlocal_solver_reference.csv) |
| Table 4, Figure 2 | [Classification](reference_results/implicit/core_dataset_summary.csv), [spectral prediction](reference_results/implicit/theory_vs_measured_summary.csv) |
| Table 5 | [Trainable recurrent models](reference_results/implicit/trainable_recurrent_summary.csv) |
| Figure 3, Figure S3 | [Width-scaling pairs](revision_results/width_pairs.csv) |
| Figure 4, Table S6 | [Graph](revision_results/ignn_first_certified_by_seed.csv), [spatial](revision_results/spatial_first_certified_by_seed.csv) |
| Tables S1–S2 | [Directed spectra](reference_results/implicit/directed_spectrum_summary.csv), [ellipse calibration](reference_results/implicit/complex_spectrum_oracle_summary.csv) |
| Table S3 | [Feed-forward comparisons](reference_results/aggregate_comparisons.csv) |
| Table S4 | [Chebyshev comparison](revision_results/chebyshev_corrected_summary.csv) |
| Table S5 | [CPU and state costs](revision_results/solver_cost_v2_summary.csv) |
| Table S7 | [Image quality](revision_results/spatial_mean_test_psnr.csv) |
| Table S8, Section 6.4 | [Per-iteration results](revision_results/fine_grid_first_pass.csv), [audit archive](revision_results/audit/) |

In the 240 matched spectral cases, Chebyshev requires fewer actions than CAT in 164 cases, ties in 42 and requires more in 34 (median Chebyshev/CAT ratio 0.913). CAT uses fixed coefficients; Chebyshev uses iteration-dependent coefficients with the same local action. GMRES uses fewer actions in the large applications but requires global Krylov algebra.

## Experiments

The following commands generate outputs under `results/`, which is excluded from version control.

### Spectral and classification benchmarks

```bash
python experiments/run_implicit_spectral_suite.py --profile standard --output results/spectral
python experiments/run_implicit_dataset_suite.py --profile standard --group all --output results/classification
python experiments/run_trainable_recurrent_control.py --profile standard --output results/trainable_recurrent
python experiments/run_chebyshev_corrected_audit.py --out results/chebyshev
python experiments/run_nonlocal_solver_reference.py --output results/nonlocal
python experiments/run_solver_cost_audit.py --anderson-config revision_results/anderson_best_config_extended.csv --seeds 5 --rhos 0.95 --repeats 9 --blas-threads 1 --out results/costs
python experiments/run_directed_spectrum_control.py --profile standard --output results/directed
python experiments/run_complex_spectrum_oracle.py --rho 0.95 --n 144 --output results/complex
python experiments/recheck_chebyshev_vision.py --repo-root . --out results/vision_audit
```

The feed-forward control uses the full suite configuration recorded in `reference_results/suite_config.json`. The retained result tables contain the four methods reported in the article; the suite configuration and metadata record the complete original run and statistical comparison family.

```bash
python run_npl_suite.py --profile full --jobs 2 --output results/feedforward
```

### Scaling and finite-tolerance sensitivity

```bash
python experiments/run_revision_scaling.py --study width --datasets breast_cancer,digits,synthetic_large --widths 32,64,128,256,512 --seeds 3 --epochs 20 --out results/revision_width
python experiments/cat_tolerance_study.py --rhos 0.8,0.95,0.99 --tolerances 1e-3,1e-4,1e-5,1e-6,1e-7 --seeds 5 --side 12 --out results/tolerance
python experiments/make_cat_scaling_compact.py --input revision_results/width_pairs.csv --output results/figure3
python experiments/make_cat_scaling_absolute.py --input revision_results/width_pairs.csv --output results/figureS3
python experiments/make_cat_tolerance_figure.py --input results/tolerance/tolerance_pairs.csv --output results/figureS1
```

### Graph classification

The official [OGB](https://ogb.stanford.edu/) dataset and temporal split are used. Run the following for each seed (0, 1, 2), changing the seed and output paths accordingly.

```bash
python experiments/run_ignn_ogbn_arxiv.py --device cpu --epochs 20 --width 32 --rho 0.95 --seed 0 --data-root data/ogb --out results/ignn_s0
python experiments/run_ignn_postcheck.py --checkpoint results/ignn_s0/trained_model.pt --data-root data/ogb --device cpu --out results/ignn_postcheck_s0
```

The option `--trust-ogb-cache` is available for locally verified OGB cache files that require the legacy PyTorch loading behaviour.

### Spatial denoising

Run the training command for seeds 0, 1 and 2, changing the output path accordingly, then summarize the three runs.

```bash
python experiments/run_spatial_implicit_denoising.py --seed 0 --steps 300 --side 80 --patch 48 --eval-side 64 --audit-side 48 --width 24 --rho 0.95 --center-weight 0.35 --sigma 0.12 --lr 0.002 --threads 2 --repeats 3 --out results/spatial_denoising_s0
python experiments/summarize_spatial_denoising.py --root results --prefix spatial_denoising_s --out results/spatial_denoising_summary
```

See [per-iteration certification](experiments/README_fine_budget.md) for the offline audits, including commands using the three archived spatial checkpoints. Additional details: [solver configurations](revision_results/README_anderson_v2.md) and [vision adjoints](revision_results/README_vision_chebyshev.md).

## Reproducibility

The repository contains compact numerical tables, 2,700 per-iteration audit records, manifests and three independently retrained spatial checkpoints. The original graph, spatial and vision checkpoint binaries are not included. Public datasets are obtained separately.

Figure 4 and Table S6 use predefined iteration budgets with one charged terminal residual evaluation. The offline per-iteration audits report the first passing iteration plus one equivalent terminal action; they do not measure continuously monitored stopping costs. Individual width-training solve convergence flags were not retained. CPU timings depend on hardware and software versions.

## Citation and license

See [CITATION.cff](CITATION.cff), [VERSION](VERSION) and [LICENSE](LICENSE) (MIT).
