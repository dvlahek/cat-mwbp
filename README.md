# Causal Adjoint Transport (CAT)

Code for the experiments reported in **Local Second-Order Adjoint Dynamics for Implicit Neural Networks**. The repository contains the implementation, experimental scripts, tests and numerical reference tables.

The CAT adjoint is a stationary two-state local iteration for the implicit backward problem `(I - J^T) lambda = b`. The frozen linear iteration belongs to the classical second-degree stationary family; the contribution examined in the article is its local neural realization and its cost relative to first-order Activation Relaxation (AR) under explicit spectral assumptions.

## Scope
 Recurrent classification, an IGNN-style graph classifier on `ogbn-arxiv` and a spatial implicit denoiser are controlled implicit-model applications. They are not three unmodified off-the-shelf DEQ architectures. CAT improves on AR in the tested reciprocal/real-spectrum regime, but is not claimed to outperform Chebyshev or GMRES in general. Offline spectral calibration and terminal error checks have costs; a single terminal check is charged in the reported new application audits.

## Install and test

Python with NumPy, SciPy, pandas, scikit-learn and Matplotlib:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/verify_revision_results.py
```

The graph and image experiments additionally require PyTorch, `ogb` (graph), `scikit-image` and `Pillow` (images). CPU timing and memory controls require `threadpoolctl` and `psutil`. Install matching versions appropriate for the machine, and document them when reproducing timings. Dataset downloads and trained weights are **not** committed.

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
python experiments/run_revision_scaling.py --study width --datasets breast_cancer,digits,synthetic_large --widths 32,64,128,256,512 --seeds 3 --epochs 20 --out results/width_scaling
python experiments/run_solver_cost_audit.py --seeds 5 --repeats 9 --blas-threads 1 --out results/costs
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

## Numerical reference tables

- `reference_results/implicit/` contains compact summaries of the original recurrent/spectral runs, **excluding superseded historical Chebyshev measurements**.
- `reference_results/` contains compact feed-forward comparisons and the original suite configuration.
- `revision_results/` contains compact corrected Chebyshev, solver-cost, IGNN and spatial-denoising results. Run `python experiments/verify_revision_results.py` to check the SHA-256 manifest and key values.
- The **revised Anderson** configurations and CPU timings are documented in [`revision_results/README_anderson_v2.md`](revision_results/README_anderson_v2.md), with `anderson_best_config_extended.csv`, `solver_cost_v2_per_seed.csv`, and `solver_cost_v2_summary.csv`. Old Anderson cells and old memory figures in historical tables are superseded; the v2 state column counts active core arrays, not GMRES's full preallocated buffers or temporary Anderson workspaces.
- The additional corrected **MNIST/CIFAR-10 Chebyshev** recheck is in [`revision_results/README_vision_chebyshev.md`](revision_results/README_vision_chebyshev.md) with five CSVs and its own manifest. Historical vision Chebyshev/CAT columns are superseded; original CAT/AR training outcomes are unchanged. Run `python experiments/recheck_chebyshev_vision.py --smoke` before reproducing the six-checkpoint comparison.

**Chebyshev correction:** Historical spectral and classification CSVs used a Chebyshev recurrence inconsistent with the standard normalized first-kind iteration. The source scripts now use the corrected recurrence; cite the independently checked `revision_results/chebyshev_corrected_*.csv` and vision recheck CSVs, not historical Chebyshev columns. CAT/AR recurrences are unchanged. Eight original fixed-recurrent dataset Chebyshev oracle comparisons are *not* reported in the manuscript pending a full corrected recheck on their original checkpoints.

### Additional diagnostics (Sections 6.3–6.4, Supplementary Table S8)
- `experiments/recheck_fine_budget_certification.py` reloads a **frozen** `ogbn-arxiv` or spatial-denoising checkpoint and checks the contraction certificate at every iteration for AR, CAT and corrected Chebyshev. This is an **offline** analysis; the existing fixed-budget/one-terminal-check results remain the published comparison. Run `python experiments/recheck_fine_budget_certification.py --selftest` and see `experiments/README_fine_budget.md` for frozen-checkpoint commands. The reported offline first-pass counts are given in `revision_results/fine_grid_first_pass.csv`. The spatial audit uses independently retrained checkpoints, not the original image models.
- `experiments/make_cat_scaling_absolute.py` plots absolute AR and CAT action counts alongside accuracy directly from the original `results/revision_width/width_pairs.csv`. It does not infer missing Chebyshev scaling data.

### Reproducibility limits
 Only compact experimental outputs are distributed here, not full datasets, model checkpoints, image outputs or complete per-step histories. Figure 4 and Table S6 report first passes on predefined budget grids. The separate per-iteration audits count the first passing iteration plus one equivalent terminal action; neither protocol measures continuously monitored distributed stopping. Individual width-training solve convergence flags were not retained. CPU times are implementation- and hardware-specific.

## Versioned archive

Release `v1.0.0` predates the revised numerical comparisons. A version-specific archive for the revision has not yet been deposited.

## Citation and license

See `CITATION.cff` and `LICENSE` (MIT). Manuscript drafts, supplementary LaTeX, reviewer correspondence and experimental attempts not reported in the article are deliberately excluded.
