# Per-iteration certification

The offline audit evaluates the relative adjoint-error certificate after every iteration on a frozen model. The reported equivalent action count is the first passing iteration plus one terminal Jacobian-transpose action. It excludes the cost of intermediate diagnostic checks.

Figure 4 and Table S6 use predefined budget grids. The graph audit uses the original graph models. All spatial comparisons in Tables S6–S8 use the same three independently retrained, archived models.

## Archived results

The trajectories, first-pass tables, manifests and retrained spatial checkpoints are in [revision_results/audit](../revision_results/audit/).

From the repository root:

```bash
python experiments/verify_archived_audits.py
python experiments/recheck_fine_budget_certification.py --selftest
```

## Graph

The archived graph checkpoints and official OGB dataset are required to repeat the frozen-model audit. Example for seed 0:

```bash
python experiments/recheck_fine_budget_certification.py --kind graph --checkpoint revision_results/audit/ignn_arxiv_w32_e200_s0/trained_model.pt --data-root data/ogb --device cpu --max-budget 120 --old-grid revision_results/audit/ignn_postcheck_e200_s0/solver_budgets_charged.csv --out results/ignn_fine_s0
```

Repeat for seeds 1 and 2. The optional `--old-grid` argument accepts the corresponding postcheck CSV. Use `--trust-ogb-cache` only with a trusted, locally verified OGB cache.

## Spatial denoising

The archived retrained checkpoint can be audited directly. Example for seed 0:

```bash
python experiments/recheck_fine_budget_certification.py --kind spatial --checkpoint revision_results/audit/spatial_reproduced_s0/trained_model.pt --old-grid revision_results/audit/spatial_reproduced_s0/solver_budgets.csv --device cpu --max-budget 200 --out results/spatial_fine_s0
```

Repeat for seeds 1 and 2. The output contains `fine_grid_all_steps.csv`, `fine_grid_first_pass.csv` and `manifest.json`. Retraining instructions are in the root README. The original spatial models are unavailable.

## Supplementary Figure S3

```bash
python experiments/make_cat_scaling_absolute.py --input revision_results/width_pairs.csv --output results/figureS3
```
