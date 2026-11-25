# Fine-grid certification (optional offline diagnostic)

This is **not** a replacement for the fixed-budget + terminal-check Figure 4 protocol. It tests on the original frozen checkpoint after **each** iteration. The first passing iteration k is reported alongside the *equivalent separately charged* fixed-budget cost k+1 Jt actions. Intermediate checks are not charged as an online monitoring algorithm, so do not describe these numbers as autonomous distributed stopping costs.

Run from the repository root:

```powershell
git pull origin main
py experiments/recheck_fine_budget_certification.py --selftest

Get-ChildItem results -Recurse -Filter trained_model.pt | Select-Object -ExpandProperty FullName
Get-ChildItem results -Recurse -Filter width_pairs.csv | Select-Object -ExpandProperty FullName
```

Graph checkpoint output (normally `results/ignn_s0/trained_model.pt`; repeat s0, s1 and s2):

```powershell
foreach ($s in 0..2) {
    py experiments/recheck_fine_budget_certification.py --kind graph --checkpoint "results/ignn_s$s/trained_model.pt" --data-root data/ogb --device cpu --max-budget 100 --out "results/ignn_fine_s$s"
}
```

For source-verified OGB caches that require the old PyTorch load behavior, add `--trust-ogb-cache` only after verifying the provenance. Provide the original graph postcheck `solver_budgets_charged.csv` as `--old-grid <path>` to verify coarse pass flags before comparing new numbers.

Spatial denoising checkpoint output (normally `results/spatial_denoising_s0/trained_model.pt`; repeat s0, s1, s2):

```powershell
foreach ($s in 0..2) {
    py experiments/recheck_fine_budget_certification.py --kind spatial --checkpoint "results/spatial_denoising_s$s/trained_model.pt" --max-budget 200 --device cpu --out "results/spatial_fine_s$s" --old-grid "results/spatial_denoising_s$s/solver_budgets.csv"
}
```

The spatial calculation uses the same held-out `TEST_NAMES[0]`, noise seed, model, and derivative of decoder loss as the original `solver_audit`. It does not retrain anything.

Each output directory contains `fine_grid_all_steps.csv`, `fine_grid_first_pass.csv`, and `manifest.json` (with checkpoint SHA256). If an old-grid comparison disagrees, investigate rather than overriding it. The graph diagnostic requires the original OGB data/split and checkpoint; the spatial diagnostic requires the original frozen checkpoint. Neither is archived in the compact GitHub results.

Figure 3 absolute actions can be regenerated, if the original width CSV is present:

```powershell
py experiments/make_cat_scaling_absolute.py --input results/revision_width/width_pairs.csv --output results/revision_width/cat_parameter_scaling_absolute
```

The script reads absolute AR and CAT backward actions from `width_pairs.csv`; it does not invent a Chebyshev width comparison.

**Do not change the manuscript's Figure 4 ratios or CAT/Chebyshev comparisons until the true checkpoint-based numbers and original-grid cross-checks are available.**
