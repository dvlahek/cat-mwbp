# Frozen-checkpoint CAT spectral transfer on the unprojected TorchDEQ DEQ-INR layer

**Status: completed, exploratory transfer audit.** This is *numerical experiment documentation*, not a manuscript, reviewer response, or claim that global spectral calibration is local. The code uses the three previously trained, unprojected native TorchDEQ DEQ-INR/MFN/SIREN-like checkpoints from `nativeall.zip`. It does not alter/retrain the models. Training originally used independent dense implicit adjoints, **not CAT**, so the frozen models were not selected using CAT performance.

## Question

Does the two-state CAT recurrence recover stable convergence for complex, nonnormal Jacobians when its coefficients are fitted on **other Jacobians**, rather than on eigenvalues of the very Jacobians being evaluated?

Two distinctly labeled transfer protocols use an identical frozen 32-dimensional native DEQ-INR layer at each pixel:

1. **Within-model, disjoint pixels:** optimize fixed real `(alpha,beta)` for the maximum characteristic-root modulus over all complex eigenvalues in **64 training-pixel** Jacobian blocks, freeze coefficients, then evaluate on **64 different held-out pixels** from the same original camera image. To avoid reusing previously inspected holdout Jacobians, the 16 pixels from the original pilot audit are excluded from the new test set.
2. **Leave-one-model-out:** optimize fixed `(alpha,beta)` on the *64 training-pixel Jacobians from each of the other two seeds* (128 blocks total). Freeze coefficients and apply them to the 64 held-out pixels of the unseen target checkpoint. **No Jacobian from the held-out model enters this calibration.**

All test spectra and singular values are obtained only after coefficients have been frozen. The optimizer uses `scipy.optimize.differential_evolution`, fixed seeds and bounds `alpha in [0.02,2]`, `beta in [0,0.98]`, with precisely the same worst-characteristic-root objective as in the exploratory oracle stress test. No test-set reoptimization, hyperparameter sweep, or coefficient selection is performed.

**Important caveat:** The models and source image were previously explored, and the tests are *not* a new independent dataset. This is a mechanistic Jacobian-transfer diagnostic. Calibration still uses global eigenvalue computations on training states. The test-time certificate still uses global SVD and does not establish an online decentralized stopping rule.

## Validated outcomes

At retrospective first exact per-iteration budget meeting a **relative adjoint** error upper bound `1e-5`, one terminal `J^T v` is included in the reported number. These values are not actual counted costs of an online check at every iteration. Unlike the earlier coarse budgets, this re-audit checks every integer budget from 1 to 120 (offline).

| Method | Seed 0 charged JTv | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| AR, alpha=1 | 20 | 21 | 26 |
| CAT, invalid nominal real interval | No pass | No pass | No pass |
| CAT, calibrated on target model's **training pixels only** | 20 | 20 | 25 |
| CAT, calibrated on **the other two models only** | 21 | 22 | 27 |

Fitted in-model CAT coefficients have `alpha = 1.017644, 0.952286, 1.012747` and `beta = 0.014955, 0.001002, 0.008653`. Leave-one-model-out coefficients also have beta close to zero. All held-out modal root factors are below 1 in these 3 models, and all transferred CAT runs pass the rigorous offline SVD certificate. The independent implementation reproduces the original upstream-model audit's maximum `||J||_2` for each of the three checkpoints to within `2e-6`. Across all finite offline diagnostics, the SVD certificate dominates the observed direct-solve relative adjoint error.

**Interpretation:** coefficient transfer prevents divergence, but produces **no substantial acceleration over AR** in these nonnormal, complex-spectrum DEQ-INR cases. Leave-one-model-out CAT needs one additional charged operation in every seed. The calibrated inertia is extremely small, indicating that a nearly first-order update is favored by the calibration objective. This is a useful negative result, not evidence of a new competitive universal DEQ solver.

## Reproduction

Run from the `cat-mwbp` repository root, where `results/official_deqinr_native_s0`, `s1` and `s2` contain the earlier verified checkpoints and manifests:

```powershell
$python = (Resolve-Path ..\CAT_revision\.venv\Scripts\python.exe).Path
& $python experiments\audit_native_deqinr_transfer.py `
  --root results `
  --torchdeq-zoo external\torchdeq\deq-zoo\deq-inr `
  --calibration-count 64 --test-count 64 `
  --exclude-prior-audit 16 --max-budget 120 `
  --out results\native_deqinr_spectral_transfer
```

The optional upstream check imports native TorchDEQ `DEQINR` and verifies its forward map and autograd `J^T v` against the explicit single-layer formula; the same script can also accept `--archive nativeall.zip` (mutually exclusive with `--root results`) and omit `--torchdeq-zoo` to audit the exact saved weights without the external dependency. The archived source has SHA-256:

`nativeall.zip` = `946e0d7abd0535e0885916a6ec4c9f701e1c272cce6a1c63f0e76e70491b9aaf`.

The output includes `transfer_first_pass.csv`, `transfer_curves.csv`, `transfer_manifest.json` with frozen coefficient values, checkpoint SHA-256 values and disjoint calibration/test pixel indices. Compact verified results are committed as [`revision_results/native_deqinr_transfer_first_pass.csv`](../revision_results/native_deqinr_transfer_first_pass.csv). Full checkpoint and output archives stay local until the eventual reproducibility deposit. Manuscripts stay local.

## What this does *not* prove

- It does not remove the need for global **training** spectra, or give local spectrum estimation.
- It does not demonstrate substantial CAT speedups over AR on this class, or superiority over GMRES, Anderson or corrected Chebyshev.
- It does not establish the performance on different DEQ architectures, unseen image datasets, or a hardware-local communication protocol.
- `first_pass` is an offline minimum among fixed-budget solutions, not an online algorithm with free stopping.