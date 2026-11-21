# Fairly calibrated AR versus CAT on native, nonsymmetric TorchDEQ — exploratory checkpoint

**Status:** completed three independent frozen trained TorchDEQ DEQ-INR checkpoints, no new training. This is code/results documentation, not a manuscript. The actual native upstream MFN/SIRENInjection model was trained separately using dense exact implicit differentiation. The checkpoints retain their original unprojected nonsymmetric recurrent weights. The *finite-width CPU training setup* uses a single camera image, same-image pixel train/test split, and is not a reproduction of TorchDEQ's full published DEQ-INR benchmark.

## Question

Can fixed CAT coefficients calibrated from **training Jacobians only**, without test eigenvalues, give a meaningful improvement over **an equally calibrated AR**, especially when CAT has non-negligible inertia? Does finite-horizon transient-aware calibration rescue performance missed by spectral-minimax fitting?

## Data separation and audit provenance

- Three existing width-32 checkpoints, each verified against its original checkpoint SHA-256 manifest.
- Same seeded per-checkpoint pixel split as the original training.
- Each calibration uses 64 training-pixel Jacobian blocks, with coefficients frozen before *any* evaluation Jacobian is constructed.
- Spectral protocol evaluates 128 held-out pixels at positions **80–207**. Previously examined held-out pixels 0–79 are excluded.
- Separate finite-horizon protocol evaluates 128 different held-out pixels at positions **208–335**. These were not inspected in the earlier transfer analysis.
- The benchmark was developed after seeing results on other pixels from the same image and models. **This is post-hoc exploratory evidence, not a genuinely independent image-domain confirmation.**
- One terminal `J^T v` per queried fixed budget is charged. The first-passing iteration is selected *retrospectively* from checks at every budget, without including the runtime/communication cost of checking every iteration online.
- The conservative relative-adjoint bound comes from the minimum singular value of the full **test** block-diagonal adjoint system and is **offline global**. It is not an online locality result.

## Equal-information spectral calibration

For each calibration protocol, AR minimizes the largest training-spectrum update modulus over `alpha` in `[0.1,1.9]` with `beta=0`. CAT minimizes the maximum magnitude of its second-degree characteristic roots with `alpha` in `[0.1,1.9]` and `beta` in `[beta_min,0.85]`. Both use **only training eigenvalues**. An additional *leave-one-model-out* control fits each method on the other two model checkpoints, with the target checkpoint excluded from calibration. Nonzero-`beta` CAT alternatives are optimized with lower bounds `0.025,0.05,0.10,0.15,0.20,0.30`.

First-passing **charged** Jacobian actions at offline relative-adjoint certificate `1e-5`:

| Method | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| AR, fixed alpha=1 | 21 | 23 | 21 |
| AR, training-spectrum tuned | 22 | 21 | 21 |
| CAT, training-spectrum tuned | 21 | 21 | 21 |
| AR, other-models spectrum tuned | 22 | 23 | 22 |
| CAT, other-models spectrum tuned | 22 | 24 | 22 |
| CAT, beta at least 0.05 | 22 | 23 | 22 |
| CAT, beta at least 0.10 | 25 | 26 | 25 |
| CAT, beta at least 0.20 | 33 | 32 | 32 |

The unconstrained CAT spectral fit chooses `beta=0.01495,0.00100,0.00865`. This is almost first-order AR. On the new test spectrum it yields only a minor asymptotic improvement and at most one charged action improvement over optimized AR. Enforcing `beta>=0.10` makes it slower in every seed.

## Fair finite-horizon calibration

Optimize both AR and CAT on exactly the **same 64 training-pixel blocks** and objective: weighted mean of per-pixel log relative residuals after 10, 16 and 22 steps (weights 0.3, 0.4, 0.3), plus `0.25 x 90th percentile` log residual, with a training-spectrum stability penalty if the characteristic-root factor is at least 0.98. AR optimizes its scalar step and CAT optimizes the step and inertia; the objective function is otherwise identical. All coefficients are frozen before evaluating 128 **different** test pixels.

| Method | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| AR, finite-horizon tuned | 22 | 24 | 24 |
| CAT, finite-horizon tuned | 21 | 25 | 24 |
| AR, fixed alpha=1 | 21 | 26 | 23 |

The finite-horizon CAT fit selects `beta=0.00960,0.00896,0.00668`. It gains one charged action in seed 0, loses one in seed 1 and ties in seed 2 compared with **equally** tuned AR.

## Interpretation

These native nonnormal TorchDEQ checkpoints *do not* exhibit a meaningful generalizable fixed-coefficient CAT speed advantage over equally calibrated AR. Inertia greater than a few hundredths worsens the measured modal factors and first-certification counts. Recovering convergence through globally fitted coefficients is valid but does not supply local calibration or a useful practical improvement. The negative result limits any claim that second-order CAT is a general DEQ solver accelerator.

The positive controlled symmetric IGNN/denoising results in this repository are separate and remain valid within their assumptions. Those results do not justify transferring a speedup assertion to arbitrary nonnormal DEQ models. Additional tuning on the same camera models would risk search-driven reporting. A future genuine extension would require a new, predeclared independent model/dataset and an implementable calibration mechanism, not more after-the-fact parameter search here.

## Reproduction

```powershell
# From cat-mwbp repository root, with Python, PyTorch, NumPy,
# SciPy, scikit-image and pandas already installed.
git pull origin main
& $python experiments\audit_native_deqinr_fair_calibration.py `
  --root results `
  --out results\native_deqinr_fair_calibration
```

Alternatively replace `--root results` with `--archive nativeall.zip` for a trusted local copy of the completed native-checkpoint archive. Optionally pass `--torchdeq-zoo external\torchdeq\deq-zoo\deq-inr` to recheck the upstream native MFN forward/Jacobian operations. Treat checkpoints and archives as trusted Python data.

Outputs: `fair_calibration_summary.csv`, `fair_calibration_curves.csv` (5,040 rows), and `fair_calibration_manifest.json`. The compact summary is versioned at `revision_results/native_deqinr_fair_calibration_summary.csv`. Full curve output, original checkpoints and the full manifold/provenance manifest are retained locally and not in the lightweight GitHub code repository.