# Standard TorchDEQ DEQ-INR layer: nonnormal frozen-adjoint diagnostic

## Purpose and limits

This is an **additional reviewer-requested architecture test** using the *original* `DEQINR`, `MFN` and `SIRENInjection` Python modules from the upstream [locuslab/torchdeq](https://github.com/locuslab/torchdeq/tree/main/deq-zoo/deq-inr) repository. Unlike our earlier controlled DEQ-INR test, this runner **does not** symmetrize, force nonnegative entries, clip the recurrent weights or bound the injection. `filter_type='siren_like'`, `n_layers=1` and `norm_type='none'` are original-module options.

The CPU experiment uses a **reduced-width, reduced-data training configuration**, not the published large-scale DEQ-INR camera experiment. It uses the same `skimage.data.camera` source image and randomized, seeded pixel split, so the held-out pixels are **not** an independent image. It must not be called a replication of the source paper or evidence of high reconstruction accuracy. No claim is made about generalization to all DEQs.

Training **does not use CAT**. The forward implicit state is iterated using the upstream `model.f`; a dense per-pixel linear solve gives the exact IFT adjoint in the training step. This avoids training a model that is specially favourable to CAT. The training protocol is ours and differs from the upstream published recipe. We verify the analytic Jacobian-transpose-vector product numerically against upstream autograd on the frozen model.

The frozen model's `J` can have **complex eigenvalues, nonzero nonnormality, negative cosine derivatives and operator norm >= 1**. We calculate each audited Jacobian block and the smallest singular value of its `I-J^T` operator in `float64`. This gives a finite-dimensional **oracle** residual-error certificate, regardless of contractivity, provided `I-J^T` is nonsingular. The global dense SVD is offline and must **not** be represented as a local, distributed, free stopping criterion. These overheads are explicitly excluded from solver timings and are not included in charged `J^Tv` counts.

Our ordinary `CAT` and `Chebyshev` controls deliberately retain a **fixed, predeclared real-interval** coefficient rule based on `--nominal-rho=0.95`. For nonreal spectra the assumptions of this rule can fail; the script records that failure instead of redesigning the model. Optional `CAT-complex-oracle` tunes two real recurrence coefficients against the *exact complex eigenvalues of the audited frozen blocks*. The latter uses forbidden oracle information, is not an available autonomous local calibration, and does not control nonnormal transients. It is a diagnostic, **not** a practical competitor.

## Local Windows PowerShell execution

From `cat-mwbp` with the existing `CAT_revision` environment:

```powershell
$python = (Resolve-Path ..\CAT_revision\.venv\Scripts\python.exe).Path

git pull origin main
& $python -m pip install -r external\torchdeq\deq-zoo\deq-inr\requirements.txt
& $python -m pip install -e external\torchdeq

& $python experiments\run_official_deqinr_nonnormal.py `
  --torchdeq-zoo external\torchdeq\deq-zoo\deq-inr `
  --smoke --seed 0 --calibrate-complex `
  --out results\official_deqinr_native_smoke
```

Smoke should print `TRAIN`, `AUDIT`, and `DONE`. Only then run the fixed 3-seed CPU protocol:

```powershell
foreach ($seed in 0,1,2) {
    & $python experiments\run_official_deqinr_nonnormal.py `
      --torchdeq-zoo external\torchdeq\deq-zoo\deq-inr `
      --seed $seed --side 48 --width 32 --steps 240 `
      --input-scale 16 --lr 0.001 --audit-pixels 16 `
      --nominal-rho 0.95 --budgets 5,10,15,20,30,40,60,80,100,120 `
      --calibrate-complex --repeats 3 --threads 2 `
      --out "results\official_deqinr_native_s$seed"

    if ($LASTEXITCODE -ne 0) { throw "Native DEQ-INR failed for seed $seed" }
}
```

If an upstream forward state **fails** to converge, or training becomes nonfinite, retain the failure log. **Do not** add symmetry or tune the recurrence to make CAT win. Such failures are themselves a limit of this unconstrained test; check checkpoint and upstream commit before interpreting. If a test already used a custom run of TorchDEQ with modified source files, remove those modifications and repeat the smoke test on a clean upstream checkout.

## Output per seed

- `manifest.json` – upstream repository commit, checkpoint SHA-256, all settings, training method, spectrum and nonnormality diagnostics, exact JTv/autograd check.
- `train_history.csv` – original-model training loss, held-out-pixel PSNR, forward iterations and actual weight norm.
- `solver_budgets.csv` – all solvers at identical *frozen* model and right-hand side with fixed budgets and one charged terminal `J^Tv`. Exact-reference errors and oracle SVD-based residual bounds are reported separately. CPU timings are implementation-specific and do not include global calibration or SVD overhead.
- `first_certified.csv` – first *tested* budget meeting oracle-certified `1e-5`, or blank for failure. These are not exact iteration optima nor independent stopping runs.
- `trained_model.pt` – tensor-only upstream-model weights, needed for replication; retain locally and archive with eventual DOI, not in GitHub source.

The figure and text for the journal are **not** placed in this public repository. Do not interpret the 3-seed protocol as full hyperparameter search or proof of native architecture scalability. Preserve all outcomes, including AR and GMRES advantages and any `CAT` instability, to support a scientifically accurate response to Reviewer 2.