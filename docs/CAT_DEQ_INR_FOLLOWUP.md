# Frozen-checkpoint DEQ-INR audit and one targeted capacity pilot

This is a continuation of the controlled TorchDEQ DEQ-INR camera experiment.
The model uses upstream `DEQINR`, `MFN` and `SIRENInjection` modules, **but**
the learned recurrent matrix is projected to a symmetric nonnegative
contraction and coordinate injection is bounded. It is not an unmodified
published DEQ-INR result. The first run had a 64x64 camera image, width 48,
120 training steps, 80% randomly selected training pixels and 20% held-out
pixels from the **same image**, and held-out PSNR of about 13.17 dB. In that
run the coarse budgets `10,20,30,40,60,80` could not resolve the crossing
between 40 and 60 iterations.

## Audit the original trained checkpoint, no further optimization

```powershell
$python = (Resolve-Path ..\CAT_revision\.venv\Scripts\python.exe).Path

git pull origin main

& $python experiments\audit_torchdeq_inr_checkpoint.py `
  --checkpoint results\deq_inr_camera_s0\trained_model.pt `
  --torchdeq-zoo external\torchdeq\deq-zoo\deq-inr `
  --start 10 --stop 65 --step 1 --repeats 3 `
  --audit-pixels 128 `
  --out results\deq_inr_camera_s0_dense
```

The script checks the SHA of the upstream TorchDEQ checkout against the
original manifest when available. It re-creates the seeded camera split and
refuses to continue if the loaded checkpoint does not reproduce its recorded
held-out PSNR to within 0.025 dB. It also checks the explicit Jacobian against
upstream `model.f` at the equilibrium and enforces the spectral hypotheses.

Outputs:

- `dense_solver_budgets.csv`: all fixed-budget solver comparisons, with
  **one terminal residual check charged** per solve and repeated CPU timings.
- `dense_first_pass.csv`: first **tested** budget meeting the conservative
  relative adjoint-error certificate; these are not adaptive online stopping
  times.
- `dense_audit_manifest.json`: checkpoint fingerprint and provenance,
  reproduction gate, spectral checks, and numerical tolerance metadata.
- `reconstruction.pdf` and `reconstruction.png`: three panels A–C containing
  ground-truth pixels, reconstructed pixels and absolute error. Figure has
  no embedded title or caption; supply the caption in LaTeX.
- `image_arrays.npz`: true image, reconstructed image and original seeded
  training/held-out indices, so figures can be regenerated.

**Important:** `dense_first_pass.csv` uses the FIRST PASS INSIDE THE SPECIFIED
budget grid. Use `--start 10 --stop 65 --step 1` for an honest dense search
including GMRES; do not interpret a 35–65 search as finding the actual
GMRES crossing. CPU timing is implementation- and device-dependent.

## One second, larger controlled capacity pilot

The baseline's image quality is modest. Run one pre-specified pilot with
larger width and a somewhat less restricted injection:

```powershell
& $python experiments\run_torchdeq_inr_cat.py `
  --torchdeq-zoo external\torchdeq\deq-zoo\deq-inr `
  --side 64 --width 96 --steps 400 `
  --rho 0.80 --injection-cap 0.42 `
  --seed 0 --audit-pixels 128 `
  --budgets 10,20,30,40,60,80 `
  --out results\deq_inr_camera_w96_s0
```

The sufficient positivity check is `rho*(1+injection_cap)+injection_cap < pi/2`:
with the above values it is 1.556, below 1.571. This is a theoretical
sufficient bound for the specific one-layer SIREN-like model under the
projected operator and bounded injections, not a guarantee of high PSNR or
of CAT outperforming other solvers.

Then audit the frozen second checkpoint and export the image:

```powershell
& $python experiments\audit_torchdeq_inr_checkpoint.py `
  --checkpoint results\deq_inr_camera_w96_s0\trained_model.pt `
  --torchdeq-zoo external\torchdeq\deq-zoo\deq-inr `
  --start 10 --stop 85 --step 1 --repeats 3 `
  --audit-pixels 128 `
  --out results\deq_inr_camera_w96_s0_dense
```

Compare the actual held-out PSNR, certified first-passing budgets and
reconstruction quality. If the pilot does not materially improve results,
report the negative finding rather than changing hyperparameters repeatedly.
For a paper claim about typical behavior across initializations, at least
three seeds are needed; these two checkpoint audits alone are diagnostics.

## Send back

From both `*_dense` folders, send `dense_first_pass.csv`,
`dense_solver_budgets.csv`, `dense_audit_manifest.json`, and
`reconstruction.png`. From the second training folder send `manifest.json`
and `train_history.csv`. Keep the `trained_model.pt` checkpoints local unless
needed for independent reproduction or auditing.