# Final CAT controls for journal revision

New experiments supplement **Local Second-Order Adjoint Dynamics for Implicit Neural Networks**. They do not overwrite historical v1.0.0 outputs. Run commands from the `cat-mwbp` repository root.

**November 2026 completion status.** The corrected Chebyshev sweep, frozen Anderson/Broyden timing study, three-seed IGNN certification and three-seed spatial denoising comparison have completed. Compact source-derived tables are committed in [revision_results](../revision_results/) with checksums and a [verification script](../experiments/verify_revision_results.py). Their interpretation and negative/limited outcomes are summarized in [REVISION_EVIDENCE.md](REVISION_EVIDENCE.md). The constrained DEQ-INR image reconstruction pilot also completed but did not demonstrate a strong reconstruction application and should not be described as a successful modern-DEQ benchmark. Historical Chebyshev output remains retained and is not corrected in place. The revision is now in manuscript/reviewer-response reconciliation, not a new numerical-search phase.

## 1. CPU time and memory on original solver benchmark

Uses frozen Anderson/Broyden selected hyperparameters from `reference_results/implicit`, matched exact-error stopping at `1e-6`, and the original `make_case` generator. Anderson-m is implemented with a bounded deque of length `m`; the script confirms its output and action count match the original routine, whose Python lists incidentally retain obsolete history.

```powershell
& $python -m pip install threadpoolctl psutil
& $python experiments\run_solver_cost_audit.py --seeds 5 --repeats 3 --blas-threads 1 --out results\solver_cost_audit
```

Outputs: `solver_cost_runs.csv`, `solver_cost_summary.csv`, `nonlocal_solver_costs.pdf`, and `manifest.json`.

**Interpretation:** Time is single-CPU-process wall clock with single-threaded BLAS, excluding case generation and exact-reference construction. Figure's estimated persistent solver state is an analytic implementation estimate, **not** peak RAM. `psutil` process RSS samples and `tracemalloc` Python allocations are exploratory because native linear-algebra buffers and allocator caching make process memory noisy. Do not convert these data into general GPU speed claims.

## 2. Corrected Chebyshev reference controls (must check before submission)

The historical spectral script `run_implicit_spectral_suite.py` used a first inertial-step coefficient differing from the normalized Chebyshev recurrence, and a momentum factor inconsistent with the updated coefficient. The new script evaluates an exact normalized first-kind Chebyshev polynomial, includes a numerical polynomial identity selftest, and compares CAT vs Chebyshev with unchanged cases, calibrations and exact-error target.

```powershell
& $python experiments\run_chebyshev_corrected_audit.py --quick --out results\cheb_check_quick
& $python experiments\run_chebyshev_corrected_audit.py --out results\cheb_check_full
```

Outputs: `chebyshev_corrected_runs.csv`, `chebyshev_corrected_pairs.csv`, `chebyshev_corrected_summary.csv`.

Do not silently replace any historical CSV or publish older Chebyshev values as corrected. Compare the complete paired table to the figures and claims in the manuscript before drafting final response to Reviewer 1. Changes may require replacing Chebyshev comparisons and regenerating figures; CAT and AR recurrence formulas are not changed.

## 3. Actual TorchDEQ zoo DEQ-INR module on image reconstruction

The experiment imports original `DEQINR`, `MFN`, and `SIRENInjection` from upstream [TorchDEQ](https://github.com/locuslab/torchdeq/tree/main/deq-zoo/deq-inr). It uses `skimage.data.camera()` (the publicly distributed image also used in the upstream DEQ-INR script), resized for CPU and with an 80/20 random pixel split. **No claim of replicating the original full DEQ-INR benchmarks is intended.**

The forward *model modules* are original TorchDEQ modules. We train the coordinate injection, readout, and recurrent matrix with CAT implicit differentiation. Each training step projects the recurrent matrix to symmetric nonnegative weights with absolute row-sum ≤ rho, and bounds the coordinate injection outputs. The cosine derivative is verified positive. This controlled adaptation permits a real spectral interval and does **not** test arbitrary learned non-normal DEQ weights. All backward solvers are evaluated on one frozen trained state and the same right-hand side, using identical Jᵀv operator and fixed action budgets. One terminal residual check is charged to each solver. The certified bound is not a fully local online stopping algorithm.

First, clone the official sources (only if not already present) and install the package into the **existing** Python environment:

```powershell
if (-not (Test-Path .\external\torchdeq\setup.py)) {
    New-Item -ItemType Directory -Force .\external | Out-Null
    git clone --depth 1 https://github.com/locuslab/torchdeq.git external\torchdeq
}
& $python -m pip install -e .\external\torchdeq scikit-image
```

Quick local test:

```powershell
& $python experiments\run_torchdeq_inr_cat.py `
  --torchdeq-zoo external\torchdeq\deq-zoo\deq-inr `
  --smoke --seed 0 --out results\deq_inr_smoke
```

Real image, local CPU pilot:

```powershell
& $python experiments\run_torchdeq_inr_cat.py `
  --torchdeq-zoo external\torchdeq\deq-zoo\deq-inr `
  --side 64 --width 48 --steps 120 --seed 0 `
  --audit-pixels 128 --rho 0.95 --out results\deq_inr_camera_s0
```

Outputs: `train_history.csv`, `solver_budgets.csv`, `trained_model.pt`, `manifest.json`. The manifest includes the PSNR of fitted and held-out pixels, the number of trainable parameters, and positivity/contractivity audits. **Do not report results before the actual TorchDEQ integration smoke test and image benchmark finish.** Check that the source revision and installed package version are frozen prior to Zenodo archiving.

## Revision decision gate

1. Corrected Chebyshev action counts have been verified in post-release CSVs; update legacy manuscript tables and figures before submission.
2. Time and memory descriptions distinguish actual measurements from analytical estimates.
3. Controlled DEQ-INR pilot converged, but reconstruction quality was weak; retain as a limitation, not successful broad-DEQ validation.
4. Original IGNN three-seed certification is retained separately with the original checkpoint hashes.
5. Only then finalize editor/reviewer responses and archive the exact release/outputs with a Zenodo DOI.