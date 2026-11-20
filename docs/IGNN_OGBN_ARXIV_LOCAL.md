# IGNN local graph-adjoint benchmark

Controlled implicit graph classifier on the real `ogbn-arxiv` citation graph. Its symmetrized, degree-normalized matrix `S` is contractive, and the implicit layer is

```
h* = tanh(X W_in + b_in + rho S h*)
```

The input projection and readout are trained with CAT implicit gradients. Then one checkpoint and one adjoint right-hand side are frozen and AR, CAT, Chebyshev and restarted GMRES are compared at matched budgets of local `J^T v` operations. The offline reference uses a float64 contraction solve. This is a controlled IGNN-style architecture, **not** an unchanged TorchDEQ/IGNN implementation, and is not evidence for arbitrary non-normal DEQ Jacobians.

## Windows PowerShell

Run from the `cat-mwbp` repository root:

```powershell
git pull
```

If you already created a virtual environment in the adjacent `CAT_revision` directory, keep using it:

```powershell
$python = (Resolve-Path ..\CAT_revision\.venv\Scripts\python.exe).Path
& $python -m pip install torch ogb
& $python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If that path does not exist, create a new one in the repository:

```powershell
py -3.13 -m venv .venv
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $python -m pip install -r requirements.txt
& $python -m pip install torch ogb
```

For NVIDIA acceleration when `torch.cuda.is_available()` is `False`, install a compatible CUDA-enabled PyTorch build following https://pytorch.org/get-started/locally/.

First run the offline synthetic smoke test without downloading data:

```powershell
& $python experiments\run_ignn_ogbn_arxiv.py `
  --smoke --device auto --epochs 2 --width 16 `
  --budgets 10,20,40 --out results\ignn_smoke
```

Once this prints `DONE`, start the real graph pilot:

```powershell
& $python experiments\run_ignn_ogbn_arxiv.py `
  --device auto --epochs 5 --width 32 --rho 0.95 --seed 0 `
  --budgets 10,20,40,80 `
  --data-root data\ogb --out results\ignn_arxiv_w32_s0
```

The first real run downloads `ogbn-arxiv` and may ask for confirmation. Send `solver_budgets.csv`, `training_history.csv` and `run_manifest.json` for analysis. The CSV contains local-action budgets, relative adjoint/gradient errors, run time, bounds, and GPU allocations. The PDF and PNG are caption-free.

## PyTorch 2.6+ OGB cache compatibility

Some released versions of `ogb.nodeproppred.NodePropPredDataset` load their
processed cache with `torch.load(path)` without specifying `weights_only`.
Newer PyTorch defaults to restricted `weights_only=True`, which may fail with
`_pickle.UnpicklingError: Unsupported operand 149`.

For a verified OGB `ogbn-arxiv` download/cache **you personally trust**,
run the existing experiment with the extra `--trust-ogb-cache` option:

```powershell
git pull origin main

& $python experiments\run_ignn_ogbn_arxiv.py `
  --device cpu --epochs 5 --width 32 --rho 0.95 --seed 0 `
  --budgets 20,30,40,50,60,70,80 `
  --data-root data\ogb --trust-ogb-cache `
  --out results\ignn_arxiv_w32_s0_corrected
```

This opt-in temporarily sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` **only
while OGB constructs and reads its dataset/split**, then restores the original
environment variable. It can execute arbitrary code when reading a malicious
pickle. Do **not** use it for third-party downloads or unverified cached
checkpoint files. The default behavior is unchanged unless the flag is given.
This is an upstream OGB/PyTorch compatibility workaround, not a CAT numerical
change.

## Chebyshev verification note

The initial IGNN pilot used a nonstandard Chebyshev momentum update. The implementation was corrected in commit `a439a6804fed7bc461fc1f23a77c2230a78fa8b2` to reproduce the normalized Chebyshev first-kind polynomials on a positive spectral interval. **Any Chebyshev plots and CSV rows generated before this correction must be regenerated and must not be cited.** The AR, CAT and GMRES numerical recurrences are unchanged. Re-run the full comparison after `git pull` before interpreting method rankings.

## Measure terminal certification overhead without retraining

The original `solver_budgets.csv` reports solve time only. Computing a
residual certificate requires one additional `J^T v` application and global
norm reductions at each check. The published five-epoch and twenty-epoch
pilot figures therefore **do not include check time**.

The script `experiments/run_ignn_postcheck.py` reads the saved checkpoint,
rebuilds the frozen graph and adjoint right-hand side, and times each
terminal residual check separately. No parameter optimization is repeated.

```powershell
git pull origin main
$python = (Resolve-Path ..\CAT_revision\.venv\Scripts\python.exe).Path

foreach ($seed in 0,1,2) {
    & $python experiments\run_ignn_postcheck.py `
      --checkpoint "results\ignn_arxiv_w32_e20_s$seed\trained_model.pt" `
      --data-root data\ogb --trust-ogb-cache `
      --device cpu --budgets 20,30,40,50,60,70,80 `
      --out "results\ignn_postcheck_s$seed"

    if ($LASTEXITCODE -ne 0) { throw "Postcheck failed for seed $seed" }
}
```

Outputs: `solver_budgets_charged.csv`, `postcheck_manifest.json` per
seed. Columns include `total_charged_jtv_actions`,
`terminal_check_seconds` and `charged_seconds`, while offline errors are
computed *after* the check timing. One terminal evaluation is not a
continuously monitored stopping algorithm. A schedule of repeated checks
would incur additional graph operations and global reductions.

## Interpretation and limits

- Each solver receives the same frozen network, graph, and adjoint right-hand side.
- Relative-error calculations use a separate offline reference; they are not a practical local stopping rule.
- The spectral interval is bounded conservatively by the known `rho`, with no exact eigensolve.
- Report actual action count, relative error, time and memory. Fixed-budget comparisons do not imply equal final accuracy.
- A single seed is a pilot, not a statistical validation.
- Keep the official OGB train/validation/test split and do not commit the raw dataset.
- Before archiving with a DOI, fix the exact Git commit, packages, configuration and confirmed result files.
