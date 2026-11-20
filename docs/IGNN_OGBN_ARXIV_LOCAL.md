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

## Interpretation and limits

- Each solver receives the same frozen network, graph, and adjoint right-hand side.
- Relative-error calculations use a separate offline reference; they are not a practical local stopping rule.
- The spectral interval is bounded conservatively by the known `rho`, with no exact eigensolve.
- Report actual action count, relative error, time and memory. Fixed-budget comparisons do not imply equal final accuracy.
- A single seed is a pilot, not a statistical validation.
- Keep the official OGB train/validation/test split and do not commit the raw dataset.
- Before archiving with a DOI, fix the exact Git commit, packages, configuration and confirmed result files.
