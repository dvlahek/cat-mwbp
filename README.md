# CAT-MWBP

Reference implementation for **Causal Adjoint Transport (CAT)** and the accompanying Metric-Wave Backpropagation (MWBP) experiments.

CAT replaces a centrally assembled hidden-layer backward pass with a local adjoint field. In feed-forward networks the field converges to the exact backpropagation gradient, but the second-order dynamics do not improve training over a matched first-order Activation Relaxation (AR) rule at the tested budgets. MWBP adds a low-rank propagating metric state. The main feed-forward experiments show that this metric changes internally while leaving the final predictions essentially unchanged.

The repository also contains the implicit/recurrent extension. In that setting the adjoint is the solution of

\[
(I-J^T)\lambda=b,
\]

so credit assignment is a genuine iterative local solve. This is the regime in which CAT's second-order dynamics become useful. The spectral experiments compare CAT with optimally relaxed first-order AR and Chebyshev semi-iteration. The classification experiments use the same datasets already present in the feed-forward study and count the actual sparse local \(J^T v\) actions used during training.

**Version 1.0.0** is the reference version for the manuscript.

## Repository layout

- `src/metric_wave/` — core NumPy implementation used by the original CAT-MWBP study.
- `experiments/run_journal_suite.py` — feed-forward benchmark and statistical analysis.
- `experiments/run_transport_probes.py` — finite-hop output-factor and Procrustes transport probes.
- `experiments/run_optional_vision_probe.py` — original MNIST/CIFAR-10 metric-transport probe.
- `experiments/run_implicit_spectral_suite.py` — implicit/recurrent spectral benchmark with ordinary and asymmetric local grids, AR, CAT, and Chebyshev controls.
- `experiments/run_implicit_dataset_suite.py` — implicit classification benchmark on the core datasets, `synthetic_large`, MNIST, and CIFAR-10.
- `reference_results/` — compact reference summaries from the reported runs.
- `tests/` — numerical checks for the local credit and metric implementations.

## Installation

The experiments use Python, NumPy, SciPy, pandas, scikit-learn, and matplotlib. A clean environment is recommended.

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Feed-forward CAT-MWBP study

A short check is:

```bash
python run_npl_suite.py --profile quick --jobs 2 --output results/quick
python experiments/run_transport_probes.py --profile quick --output results/transport_quick
```

The full non-vision study is:

```bash
python run_npl_suite.py --profile full --jobs 4 --output results/full
python experiments/run_transport_probes.py --profile full --output results/transport_full
```

The main suite contains 861 completed runs. CAT64 is practically equivalent to BP-Momentum under the predeclared accuracy and loss margins. At the same local sweep budget, CAT64 does not improve training over AR64. The propagating MWBP metric is also indistinguishable from the non-propagating and energy-matched controls in predictive performance.

## Implicit/recurrent spectral study

The standard spectral run is:

```bash
python experiments/run_implicit_spectral_suite.py \
  --profile standard \
  --output results/implicit_spectral
```

The benchmark uses periodic local 2-D recurrent graphs, including a non-bipartite asymmetric graph. Each solver iteration performs one local \(J^T v\) action. The standard run compares first-order AR, second-order CAT, and Chebyshev semi-iteration over recurrent spectral radii from 0.50 to 0.99.

At a relative implicit-gradient error of \(10^{-6}\), the standard reference run gives a CAT-bound/AR-oracle speedup of 8.83x for the asymmetric linear graph at \(\rho=0.99\). In the nonlinear tanh problem, the corresponding speedups are 3.28x at \(\rho=0.95\), 3.82x at \(\rho=0.98\), and 3.34x at \(\rho=0.99\). CAT-oracle remains within a few percent of the Chebyshev control.

## Implicit classification on the manuscript datasets

Run the core and scaling datasets with:

```bash
python experiments/run_implicit_dataset_suite.py \
  --profile standard \
  --group core \
  --output results/implicit_core
```

Run the real-image probes with:

```bash
python experiments/run_implicit_dataset_suite.py \
  --profile standard \
  --group vision \
  --output results/implicit_vision
```

The recurrent operator is a fixed sparse local ring. Each hidden unit communicates with four neighbours, at offsets \(\pm1\) and \(\pm2\). AR and CAT use the same recurrent graph, initialization, minibatch order, stopping tolerance, and spectral envelope. The reported solver count is the actual number of sparse local \(J^T v\) calls.

Across the eight core/scaling datasets and three seeds, CAT uses fewer local adjoint actions in all 24 paired runs, with a median speedup of 2.67x and no change in test accuracy. The MNIST and CIFAR-10 runs add six paired comparisons. CAT is faster in all six, with mean dataset-level speedups of 1.77x on MNIST and 1.40x on CIFAR-10, again with unchanged test accuracy.

The image models are controlled implicit-gradient probes, not competitive vision architectures. Their role is to test solver cost at fixed predictive behaviour.

## Reference results

Compact summaries of the standard implicit runs are in `reference_results/implicit/`. Full run-level files, solver audits, figures, and checkpoints are regenerated by the scripts and are intentionally not stored in the repository.

## Scope

The feed-forward and implicit results should be read together. CAT does not provide a general optimizer advantage on ordinary feed-forward networks. Its advantage appears when the adjoint itself is an iterative implicit/recurrent solve, especially as the recurrent Jacobian approaches the critical regime. The heavy-ball and Chebyshev acceleration formulas used in the spectral analysis are classical. The contribution is the identification and validation of this regime for local CAT credit dynamics.

MWBP remains a bounded negative result in the present low-rank output-sourced setting. The code keeps those experiments because they establish where propagating update geometry does and does not affect learning.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The code is released under the MIT License.
