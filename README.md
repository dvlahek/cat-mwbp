# CAT-MWBP

Reference implementation for **Causal Adjoint Transport (CAT)** and the accompanying Metric-Wave Backpropagation (MWBP) experiments.

CAT represents backward credit by a local adjoint state. In feed-forward networks this state converges to the exact backpropagation gradient, but the tested second-order dynamics do not improve training over matched first-order Activation Relaxation (AR). In implicit and recurrent networks the backward problem instead requires the iterative solution of

\[
(I-J^T)\lambda=b.
\]

This is the setting in which the second-order CAT state becomes useful. Relative to AR, CAT stores one additional local adjoint vector, uses the same neighbouring \(J^T v\) action, and requires no growing solver history. Its measured advantage in the real-spectrum near-critical regime is a substantial reduction in the number of local actions needed for the same exact gradient.

The intended computation model is relevant to reciprocal physical learning systems. In such substrates, neighbouring interactions can be implemented by the physical couplings themselves and reciprocal stiffness or coupling operators are naturally symmetric. Global Krylov inner products, dense quasi-Newton state, or Anderson least-squares reductions require additional readout and aggregation. The repository therefore distinguishes the fixed-memory local CAT/AR comparison from general solver references.

**Version 1.0.0** is the reference version for the manuscript.

## Repository layout

- `src/metric_wave/` contains the NumPy implementation used by the feed-forward experiments.
- `experiments/run_journal_suite.py` runs the main feed-forward benchmark.
- `experiments/run_transport_probes.py` runs the metric-transport controls.
- `experiments/run_optional_vision_probe.py` contains the original MNIST/CIFAR-10 transport probe.
- `experiments/run_implicit_spectral_suite.py` runs the implicit spectral benchmark.
- `experiments/run_implicit_dataset_suite.py` runs the fixed-recurrent classification benchmark.
- `experiments/run_nonlocal_solver_reference.py` compares AR, CAT, GMRES, good Broyden, and tuned Anderson at a common exact-error target.
- `experiments/run_trainable_recurrent_control.py` trains the local recurrent edge weights on three classification datasets.
- `experiments/run_directed_spectrum_control.py` tests directed complex-spectrum recurrent operators outside the main real-spectrum theory.
- `reference_results/` contains compact summaries of the reported runs.
- `tests/` contains numerical checks for the local credit and metric implementations.

## Installation

```bash
python -m venv .venv
```

On Windows:

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Feed-forward study

A short check is:

```bash
python run_npl_suite.py --profile quick --jobs 2 --output results/quick
python experiments/run_transport_probes.py --profile quick --output results/transport_quick
```

The main non-vision suite contains 861 completed runs. CAT64 is practically equivalent to BP-Momentum under the predeclared accuracy and loss margins. At the same local sweep budget, CAT64 does not improve training over AR64. The propagating MWBP metric is also indistinguishable from the non-propagating and energy-matched controls in predictive performance.

## Implicit spectral study

```bash
python experiments/run_implicit_spectral_suite.py --profile standard --output results/implicit_spectral
```

The benchmark uses periodic local 2-D recurrent graphs. One is an ordinary square grid. The other adds a symmetric diagonal pair, which removes the bipartite spectral symmetry while keeping the recurrent matrix symmetric. Each solver iteration performs one local \(J^T v\) action.

At relative adjoint or implicit-gradient error \(10^{-6}\), the conservative comparison gives a CAT-bound/AR-oracle speedup of 8.83x on the non-bipartite linear graph at \(\rho=0.99\). When both local methods receive the exact spectral interval, the corresponding AR-oracle/CAT-oracle ratio is 10.13x. The 11.74x value reported for the ordinary grid is a separate CAT-bound/AR-oracle control. CAT-oracle remains close to the Chebyshev reference.

## Fixed-recurrent classification

```bash
python experiments/run_implicit_dataset_suite.py --profile standard --group core --output results/implicit_core
```

The recurrent operator is a sparse degree-four local ring. Across the eight core/scaling datasets and three seeds, CAT uses fewer local adjoint actions in all 24 paired runs. The median AR/CAT action ratio is 2.67x and paired test accuracy is unchanged. MNIST gives a 1.77x reduction with the same accuracy. CIFAR-10 is retained only as a high-dimensional solver probe.

## Trainable recurrent control

```bash
python experiments/run_trainable_recurrent_control.py --profile standard --output results/trainable_recurrent
```

The degree-four recurrent edge weights are trained jointly with the classifier on `moons`, `circles`, and `breast_cancer`, using 20 seeds for each dataset. CAT uses fewer local \(J^T v\) actions in all 60 paired runs. The pooled median AR/CAT action ratio is 2.13x with a 95% bootstrap interval of [1.96, 2.30]. Paired test accuracy is unchanged.

## General solver reference

```bash
python experiments/run_nonlocal_solver_reference.py --output results/nonlocal_solver_reference
```

All methods are evaluated at the same exact target of \(10^{-6}\). The linear problems use relative adjoint error. The tanh problems use relative implicit-gradient error. GMRES uses full Arnoldi, good Broyden uses a tuned full inverse update, and Anderson is tuned over memory, mixing, and least-squares regularization.

Only \(J^T v\) actions are counted. GMRES inner products and orthogonalization, Broyden dense inverse updates, and Anderson history and least-squares work are not charged. GMRES and good Broyden therefore require fewer Jacobian actions than CAT on these controlled systems. CAT is not presented as a lower-action replacement for such general solvers. Its comparison of interest is the fixed-memory local one against matched first-order AR.

## Directed-spectrum scope control

```bash
python experiments/run_directed_spectrum_control.py --profile standard --output results/directed_spectrum
```

The directed experiment uses a translation-invariant nonsymmetric operator with complex spectrum and a heterogeneous non-normal local operator. AR and CAT intentionally retain the real-interval coefficients used in the symmetric study.

At \(\rho=0.80\), CAT converges in all 20 directed cases and the pooled median AR/CAT action ratio is 1.42x. At \(\rho=0.95\), CAT converges in 4 of 20 cases. At \(\rho=0.99\), it converges in none of the 20 cases, while AR converges throughout. This result defines the scope of the present real-interval CAT parameterization. Classical nonsymmetric Chebyshev and second-degree stationary methods can instead use complex spectral enclosures. The experiment does not claim that second-order acceleration is impossible for complex-spectrum systems.

## Reference results

Compact summaries are stored in `reference_results/implicit/`. Full run-level files, audits, figures, and checkpoints are regenerated by the experiment scripts and are not committed to the repository.

## Scope

The central positive result concerns fixed-memory local implicit credit. In reciprocal or symmetrizable near-critical systems, CAT adds one local adjoint vector relative to AR and substantially reduces the number of neighbouring Jacobian actions needed for the same exact gradient. The fixed-recurrent experiments favor CAT in all 24 core/scaling pairs and the trainable-recurrent control favors CAT in all 60 pairs without changing test accuracy.

The feed-forward and general-solver controls define the boundaries of this advantage. CAT does not provide a general optimizer advantage on ordinary feed-forward networks, and it is not a replacement for GMRES or quasi-Newton methods when their global operations and larger solver state are available.

The heavy-ball and Chebyshev formulas are classical. The contribution is not a new linear-solver theorem. It is the formulation and validation of a fixed-memory local credit dynamics, together with the identification of the regime in which its second-order state reduces neighbouring Jacobian actions relative to matched first-order relaxation.

MWBP remains a bounded negative result in the low-rank output-sourced setting studied here.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The code is released under the MIT License.
