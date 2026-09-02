# CAT

Reference code for **Causal Adjoint Transport (CAT)**, a two-state local adjoint dynamics for exact gradient computation in implicit and recurrent neural networks.

CAT represents backward credit by a local adjoint state. In an ordinary feed-forward network this state converges to the exact backpropagation gradient, but the tested second-order dynamics do not improve training over matched first-order Activation Relaxation (AR). The useful regime appears in implicit and recurrent networks, where the backward problem requires the iterative solution of

$$(I - J^\top)\lambda = b.$$

For a frozen implicit system, the two-state CAT recurrence is not a new numerical iteration. It is an instance of the classical linear second-degree stationary family analyzed by Manteuffel. The CAT name refers to its use as a persistent local neural adjoint dynamics and to the common feed-forward/implicit formulation used in the experiments.

Relative to AR, CAT stores one additional local adjoint vector, uses the same neighbouring $J^\top v$ action, and requires no growing solver history. In reciprocal or symmetrizable near-critical systems this reduces the number of local actions needed for the same exact gradient.

Chebyshev semi-iteration is the closest equal-locality numerical control. Once the spectral information is supplied, it also uses two vectors and local $J^\top v$ actions, and its action count is essentially the same as CAT in the spectral experiments. CAT uses fixed coefficients after calibration, whereas Chebyshev follows an iteration-dependent coefficient schedule. A scheduled Chebyshev implementation therefore requires the participating units to share the same coefficient index, while CAT does not. This is a structural difference in the local update, not an action-count advantage over Chebyshev.

The intended computation model is motivated by physical learning systems with local reciprocal couplings. Neighbouring interactions can be implemented directly by the substrate, while Krylov inner products, dense quasi-Newton state, or Anderson least-squares reductions require additional readout and aggregation. These examples motivate the operation model only. The repository does not claim a measured hardware speed, energy, or communication advantage.

**Version 1.0.0** is the manuscript reference release.

## Main files

- `src/metric_wave/` - NumPy implementation shared by the feed-forward experiments.
- `run_npl_suite.py` - main feed-forward benchmark.
- `experiments/run_implicit_spectral_suite.py` - implicit spectral benchmark.
- `experiments/run_implicit_dataset_suite.py` - fixed-recurrent classification benchmark.
- `experiments/run_trainable_recurrent_control.py` - trainable recurrent-edge control.
- `experiments/run_nonlocal_solver_reference.py` - exact-error AR, CAT, GMRES, good-Broyden, and tuned-Anderson comparison.
- `experiments/run_directed_spectrum_control.py` - directed normal and non-normal complex-spectrum control with the original real-interval parameters.
- `experiments/run_complex_spectrum_oracle.py` - oracle elliptic and exact-spectrum tuning for the directed normal and heterogeneous non-normal controls.
- `reference_results/implicit/` - compact summaries of the reported standard runs and manuscript audits.

## Mapping to the manuscript

| Script | Reproduces |
|---|---|
| `experiments/run_implicit_spectral_suite.py` | Table 2, Figure 1 |
| `experiments/run_nonlocal_solver_reference.py` | Table 3 |
| `experiments/run_implicit_dataset_suite.py` | Table 4, Figure 2, MNIST and CIFAR-10 results |
| `experiments/run_trainable_recurrent_control.py` | Table 5 |
| `experiments/run_directed_spectrum_control.py` | Supplementary Table S1 |
| `experiments/run_complex_spectrum_oracle.py` | Supplementary Table S2, Figure 3 |
| `run_npl_suite.py` | Supplementary Table S3 |

## Installation

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the numerical checks with:

```bash
python -m unittest discover -s tests -v
```

## Feed-forward study

```bash
python run_npl_suite.py --profile full --jobs 2 --output results/feedforward_full
```

The main non-vision suite contains 861 completed runs. CAT64 is practically equivalent to BP-Momentum under the predefined accuracy and loss margins. At the same local sweep budget, CAT64 does not improve training over AR64.

For a short installation check, use --profile quick.

## Implicit spectral study

```bash
python experiments/run_implicit_spectral_suite.py --profile standard --output results/implicit_spectral
```

The benchmark uses periodic local 2-D recurrent graphs. One is an ordinary square grid. The second adds a symmetric diagonal pair, removing the bipartite spectral symmetry while keeping the recurrent matrix symmetric. Each solver iteration performs one local $J^\top v$ action.

At relative adjoint or implicit-gradient error $10^{-6}$, CAT-bound requires 8.83 times fewer actions than AR-oracle on the non-bipartite linear graph at $\rho = 0.99$. When both local methods receive the exact spectral interval, the corresponding AR-oracle/CAT-oracle ratio is 10.13. The separate ordinary-grid AR-oracle/CAT-bound control gives 11.74.

## Fixed and trainable recurrent classification

Fixed recurrent operator:

```bash
python experiments/run_implicit_dataset_suite.py --profile standard --group core --output results/implicit_core
```

CAT uses fewer local adjoint actions in all 24 core/scaling pairs. The median AR/CAT action ratio is 2.67 and paired test accuracy is unchanged. Across the eight dataset-level summaries, the median recurrent Jacobian radius and oracle AR/CAT action ratio are perfectly rank ordered. More strongly, the action reduction follows the asymptotic second-degree prediction computed from the exact sample-specific spectral intervals: the eight dataset-level theoretical and measured oracle ratios have Pearson correlation 0.997. The measured ratio is approximately 0.81-0.84 of the asymptotic prediction across all eight datasets. MNIST gives a 1.77 ratio with the same accuracy.

Trainable recurrent operator:

```bash
python experiments/run_trainable_recurrent_control.py --profile standard --output results/trainable_recurrent
```

The degree-four recurrent edge weights are trained jointly with the classifier on `moons`, `circles`, and `breast_cancer`, using 20 seeds per dataset. All 60 paired runs favour CAT in local action count. The pooled median AR/CAT ratio is 2.13 with a 95% interval of [1.96, 2.30]. Paired test accuracy is unchanged.

## General solver reference

```bash
python experiments/run_nonlocal_solver_reference.py --output results/nonlocal_solver_reference
```

All methods are evaluated at the same exact target of $10^{-6}$. Only $J^\top v$ actions are counted. GMRES inner products and orthogonalization, Broyden dense inverse updates, and Anderson history and least-squares work are deliberately not charged. GMRES and good Broyden therefore require fewer Jacobian actions than CAT on these controlled systems. This is not presented as a wall-time comparison or as a CAT advantage over general-purpose solvers.

## Directed and complex-spectrum controls

The first control keeps the real-interval coefficients used in the symmetric study:

```bash
python experiments/run_directed_spectrum_control.py --profile standard --output results/directed_spectrum
```

At $\rho = 0.80$, CAT converges in all 20 directed cases and the pooled median AR/CAT action ratio is 1.42. Near criticality, the same real-interval coefficients lose stability as the spectrum becomes complex.

The oracle diagnostic tests the parameterization without changing the two-state CAT recurrence:

```bash
python experiments/run_complex_spectrum_oracle.py --output results/complex_spectrum_oracle
```

For the translation-invariant directed linear operator at $\rho = 0.95$, the real-interval coefficients have a predicted worst characteristic-root modulus of 1.030 and fail in all 20 right-hand-side trials. Oracle elliptic tuning reduces the factor to 0.878 and restores exact-error convergence in all 20 trials with median 106 local actions. Direct optimization on the exact discrete spectrum gives the same median count.

The same test is repeated on five heterogeneous non-normal local operators, with four right-hand sides for each operator. The normalized commutator measure $\eta_{\mathrm{NN}}$ (Eq. 20 in the article) ranges from 0.058 to 0.070. Real-interval CAT converges in 8 of 20 trials. Oracle elliptic tuning reduces the predicted root factor to 0.867-0.882 and restores convergence in all 20 trials, with median 106.5 local actions. Exact-spectrum tuning also converges in all 20 trials with median 104.

These diagnostics show that the near-critical directed failures in the tested operators are primarily parameterization failures rather than failures of the two-state recurrence. The spectral enclosures are oracle information. The code does not provide a local enclosure estimator or a guarantee for arbitrary strongly non-normal implicit Jacobians.

## Scope

The central positive result concerns fixed-memory local implicit credit. CAT adds one local adjoint vector relative to AR and can substantially reduce the number of neighbouring Jacobian actions required near criticality without changing the exact gradient target or predictive accuracy. The measured reduction also follows the spectral dependence predicted by the classical second-degree contraction factors across the fixed-recurrent datasets.

The controls define the boundaries of that result. CAT does not provide a general optimizer advantage on feed-forward networks. Chebyshev reaches essentially the same local action count when its coefficient schedule is available. GMRES and quasi-Newton methods can use fewer Jacobian actions when global operations and larger solver state are available. Complex spectra require a spectral parameterization appropriate to the complex domain, and the current complex-spectrum experiments use oracle enclosures.

The heavy-ball, Chebyshev, and nonsymmetric second-degree acceleration principles used in the analysis are classical. The contribution is the neural computation model, the use of a stationary two-state local adjoint dynamics, and the identification and validation of the regimes in which the additional local state is useful relative to matched first-order relaxation.

## Material not reported in this article

The repository also contains an earlier Metric-Wave Backpropagation (MWBP) line of experiments that is not part of the manuscript. It is retained for completeness and is not required to reproduce any reported result.

- `experiments/run_transport_probes.py` - metric-transport controls
- `experiments/run_optional_vision_probe.py` - MNIST/CIFAR-10 transport probe

```bash
python experiments/run_transport_probes.py --profile quick --output results/transport_quick
```

The package directory `src/metric_wave/` keeps its original name for import compatibility; it contains the shared NumPy implementation used by all feed-forward experiments. The repository name also dates from this earlier line of work.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The code is released under the MIT License.
