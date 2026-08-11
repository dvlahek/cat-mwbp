# CAT-MWBP

Reference implementation and reproducibility code for **Causal Adjoint
Transport with Metric-Wave Backpropagation (CAT-MWBP)**.

CAT propagates the output error through neighbouring layers with local
second-order adjoint dynamics. Its equilibrium recovers the backpropagation
gradient. MWBP uses the resulting block gradients as sources for a symmetric
low-rank metric field that preconditions each local parameter update. The code
also implements the matched first-order Activation Relaxation (AR) baseline,
standard backpropagation, feedback-alignment controls, local auxiliary heads,
non-propagating metric ablations, and output-factor transport probes.

This is a numerical optimization model inspired by finite-speed wave
propagation. It does not model physical gravitational waves.

## Repository contents

- `src/metric_wave/` contains the NumPy implementation.
- `experiments/run_journal_suite.py` runs the validation-controlled benchmark
  and produces all statistical tables and figures.
- `experiments/run_transport_probes.py` tests finite-hop output-factor support
  and Procrustes-transported gauge fields.
- `experiments/run_optional_vision_probe.py` runs the optional real-data MNIST
  or CIFAR-10 probe.
- `tests/` contains numerical tests for locality, finite-hop support,
  positive-definite metric action, the CAT-to-AR overdamped limit, Woodbury
  inversion, and full-reach transport.
- `reference_results/` contains compact non-vision outputs from the reported
  861-run protocol. Large epoch histories and checkpoints are intentionally
  omitted.

## Installation

The reported run used Python 3.13.7. Create a clean environment and install the
pinned dependencies.

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

## Quick verification

Run the unit tests first.

```bash
python -m unittest discover -s tests -v
```

Then run the short benchmark and transport probe.

```bash
python run_npl_suite.py --profile quick --jobs 2 --output results/quick
python experiments/run_transport_probes.py --profile quick --output results/transport_quick
```

The quick profile is a functional check. Its numerical results are not the
reported full-study estimates.

## Full non-vision protocol

```bash
python run_npl_suite.py --profile full --jobs 4 --output results/full
python experiments/run_transport_probes.py --profile full --output results/transport_full
```

The first command schedules 861 runs across seven core datasets and one larger
synthetic scaling dataset. It also performs the CAT/AR equal-budget comparison,
the independently implemented overdamped-limit audit, equivalence tests,
dataset-level paired tests, hierarchical bootstrap intervals, Friedman tests,
and multiple-comparison correction. The run is checkpointed and can be resumed
with the same command and output directory. Runtime depends strongly on the
number of worker processes and available CPU cores.

Generated output includes raw run-level CSV files, validation histories,
summary tables, figures, LaTeX tables, environment metadata, and SHA-256
checksums.

## Optional MNIST and CIFAR-10 probes

The vision probe downloads the real datasets and does not substitute synthetic
data. The three-hop condition has partial support in the six-block networks and
must be described as **partial three-hop output-factor transport**.

MNIST configuration used for the reported partial-support comparison:

```bash
python experiments/run_optional_vision_probe.py \
  --dataset mnist \
  --seeds 0 1 2 3 4 \
  --epochs 40 \
  --subsample 8000 \
  --methods instantaneous partial_3hop local_output_only \
  --output results/vision_mnist.csv
```

CIFAR-10 configuration:

```bash
python experiments/run_optional_vision_probe.py \
  --dataset cifar10 \
  --seeds 0 1 2 \
  --epochs 40 \
  --subsample 8000 \
  --methods instantaneous partial_3hop local_output_only \
  --output results/vision_cifar10.csv
```

For an all-block transport check, add `full_reach` or `relaxed_10hop` to the
method list. These conditions are separate from the reported three-hop vision
result.

## Experimental safeguards

- Data are split into training, validation, and test partitions using a
  60/15/25 ratio.
- Feature scaling is fitted on the training partition only.
- The test partition is evaluated after the lowest-validation-loss checkpoint
  is restored.
- CAT and AR are compared at equal synchronous local-sweep and one-hop
  Jacobian-action budgets.
- The AR rate is fixed to the analytically derived overdamped CAT rate.
- The energy-matched local metric calibration uses training data only.
- Random seeds and the complete full-profile configuration are written to the
  output directory.

## Interpreting the results

The experiments support finite-step local credit transport and convergence to
the exact-backpropagation equilibrium. They do not establish predictive
superiority of CAT or spatial metric propagation. Comparisons should be based
on paired differences, uncertainty intervals, equivalence tests, runtime, and
the distinction between credit transport and metric transport.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The implementation is
released under the MIT License.

