# CAT-MWBP Neural Processing Letters reproducibility suite

- Completed runs: 861 / 861
- Failed runs: 0
- Configuration hash: `7066c96c3bd2a8c7`
- Test set used only after restoring the best validation checkpoint.
- Feature scaler fitted only on the optimization-training partition.
- Aggregate confidence intervals resample datasets first and seeds second.
- Equivalence margins and the AR overdamped rate were fixed in the configuration before test evaluation.
- CAT and AR use equal synchronous local sweeps and equal one-hop transpose-Jacobian action counts.

## Aggregate result

Lowest core test-loss rank: BP-Momentum (4.414).
Dataset-level Friedman test: chi-square=53.9231, p=1.21912e-07.

## Primary aggregate comparisons

- FA-Momentum vs BP-Momentum: delta loss +0.0829575, hierarchical 95% CI [+0.0331348, +0.151203], dataset wins 0/7.
- DFA-Momentum vs BP-Momentum: delta loss +0.115663, hierarchical 95% CI [+0.0248015, +0.244728], dataset wins 0/7.
- LocalHead-Momentum vs BP-Momentum: delta loss +0.113753, hierarchical 95% CI [+0.0397903, +0.203454], dataset wins 0/7.
- AR64-Momentum vs BP-Momentum: delta loss -0.000158682, hierarchical 95% CI [-0.000680374, +0.000106797], dataset wins 3/7.
- CAT40-Momentum vs BP-Momentum: delta loss +0.00106664, hierarchical 95% CI [-0.000194005, +0.00251357], dataset wins 1/7.
- CAT64-Momentum vs BP-Momentum: delta loss +2.41812e-05, hierarchical 95% CI [-4.19773e-05, +8.11706e-05], dataset wins 2/7.
- BP-MWBP vs BP-Momentum: delta loss +0.000141144, hierarchical 95% CI [-0.000536385, +0.000866584], dataset wins 3/7.
- CAT64-Momentum vs AR64-Momentum: delta loss +0.000182863, hierarchical 95% CI [-4.39106e-05, +0.000675762], dataset wins 4/7.
- CAT64-Momentum vs CAT40-Momentum: delta loss -0.00104246, hierarchical 95% CI [-0.00247776, +0.000232056], dataset wins 6/7.
- CAT64-MWBP vs CAT64-Momentum: delta loss +0.000212233, hierarchical 95% CI [-0.000437844, +0.000935742], dataset wins 3/7.
- CAT64-MWBP vs CAT64-MWBP-local: delta loss -0.000288639, hierarchical 95% CI [-0.00122433, +0.000395716], dataset wins 4/7.
- CAT64-MWBP vs CAT64-MWBP-local-EM: delta loss -2.45884e-05, hierarchical 95% CI [-0.000521022, +0.00050053], dataset wins 3/7.

## Practical equivalence to BP

- AR64-Momentum, test_accuracy, margin +/-0.005: supported (TOST p=1.79827e-09; difference -9.52381e-05).
- AR64-Momentum, test_loss, margin +/-0.001: supported (TOST p=0.00140498; difference -0.000158682).
- CAT40-Momentum, test_accuracy, margin +/-0.005: supported (TOST p=0.000264476; difference -0.00137837).
- CAT40-Momentum, test_loss, margin +/-0.001: not supported (TOST p=0.542012; difference +0.00106664).
- CAT64-Momentum, test_accuracy, margin +/-0.005: supported (TOST p=2.29561e-12; difference -3.1746e-05).
- CAT64-Momentum, test_loss, margin +/-0.001: supported (TOST p=1.67653e-08; difference +2.41812e-05).

## Six-block gradient validation

- CAT40: relative error 0.0834836, cosine 0.997487.
- CAT64: relative error 0.00408823, cosine 0.999994.
- CAT100: relative error 1.69767e-05, cosine 1.000000.
- CAT160: relative error 6.11643e-10, cosine 1.000000.

## Activation Relaxation comparison

- AR64: relative error 0.00727585, cosine 0.999981 at the same local sweep budget.
- CAT64: relative error 0.00408823, cosine 0.999994 at the same local sweep budget.
- CAT64-Momentum vs AR64-Momentum: delta loss +0.000182863, hierarchical 95% CI [-4.39106e-05, +0.000675762], median runtime ratio 1.291.

## Overdamped-limit audit

- Configured AR rate: 0.16.
- Derived CAT overdamped rate: 0.16.
- Maximum absolute gradient difference across 81 audit cases: 0.
- Maximum relative gradient difference: 0.

## Energy-matched ablation

- Mean calibrated local coupling: 2.250650.
- Mean relative calibration mismatch: 0.000153997.

## Interpretation guardrail

Do not claim optimizer superiority from ranks alone. Report paired differences, confidence intervals, corrected p-values, runtime, early-stopping behavior, and the distinction between local credit transport and metric propagation.
