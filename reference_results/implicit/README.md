# Implicit/recurrent reference summaries

These CSV files are compact summaries of the standard runs reported with the manuscript.

- `spectral_asymmetry_audit.csv` records the spectral intervals and AR step sizes for the ordinary and non-bipartite symmetric 2-D recurrent graphs. The historical file name is retained for compatibility with the original run output.
- `core_dataset_summary.csv` contains the eight fixed-operator core/scaling classification summaries over three seeds.
- `vision_dataset_summary.csv` contains the MNIST and CIFAR-10 solver-probe summaries over three seeds.
- `trainable_recurrent_summary.csv` contains the 20-seed trainable recurrent control on moons, circles, and breast cancer.
- `nonlocal_solver_reference.csv` contains the exact-error AR, CAT, GMRES, good-Broyden, and tuned-Anderson comparison.
- `anderson_best_config.csv` and `broyden_best_config.csv` record the selected reference-solver settings.
- `directed_spectrum_summary.csv` contains the directed normal and non-normal controls with the original real-interval parameters.
- `complex_spectrum_oracle_summary.csv` contains the final oracle-ellipse and exact-spectrum diagnostics for the translation-invariant directed normal operator and the heterogeneous non-normal controls at rho=0.95.

The experiment scripts regenerate the full run-level CSV files, audits, figures, and checkpoints. Those larger generated files are not committed here.
