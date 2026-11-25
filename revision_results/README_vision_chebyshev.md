# MNIST and CIFAR-10 adjoint comparisons

The normalized first-kind Chebyshev recurrence was evaluated on six deterministically retrained CAT-bound models (three seeds each for MNIST and CIFAR-10). Eight held-out examples per checkpoint were compared at the same frozen adjoint and relative implicit-gradient-error tolerance of 1e-6. CAT required fewer actions than first-order AR in 48/48 cases. Chebyshev required fewer actions than CAT in 35/48 and tied in 13/48 cases; none required more.

The medians of the per-sample Chebyshev/CAT ratios, first within seeds then across three seeds, are **0.925 (MNIST)** and **0.917 (CIFAR-10)**. The CAT model accuracies are 0.9077 and 0.3728. The 144 solution counts and errors were independently checked against the saved local NPZ checkpoints.

Files in this folder: `vision_cat_retraining.csv`, `vision_corrected_chebyshev_actions.csv`, `vision_corrected_chebyshev_pairs.csv`, `vision_corrected_chebyshev_seed_summary.csv`, `vision_corrected_chebyshev_dataset_summary.csv`, and `vision_chebyshev_manifest.json`. The manifest contains hashes of the committed CSV text and local checkpoint files.

Reproduce with:

```bash
python experiments/recheck_chebyshev_vision.py --smoke
python experiments/recheck_chebyshev_vision.py --repo-root . --out results/chebyshev_vision_recheck
```

The audit uses the same frozen CAT-trained model for all three solvers within each seed. The vision checkpoint binaries are not included.
