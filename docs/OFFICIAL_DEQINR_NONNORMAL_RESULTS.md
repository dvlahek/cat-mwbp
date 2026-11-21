# Unprojected upstream TorchDEQ DEQ-INR: three-seed measured outcome

**Status:** completed local native-source runs (seeds 0, 1, 2) and one smoke run. This is numerical evidence and provenance, **not manuscript text**. The original model code is imported from upstream TorchDEQ at commit `4f6bd5fa66dd991cad74fcc847c88061764cf8db`. The **weights and injection are not projected or symmetrized**. The compact 48×48 camera/siren-like configuration, custom IFT training, and same-image 80/20 pixel split are **not** a reproduction of the original published training benchmark.

**Uploaded source archive SHA-256:** `946e0d7abd0535e0885916a6ec4c9f701e1c272cce6a1c63f0e76e70491b9aaf` (`nativeall.zip`). Full per-budget CSVs, checkpoints and training logs remain in the local archive, not in this minimal repository. The checkpoints' SHA-256 values are recorded in [the concise results CSV](../revision_results/official_deqinr_nonnormal_summary.csv), and all three fingerprints were checked against the actual archive entries.

## Numerical facts

- 240 independent exact-IFT training steps, 1,281 trainable parameters, 3 seeds. Held-out **same-image pixel** PSNR = 23.4293, 21.9360, 23.1882 dB.
- Each frozen adjoint audit uses 16 distinct held-out pixels (512 total eigenvalues across 32×32 real Jacobian blocks). The explicit matrix-free `J^Tv` agrees with upstream automatic differentiation to relative discrepancy `<3e-16`.
- Frozen `J` has complex eigenvalues in all 16 audited blocks per seed. Largest absolute imaginary eigenvalue across each audit = 0.5271, 0.5326, 0.4882. Largest measured `||J||_2` = 1.1125, 1.1030, 1.1035. The spectral radii are below one in these samples, but the nominal real interval `[0.05,1.95]` for `I-J^T` does **not** contain the complex spectrum.
- Smallest audited `sigma_min(I-J^T)` = 0.2704, 0.3114, 0.3307. Only the **offline global singular-value bound** is used for terminal certification; it is not a practical local stopping criterion.
- Three timing repeats on a Windows CPU, but small-scale millisecond comparisons depend on implementation. Only first **tested** passing budget on grid `5,10,15,20,30,40,60,80,100,120` is reported. Every solve is charged one extra `J^Tv` for its terminal residual check.

| Method | Seed 0 charged JTv | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| AR (alpha=1) | 31 | 31 | 31 |
| CAT with predeclared real-interval coefficients | no pass through 120 | no pass through 120 | no pass through 120 |
| Corrected scheduled Chebyshev on the same real interval | no pass through 120 | no pass through 120 | no pass through 120 |
| CAT, **global complex-spectrum oracle** | 31 | 21 | 21 |
| Restarted GMRES | 31 | 21 | 21 |

The real-interval CAT error *grows* rather than merely missing a strict numerical tolerance: relative gradient error at 120 actions is about `1.07e10`, `1.60e9` and `4.05e7` in the three seeds. By contrast, complex-oracle CAT reaches the `1e-5` **relative adjoint** certificate in all three seeds. In seeds 1 and 2, it passes on the measured budget grid with 21 rather than AR's 31 charged actions. This is **not** an independent test of coefficient generalization: optimization used the exact eigenvalues of all 16 audited blocks, including evaluation samples. It is **not** a locally computed, practical CAT calibration, and its precomputation cost is excluded from timing.

For every finite result in the source CSVs, the recorded global-SVD certified adjoint bound dominates the observed relative adjoint error. The first-pass tables agree with the reported per-budget passes; original checkpoint SHA-256 hashes also match.

## Interpretation and limitations

This genuinely tests an **unprojected, nonnormal and complex-spectrum** instance of the upstream DEQ-INR layer, and gives a valuable negative result for using real-interval CAT outside its theoretical domain. It does not show that fixed-coefficient CAT is the universally preferred backward solver on modern DEQs. Nor does the oracle eigenvalue fit supply an implementable local calibration or show robust out-of-sample transfer of spectral coefficients. Further research would require **training/calibration-independent held-out Jacobian blocks**, a deployable enclosure/calibration mechanism and broader architectures.

The native module experiment and the earlier constrained DEQ-INR experiment answer different questions and must not be conflated. No manuscript, supplementary manuscript or response to reviewers is stored on GitHub.
