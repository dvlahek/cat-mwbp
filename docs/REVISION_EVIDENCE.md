# CAT revision evidence, November 2026

This is the **post-v1.0.0 journal-revision checkpoint** for *Local Second-Order Adjoint Dynamics for Implicit Neural Networks*. Original v1.0.0 outputs and code paths have been retained for reproducibility. Some historical Chebyshev values must **not** be used as corrected Chebyshev benchmarks. The controlled results below are not claims of universal solver superiority or unmodified DEQ coverage.

**Verifiable compact data:** [`../revision_results/`](../revision_results/) and `python experiments/verify_revision_results.py`. The compact CSVs have per-file SHA-256 digests in `revision_results/MANIFEST.json`. The full training histories, model checkpoints, and image figures belong in the eventual archival deposit, not in this lightweight GitHub snapshot. No Zenodo DOI has yet been minted by this update.

## 1. Corrected Chebyshev baseline — mandatory correction

The historical `experiments/run_implicit_spectral_suite.py` implements an inconsistent Chebyshev recurrence. Do not use the historical Chebyshev counts in main-text tables or figures as if correct. For the same cases, `experiments/run_chebyshev_corrected_audit.py` validates the standard first-kind Chebyshev polynomial and re-runs the baseline.

The **240** matched CAT-versus-Chebyshev comparisons comprise 164 cases in which corrected Chebyshev uses fewer local actions, 42 ties, and 34 cases in which CAT uses fewer. The pooled median `Chebyshev/CAT` action ratio is **0.9130**. Chebyshev is an equal-locality, two-vector control with iteration-dependent coefficients. CAT has **constant** coefficients after spectral calibration. The experiments do not show that CAT generally outperforms corrected Chebyshev. This distinction must be reflected in the manuscript, figure labels, and Reviewer 1 reply.

See `revision_results/chebyshev_corrected_pairs.csv` and `revision_results/chebyshev_corrected_summary.csv`. Reproduce with:

```bash
python experiments/run_chebyshev_corrected_audit.py --out results/cheb_check_full
```

## 2. Anderson, GMRES and Broyden — runtime and state

The final `experiments/run_solver_cost_audit.py` uses selected fixed hyperparameters in `reference_results/implicit/` on the original linear and tanh 12x12 symmetric graph controls, across five seeds, three timing repeats and single-threaded BLAS. Bounded Anderson-m history is checked numerically against the original implementation, so avoid comparing against accidentally unbounded history storage.

For tanh with `rho=0.95`, the median actions / time are: AR **160 / 1.676 ms**, CAT **39 / 0.405 ms**, Anderson-m **158 / 9.839 ms**, GMRES **24 / 1.503 ms**, and good Broyden **29 / 1.126 ms**. This is not a general-performance ranking: the `n=144` small-system setup, CPU implementation, BLAS configuration, and solver-library choices matter. The estimated persistent solver state is a structural count, **not** a measured physical RAM peak; Python-allocated memory and process RSS are separately sampled but also have limitations.

See `revision_results/solver_cost_summary.csv` and `docs/CAT_FINAL_EXPERIMENTS.md`. Reproduce with:

```bash
python experiments/run_solver_cost_audit.py --seeds 5 --repeats 3 --blas-threads 1 --out results/solver_cost_audit
```

## 3. Graph learning, OGB `ogbn-arxiv`

The model is a **controlled symmetric contractive IGNN-style layer**, trained on the official OGB split. It is not a standard unmodified DEQ architecture. Three independently trained checkpoints (seeds 0–2, width 32, 20 epochs, `rho=0.95`) have **169,343 graph nodes**, **5,418,976 equilibrium-state variables** and **5,448 trainable parameters**, with mean test accuracy **52.66%**. Do not equate equilibrium-state size with trainable parameter count.

At a predeclared **`1e-5` conservative relative *adjoint* error certificate**, with one additional terminal residual `J^T v` and global reductions charged per fixed-budget run, all three seeds first pass on the tested grid with **AR 81**, **CAT 51**, **Chebyshev 51** and **GMRES 31** total actions. Median within-seed AR/CAT time speedup is **1.55x**. The costs do **not** establish free fully local online stopping; continuously checking convergence incurs additional global operations. Time values are single CPU timing passes per budget, not a hardware-independent speed claim or repeated-sample confidence interval.

See `revision_results/ignn_*`, `docs/IGNN_OGBN_ARXIV_LOCAL.md`, `experiments/run_ignn_postcheck.py` and the full original run artifacts in the archival package when published.

## 4. Spatial equilibrium network, real-image denoising

Three seeds trained 300 steps on source-disjoint images bundled with `scikit-image`. The **controlled, contractive spatial implicit network** (11,953 parameters, 55,296 frozen audit-state variables) has a symmetric neighborhood recurrence. It is not an off-the-shelf modern vision DEQ. The test images do not overlap the train or validation source images.

At the same terminal `1e-5` conservative adjoint certificate, charged actions are **AR 161, CAT 46, corrected Chebyshev 46** for all three seeds and **GMRES 31–36**. Median within-seed AR/CAT CPU time ratio is **3.56x**, contingent on the tested implementation. The Gaussian corruption is sigma 0.12. Test-set mean PSNR for CAT-trained models is **27.144, 26.873, 26.320 dB** for seeds 0, 1 and 2, versus the *post hoc best tested Gaussian width 1.2* at **26.841, 26.862, 26.893 dB**. Thus the network **does not consistently outperform Gaussian smoothing**. The post hoc best test Gaussian should not be described as a preselected validation baseline. See `docs/SPATIAL_DENOISING_CAT.md` and `revision_results/spatial_*`.

```bash
python experiments/run_spatial_implicit_denoising.py --seed 0 --steps 300 --side 80 --patch 48 --eval-side 64 --audit-side 48 --width 24 --rho 0.95 --center-weight 0.35 --sigma 0.12 --lr 0.002 --threads 2 --repeats 3 --out results/spatial_denoising_s0
python experiments/summarize_spatial_denoising.py --root results --prefix spatial_denoising_s --out results/spatial_denoising_summary
```

Repeat the training command for `--seed 1` and `--seed 2`, with corresponding output names, before running the summarizer.

## 5. DEQ-INR — negative/control result

An original TorchDEQ DEQ-INR module was imported, but its recurrent operator and coordinate injection were **constrained** to permit a real-spectrum contractive CAT calibration. The image reconstruction is weak (held-out PSNR approximately 13.17 dB for widths 48 and 96). A simple linear coordinate regression matches/exceeds this performance, and the larger model loses in solver action count to AR. It is **not** evidence of successful image reconstruction or a CAT advantage for arbitrary TorchDEQ. At most it belongs in supplementary limitations or a transparent negative/transfer control; do not count it as an independent successful benchmark.

## 6. Unprojected upstream TorchDEQ DEQ-INR — native complex/nonnormal stress result

A second, deliberately **unprojected** experiment uses original upstream DEQ-INR/MFN/SIRENInjection modules. It does not symmetrize or project the recurrent matrix, and the model is trained by an independent exact dense IFT adjoint, not by CAT. This compact CPU setup is not the published full DEQ-INR training benchmark. On a single camera image (same-image train/holdout pixel split), three model seeds achieve held-out PSNR 23.43, 21.94 and 23.19 dB with 1,281 trainable parameters. Frozen Jacobian blocks have complex eigenvalues and are nonnormal. Their maximum `||J||_2` is about 1.10–1.11.

At the first tested budget reaching an **oracle-SVD-based relative adjoint certificate** of `1e-5`, AR uses 31 charged actions in all three seeds. **CAT and corrected Chebyshev with the predeclared real interval fail to converge even by 120 tested iterations.** An optional **global complex-spectrum, in-sample oracle calibration** recovers CAT convergence at 31/21/21 actions, matching GMRES action budgets on this grid. The oracle optimizer uses *all audited eigenvalues, including the evaluation blocks*, and the SVD stopping bound is also nonlocal. This is useful for identifying a failure mode, not a deployable or generalizing CAT advantage. See [native DEQ-INR audit](OFFICIAL_DEQINR_NONNORMAL_RESULTS.md) and [compact numeric CSV](../revision_results/official_deqinr_nonnormal_summary.csv). Full result archive checksum is documented there.

This increases the external architectural scope, **but does not close a request for three or four successful unmodified modern DEQ applications**. No universal CAT advantage should be claimed from this negative control.

### Held-out coefficient transfer on the native nonnormal models

A follow-on [transfer audit](NATIVE_DEQINR_SPECTRAL_TRANSFER.md) freezes coefficients chosen **without evaluation Jacobian eigenvalues**. Calibration from the same model's 64 *training-pixel* Jacobian blocks gives CAT first-certification counts **20, 20, 25** versus AR's **20, 21, 26** across three models. More stringent calibration on *only the other two models* gives CAT **21, 22, 27**. The nominal invalid real-interval CAT does not pass within 120 iterations. Thus spectral fitting from separate data can restore **stability**, but does **not** show meaningful acceleration or local calibration. Training-calibration eigenvalues and evaluation SVD still require global information. These checkpoints share the same image source, so do not present this as a new independent visual benchmark. The 64-pixel transfer test excludes the 16 previously inspected holdout pixels from the earlier oracle pilot. See [compact results](../revision_results/native_deqinr_transfer_first_pass.csv).

### Fair equal-information AR versus CAT calibration — completed negative result

A targeted [post-hoc native TorchDEQ follow-up](NATIVE_DEQINR_FAIR_CALIBRATION.md) compares **optimized AR** against CAT using exactly the same *training-only* calibration Jacobians and no evaluation-spectrum feedback. Each model uses 64 training pixels to choose coefficients, then tests (a) 128 previously uninspected held-out pixels for spectral minimax calibration, and (b) a separate 128 for finite-horizon residual calibration. Spectral-fitted AR first-pass charged JTv counts are **22/21/21** against CAT **21/21/21**. The leave-one-model-out fits give AR **22/23/22** versus CAT **22/24/22**. Constraining CAT inertia to beta >= 0.10 worsens its counts to **25/26/25**. With the same finite-horizon training objective, optimized AR needs **22/24/24** and CAT **21/25/24** operations. The selected CAT inertia is beta <= 0.015 in all six fitted experiments. Thus no meaningful practical CAT speed advantage over equally calibrated AR was demonstrated on these native nonnormal TorchDEQ checkpoints. This is a **negative** result, not a new positive benchmark. It remains a same-image, reused-checkpoint exploratory diagnostic with **global training-spectrum calibration, offline global-SVD certification and retrospective stopping**. [Code](../experiments/audit_native_deqinr_fair_calibration.py) and [compact CSV](../revision_results/native_deqinr_fair_calibration_summary.csv) are public. Full curves and models are local.

## What the editor and reviewers can safely infer

- **Strong controlled result:** CAT reduces local adjoint actions versus AR on tested contractive real-spectrum systems, particularly close to criticality; advantages occur in tabular, large-graph and spatial-image tasks.
- **Important negative result:** corrected Chebyshev is often slightly superior in action count and uses the same local Jacobian actions with a coefficient schedule. GMRES often needs fewer `J^T v` actions but has global Krylov operations and a larger history; neither is globally dominated by CAT.
- **Important limitation:** reference spectra or conservative enclosures are provided or assumed. Neither global stopping reduction nor spectral calibration is free. Nonnormal complex-spectrum behavior is not covered by the real-interval bound. The added image models are controlled rather than unmodified off-the-shelf DEQs.
- **Journal revision gap:** reviewer-requested breadth of 3–4 *modern independent DEQ architectures* is only **partially** met. Be candid in the response. Scaling plots need to label the number of learned parameters and the Jacobian/iteration counts separately; measured time and state estimates must not be conflated.
- **Release:** keep v1.0.0 historical results and corrected post-release audits separately, then mint an archival DOI after preparing/checking the frozen release. Do not announce a DOI that is not available.

## Submission handoff

Before resubmission, reconcile the full main `.tex`, supplement, figures and reviewer response against these compact tables. Correct or remove the original Chebyshev figure and table comparisons. Describe fixed coefficients versus scheduled Chebyshev, offline calibration, discrete budget first-crossings and one terminal check. The manuscript should not claim that CAT is universally fastest, that denoising beats all filters, or that three or four modern off-the-shelf architectures were tested.