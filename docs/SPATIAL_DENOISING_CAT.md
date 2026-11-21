# Controlled spatial implicit denoising – final revision audit

**Status:** three completed seeds, 300 training steps per seed, all four solvers certified on the same frozen adjoint system for each seed. Intended as an additional controlled application, not as a claim about general or unmodified DEQ architectures.

## Setup and provenance

- Spatial equilibrium `h* = tanh(U_theta(features(noisy)) + rho S h*)`, with `rho=0.95`, symmetric periodic five-point averaging stencil `S`, central weight 0.35, and a convolutional encoder `U_theta`. A local convolutional decoder predicts a bounded correction to a Gaussian 0.7 image.
- Because `S` is symmetric and doubly stochastic, `||S||_2 <= 1`. Since `||diag(tanh')||_2 <= 1`, the true Jacobian satisfies `||J||_2 <= rho < 1`. Where derivatives are positive, the Jacobian is diagonally similar to a symmetric matrix. CAT and Chebyshev use the conservative interval `[1-rho, 1+rho]` for `I-J^T`.
- Train images `camera`, `coins`, `moon`, `page`, `text`, `brick`, `grass`, `gravel`. Validation images `clock`, `chelsea`. Unused-in-pilot final test images `retina`, `immunohistochemistry`, `hubble_deep_field`, `cell`. All are bundled with `scikit-image` and resized to fixed dimensions. The train, validation and test sets are disjoint by source image, not merely pixel.
- Pixel noise `N(0,0.12²)` clipped to `[0,1]`. Training 300 steps, crops 48×48 from resized 80×80 source images. Test images 64×64. The best model checkpoint is chosen by validation PSNR, not by test PSNR.
- Three independently initialized models at seeds 0, 1 and 2. Each has 11,953 learned parameters. Solver audit uses the held-out `retina` image at 48×48 with a fixed (seed-dependent) noise realization. The state has 24×48×48 = 55,296 scalar variables.
- CPU, PyTorch 2.10.0, two threads, three solver timing repeats at each budget in this run. Individual timings are hardware- and implementation-dependent.
- The independent finite-difference check of the CAT implicit gradient passes at relative error `1.213e-09`. Conservatively certified reference adjoint bounds were `1.99e-09`, `1.91e-09` and `1.86e-09` for seeds 0, 1 and 2.

## Fixed-budget adjoint benchmark

Each solver gets exactly the same trained checkpoint, loss-gradient RHS and Jacobian-vector product. A terminal residual check costs one additional `J^Tv` operation and global norms. The bound `||error|| <= ||residual||/(1-rho)` is converted into a relative bound by dividing by a verified lower bound on the unknown solution norm. The criterion is `< 1e-5`.

| Solver | Seed 0 actions | Seed 1 actions | Seed 2 actions | Median charged time |
|---|---:|---:|---:|---:|
| AR | 161 | 161 | 161 | 34.878 ms |
| CAT | 46 | 46 | 46 | 9.876 ms |
| Corrected Chebyshev | 46 | 46 | 46 | 9.997 ms |
| Restarted GMRES | 36 | 36 | 31 | 22.959 ms |

AR/CAT action ratio is 3.50 across all three seeds. The median of within-seed CPU time ratios AR/CAT is **3.56** (individual ratios approximately 3.30, 3.62 and 3.56). Chebyshev is **equivalent in charged action count** and slightly faster/slower by seed. GMRES needs fewer actions but uses global inner products, orthogonalization and solves.

**Qualification:** these are first passing points in a *preselected discrete budget grid*. They are not exact optimal iteration counts. Only one terminal certificate per fixed-budget solve is charged. A hypothetical algorithm that checks all intervening budgets online would pay more for monitoring. Memory entries in the CSV are algorithmic state estimates, not measured process peak RAM. This is a one-image frozen-solver audit for each of three model seeds.

## Denoising quality

For each seed, mean PSNR (dB) over four held-out source images:

| Seed | Noisy | Gaussian 0.7 | Gaussian 1.2 | Gaussian 1.8 | Controlled spatial implicit |
|---|---:|---:|---:|---:|---:|
| 0 | 19.156 | 25.572 | 26.841 | 26.085 | 27.144 |
| 1 | 19.117 | 25.518 | 26.862 | 26.109 | 26.873 |
| 2 | 19.168 | 25.548 | 26.893 | 26.225 | 26.320 |

The trained model consistently improves the noisy input by about 7–8 dB, but **does not consistently outperform Gaussian 1.2**, which was the strongest of the tested Gaussian baselines on the final test images. The differences to Gaussian 1.2 are `+0.303`, `+0.011`, and `-0.572` dB. A Gaussian 1.8 filter was selected using the *validation* set, which is different from a post hoc best filter on test. The model beats that validation-selected filter on test in all three seeds, but is poorer on average on the validation images. We do not claim denoising superiority or state-of-the-art image restoration.

No external training dataset was downloaded. Earlier exploratory pilot development used `astronaut`, `coffee` and `rocket` instead of the four final test images. The model configuration used in this three-seed campaign was then held fixed. This is still a limited small-data evaluation, not an independent multi-dataset benchmark or a published DEQ image-restoration model.

## Recommended inclusion

- Main manuscript: the previously completed OGB `ogbn-arxiv` IGNN result remains the strongest large-state application. If room allows, present **one concise paragraph** with the spatial denoising action ratios and a clear note that reconstruction does not consistently beat Gaussian 1.2.
- Supplementary Information: spatial architecture definition, provenance, complete PSNR table, first-certified solver table, and the two-panel figure. Do not make a stronger inference about arbitrary DEQ networks.
- Reviewer response: cite the three distinct task domains covered by trained controlled implicit models (tabular classification, graph learning, spatial denoising), and explicitly acknowledge that this does not provide 3–4 independent off-the-shelf modern DEQ architectures. The breadth request is strengthened, not guaranteed fully resolved.

## Local reproduction (Windows PowerShell)

```powershell
$python = (Resolve-Path ..\CAT_revision\.venv\Scripts\python.exe).Path
& $python -m pip install numpy pandas scipy scikit-image matplotlib pillow

git pull origin main
foreach ($seed in 0,1,2) {
  & $python experiments\run_spatial_implicit_denoising.py `
      --seed $seed --steps 300 --side 80 --patch 48 `
      --eval-side 64 --audit-side 48 --width 24 --rho 0.95 `
      --center-weight 0.35 --sigma 0.12 --lr 0.002 `
      --threads 2 --repeats 3 `
      --out "results\spatial_denoising_s$seed"
  if ($LASTEXITCODE -ne 0) { throw "Spatial denoising seed $seed failed" }
}
& $python experiments\summarize_spatial_denoising.py `
    --root results --prefix spatial_denoising_s `
    --out results\spatial_denoising_summary
```

Run `--smoke` to verify installation on a small synthetic training run that still uses built-in real images for the denoising input. All output paths under `results/` are ignored by Git.

## Suggested manuscript paragraph (English)

> To examine a distinct visual task, we trained a controlled contractive spatial equilibrium network for Gaussian image denoising on source-disjoint images from the scikit-image collection. The equilibrium state comprised 55,296 variables in the frozen-solver audit. For a terminally certified relative adjoint-error bound of 1e-5, CAT used 46 charged Jacobian-transpose actions across all three trained seeds, versus 161 for activation relaxation. Corrected Chebyshev iteration also required 46 actions, while restarted GMRES required 31–36. The median paired CPU-time ratio between AR and CAT was 3.56 on the tested implementation. The models improved mean PSNR over the noisy inputs, but did not consistently outperform the strongest tested Gaussian smoothing baseline. These experiments probe backward-solver behavior in a constrained symmetric spatial equilibrium architecture and do not establish performance for general nonnormal DEQ networks.

## Suggested reviewer-response paragraph (English)

> We extended the evaluation beyond the original controlled classification studies with an implicit graph model on ogbn-arxiv and a separate contractive spatial equilibrium denoising task. The latter was run with three independently trained seeds, source-disjoint real images, an explicit finite-difference gradient check, and terminal residual certification that charges its Jacobian-transpose and norm-reduction costs. CAT reduced the certified action count from 161 (AR) to 46 for every seed, matching scheduled Chebyshev while GMRES needed fewer actions with higher measured CPU cost. We report the denoising PSNR and simple-filter baselines, including the seed where Gaussian smoothing performed better, rather than implying state-of-the-art restoration. Both additional models were deliberately restricted to real-spectrum contractive operators. We do not claim validation on unrestricted or off-the-shelf DEQ architectures.