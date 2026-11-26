# A9: Signed dual residual correction and full tied-parameter gradient

Let (I-T)x*=f, c=||T||_infty<1, g=Gx*, r=f-(I-T)x. For any approximate dual operator Z, define F=G-Z(I-T). Then

    g - (Gx+Zr) = F(I-T)^(-1)r

and the L1 error is bounded by E=||F||_entry,1 * ||r||_infty/(1-c). For the uncorrected gradient the bound is ||Zr||_1+E. A standard uniform bound uses Z=0. If the norm of the reported gradient exceeds its absolute error bound E, its relative error is bounded by E/(||g_report||_1-E). This is a classical dual-weighted residual identity applied to the implicit gradient, not a new general numerical theorem. See https://arxiv.org/abs/2001.01929 and https://doi.org/10.21136/AM.2017.0173-17.

## Pilot

Run: python research/async_cat/a9_dual_certificate.py --out a9_results.json

Frozen 12x12 tanh equilibrium, periodic four-neighbor grid and modular control with weakened cross-block couplings, width-4 blocks, rho=.95,.99,.999, seeds 40--42. The model has four shared parameters (horizontal and vertical coupling, input gain, bias), synthetic input and regression target, squared-error loss. G contains the derivatives with respect to all four parameters. Central finite differences check all four components for seed 40 at rho=.95 on both topologies. This is an actual full gradient for this small model, not a training experiment or evidence for high-dimensional parameter spaces.

Both CAT and Chebyshev receive the identical block splitting and conservative spectral interval. A dual operator is computed using 8 or 32 Chebyshev steps on four RHS. Uniform, signed (uncorrected), corrected gradient certificates and direct forward sensitivity use relative L1 tolerance 1e-5. Uniform references are repeated for both K settings; 234 recorded rows, 198 distinct solver/policy/budget configurations after deduplicating the 36 repeated uniform references. 234/234 converge; max actual relative error 9.819962021442566e-6; max certified relative error 9.968513213146429e-6. No observed violation beyond numerical tolerance. Not interval arithmetic.

Corrected primary-phase median reduction (uniform rounds / corrected rounds): CAT 1.60764 (K=8), 9.5 (K=32); Cheb 1.62092 and 10. Median dual residual entrywise norm relative to original G: .0123634 and 6.93505e-9. However including setup, all 144 paired signed/corrected comparisons lose in scalar sends versus uniform. Median uniform/corrected ratios CAT .634134 and .291045; Cheb .643386 and .298507. Direct sensitivity median sends / ordinary Cheb uniform sends =4.

The communication proxy is one scalar per distinct boundary source coordinate per destination block per vector application; a transpose product uses the reversed aggregate communication. Each primary iteration and certificate check is charged separately (no product reuse); the dual setup charges p*(K+1) vector applications including its residual check; direct sensitivity charges p products for both step and check. Reductions, local computation, memory, initial block setup and real network latency are excluded. Same-block setup is shared. A different implementation could reuse products, and multiple RHS with identical T and G could amortize dual setup, but neither is demonstrated here.

## Decision

A9 gives a valid error control mechanism but no net communication advantage in the tested single-loss scenario. Do not expand to large training or claim TNNLS readiness from A1--A9. Preserve this negative result. The next research question, if pursued, must be genuinely different: e.g. tracking the changing exact adjoint across optimization steps under bounded Jacobian/RHS drift, with equally warm-started and restart-aware Chebyshev controls, or a justified multi-loss reuse regime. Repeated scalar tuning or goal-oriented dual setup for a single loss is not supported by these results.
