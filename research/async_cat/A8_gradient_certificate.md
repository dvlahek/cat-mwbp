# A8: Goal-oriented residual certificates and event-triggered communication

For A=I-B=M-R, B>=0, rho(B)<1, and a fixed affine parameter gradient g=g0+Gx, set v=|G|^T 1. If w>=0 and A^T w>=v, then A^{-1}>=0 implies w>=A^{-T}v. For a common boundary cache xbar and componentwise sender discrepancy bound |x-xbar|<=tau on the cross-block source coordinates, the exact gradient error satisfies

    ||g(x)-g(x*)||_1 <= w^T (|b-Mx+R*xbar|+|R|tau) = E.

When ||g(x)||_1>E, the relative error is at most E/(||g(x)||_1-E). The certificate is independent of the solver. For B>=0 and ||B^T||_infty<=rho<1, a constructive supersolution is

    w_K = sum_{i=0}^{K-1}(B^T)^i v + ||(B^T)^K v||_infty/(1-rho) * 1.

Indeed applying A^T to the sum gives v-(B^T)^K v, and A^T 1 >= (1-rho)1. Computation requires K transpose applications and a global maximum. Floating-point tests use padding, not a rigorous interval arithmetic implementation. For signed operators a comparison-matrix extension would require separate hypotheses (e.g. || |B| ||<1 in a suitable norm).

The communication contribution is omega^T tau, omega=|R|^T w. The pilot compares full communication, uniform geometric thresholds, and inverse-omega weighted thresholds with equal total omega^T tau at each step. This allocation is a heuristic, not an optimized message-count solution. The full and suppressed variants all use the same certificate. The exact fixed-point objective is unchanged. A stable linear recurrence with a geometrically vanishing perturbation converges under consistent cache delivery; no theorem for arbitrary in-flight messages is asserted.

## Reproduce

    python research/async_cat/a8_gradient_certificate.py --out a8_results.json

216 solves: A7 grid and modular topology, linear and tanh, rho .95/.99/.999, seeds 30/31, width 4, AR/CAT/Cheb, three policies. The fixed goal is four selected drive-gradient coordinates (not the full trained-model parameter gradient). K=16, relative L1 tolerance 1e-5, certificate checked every five rounds, initial threshold .05*||f||_infty with solver-specific geometric decay. All 216 converge. Maximum actual relative error 5.69561e-6; maximum certified relative error 9.99289e-6. No observed certificate violation. Relative certificate/actual error median 12.03549; uniform global residual bound/goal certificate median 1.20967; goal/oracle dual-weight bound median 3.89806.

Median full/weighted scalar sends: AR 1.27048, CAT 1.01231, Cheb 1.01480. Including the K-step setup proxy on both variants: 1.19106, 1.00742, 1.00993. Weighted versus uniform median sends ratio .999878, 40/72 weighted wins. CAT versus same-policy Cheb: 4/24 full wins (20 ties), 11/24 uniform wins (1 tie), 12/24 weighted wins (0 ties); median Cheb/CAT send ratios 1, .999650, .999936.

Communication is a scalar-per-destination-block proxy, with immediate reliable consistent delivery; synchronized rounds and stopping reductions are simulated/offline, not actual network measurements. Setup includes K transpose-product cross-block scalar sends only, not reduction or coefficient-distribution cost. The oracle dual solve is an offline diagnostic and not used for stopping. No real neural training, no general asynchronous proof, no claim of novelty. Relevant prior work: Reducing Communication in Distributed Asynchronous Iterative Methods (2016), https://www.sciencedirect.com/science/article/pii/S1877050916309887; Communication-efficient algorithms for solving pressure Poisson equation (2022), https://doi.org/10.1371/journal.pone.0277940; Distributed convergence detection based on global residual error under asynchronous iterations, https://arxiv.org/abs/2312.17558.

Decision: weighted threshold heuristic adds negligible benefit and does not provide CAT-specific advantage. Avoid scaling this variant. Next investigate the certificate's local tail bound using blockwise supersolutions and compare against a simple uniform residual certificate including complete setup cost; require substantial improvement on actual parameter gradients before proceeding to a trained neural network or TNNLS claim.
