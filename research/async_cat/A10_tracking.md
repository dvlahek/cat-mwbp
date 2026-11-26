# A10: Continuous adjoint tracking

For one heavy-ball update at each time t, let A_t x*_t=b_t, e_t=x_t-x*_t, d_t=x*_t-x*_(t-1). Then
e_t=((1+beta)I-alpha A_t)e_(t-1)-beta e_(t-2)-(I-alpha A_t)d_t+beta d_(t-1).
For constant A and drift d, if the homogeneous recurrence is stable, e_inf=(I-((1-beta)/alpha)A^-1)d. This follows by substituting the constant particular solution. Scalar self-test in the script. Individual spectral stability does not prove stability under switching A_t.

Related literature: Madden, Becker & Dall'Anese, Bounds for the Tracking Error of First-Order Online Optimization Methods, https://doi.org/10.1007/S10957-021-01836-9; Single-loop methods for bilevel parameter learning, https://tuomov.iki.fi/mathematics/bilevel_general.pdf. Warm starts and persistent momentum are not themselves new.

## Reproduction

python research/async_cat/a10_tracking.py --out a10_results.json

Two model families, three rho values (.95,.99,.999), three drift regimes (smooth, jump, noise), 120 time points, 24 states, three action budgets (1,4,8 per point), 10 methods, 3 test seeds (50--52), 162 matched configurations, 1620 recorded test trajectories. Linear control has fixed symmetric A and moving target; its G=I, so its gradient metric is an adjoint proxy. Tanh family has a prescribed parameter path, exact numerical equilibrium at each point, actual full three-parameter gradient (coupling, input gain, bias), not an optimizer-driven training trajectory. No communication latency in A10. All methods start at zero. Exclude the first 20 time points from NRMSE but charge all backward actions. Score sqrt(sum squared errors / sum squared reference norms).

Tuning per model family, rho, regime and budget uses only seeds 48,49. AR tunes step size from four options; CAT tunes beta as a fraction of conservative q^2 with alpha=(1+beta)/center; Cheb restart tunes extrapolation from five options including zero. This is limited grid calibration, not optimal tuning of all possible Chebyshev variants; regime is known at calibration. Continuous Cheb carries state and its coefficient recurrence across time points. Its coefficients approach the conservative CAT stationary coefficients, even when the matrices change.

## Findings

All recorded NRMSE values finite, NOT all trajectories stable: gradient NRMSE >1e6 in 135 CAT-reset, 108 Cheb-extrap05, 144 Cheb-extrap10 of 162 each. All other methods had zero such cases. The threshold is a diagnostic not a convergence theorem. No trajectory removed. CAT-reset reuses the position but resets the predecessor every new problem; this can repeatedly amplify endpoint transients. For budget one its effective update is Richardson with the original CAT alpha, which can be unstable.

Tanh tuned CAT vs tuned restarted Cheb median paired gradient NRMSE ratios (Cheb/CAT): 1.7046, .6673, 3.9908 at budgets 1,4,8; CAT wins 27/27, 8/27, 24/27. Against continuous Cheb: 1.0, 1.0001, 2.4378; CAT wins 13/27,14/27,18/27 (ties within absolute ratio tolerance 1e-8: 0,9,6). Against the per-test-case best of tuned AR, tuned restarted Cheb and continuous Cheb (offline oracle comparator), median ratios .9999895,.6672675,2.0507413; wins 13,7,18 of 27, ties 0,0,3. Linear controls favor tuned AR: median paired AR/CAT ratios .8221,.9051,.9057.

This is exploratory: seeds held out from the finite tuning grid but the design and analysis were developed in the same research session. No claim of statistical generalization or paper readiness. Next gate: freeze calibration and test on independently generated optimizer-driven parameter trajectories, with strong restart-aware and continuous Cheb and a charged warm-start Krylov baseline; evaluate both gradient accuracy and training outcomes. A10 is not evidence of arbitrary asynchronous stability.
