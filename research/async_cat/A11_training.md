# A11: Actual implicit training and safeguarded momentum

For a fixed current system A x=b, r=b-Ax and a proposed CAT direction p=alpha*r+beta*v, choose eta=(Ap)^T r/||Ap||^2 when Ap is nonzero and eta=0 otherwise. The residual norm after x+=eta*p is no larger than either the original residual or the residual after the full proposed step. This is scalar least-squares minimization, not a novel general theorem. It does not guarantee strict decrease, gradient-error decrease, or dominance over equal-action solvers. If A changes between optimizer steps, the initial residual must be recomputed. SafeCAT uses one initial operator action plus one action per line search, updating the residual algebraically within the frozen system; global dot products are required.

A stronger prospective safeguard (not implemented in A11) minimizes ||r-A[r,v]c|| over c in R^2. Since c=(1,0) is feasible, for A=I-B with ||B||_2<=rho<1 the resulting residual is at most rho*||r||. The unmodified CAT direction is also feasible. This guarantees contraction per such substep, including when v is unhelpful; it charges two operator actions per substep (plus initialization per system) and requires small global least squares. It does not imply superiority over two equal-budget GMRES or Chebyshev actions. Related subspace recycling methods: https://doi.org/10.1137/040607277 and https://arxiv.org/abs/2001.10347.

## Reproduce

python research/async_cat/a11_train.py --out a11_results.json

A synthetic teacher-student regression model on a 24-node ring, four training inputs and four disjoint validation inputs, three shared trainable parameters (coupling, input gain, bias), projected full-batch gradient descent for 60 steps. Teacher coupling=.98*rho, gain=1.3, bias=.006. Initial coupling=.95*rho, gain=1, bias=0. The exact forward equilibrium is computed to numerical tolerance. rho=.95,.99,.999; seeds 60--62; learning rates .03 and .15; backward action budgets 4 and 8; nine approximate solvers and an offline exact-gradient reference, 360 training trajectories including 36 exact reference rows (18 distinct exact trajectories duplicated across budgets). Each approximate method charges 240 or 480 batched operator actions, including warm-start residual evaluation for GMRES, MR and SafeCAT. An action applies the block-diagonal operator for all four samples. GMRES uses double orthogonalization and least squares; no timing, communication, or reduction cost measured. The A10 smooth/tanh tuning choices are frozen for CAT and restarted Chebyshev. SafeCAT uses the original conservative momentum with least-squares line search, MR is its zero-momentum control. All approximate solvers warm-start from previous x, with specified momentum persistence or restart; Cheb-tuned additionally extrapolates. Exact adjoints at each method's own current parameters are offline gradient diagnostics, not used to steer approximate methods. Exact-reference direct solve has no comparable charged operator-action count.

## Results

All 360 trajectories completed. At budgets 4/8, median gradient NRMSE:
- CAT-tuned .562394/.156331
- Cheb-continuous 1.017916/.350200
- GMRES .125772/.003966
- MR .399631/.181397
- SafeCAT .322573/.020621

SafeCAT vs CAT-tuned: median paired error ratio 2.37824/9.26190, wins 13/18 and 18/18. SafeCAT vs MR: 1.05452/7.30765, wins 9/18 and 18/18. SafeCAT vs continuous Cheb: 5.24931/15.50686, wins 15/18 and 18/18. GMRES beats SafeCAT in all 36 configurations; median paired SafeCAT/GMRES error ratio is approximately 2.68/5.61 if computed as reciprocal of the median GMRES/SafeCAT ratio. Maximum observed positive residual-increase violation is zero. The residual guarantee is within each frozen inner system only.

Validation losses do not establish a CAT advantage. At lr=.03 the exact-gradient reference median validation loss is .010312 whereas AR has approximately .00032, suggesting outer-optimizer step size is a substantial confound; at lr=.15 the exact reference median loss is .019109. Different approximate methods visit different parameter trajectories; gradient NRMSE is relative to the exact gradient at each method's own trajectory, not a common sequence. These small synthetic runs cannot establish generalization or training superiority. The next gate should separately calibrate a stable outer optimizer using exact-gradient runs on new calibration seeds, freeze it before held-out testing, add recycling GMRES and a rank-two safeguard if warranted, and measure actual gradient error and optimization quality under charged equal cost. No TNNLS novelty or universal dominance claim.
