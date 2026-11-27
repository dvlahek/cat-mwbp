# A14: Noncommuting loss curvature and collocated learning

Consider h'=P theta-F h, F=F^T>0, Q=Q^T>0, P full column rank, and loss (h-y)^T Q(h-y)/2. The equilibrium objective has Hessian H=P^T F^-1 Q F^-1 P>0. The exact forward lead h+F^-1 h'=F^-1 P theta requires an inverse application and is an oracle unless F is locally invertible.

## Exact lead and its limitation

After shifting to the unique stationary solution, the lead-based adjoint/parameter system is lambda'=-F lambda+Q F^-1 P theta, theta'=-eta P^T lambda. If FQ=QF, S=QF^-1 is SPD and commutes with F. Energy V=||theta||^2/(2 eta)+lambda^T S^-1 lambda/2 satisfies V'=-lambda^T S^-1 F lambda. The largest invariant set in lambda=0 is the origin (P has full column rank). Thus the reduced linear system is Hurwitz for every eta>0; the forward states add stable -F eigenvalues.

Counterexample: F=diag(eps,1), Q=[[1.1,-2],[-2,4.1]], P=(1,1)^T, 0<eps<=.1. Both F and Q SPD. Let u=QF^-1P. The cubic is s^3+(1+eps)s^2+(eps+eta C)s+eta D, C=u1+u2=-.9/eps+2.1<0, D=u1+eps*u2=1.1/eps-4+4.1 eps>0. Routh-Hurwitz gives the exact open stability interval

    0<eta<eps*(1+eps)/(2/eps-5.2+2 eps).

Raw threshold ~eps^2/2; normalized threshold eta*H ->.55, with H=D/eps. Noncommutation alone is not sufficient for instability; this is a counterexample to the generalization of the scalar all-gain claim.

## Collocated adjoint dynamics

Define K=F Q^-1 F, SPD. Use lambda'=P theta-F y-K lambda, theta'=-eta P^T lambda. Equilibrium lambda=K^-1(P theta-F y)=F^-1 Q(F^-1 P theta-y) recovers the exact implicit adjoint. Shift by the stationary pair of lambda,theta (lambda* may be nonzero). V=||theta_error||^2/(2 eta)+||lambda_error||^2/2 has derivative -lambda_error^T K lambda_error. LaSalle/full column rank gives asymptotic stability for every eta>0. This is a classical collocated saddle/Arrow-Hurwicz flow, not a priority claim. It uses two F applications plus a Q solve per adjoint update; for Q=I, cond_2(K)=cond_2(F)^2. If forward dynamics is retained its eigenvalues include -F. No claim of eliminating critical slowing or of a local nonlinear algorithm.

## Exact discrete boundary

For step h>0, update lambda_plus=(I-hK)lambda+hP theta, then theta_plus=theta-h eta P^T lambda_plus (centered variables). The update matrix is Schur stable if and only if

    2hK+h^2 eta PP^T < 4I

in the strict Loewner sense, under K SPD, eta>0, and P full column rank. Proof: for eigenvalue z !=1, lambda component l !=0 and
[(z-1)^2 I+h(z-1)K+h^2 eta z PP^T]l=0.
Normalize l and project with its conjugate transpose. Let k=l^* K l>0 and p=||P^T l||^2>=0. The scalar quadratic is z^2+(-2+hk+h^2 eta p)z+1-hk=0. If p>0, the strict quadratic Jury conditions follow from 2hk+h^2 eta p<4. If p=0, its roots are 1 and 1-hk, the latter strictly inside the disk. z=1 is impossible for the update matrix: P^T l=0, K l=P theta imply l=0 and then theta=0. Conversely, at z=-1 the symmetric matrix polynomial is 4I-2hK-h^2 eta PP^T. If this is not SPD, its smallest eigenvalue is nonpositive. As real z tends to minus infinity the matrix polynomial is positive definite, hence a singular point exists at real z<=-1 by continuity. Reconstruct theta=-h eta z P^T l/(z-1) to obtain an update eigenvector. Thus stability fails. This is a matrix stability result for the specified semi-implicit update, not all discretizations.

## Verification

python research/async_cat/a14_multimode.py --out a14_results.json

430 checks: seven gap values, five gain ratios around exact lead boundary and five gains for the collocated flow per gap; 60 random SPD F,Q and full-column-rank P cases (dimensions 4--8, parameter dimensions 1--3), collocated and commuting lead continuous controls, four discrete step ratios around the exact threshold. 120 discrete stable and 120 unstable cases, all predicted correctly. Tests use floating-point eigensolves, not interval arithmetic.

Prior work: Stern et al., Physical learning beyond the quasistatic limit, https://doi.org/10.1103/PhysRevResearch.4.L022037; Arrow-Hurwicz convergence, https://doi.org/10.1080/02331934.2023.2215799; discrete Arrow-Hurwicz-Uzawa stability, https://link.springer.com/article/10.1007/s10107-023-02051-7. The present derivation is an application to implicit adjoints, not an established novelty claim.

Next gate: compare actual discrete decay and work, not only gain ceilings, against ordinary first-order, CAT and standard preconditioned saddle methods on task-normalized multimode systems, including noncommuting and nonlinear controls. No Nature Communications readiness.
