# A13: Coupled learning near criticality

This is an exact scalar continuous-time audit, not a Nature Communications novelty or performance claim. Consider h*=rho h*+theta, loss (h*-y)^2/2, epsilon=1-rho>0. The unique target equilibrium is h=y, theta=epsilon*y, lambda=v=0. Use normalized variables h-y, z=epsilon*lambda and theta-epsilon*y. The equilibrium Hessian in theta is epsilon^-2, so define normalized outer gain k=eta/epsilon^2. Let f=epsilon, b=a*epsilon for first-order adjoint; c=alpha*epsilon, d=gamma for second-order adjoint.

First order: h'=theta-f*h, z'=b*(h-z), theta'=-k*f*z.
Second order: h'=theta-f*h, z''+d*z'+c*z=c*h, theta'=-k*f*z.

Characteristic polynomials are P1=s(s+f)(s+b)+k*f*b and P2=s(s+f)(s^2+d*s+c)+k*f*c. For positive coefficients the cubic and quartic Routh-Hurwitz criteria give exactly

    0 < k < f+b (first order)
    0 < k < d*(c+f*d+f^2)/(d+f)^2 (second order).

At equality each has a purely imaginary conjugate pair (frequencies sqrt(f*b) and sqrt(f*c/(d+f)), respectively); the remainder of the factors is stable. This is a linear stability boundary, not a nonlinear Hopf bifurcation claim.

With a fixed, alpha fixed and critical damping d=2 sqrt(c), raw eta thresholds scale as epsilon^3 (first order) and epsilon^(5/2) (second order). BUT the first-order control with a=(1/2)sqrt(alpha/epsilon) has the same low-frequency lag as the critically damped second-order control and the same exponent 5/2. For all f,c,d>0, matching low-frequency lag means b=c/d, from the transfer functions b/(s+b) and c/(s^2+d*s+c). The difference of the two normalized stability margins times d*(d+f)^2 is

    (2*d*f+f^2)*c + f^2*d*(d+f) > 0.

Thus the matched-lag first-order controller always permits a strictly greater outer gain. Matching critically damped pole decay instead gives b=sqrt(c) and likewise no advantage for the second-order gain margin. Neither comparison is an equal-cost implementation comparison. Gain margins do not alone measure training speed, and raw eta exponents depend on parametrization.

## Forward phase-lead control

Use z'=b*(h+tau*h'-z) instead of z'=b*(h-z), with h' available from the scalar forward vector field. The characteristic polynomial is P1+k*f*b*tau*s. The exact stability condition is k>0 and k*(1-(f+b)*tau)<f+b. For tau=1/f, the polynomial factors as (s+f)*(s^2+b*s+k*b), stable for all positive k. This is classical phase lead / pole cancellation, not established novelty. In the special case b=f, the optimal spectral abscissa over stable k without phase lead is -f/3 (double root at k=4f/27); with exact lead it is -f/2 (double root at k=f/4). Thus even eliminating the gain ceiling does not eliminate critical slowing of the optimal asymptotic rate in this comparison. The extrapolation requires the known scalar gap and may amplify noise. No discrete-time stability, nonlinear robustness or actual wall-time advantage is demonstrated.

## Reproduction

python research/async_cat/a13_critical_scaling.py --out a13_results.json

Requires numpy and sympy. Exact symbolic characteristic-polynomial and difference identities; 162 floating-point root checks over 13 gaps from 1e-1 to 1e-7, including below/above threshold, phase-lead and double-root controls. Fitted slopes on last five gaps: 3, 2.5011961309, 2.5005985528 and 2.5005990430 for fixed first order, lag-matched first order, critical second order and decay-matched first order respectively. These are diagnostic numerical fits to analytic formulas, not independent empirical scaling laws.

## Prior art and next gate

Stern et al., Physical learning beyond the quasistatic limit, Phys. Rev. Research (2022), https://doi.org/10.1103/PhysRevResearch.4.L022037, already study critical learning rates and oscillatory physical learning. Continual Equilibrium Propagation https://arxiv.org/abs/2005.04168 and A neuronal least-action principle for real-time learning in cortical circuits https://pmc.ncbi.nlm.nih.gov/articles/PMC11661794/ address simultaneous dynamics and prospective states. Continuous-Time Analysis for Minimax and Bilevel Problems https://arxiv.org/abs/2605.20898 and Analyzing inexact hypergradients for bilevel learning https://doi.org/10.1093/imamat/hxad035 are relevant to joint timescale analysis (see the journal page for the latter: https://academic.oup.com/imamat/article/89/1/254/7456287). No claim of priority for the scalar calculation.

Decision: the original hypothesis of an intrinsic second-order improvement in the critical gain exponent fails the matched-lag control for this scalar model. Do not launch a large benchmark to confirm it. Next gate: multimode systems with noncommuting task Hessian and parameter couplings, equal-cost discrete dynamics and noise/robustness controls, to establish whether any advantage remains. Phase lead alone is not a new method. No claim of Nature Communications readiness.
