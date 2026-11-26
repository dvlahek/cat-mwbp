# A3: Shifted predictive adjoint with one-tick uniform delay

## Scope

A conditional mathematical result, not a general asynchronous convergence theorem or a novelty claim. Start from (I-B)x=b, B=J^T with real spectrum in [-s,rho], 0<=s<=rho<1. Define C=(B+sI)/(1+s), f=b/(1+s), epsilon=(1-rho)/(1+s). Then (I-C)x=f is exactly equivalent to the original adjoint problem and sigma(C) lies in [0,1-epsilon]. The shift uses only a local self term; no new graph edges. It requires a valid lower spectral bound (the conservative choice s=rho is allowed).

## Predictive recurrence

For simultaneous updates and uniformly one-tick old samples of ALL components in C (including the deliberately delayed local diagonal term):

    x[t+1] = x[t] + beta*(x[t]-x[t-1])
             + alpha*(f-x[t]+C*(2*x[t-1]-x[t-2])).

The self state in the residual -x[t] is fresh. Receivers need two consecutive samples per neighbor (or equivalent transmitted difference), and local history. A single new B-vector product suffices per tick. This is not an event-driven implementation. A constant history equal to the exact solution is a fixed point.

## Theorem

For 0<epsilon<=1/16 choose alpha=1/2, beta=1-sqrt(epsilon). Every root of the modal polynomial

    p_z(r)=r^3+(alpha-1-beta)*r^2+(beta-2*alpha*z)*r+alpha*z

has modulus strictly less than q=1-sqrt(epsilon)/4 for every z in [0,1-epsilon]. Consequently the lifted recurrence has spectral radius < q, including for a nondiagonalizable B (this is not a uniform finite-time norm bound). Since s<=rho<1, the certified gap is Omega(sqrt(1-rho)). This recovers the square-root criticality order in this uniform delay model.

Proof: set t=sqrt(epsilon)/4, 0<t<=1/16, q=1-t, beta=1-4t. Write p=r^3+a*r^2+b0*r+c0. The strict Schur conditions for the scaled polynomial p(q*r)/q^3 are

    |c0| < q^3,
    q^3+a*q^2+b0*q+c0 > 0,
    q^3-a*q^2+b0*q-c0 > 0,
    q^6-b0*q^4+a*c0*q^2-c0^2 > 0.

The first follows from |c0|<=1/2 and q>=15/16. The next two are affine in z; the last is concave in z, so it suffices to prove their positivity at z=0 and z=1-16t^2. The certificate script uses exact rational Bernstein coefficients on u=16t in [0,1]. All coefficients are nonnegative and the final coefficient is positive, implying strict positivity for u>0. The polynomials in this script are formed with exactly representable small dyadic coefficients before rational conversion. Reproduction checks the six endpoint polynomials.

Without the shift, at the critical limit beta=1, alpha=1/2, q=1, the last Schur expression equals alpha^2*z*(1-z), which is negative for z<0. Thus negative modes invalidate this particular unshifted calibration near criticality.

## Baseline interpretation

A2 gives the exact minimax spectral radius for the ORIGINAL fresh-self, one-tick stale-neighbor two-state recurrence without prediction or spectral shift. It is not the minimax bound for all possible communication algorithms. The new recurrence uses more temporal information. The reported rate ratios log(q_new_bound)/log(q_AR) are modal asymptotic comparisons in ticks, not measured speedups. Communication, history storage, and local computation must be accounted for separately.

## Limitation and next gate

Uniform two-tick extrapolation (3*x[t-2]-2*x[t-3]) with these same coefficients can be unstable. Arbitrarily mixed ages, skipped activations, FIFO event-driven messages, non-normal finite-step behavior, and actual neural training remain open. A proof for uniform one-tick delay cannot be transferred to them. Predictor extrapolation is already known in asynchronous optimization; literature novelty has not been established.

Relevant references: Chow, Frommer & Szyld (2021), arXiv:2009.02015; Wolfson-Pou & Chow (2025), DOI:10.1137/24M1669669; related momentum extrapolation in distributed optimization, https://www.sciencedirect.com/science/article/pii/S0378779621001747.

Run:

    python research/async_cat/shifted_predictor.py

18 interval configurations, 1001 numerical modal points per interval in addition to the exact polynomial certificate. The next experimental gate is paired finite-tolerance comparison on uniform delays including a predictor without momentum and accounting of extra temporal history. Do not claim TNNLS readiness before realistic delay validation and comparisons with strong asynchronous methods.
