# A2: Exact minimax calibration and limitations

Status: proved for the stated uniform-delay model; novelty relative to the numerical-analysis literature is not established. This is not yet a generally asynchronous solver.

## Uniform one-tick stale neighbors

Assume a diagonal-free frozen adjoint operator B=J^T, real spectrum in [-s,rho], with 0<=s<=rho<1 and rho>0. Update every row simultaneously, using the fresh self-state and local predecessor and uniformly one-tick stale neighbor states. For an eigenvalue z of B, the error polynomial is

    p_z(r) = r^2 - (1-alpha+beta) r + beta-alpha*z.

Over alpha>0, beta>=0, the exact minimax modal root radius on the interval is

    alpha* = 2/(2+s-rho)
    beta*  = (rho-s)/(2+s-rho)
    q*^2   = (rho+s)/(2+s-rho).

Proof: for any 0<q<1, closed-disk Schur conditions for every z reduce to

    beta + alpha*s <= q^2
    q^2 + beta - alpha*rho >= q*abs(1-alpha+beta).

Eliminating beta by its upper bound gives

    alpha*((1+s)*q-(rho+s)) >= q*(1-q)^2
    alpha*((1+s)*q+(rho+s)) <= q*(1+q)^2.

The first coefficient must be positive. Combining the inequalities yields

    (2+s-rho)*q^2 >= rho+s.

At the proposed optimum alpha*=1+beta*, and beta*-alpha*z ranges over [-q*^2,q*^2], so the bound is attained. This also covers nondiagonalizable B as a statement about the spectral radius, not a uniform finite-step norm bound.

For first-order relaxation beta=0, the two Schur inequalities imply

    alpha*(q-rho) >= q*(1-q)
    alpha*(q+rho) <= q*(1+q).

Together they give q^2>=rho; alpha=1 attains q=sqrt(rho). Thus the second-order optimum strictly improves the modal rate iff s<rho. The improvement vanishes on symmetric spectral intervals. Since 1-q*^2=2*(1-rho)/(2+s-rho), fixed coefficients do not recover square-root criticality scaling for this delayed model. For s=c*rho, the asymptotic first-order/second-order tick ratio tends to 2/(1+c), not infinity.

## Failure under a two-tick delay

At alpha=1+beta and z=rho, uniform two-tick stale reads yield

    p(r)=r^3+beta*r-alpha*rho.

For beta>=0 this polynomial has a unique positive root t, and its other two roots have modulus sqrt(t^2+beta). Stability requires beta<1 and alpha*rho<sqrt(1-beta), by monotonicity of p on the positive axis. Hence the one-tick optimal parameters need not work even with a fixed two-tick delay. For the 12x12 non-bipartite grid (s=rho/2), the positive mode gives maximum root moduli 1.044347, 1.127629, 1.149307 for rho=.8,.95,.99 respectively.

## Sufficient arbitrary bounded-staleness certificate

Suppose B is diagonal-free with absolute row sums c_i<=c<1. Each row updates at least once every H ticks; neighbor reads are at most D ticks old. The previous local state is from just before that row's preceding update. Let alpha_i>0 and beta_i>=0, possibly time-varying. A uniform sufficient condition is

    |1-alpha_i+beta_i| + beta_i + alpha_i*c_i <= eta < 1.

For e=x-x*, each updated row is bounded by eta times the maximum absolute error over the bounded read history, including the stored local predecessor. Non-updated rows retain their state. Starting from bounded initial history, the history envelope is nonincreasing. Every H ticks all rows have been updated, and after an additional max(D,H+1) ticks no pre-epoch state is used. Hence the envelope contracts by at least eta per such epoch. The fixed point remains the exact adjoint. The result is only sufficient, not necessary.

For alpha=1+beta and c_i<=c, the certificate reduces to (1+beta)c+beta<1. In fact for c_i>0, the absolute-value bound is never smaller than c_i over alpha>0,beta>=0: minimizing over alpha gives alpha=1+beta and the bound c_i+(1+c_i)*beta. Therefore this certificate alone cannot establish acceleration over AR in the worst-case history envelope.

## Reproduction and gate

    python research/async_cat/uniform_delay_optimal.py

24 interval configurations checked against direct modal roots and independent linear-programming feasibility for the closed Schur disk, including the optimized first-order baseline. Maximum observed discrepancy 5.744e-10. For the non-bipartite 12x12 grid, s=rho/2. The original pilot used randomized per-edge stale reads, which is not the uniform-delay model here.

A2 conclusion: exact one-tick optimum is a useful control, not a publication-ready new method. A3 must change the information/update mechanism rather than merely select a larger constant momentum. Candidate directions include timestamped local prediction with a separately proved safeguard, and local block elimination with explicit accounting of additional communication. Compare against tuned asynchronous Richardson and asynchronous Chebyshev, and keep delay-model assumptions explicit.

Relevant prior work: Chow, Frommer and Szyld (2021), https://arxiv.org/abs/2009.02015; Wolfson-Pou and Chow (2025), https://doi.org/10.1137/24M1669669. The delayed-gradient setting in arXiv:2610.37444 is related but differs from the fresh-self/stale-neighbor model above.
