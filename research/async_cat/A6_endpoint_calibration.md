# A6: Endpoint construction, paired evaluation, and polynomial obstruction

For the shifted operator C=(B+sI)/(1+s), let L=(rho+s)/(1+s). Replace the A3 predictor by (1+gamma)x[k-2]-gamma*x[k-3]. The modal cubic is r^3+(alpha-1-beta)r^2+(beta-alpha(1+gamma)z)r+alpha*gamma*z.

Choose q in (0,1) solving L=q^2(2-q)/(1-q+q^2), alpha=1-q+q^2, beta=q^2, gamma=q/(2-q). The defining function increases from 0 to 1 (equivalently 1-L=(1-q)^2(1+q)/(1-q+q^2), strictly decreasing). At z=0 the polynomial factors as r*(r^2-q*r+q^2); at z=L it is (r-q)^2*(r+q). Put u=z/L. The cubic coefficients are A=-q, B=q^2*(1-2*u), C0=q^3*u. The four strict scaled Schur conditions for radius q have margins q^3*(1-u), q^3*(1-u), 3*q^3*(1-u), q^6*u*(1-u), respectively. They are positive in the open interval, and the explicit endpoint roots have modulus at most q. The worst modal radius is therefore exactly q. This is a constructive calibration, not a proved global minimax optimum over all alpha,beta,gamma. Its critical gap is asymptotic to sqrt((1-L)/2). The repeated upper endpoint root introduces a finite-step transient.

## Pilot

Run: python research/async_cat/a6_predictor.py --out a6_results.json

Conservative s=rho; 8x8 non-bipartite grid; five new RHS seeds 10--14; schedule seeds 1010--1014; rho=.95,.99,.999; transit maxima 1,8,32. Calibration uses only the interval. Four methods, 180/180 converged at relative adjoint error <=1e-6. Medians in logical levels (Cheb / fresh CAT / A3 / A6):
- .95: 45 / 51 / 196 / 141
- .99: 103 / 117 / 442 / 328
- .999: 325 / 373 / 1102 / 1050

A6 loses to fresh Chebyshev in all 45 paired simulated completion times. Median paired A6/Cheb time ratios range 1.5813--1.6768 over the nine cells. No real network measurements; stopping and buffer overhead excluded. No held-out neural training; no novelty claim.

## Why more tuning may not be the right research direction

For the restricted class of linear, globally scalar-coefficient algorithms starting at zero, with RHS available locally at version 1, local linear combinations of existing states, and every B multiplication applied only to versions at most k-2 when constructing version k, induction gives x_k=P_k(B)b with deg(P_k)<=floor((k-1)/2). Thus the error polynomial E_k(z)=1-(1-z)P_k(z) has degree at most m=floor((k-1)/2)+1 and E_k(1)=1. Over a full symmetric real interval [-rho,rho], the classical Chebyshev minimax bound is max|E_k|>=1/T_m(1/rho). For symmetric operators this is the worst spectral relative error across admissible systems/RHS. A fresh Chebyshev chain attains the corresponding bound using m logical steps, whereas a lag-two method needs at least 2m-1 logical versions to reach degree m (except m=1). In the ideal homogeneous latency-only limit, both have the same minimum communication-depth order, not a general pipeline speedup. This does not cover node-dependent coefficients, adaptive Krylov methods, local block solves, multiple RHS, graph-specific preconditioners, or instance-specific spectra. Nor is it a universal physical wall-time bound. The approximation-theoretic result is classical; its application here identifies the limitation of the current model.

Decision: do not claim A6 beats Chebyshev. Next investigate graph-local block elimination/preconditioning with explicit communication charges, or another genuinely different information model, rather than endless scalar momentum tuning. Include block-preconditioned Chebyshev and AR as fair controls.
