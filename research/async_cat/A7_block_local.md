# A7: Block-local adjoint pilot

Scope: frozen systems only; classical block Jacobi splitting, not a new asynchronous algorithm.

For B>=0 and ||B||_infty<=rho<1 let B_d contain the diagonal blocks of B, M=I-B_d, R=B-B_d, T=M^-1 R, f=M^-1 b. The exact adjoint satisfies x=f+Tx. Since M^-1=sum_{k>=0} B_d^k>=I entrywise, T>=0 and T1=1-M^-1(I-B)1. Consequently ||T||_infty <= 1-(1-rho) min_i (M^-1 1)_i <=rho, with equality in the first bound if B1=rho*1. The exact row sums are locally computable; using their common maximum requires a setup reduction. For B=W D with symmetric W and positive diagonal D, ||W||_2<=rho and D<=I, the similarity by D^(1/2) yields SPD A_s and SPD block principal preconditioner M_s. Thus I-T is similar to M_s^(-1/2) A_s M_s^(-1/2), with real positive spectrum. These are classical splitting properties.

## Experiment

Run python research/async_cat/a7_block_local.py --out a7_results.json

12x12 periodic four-neighbor grid, rho=.95,.99,.999, three RHS seeds 20--22; linear and tanh-derived Jacobians. Modular control weakens edges crossing predefined 4x4 blocks by factor .1 and puts removed mass on the diagonal, preserving symmetry and row sums. Widths 1,3,4,6; width 1 includes diagonal preconditioning in the modular case. Shared bound calibration [1-c,1+c], c=||T||_infty, and independent oracle calibration from generalized symmetric eigensolve. AR, CAT, Chebyshev use the identical splitting and calibration. Direct-reference relative adjoint error <=1e-6, cap 30000. Tanh RHS is synthetic, not derived from a trained loss.

864/864 converged. 144 matched CAT/Cheb pairs per calibration:
- bound: CAT wins 75, ties 25, loses 44; median Cheb/CAT 1.01258.
- oracle: CAT wins 5, ties 14, loses 125; median Cheb/CAT 0.88889.
- width 4 bound: median communication-volume reduction relative to width 1 is 15.184x for CAT and 15.819x for Cheb across 36 paired cases.
- modular linear rho=.99, width 4: median AR/CAT/Cheb rounds 41/18/17 vs 1053/101/99 at width 1; c decreases from .99 to .720873.

Communication proxy counts one scalar per distinct source node per destination block per logical level, assuming aggregation and reuse. Edge reads count original cross-block nonzero entries. Local dense multiplication and setup proxies count arithmetic scale, not measured times. No actual network, scheduling, setup communication, memory/packet overhead, or stopping reduction is benchmarked. The current dense simulator is not a scalable implementation.

## Decision

The principal gain is the common preconditioner, not CAT. No strong evidence of a unique CAT advantage, especially with oracle calibration. Do not claim a novel block solver or TNNLS readiness. Next distinguish exact boundary elimination from ordinary block Jacobi (including the equivalence of nonzero spectra), and assess whether a genuinely new information/communication model can improve on identically preconditioned Chebyshev or whether to pivot to a certified local-gradient error / communication-budget problem. Preserve negative results. 
