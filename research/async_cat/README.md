# Asynchronous local adjoint solvers

Research branch based on release v1.1.0. The objective is to determine whether a delay-robust local adjoint recurrence can improve on asynchronous first-order and second-order numerical baselines while preserving the exact implicit-gradient target.

## Reproduce the bounded-staleness pilot

From the repository root:

```bash
python research/async_cat/run_async_pilot.py --profile quick
python research/async_cat/run_async_pilot.py --profile standard --out results/async_standard
```

The quick profile contains 144 frozen linear adjoint solves on a periodic non-bipartite 12 x 12 grid. The standard profile adds tanh-derived adjoints and five seeds (720 solves). Both use prescribed recurrent radii 0.80, 0.95 and 0.99, activation probabilities 1 and 0.5, and maximum stale-read ages 0, 1, 2 and 4 simulator ticks. A node is forced to update after seven idle ticks. The same activation and stale-read draws are used across methods. The current self-state is fresh; neighboring states can be stale. Chebyshev coefficients depend on each node's own update counter.

This is a discrete bounded-staleness simulator, not an implementation of an actual FIFO message network. The simulator has a global tick, but the update rule does not require nodes to synchronize their local counters. The stale-read model can access an older snapshot after a newer one. A realistic event-driven message model is a separate validation requirement.

The stopping diagnostic uses direct-reference relative adjoint and drive-gradient errors (both <= 1e-6). The budget is 3000 ticks. Report convergence, divergence and budget exhaustion separately. The raw CSV reports updated rows, equivalent full sweeps and edge reads; these are not measured wall-clock time or network bytes. Conditional medians among successful runs must not be interpreted as unconditional speedups.

## One-tick uniform-delay control

Let B = J^T and use a fresh local self-state and previous local state, but delay all neighbor reads by exactly one tick. For simultaneous updates on the diagonal-free grid, an eigenmode of B with eigenvalue z satisfies

    e[t+1] = (1-alpha+beta) e[t] + (alpha*z-beta) e[t-1].

For the symmetric conservative interval, alpha = 1+beta and the characteristic polynomial reduces to r^2 = (1+beta)z-beta. Therefore stability on the full real interval [-rho,rho] requires beta < (1-rho)/(1+rho), for beta >= 0 and 0 < rho < 1. This condition is exact for this particular uniform-delay model, not a guarantee for arbitrary asynchronous schedules.

```bash
python research/async_cat/uniform_delay_modes.py
```

## Prior work and research gate

- E. Chow, A. Frommer and D. B. Szyld, *Asynchronous Richardson iterations: theory and practice*, Numerical Algorithms (2021), https://arxiv.org/abs/2009.02015.
- J. Wolfson-Pou and E. Chow, *Asynchronous Semi-iterative Methods and the Asynchronous Chebyshev Method with Multigrid Preconditioning*, SIAM Journal on Scientific Computing (2025), https://doi.org/10.1137/24M1669669.

Asynchronous second-order Richardson and Chebyshev iterations are established methods. Asynchronous execution of CAT alone is not a novelty claim. A new method must provide a nontrivial delay-aware stability or convergence result and an advantage over appropriately tuned asynchronous controls, with calibration separated from held-out evaluation. The pilot is exploratory and does not establish a new solver.
