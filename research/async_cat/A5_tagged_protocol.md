# A5: Switching obstruction and version-tagged execution

## Exact switching counterexample

Use A3 with epsilon=1/10000, alpha=1/2, beta=99/100 and shifted eigenvalue z=2/3. The delay-0 and delay-1 companion matrices M0 and M1 are separately stable (radii 0.9949874371 and 0.9801545092). The periodic sequence 00000111 has product T=M1^3 M0^5. Exact rational arithmetic gives det(I+T)=-36302401531425971/3645000000000000000 < 0, implying a real eigenvalue below -1. The period radius is 1.04766605968. Thus bounded delay alone does not suffice for the A3 calibration. This is a scalar modal counterexample within the permitted interval; it is not a claim about arbitrary random schedules.

## Version-tagged local pipeline

Every node computes its own logical version k, using its local versions k-1 and k-2 and exact neighbor versions k-2 and k-3, rather than whichever message arrived most recently. The shifted diagonal term also uses the prescribed local history. Initialize versions -2,-1,0 to zero everywhere. Each node advances when all its dependencies are present, and sends each new scalar state to its neighbors with a version tag. No common physical clock or global barrier is required, but logical version counters and history storage are required. Reliable eventual delivery and fair processing guarantee progress by induction over k; induction also proves equality to the uniform logical-delay recurrence from A3. Hence the A3 spectral theorem applies regardless of the physical delivery schedule, subject to these assumptions. This does not prove acceleration under unrestricted stale-read updates.

If each message arrives within D time units of sending, and ready-to-completion time is at most H, the latest finish time across nodes satisfies T_k <= max(T_(k-1), T_(k-2)+D)+H for k>=3. With T_1<=H, T_2<=2H, this yields T_k<=k*max(H,(D+H)/2). Fresh-version Jacobi/Chebyshev instead have the simple bound k*(D+H). These are upper bounds, not a proof of relative wall-time advantage. On bidirectional edges, dependency constraints imply adjacent logical version skew at most two, so bounded version retention is possible; buffer implementation is not yet tested.

## Paired pilot

Run: python research/async_cat/a5_tagged_protocol.py

6x6 non-bipartite periodic grid, two RHS seeds, rho=.95,.99,.999, two transit ranges [0,1] and [0,8], independent processing in [.1,1], FIFO transit arrival enforced per edge. Same random draws per logical version for each method. All methods use the conservative symmetric interval. A logical DAG computes finish timestamps using exact dependencies, and values using equivalent vector recurrences. This is a simulator, not a measured network implementation. The output seed field is the schedule seed (100 or 101); RHS seed is schedule seed minus 100. Direct reference relative adjoint error <=1e-6, cap 30000 logical levels. Counts assume a fixed number of levels per node, excluding stopping, packet headers, buffer costs, and actual concurrent execution overhead.

48 runs: AR, fresh-version Chebyshev and predictive CAT each converge 12/12. The predictor without momentum converges 8/12 (four budget exhaustions at rho=.999). Median levels (AR / Cheb / predictive CAT): 263 / 45 / 196 at .95; 1372 / 102.5 / 442 at .99; 13808.5 / 325 / 1102 at .999. Predictive CAT loses to fresh-version Chebyshev in all six aggregate latency/radius cells; median paired simulated time ratios CAT/Cheb range 1.705 to 2.263. It improves on AR but not the strongest control.

## Decision

A3 can be executed asynchronously with version dependencies, but this does not establish novelty (dataflow/version tagging is an established mechanism) or superiority over Chebyshev. Do not scale to real neural models yet. Next gate: optimize the delayed polynomial under the same conservative interval, include fresh-version CAT and pipelined polynomial baselines, and establish an advantage at equal communication and accuracy. The current result is a conditional stability construction plus a negative performance finding, not a TNNLS-ready contribution.
