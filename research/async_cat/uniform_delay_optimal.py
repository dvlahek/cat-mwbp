#!/usr/bin/env python3
"""Minimax calibration for fresh self / uniformly one-tick stale neighbors."""
import json
import numpy as np
from scipy.optimize import linprog


def optimal(rho, s):
    if not 0 <= s <= rho < 1 or rho == 0:
        raise ValueError('require 0 <= s <= rho < 1 and rho > 0')
    d = 2+s-rho
    return 2/d, (rho-s)/d, np.sqrt((rho+s)/d)


def modal_radius(alpha, beta, rho, s):
    z = np.linspace(-s, rho, 2001)
    a = 1-alpha+beta
    disc = np.sqrt((a*a-4*(beta-alpha*z)).astype(complex))
    return float(np.max(np.maximum(abs((a+disc)/2), abs((a-disc)/2))))


def feasible(q, rho, s, ar=False):
    # Closed Schur disk inequalities on both interval endpoints, q > 0.
    A = [[s, 1], [rho-q, q-1], [rho+q, -1-q]]
    b = [q*q, q*q-q, q*q+q]
    return linprog([0, 0], A_ub=A, b_ub=b,
                   bounds=[(0,None),(0,0) if ar else (0,None)],
                   method='highs', options={'primal_feasibility_tolerance':1e-9,
                                            'dual_feasibility_tolerance':1e-9}).success


def lp_radius(rho, s, ar=False):
    lo, hi = 0., 1.
    for _ in range(36):
        q = (lo+hi)/2
        if feasible(q,rho,s,ar): hi = q
        else: lo = q
    return hi


def grid_min(side, rho):
    t = 2*np.pi*np.arange(side)/side
    z = rho*(.30*(np.cos(t[:,None])+np.cos(t[None,:]))+.40*np.cos(t[:,None]-t[None,:]))
    return -float(z.min())


def main():
    rows = []
    worst = 0.
    for rho in (.2,.5,.8,.95,.99,.999):
        for s in (0., .25*rho, grid_min(12,rho), rho):
            a,b,q = optimal(rho,s)
            lp = lp_radius(rho,s)
            arlp = lp_radius(rho,s,True)
            err = max(abs(lp-q),abs(arlp-np.sqrt(rho)),abs(modal_radius(a,b,rho,s)-q))
            worst = max(worst,err)
            assert err < 2e-6, (rho,s,err)
            if s == grid_min(12,rho):
                beta0 = ((np.sqrt(1+rho)-np.sqrt(1-rho))/(np.sqrt(1+rho)+np.sqrt(1-rho)))**2
                rows.append(dict(rho=rho,s=s,alpha=a,beta=b,q_opt=q,q_ar=np.sqrt(rho),
                                 q_sync_cat_delayed=modal_radius(1+beta0,beta0,rho,s),
                                 asymptotic_action_ratio=float(np.log(q)/np.log(np.sqrt(rho)))))
    print(json.dumps(dict(status='UNIFORM_DELAY_MINIMAX_PASS',cases=24,max_error=worst,grid=rows),indent=2))


if __name__ == '__main__':
    main()
