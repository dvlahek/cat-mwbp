#!/usr/bin/env python3
"""Exact modal stability diagnostic for simultaneous one-tick stale neighbor reads."""
import json
import numpy as np
from run_async_pilot import problem


def modes(side, rho):
    B = problem(side, rho, 0, 'linear')[-1]
    z = np.linalg.eigvalsh(B)
    beta = ((np.sqrt(1+rho)-np.sqrt(1-rho)) /
            (np.sqrt(1+rho)+np.sqrt(1-rho)))**2
    alpha = 1+beta
    radius = float(np.sqrt(np.max(np.abs(alpha*z-beta))))
    return dict(side=side, rho=rho, lambda_min=float(z.min()),
                lambda_max=float(z.max()), alpha=float(alpha), beta=float(beta),
                full_interval_beta_limit=(1-rho)/(1+rho),
                grid_beta_limit=float((1+z.min())/(1-z.min())),
                cat_delay_radius=radius, ar_delay_radius=float(np.sqrt(np.max(np.abs(z)))),
                cat_stable=radius<1)


def selftest():
    for rho in (.8,.95,.99):
        B = problem(4, rho, 0, 'linear')[-1]
        r = modes(4,rho)
        n = len(B)
        I = np.eye(n); Z = np.zeros_like(I)
        lifted = np.block([[Z,r['alpha']*B-r['beta']*I],[I,Z]])
        actual = float(np.max(np.abs(np.linalg.eigvals(lifted))))
        assert abs(actual-r['cat_delay_radius'])<1e-10


if __name__=='__main__':
    selftest()
    print(json.dumps([modes(12,r) for r in (.8,.95,.99)],indent=2))
