#!/usr/bin/env python3
"""A3: shifted one-tick predictive adjoint, exact interval certificate and modal tests."""
import json
import numpy as np
from numpy.polynomial import Polynomial as P
from math import comb
from fractions import Fraction as F


def coeff(rho,s):
    eps=(1-rho)/(1+s)
    if not (0 <= s <= rho < 1 and 0 < eps <= 1/16):
        raise ValueError('requires 0 <= s <= rho < 1, 0 < shifted gap <= 1/16')
    t=np.sqrt(eps)/4
    return .5,1-4*t,1-t,eps


def radius(alpha,beta,z,delay=1):
    # Uniform stale linear extrapolation x[t-d]+d*(x[t-d]-x[t-d-1]).
    if delay==0:
        p=[1,-(1-alpha+beta+alpha*z),beta]
    else:
        p=np.zeros(delay+3)
        p[0]=1;p[1]=-(1-alpha+beta);p[2]=beta
        p[-2]-=alpha*z*(delay+1);p[-1]+=alpha*z*delay
    return float(max(abs(np.roots(p))))


def bernstein(poly):
    a=[F(x)/16**i for i,x in enumerate(poly.coef)]
    n=len(a)-1
    return [sum(a[j]*F(comb(i,j),comb(n,j)) for j in range(i+1)) for i in range(n+1)]


def certificate():
    # All coefficients exact dyadic rationals; u=16*t in [0,1].
    t=P([0,1]);q=1-t;alpha=.5;beta=1-4*t
    a=alpha-1-beta
    out={}
    for name,z in [('zero',P([0])),('upper',1-16*t*t)]:
        b=beta-2*alpha*z;c=alpha*z
        for key,poly in [('p_plus',q**3+a*q*q+b*q+c),('p_minus',q**3-a*q*q+b*q-c),('jury',q**6-b*q**4+a*c*q*q-c*c)]:
            bc=bernstein(poly)
            assert min(bc)>=0 and bc[-1]>0, (name,key,bc)
            out[name+'_'+key]=[str(x) for x in bc]
    return out


def main():
    cert=certificate();rows=[];max_excess=-np.inf
    for rho in (.95,.99,.999,.9999,.99999,.999999):
        for c in (0.,.5,1.):
            s=c*rho;alpha,beta,q,eps=coeff(rho,s);zmax=1-eps
            zgrid=np.linspace(0,zmax,1001)
            actual=max(radius(alpha,beta,z) for z in zgrid)
            max_excess=max(max_excess,actual-q)
            assert actual<=q+1e-10
            # Compare with A2 exact minimax of the original stale-neighbor recurrence.
            old=np.sqrt((rho+s)/(2+s-rho));ar=np.sqrt(rho)
            rows.append(dict(rho=rho,s=s,alpha=alpha,beta=beta,shifted_gap=eps,
                             certified_q=q,observed_grid_radius=actual,
                             original_ar_q=ar,original_a2_q=old,
                             certified_rate_ratio=float(np.log(q)/np.log(ar)),
                             delay2_at_positive_endpoint=radius(alpha,beta,zmax,2)))
    print(json.dumps(dict(status='SHIFTED_ONE_TICK_CERTIFICATE_PASS',intervals=len(rows),max_radius_minus_bound=float(max_excess),bernstein=cert,rows=rows),indent=2))


if __name__=='__main__':main()
