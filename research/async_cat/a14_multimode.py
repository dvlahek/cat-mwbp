#!/usr/bin/env python3
"""A14: noncommuting loss curvature, collocated adjoint and discrete stability."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import brentq


def lead(F,Q,P,eta):
    n,m=P.shape
    return np.block([[-F,Q@np.linalg.solve(F,P)],[-eta*P.T,np.zeros((m,m))]])


def collocated(K,P,eta):
    n,m=P.shape
    return np.block([[-K,P],[-eta*P.T,np.zeros((m,m))]])


def discrete(K,P,eta,h):
    n,m=P.shape; U=np.eye(n)-h*K
    return np.block([[U,h*P],[-h*eta*P.T@U,np.eye(m)-h*h*eta*P.T@P]])


def abscissa(M):return float(np.max(np.linalg.eigvals(M).real))
def radius(M):return float(np.max(abs(np.linalg.eigvals(M))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('a14_results.json'));args=ap.parse_args()
    P=np.ones((2,1));Q=np.array([[1.1,-2.],[-2.,4.1]])
    rows=[];checks=0
    for eps in (.1,.03,.01,.003,.001,.0003,.0001):
        F=np.diag([eps,1.]);u=(Q@np.linalg.solve(F,P)).ravel();H=float((P.T@np.linalg.solve(F,Q@np.linalg.solve(F,P))).item())
        a=1+eps;b=eps;C=u[0]+u[1];D=u[0]+eps*u[1]
        limit=a*b/(D-a*C)
        assert H>0 and C<0 and D>0
        for scale in (.5,.99,1.01,2.,100.):
            eta=limit*scale;M=lead(F,Q,P,eta)
            assert np.max(abs(np.poly(M)-[1,a,b+eta*C,eta*D]))<1e-10
            z=abscissa(M);assert (z<0)==(scale<1);checks+=1
        K=F@np.linalg.solve(Q,F);K=(K+K.T)/2
        for eta in (.001,.1,1.,10.,100.):
            assert abscissa(collocated(K,P,eta))<0;checks+=1
        assert np.allclose(np.linalg.solve(K,P),np.linalg.solve(F,Q@np.linalg.solve(F,P)))
        rows.append(dict(epsilon=eps,eta_lead_limit=float(limit),normalized_gain_limit=float(H*limit),loss_hessian=H,lead_high_gain_abscissa=abscissa(lead(F,Q,P,100*limit)),collocated_abscissa_at_eta1=abscissa(collocated(K,P,1.)),K_condition=float(np.linalg.cond(K))))
    rng=np.random.default_rng(140);discrete_rows=[]
    for i in range(60):
        n=4+i%5;m=1+i%3
        U,_=np.linalg.qr(rng.normal(size=(n,n)));V,_=np.linalg.qr(rng.normal(size=(n,n)))
        F=(U*np.geomspace(.03,1,n))@U.T;Q=(V*np.geomspace(.2,3,n))@V.T;P=rng.normal(size=(n,m));eta=10**rng.uniform(-2,2)
        K=F@np.linalg.solve(Q,F);K=(K+K.T)/2
        assert abscissa(collocated(K,P,eta))<0;checks+=1
        # Q commuting with F: exact phase lead is stable for every positive gain.
        Qc=(U*np.geomspace(.2,3,n))@U.T
        assert abscissa(lead(F,Qc,P,eta))<0;checks+=1
        def margin(h):return float(np.linalg.eigvalsh(2*h*K+h*h*eta*(P@P.T))[-1]-4)
        hi=1.
        while margin(hi)<0:hi*=2
        hc=brentq(margin,0,hi)
        for scale in (.5,.99,1.01,2.):
            h=hc*scale;r=radius(discrete(K,P,eta,h))
            assert (r<1)==(scale<1),(i,scale,r)
            discrete_rows.append(dict(case=i,n=n,m=m,eta=float(eta),h_limit=float(hc),scale=scale,radius=r));checks+=1
    data=dict(status='A14_MULTIMODE_PASS',checks=checks,counterexample=rows,discrete_rows=discrete_rows,discrete_stable=sum(r['radius']<1 for r in discrete_rows),discrete_unstable=sum(r['radius']>1 for r in discrete_rows))
    args.out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');print('A14_MULTIMODE_PASS',checks,'checks');print('discrete stable/unstable',data['discrete_stable'],data['discrete_unstable']);print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
