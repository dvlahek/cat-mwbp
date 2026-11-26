#!/usr/bin/env python3
"""A7: block-local adjoint and fair same-preconditioner controls (offline pilot)."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh
from scipy.linalg import solve


def blocks(side, width):
    assert side % width == 0
    g=np.arange(side**2).reshape(side,side)
    return [g[i:i+width,j:j+width].ravel() for i in range(0,side,width) for j in range(0,side,width)]


def system(side,rho,kind,problem,seed):
    n=side**2;g=np.arange(n).reshape(side,side);W=np.zeros((n,n))
    for axis in (0,1):
        for shift in (-1,1):
            W[np.arange(n),np.roll(g,shift,axis).ravel()]=.25*rho
    if kind=='modular':
        owner=np.empty(n,int)
        for i,idx in enumerate(blocks(side,4)):owner[idx]=i
        W[owner[:,None]!=owner[None,:]]*=.1
        W+=np.diag(rho-W.sum(axis=1))
    rng=np.random.default_rng(seed);d=np.ones(n)
    if problem=='tanh':
        u=.03*rng.normal(size=n);h=np.zeros(n)
        for _ in range(50000):
            nxt=np.tanh(W@h+u)
            if np.linalg.norm(nxt-h)<1e-12:
                h=nxt;break
            h=nxt
        else:raise RuntimeError('forward fixed point failed')
        d=1-h*h
    b=rng.normal(size=n);b=.85*b/np.linalg.norm(b)+.15/np.sqrt(n)
    B=W*d[None,:];A=np.eye(n)-B
    return A,b,d,solve(A,b)


def prepare(A,b,d,parts,rho):
    n=len(b);M=np.zeros_like(A);owner=np.empty(n,int)
    for i,idx in enumerate(parts):
        M[np.ix_(idx,idx)]=A[np.ix_(idx,idx)];owner[idx]=i
    R=M-A;T=np.zeros_like(A);f=np.zeros(n);leak=np.zeros(n)
    edge_reads=0;sends=0;local_mults=0;setup_proxy=0
    for i,idx in enumerate(parts):
        boundary=np.flatnonzero(np.any(R[idx]!=0,axis=0));K=M[np.ix_(idx,idx)]
        T[np.ix_(idx,boundary)]=solve(K,R[np.ix_(idx,boundary)])
        f[idx]=solve(K,b[idx]);leak[idx]=solve(K,np.ones(len(idx)))
        edge_reads+=np.count_nonzero(R[idx]);sends+=len(boundary)
        local_mults+=len(idx)*len(boundary);setup_proxy+=len(idx)**3+len(idx)**2*(len(boundary)+2)
    c=float(np.max(np.sum(abs(T),axis=1)))
    # T is similar to a symmetric matrix via D and the SPD block preconditioner.
    sd=np.sqrt(d);As=sd[:,None]*A/sd[None,:];Ms=sd[:,None]*M/sd[None,:]
    ev=eigvalsh(As,Ms);actual=float(max(abs(1-ev)))
    assert np.min(ev)>0 and actual<=c+1e-10 and c<=rho+1e-10
    assert np.min(T)>=-1e-13
    assert np.max(abs(T.sum(axis=1)-(1-solve(M,A@np.ones(n)))))<1e-10
    if np.max(abs(A@np.ones(n)-(1-rho)))<1e-10:
        assert abs(c-(1-(1-rho)*min(leak)))<1e-10
    return T,f,c,actual,edge_reads,sends,local_mults,setup_proxy,float(ev.min()),float(ev.max())


def run(T,f,mu,L,exact,method,cap=30000):
    q=(np.sqrt(L)-np.sqrt(mu))/(np.sqrt(L)+np.sqrt(mu));center=(L+mu)/2;half=(L-mu)/2
    x=np.zeros_like(f);old=x.copy();a=1/center;norm=np.linalg.norm(exact)
    for k in range(1,cap+1):
        r=f-x+T@x
        if method=='AR':a,m=1/center,0.
        elif method=='CAT':a,m=4/(np.sqrt(L)+np.sqrt(mu))**2,q*q
        elif k==1:a,m=1/center,0.
        else:
            factor=half*half/(2 if k==2 else 4);an=1/(center-factor*a);m=factor*a*an;a=an
        new=x+a*r+m*(x-old);old,x=x,new
        err=float(np.linalg.norm(x-exact)/norm)
        if err<=1e-6 or not np.isfinite(err) or err>1e8:break
    return k,err


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('a7_results.json'));args=p.parse_args();rows=[];side=12
    for kind in ('grid','modular'):
        for problem in ('linear','tanh'):
            for rho in (.95,.99,.999):
                for seed in (20,21,22):
                    A,b,d,exact=system(side,rho,kind,problem,seed)
                    for width in (1,3,4,6):
                        T,f,c,actual,reads,sends,mults,setup,mu,L=prepare(A,b,d,blocks(side,width),rho)
                        for calibration,lo,hi in [('bound',1-c,1+c),('oracle',mu,L)]:
                            for method in ('AR','CAT','Cheb'):
                                k,err=run(T,f,lo,hi,exact,method)
                                rows.append(dict(kind=kind,problem=problem,rho=rho,seed=seed,width=width,calibration=calibration,method=method,c=c,actual_radius=actual,mu=lo,L=hi,rounds=k,error=err,converged=err<=1e-6,edge_reads=k*reads,scalar_sends=k*sends,local_dense_mults_proxy=k*mults,setup_proxy=setup))
    args.out.write_text(json.dumps(dict(status='A7_BLOCK_LOCAL_EXECUTED',rows=rows),indent=2,default=lambda x:x.item())+'\n')
    print('runs',len(rows),'converged',sum(r['converged'] for r in rows))
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in ('kind','problem','rho','seed','width','calibration')), {})[r['method']]=r
    for cal in ('bound','oracle'):
        for width in (1,3,4,6):
            g=[v for k,v in groups.items() if k[-2:]==(width,cal)]
            ratios=[v['Cheb']['rounds']/v['CAT']['rounds'] for v in g]
            print(cal,width,'wins',sum(x>1 for x in ratios),'ties',sum(x==1 for x in ratios),'of',len(g),'median Cheb/CAT',float(np.median(ratios)))
if __name__=='__main__':main()
