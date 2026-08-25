#!/usr/bin/env python3
"""Reference solvers for the implicit CAT benchmark.

The controlled systems use the same symmetric non-bipartite 12x12 grid as the
spectral study. Every reported count is the number of J^T v actions needed to
reach a common exact target: relative adjoint error <= 1e-6 for linear systems
and relative implicit-gradient error <= 1e-6 for tanh systems.

GMRES, good Broyden, and Anderson are given their usual global vector algebra
for free in this count. The experiment therefore compares Jacobian actions,
not wall-clock cost or locality.
"""

from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

TOL=1e-6


def grid(side,rho):
    n=side*side; W=np.zeros((n,n))
    idx=lambda r,c:(r%side)*side+(c%side)
    for r in range(side):
        for c in range(side):
            i=idx(r,c)
            for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                W[i,idx(rr,cc)]=.15*rho
            W[i,idx(r-1,c+1)]=.20*rho
            W[i,idx(r+1,c-1)]=.20*rho
    return csr_matrix(W)


def tanh_state(W,u):
    h=np.zeros_like(u)
    for _ in range(30000):
        new=np.tanh(W@h+u)
        if np.linalg.norm(new-h)<=1e-12*(1+np.linalg.norm(new)): return new
        h=new
    raise RuntimeError("tanh equilibrium did not converge")


def make_case(problem,rho,seed,side=12):
    W=grid(side,rho); n=W.shape[0]
    rng=np.random.default_rng(seed+100000*(problem=="tanh"))
    if problem=="linear":
        q=rng.normal(size=n); q/=np.linalg.norm(q)+1e-30
        rhs=.85*q+.15*np.ones(n)/np.sqrt(n); rhs/=np.linalg.norm(rhs)+1e-30
        exact=np.asarray(spsolve(eye(n,format="csr")-W,rhs))
        ev=np.linalg.eigvalsh(W.toarray()); d=None
        jtv=lambda v:np.asarray(W@v)
    else:
        u=.05*rng.normal(size=n); h=tanh_state(W,u); d=1-h*h
        readout=rng.normal(size=n); readout/=np.linalg.norm(readout)+1e-30
        rhs=(float(readout@h)-1)*readout
        exact=np.linalg.solve(np.eye(n)-W.toarray()@np.diag(d),rhs)
        sd=np.sqrt(np.maximum(d,1e-15))
        ev=np.linalg.eigvalsh(sd[:,None]*W.toarray()*sd[None,:])
        jtv=lambda v:np.asarray(W@(d*v))
    mu,L=1-float(ev.max()),1-float(ev.min())
    def err(x):
        if d is None: return np.linalg.norm(x-exact)/(np.linalg.norm(exact)+1e-30)
        eg=d*exact
        return np.linalg.norm(d*x-eg)/(np.linalg.norm(eg)+1e-30)
    return rhs,exact,jtv,mu,L,err


def coeff(mu,L):
    sm,sL=np.sqrt(mu),np.sqrt(L)
    return 2/(L+mu),4/(sL+sm)**2,((sL-sm)/(sL+sm))**2


def stationary(rhs,jtv,error,alpha,beta,max_actions=12000):
    x=np.zeros_like(rhs); prev=np.zeros_like(rhs)
    for k in range(max_actions+1):
        if error(x)<=TOL: return k,x
        if k==max_actions: break
        r=rhs-(x-jtv(x)); new=x+alpha*r+beta*(x-prev); prev,x=x,new
    return max_actions,x


def gmres_exact(rhs,jtv,error):
    n=len(rhs); bnorm=np.linalg.norm(rhs)
    V=np.zeros((n,n+1)); H=np.zeros((n+1,n)); V[:,0]=rhs/bnorm
    g=np.zeros(n+1); g[0]=bnorm; x=np.zeros(n)
    for k in range(n):
        w=V[:,k]-jtv(V[:,k])
        for j in range(k+1):
            H[j,k]=V[:,j]@w; w-=H[j,k]*V[:,j]
        H[k+1,k]=np.linalg.norm(w)
        if H[k+1,k]>1e-14: V[:,k+1]=w/H[k+1,k]
        y=np.linalg.lstsq(H[:k+2,:k+1],g[:k+2],rcond=None)[0]
        x=V[:,:k+1]@y
        if error(x)<=TOL or H[k+1,k]<=1e-14: return k+1,x
    return n,x


def anderson(rhs,jtv,error,m,beta,reg,max_actions=12000):
    x=np.zeros_like(rhs); xs=[]; fs=[]; gs=[]
    for k in range(max_actions+1):
        if error(x)<=TOL: return k,x
        if k==max_actions: break
        f=rhs+jtv(x); g=f-x
        xs.append(x.copy()); fs.append(f.copy()); gs.append(g.copy())
        keep=min(m,len(xs)); X=np.column_stack(xs[-keep:]); F=np.column_stack(fs[-keep:]); G=np.column_stack(gs[-keep:])
        if keep==1: x=(1-beta)*X[:,0]+beta*F[:,0]; continue
        A=G.T@G+reg*np.eye(keep)
        K=np.block([[np.zeros((1,1)),np.ones((1,keep))],[np.ones((keep,1)),A]])
        q=np.zeros(keep+1); q[0]=1
        try: a=np.linalg.solve(K,q)[1:]
        except np.linalg.LinAlgError: a=np.linalg.lstsq(K,q,rcond=None)[0][1:]
        x=(1-beta)*(X@a)+beta*(F@a)
    return max_actions,x


def broyden(rhs,jtv,error,scale,damping,max_actions=12000):
    n=len(rhs); x=np.zeros(n); H=scale*np.eye(n)
    if error(x)<=TOL: return 0,x
    f=x-jtv(x)-rhs; actions=1
    while actions<=max_actions:
        s=-damping*(H@f); xn=x+s
        if error(xn)<=TOL: return actions,xn
        if actions==max_actions: return actions,xn
        fn=xn-jtv(xn)-rhs; actions+=1
        y=fn-f; Hy=H@y; sH=s@H; den=float(sH@y)
        if abs(den)>1e-12*(np.linalg.norm(sH)*np.linalg.norm(y)+1e-30):
            H+=np.outer(s-Hy,sH)/den
        x,f=xn,fn
    return actions,x


def best_sweep(rows,keys):
    df=pd.DataFrame(rows)
    stat=df.groupby(keys,as_index=False).agg(median=("actions","median"),mean=("actions","mean"),conv=("converged","mean"))
    valid=stat[stat.conv==1]
    pick=(valid if len(valid) else stat).sort_values(["conv","median","mean"],ascending=[False,True,True]).iloc[0]
    return pick


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("results/nonlocal_solver_reference"))
    p.add_argument("--quick",action="store_true")
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    rhos=(.95,) if a.quick else (.80,.95,.99)
    seeds=(0,1) if a.quick else tuple(range(5))
    and_mem=(5,20) if a.quick else (3,5,10,20,50)
    and_beta=(.8,1.) if a.quick else (.5,.8,1.)
    and_reg=(1e-10,1e-6) if a.quick else (1e-10,1e-8,1e-6,1e-4,1e-2)
    broy_scale=(.5,1.,1.5) if a.quick else (.25,.5,1.,1.5,2.)
    broy_damp=(.8,1.) if a.quick else (.5,.8,1.)
    table=[]
    for problem in ("linear","tanh"):
        for rho in rhos:
            base=[]; aa=[]; bb=[]
            for seed in seeds:
                rhs,exact,jtv,mu,L,error=make_case(problem,rho,seed,8 if a.quick else 12)
                ar,ca,cb=coeff(mu,L)
                for name,fn in (
                    ("AR-oracle",lambda:stationary(rhs,jtv,error,ar,0)),
                    ("CAT-oracle",lambda:stationary(rhs,jtv,error,ca,cb)),
                    ("GMRES",lambda:gmres_exact(rhs,jtv,error)),
                ):
                    actions,x=fn(); base.append(dict(method=name,seed=seed,actions=actions,converged=error(x)<=TOL,target_error=error(x)))
                for m in and_mem:
                    for beta in and_beta:
                        for reg in and_reg:
                            actions,x=anderson(rhs,jtv,error,m,beta,reg)
                            aa.append(dict(seed=seed,m=m,beta=beta,reg=reg,actions=actions,converged=error(x)<=TOL))
                for scale in broy_scale:
                    for damping in broy_damp:
                        actions,x=broyden(rhs,jtv,error,scale,damping)
                        bb.append(dict(seed=seed,scale=scale,damping=damping,actions=actions,converged=error(x)<=TOL))
            ad=best_sweep(aa,["m","beta","reg"]); bd=best_sweep(bb,["scale","damping"])
            row={"problem":problem,"rho":rho}
            bdf=pd.DataFrame(base)
            for method,g in bdf.groupby("method"): row[method]=float(g.actions.median())
            row.update(Anderson=float(ad["median"]),Anderson_memory=int(ad.m),Anderson_mixing=float(ad.beta),Anderson_regularization=float(ad.reg),
                       Broyden=float(bd["median"]),Broyden_inverse_scale=float(bd.scale),Broyden_damping=float(bd.damping))
            table.append(row)
            print(problem,rho,row)
    pd.DataFrame(table).to_csv(a.output/"solver_reference.csv",index=False)


if __name__=="__main__":
    main()
