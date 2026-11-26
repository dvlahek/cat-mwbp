#!/usr/bin/env python3
"""A8: goal-oriented residual certificate and event-triggered communication pilot."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from a7_block_local import system,blocks,prepare


def setup(side,rho,kind,problem,seed,K):
    A,b,d,xstar=system(side,rho,kind,problem,seed)
    parts=blocks(side,4);n=len(b);M=np.zeros_like(A);cost=np.zeros(n,int)
    for idx in parts:
        M[np.ix_(idx,idx)]=A[np.ix_(idx,idx)]
    R=M-A;B=np.eye(n)-A
    for idx in parts:cost+=np.any(R[idx]!=0,axis=0)
    T,f,c,*_=prepare(A,b,d,parts,rho)
    # Four selected drive-gradient coordinates, fixed before evaluation.
    v=np.zeros(n);sel=np.array([0,1,side,side+1]);v[sel]=d[sel]
    assert np.min(B)>=-1e-14 and np.max(B.sum(axis=0))<=rho+1e-12
    w=np.zeros(n);h=v.copy()
    for _ in range(K):w+=h;h=B.T@h
    w+=np.max(h)/(1-rho)
    # Tiny floating-point safety padding; not interval arithmetic.
    w+=max(0.,float(np.max(v-A.T@w)))/(1-rho)+1e-12/(1-rho)
    assert np.min(A.T@w-v)>=-1e-10
    wo=solve(A.T,v);assert np.min(w-wo)>=-1e-9
    influence=R.T@w;active=cost>0
    factor=np.ones(n);weights=influence[active]
    factor[active]=np.clip(np.mean(weights)/np.maximum(weights,1e-30),.1,10.)
    factor[active]*=np.sum(weights)/np.dot(weights,factor[active])
    assert abs(influence@factor-influence.sum())<1e-8*(1+influence.sum())
    return A,b,d,xstar,M,R,T,f,c,w,wo,v,cost,factor


def run(case,method,policy,rho,K,tol=1e-5,cap=30000):
    A,b,d,exact,M,R,T,f,c,w,wo,v,cost,factor=case
    x=np.zeros_like(b);old=x.copy();cache=x.copy();q=(np.sqrt(1+c)-np.sqrt(1-c))/(np.sqrt(1+c)+np.sqrt(1-c));a=1.;sends=0
    norm=float(v@abs(exact));assert norm>0
    base=.05*np.linalg.norm(f,np.inf);max_violation=0.;max_rel_violation=0.
    rate=c if method=='AR' else q
    for k in range(1,cap+1):
        r=f-x+T@cache
        if method=='AR':a,m=1.,0.
        elif method=='CAT':a,m=1+q*q,q*q
        elif k==1:a,m=1.,0.
        else:
            ff=c*c/(2 if k==2 else 4);an=1/(1-ff*a);m=ff*a*an;a=an
        new=x+a*r+m*(x-old);old,x=x,new
        tau=np.zeros_like(x) if policy=='full' else base*rate**k*(factor if policy=='weighted' else np.ones_like(x))
        mask=cost>0 if policy=='full' else (cost>0)&(abs(x-cache)>tau)
        cache[mask]=x[mask];sends+=int(cost[mask].sum())
        # Cross-block cache is consistent and immediately delivered in this pilot.
        if k%5!=0 and k!=cap:continue
        assert np.max((abs(x-cache)-tau)[cost>0])<=1e-12
        residual=abs(b-M@x+R@cache)+R@tau
        E=float(w@residual); Eo=float(wo@residual); Eu=float(v.sum()*max(residual)/(1-rho))
        true=float(v@abs(x-exact));g=float(v@abs(x));rel=E/(g-E) if g>E else float('inf')
        max_violation=max(max_violation,true-E);max_rel_violation=max(max_rel_violation,true/norm-rel)
        if not np.isfinite(E) or E>1e12:break
        if rel<=tol:break
    assert max_violation<1e-9 and max_rel_violation<1e-9
    return dict(method=method,policy=policy,rounds=k,checks=k//5,sends=sends,setup_sends=K*int(cost.sum()),actual_relative_error=true/norm,certified_relative_error=rel,certified_abs=E,oracle_abs=Eo,uniform_abs=Eu,converged=rel<=tol,max_violation=max_violation)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('a8_results.json'));p.add_argument('--K',type=int,default=16);args=p.parse_args();rows=[]
    for kind in ('grid','modular'):
        for problem in ('linear','tanh'):
            for rho in (.95,.99,.999):
                for seed in (30,31):
                    case=setup(12,rho,kind,problem,seed,args.K)
                    for method in ('AR','CAT','Cheb'):
                        for policy in ('full','uniform','weighted'):
                            rows.append(dict(kind=kind,problem=problem,rho=rho,seed=seed,**run(case,method,policy,rho,args.K)))
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in ('kind','problem','rho','seed','method')), {})[r['policy']]=r
    summary=[]
    for method in ('AR','CAT','Cheb'):
        for pol in ('uniform','weighted'):
            g=[x for k,x in groups.items() if k[-1]==method and x['full']['converged'] and x[pol]['converged']]
            ratio=[x['full']['sends']/x[pol]['sends'] for x in g];total=[(x['full']['sends']+x['full']['setup_sends'])/(x[pol]['sends']+x[pol]['setup_sends']) for x in g]
            summary.append(dict(method=method,policy=pol,matched=len(g),wins=sum(x>1 for x in ratio),median_sends_ratio=float(np.median(ratio)),median_with_setup=float(np.median(total))))
    ww=[g['weighted']['sends']/g['uniform']['sends'] for g in groups.values() if g['weighted']['converged'] and g['uniform']['converged']]
    data=dict(status='A8_GRADIENT_CERTIFICATE_EXECUTED',K=args.K,rows=rows,summary=summary,weighted_vs_uniform_median=float(np.median(ww)),weighted_wins=sum(x<1 for x in ww),max_certificate_violation=max(r['max_violation'] for r in rows))
    args.out.write_text(json.dumps(data,indent=2,default=lambda x:x.item())+'\n');print('runs',len(rows),'converged',sum(r['converged'] for r in rows));print(json.dumps({k:v for k,v in data.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
