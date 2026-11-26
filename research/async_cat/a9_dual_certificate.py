#!/usr/bin/env python3
"""A9: signed dual residual bounds for the full gradient of a tied equilibrium model."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import solve


def problem(side,rho,kind,seed):
    n=side**2; grid=np.arange(n).reshape(side,side); eye=np.eye(n)
    owner=((grid//side)//4*(side//4)+(grid%side)//4).ravel()
    mats=[]
    for axis in (0,1):
        S=np.zeros((n,n))
        for shift in (-1,1):S[np.arange(n),np.roll(grid,shift,axis).ravel()]=.5
        if kind=='modular':
            S[owner[:,None]!=owner[None,:]]*=.1;S+=np.diag(1-S.sum(axis=1))
        mats.append(S)
    rng=np.random.default_rng(seed);u=.05*rng.normal(size=n);target=.1*rng.normal(size=n)
    def equilibrium(theta):
        W=theta[0]*mats[0]+theta[1]*mats[1];h=np.zeros(n)
        for _ in range(100000):
            nxt=np.tanh(W@h+theta[2]*u+theta[3]);err=np.max(abs(nxt-h));h=nxt
            if err<1e-13:return h,W
        raise RuntimeError('forward failed')
    theta=np.array([rho/2,rho/2,1.,0.]);h,W=equilibrium(theta);d=1-h*h
    A=eye-W*d[None,:];b=(h-target)/n;G=d*np.stack([mats[0]@h,mats[1]@h,u,np.ones(n)])
    exact=G@solve(A,b);M=A*(owner[:,None]==owner[None,:]);R=M-A
    T=solve(M,R);f=solve(M,b);c=float(np.max(T.sum(axis=1)))
    assert np.min(T)>=-1e-12 and c<1
    # one scalar per unique source node per destination block; symmetric support
    sends=sum(np.count_nonzero(np.any(R[owner==i]!=0,axis=0)) for i in np.unique(owner))
    if seed==40 and rho==.95:
        for j in range(4):
            delta=np.zeros(4);delta[j]=1e-6
            hp,_=equilibrium(theta+delta);hm,_=equilibrium(theta-delta)
            fd=(np.sum((hp-target)**2)-np.sum((hm-target)**2))/(4e-6*n)
            assert abs(fd-exact[j])<1e-7,(fd,exact[j])
    return T,f,G,c,exact,int(sends)


def iterate(T,f,c,method):
    # works for column RHS (primal or transpose sensitivity), with shared spectral interval
    x=np.zeros_like(f);old=x.copy();a=1.;q=(np.sqrt(1+c)-np.sqrt(1-c))/(np.sqrt(1+c)+np.sqrt(1-c))
    for k in range(1,30001):
        r=f-x+T@x
        if method=='CAT':a,m=1+q*q,q*q
        elif k==1:a,m=1.,0.
        else:
            ff=c*c/(2 if k==2 else 4);an=1/(1-ff*a);m=ff*a*an;a=an
        new=x+a*r+m*(x-old);old,x=x,new
        yield k,x


def bound(g,E):
    norm=np.linalg.norm(g,1)
    return float(E/(norm-E)) if norm>E else float('inf')


def run(case,K,method,tol=1e-5):
    T,f,G,c,exact,sends=case;p=len(G);norm=np.linalg.norm(exact,1)
    # Dual solve Z(I-T)=G, K steps, with the same Chebyshev calibration.
    Z=np.zeros_like(G)
    if K:
        for k,z in iterate(T.T,G.T,c,'Cheb'):
            if k==K:Z=z.T;break
    F=G-Z+Z@T if K else G
    tail=float(np.abs(F).sum()/(1-c));uniform=float(np.abs(G).sum()/(1-c))
    rows={};max_violation=0.
    for k,x in iterate(T,f,c,method):
        r=f-x+T@x;g=G@x;cor=Z@r;rest=tail*np.max(abs(r));plain=uniform*np.max(abs(r))
        for policy,gg,E,setup in [('uniform',g,plain,0),('signed',g,np.linalg.norm(cor,1)+rest,p*(K+1)),('corrected',g+cor,rest,p*(K+1))]:
            if policy in rows:continue
            actual=float(np.linalg.norm(gg-exact,1));max_violation=max(max_violation,actual-E)
            if bound(gg,E)<=tol or k==30000:
                rows[policy]=dict(method=method,policy=policy,K=K,rounds=k,cert=bound(gg,E),actual=actual/norm,converged=bound(gg,E)<=tol,scalar_sends=(2*k+setup)*sends,setup_sends=setup*sends,tail_ratio=tail/uniform)
        if len(rows)==3:break
    assert max_violation<=1e-10
    return list(rows.values())


def sensitivity(case,tol=1e-5):
    T,f,G,c,exact,sends=case;p=len(G);norm=np.linalg.norm(exact,1)
    for k,z in iterate(T.T,G.T,c,'Cheb'):
        Z=z.T;F=G-Z+Z@T;g=Z@f;E=float(np.abs(F).sum()*np.max(abs(f))/(1-c));actual=float(np.linalg.norm(g-exact,1));assert actual<=E+1e-10
        rel=bound(g,E)
        if rel<=tol or k==30000:break
    return dict(method='ForwardSensitivity',policy='direct',K=0,rounds=k,cert=rel,actual=actual/norm,converged=rel<=tol,scalar_sends=2*k*p*sends,setup_sends=0,tail_ratio=None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('a9_results.json'));args=ap.parse_args();rows=[]
    for kind in ('grid','modular'):
        for rho in (.95,.99,.999):
            for seed in (40,41,42):
                case=problem(12,rho,kind,seed)
                for method in ('CAT','Cheb'):
                    for K in (8,32):
                        rows.extend(dict(kind=kind,rho=rho,seed=seed,**r) for r in run(case,K,method))
                rows.append(dict(kind=kind,rho=rho,seed=seed,**sensitivity(case)))
    summary=[]
    for method in ('CAT','Cheb'):
        for K in (8,32):
            group=[r for r in rows if r['method']==method and r['K']==K];pairs={}
            for r in group:pairs.setdefault((r['kind'],r['rho'],r['seed']),{})[r['policy']]=r
            for policy in ('signed','corrected'):
                ratios=[v['uniform']['scalar_sends']/v[policy]['scalar_sends'] for v in pairs.values() if v['uniform']['converged'] and v[policy]['converged']]
                summary.append(dict(method=method,K=K,policy=policy,matched=len(ratios),wins=sum(r>1 for r in ratios),median_ratio=float(np.median(ratios))))
    data=dict(status='A9_DUAL_CERTIFICATE_EXECUTED',rows=rows,summary=summary)
    args.out.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.item())+'\n');print('runs',len(rows),'converged',sum(r['converged'] for r in rows));print(json.dumps(summary,indent=2,default=lambda x:x.item()))
if __name__=='__main__':main()
