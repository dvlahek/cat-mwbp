#!/usr/bin/env python3
"""A10: equal-action tracking on held-out moving frozen adjoint sequences."""
import argparse,json
from pathlib import Path
import numpy as np

METHODS=('AR-warm','CAT-reset','CAT-persistent','Cheb-restart','Cheb-continuous','Cheb-extrap05','Cheb-extrap10')

def sequence(rho,regime,seed,kind,n=24,steps=120):
    rng=np.random.default_rng(seed); t=np.arange(steps); mu=1-rho; L=1+rho
    if kind=='linear':
        Q,_=np.linalg.qr(rng.normal(size=(n,n))); vals=np.linspace(mu,L,n); A0=(Q*vals)@Q.T
        u,v,w=rng.normal(size=(3,n));u/=np.linalg.norm(u);v/=np.linalg.norm(v);w/=np.linalg.norm(w)
        targets=u[None,:]+.5*np.sin(t[:,None]*.025)*v+.3*np.cos(t[:,None]*.041)*w
        if regime=='jump': targets+=.8*((t//20)%2)[:,None]*w
        if regime=='noise': targets+=.1*rng.normal(size=targets.shape)/np.sqrt(n)
        A=np.repeat(A0[None,:,:],steps,axis=0); b=np.einsum('tij,tj->ti',A,targets); G=np.repeat(np.eye(n)[None,:,:],steps,axis=0)
    else:
        S=np.zeros((n,n));i=np.arange(n);S[i,(i-1)%n]=S[i,(i+1)%n]=.5
        u=.05*rng.normal(size=n); y=.1*rng.normal(size=n);h=np.zeros(n);As=[];bs=[];Gs=[]
        for j in t:
            coupling=rho*(.98+.02*np.cos(.035*j));gain=1+.15*np.sin(.025*j);bias=.015*np.sin(.041*j)
            if regime=='jump':bias+=.025*((j//20)%2)
            if regime=='noise':bias+=.004*rng.normal()
            for _ in range(30000):
                new=np.tanh(coupling*(S@h)+gain*u+bias)
                if np.max(abs(new-h))<1e-12:h=new;break
                h=new
            else:raise RuntimeError('equilibrium failed')
            d=1-h*h;As.append(np.eye(n)-coupling*S*d[None,:]);bs.append((h-y)/n);Gs.append(d*np.stack([S@h,u,np.ones(n)]))
        A=np.array(As);b=np.array(bs);G=np.array(Gs);targets=np.linalg.solve(A,b[...,None])[...,0]
    return A,b,G,targets,mu,L

def run(seq,budget,method,setting=None):
    A,b,G,star,mu,L=seq; n=len(b[0]);center=(L+mu)/2;half=(L-mu)/2;q=(np.sqrt(L)-np.sqrt(mu))/(np.sqrt(L)+np.sqrt(mu));ca=4/(np.sqrt(L)+np.sqrt(mu))**2
    # All methods start with identical zero states; first 20 time points excluded from scores.
    x=np.zeros(n);old=x.copy();last=x.copy();older=x.copy();a=1/center;count=0; es=[];gs=[]
    for t in range(len(b)):
        if method in ('CAT-reset','Cheb-restart','Cheb-extrap05','Cheb-extrap10','Cheb-tuned') :
            if method.startswith('Cheb-extrap') and t>=2:x=last+(.5 if method.endswith('05') else 1.)*(last-older)
            if method=='Cheb-tuned' and t>=2:x=last+setting*(last-older)
            old=x.copy();count=0;a=1/center
        for _ in range(budget):
            r=b[t]-A[t]@x;count+=1
            if method=='AR-warm': aa,m=1/center,0.
            elif method=='AR-tuned': aa,m=setting/center,0.
            elif method=='CAT-tuned': m=setting*q*q;aa=(1+m)/center
            elif method.startswith('CAT'): aa,m=ca,q*q
            elif count==1:aa,m=1/center,0.
            else:
                ff=half*half/(2 if count==2 else 4);aa=1/(center-ff*a);m=ff*a*aa
            a=aa;new=x+aa*r+m*(x-old);old,x=x,new
        older,last=last,x.copy()
        if t>=20:
            es.append(float(np.linalg.norm(x-star[t])**2));gs.append(float(np.linalg.norm(G[t]@(x-star[t]))**2))
    return dict(adjoint_nrmse=float(np.sqrt(sum(es)/np.sum(star[20:]**2))),gradient_nrmse=float(np.sqrt(sum(gs)/np.sum(np.einsum('tij,tj->ti',G[20:],star[20:])**2))),actions=len(b)*budget)

def selftest():
    # Constant-drift analytic steady error, scalar mode.
    a,beta,lam,d=.8,.3,.7,.01;x=old=0.
    for t in range(1,1001):new=x+a*lam*(t*d-x)+beta*(x-old);old,x=x,new
    assert abs(x-1000*d-(1-(1-beta)/(a*lam))*d)<1e-12

def tuning(rho,regime,kind,budget,cal):
    # Independent calibration seeds, same loss and budget for all families.
    options={'AR-tuned':(.25,.5,.75,1.),'CAT-tuned':(0.,.25,.5,.75,1.),'Cheb-tuned':(0.,.1,.25,.5,1.)}
    return {m:min(values,key=lambda v:np.mean([run(seq,budget,m,v)['gradient_nrmse'] for seq in cal])) for m,values in options.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('a10_results.json'));args=p.parse_args();selftest();rows=[]
    for kind in ('linear','tanh'):
        for rho in (.95,.99,.999):
            for regime in ('smooth','jump','noise'):
                cal=[sequence(rho,regime,seed,kind) for seed in (48,49)]
                settings={budget:tuning(rho,regime,kind,budget,cal) for budget in (1,4,8)}
                for seed in (50,51,52):
                    seq=sequence(rho,regime,seed,kind)
                    for budget in (1,4,8):
                        for method in METHODS+tuple(settings[budget]):
                            setting=settings[budget].get(method);rows.append(dict(kind=kind,rho=rho,regime=regime,seed=seed,budget=budget,method=method,setting=setting,**run(seq,budget,method,setting)))
    assert all(np.isfinite(r['gradient_nrmse']) and np.isfinite(r['adjoint_nrmse']) for r in rows)
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in ('kind','rho','regime','seed','budget')), {})[r['method']]=r
    summary=[]
    for kind in ('linear','tanh'):
        for budget in (1,4,8):
            g=[v for k,v in groups.items() if k[0]==kind and k[-1]==budget]
            for metric in ('adjoint_nrmse','gradient_nrmse'):
                for control in ('CAT-reset','Cheb-restart','Cheb-continuous','Cheb-extrap05','Cheb-extrap10'):
                    ratio=np.array([v[control][metric]/v['CAT-persistent'][metric] for v in g]);summary.append(dict(kind=kind,budget=budget,metric=metric,control=control,median_ratio=float(np.median(ratio)),wins=int(np.sum(ratio>1+1e-8)),ties=int(np.sum(abs(ratio-1)<=1e-8)),n=len(g)))
    data=dict(status='A10_TRACKING_EXECUTED',rows=rows,summary=summary);args.out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');print('runs',len(rows));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
