"""A6 exploratory spectral-only calibration and paired tagged-DAG controls."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq


def calibration(rho):
    L=2*rho/(1+rho)
    q=brentq(lambda q:q*q*(2-q)/(1-q+q*q)-L,0,1,xtol=1e-15)
    a=1-q+q*q;b=q*q;delta=q**3/L
    return a,b,delta/a,q,L


def certificate(rho):
    a,b,g,q,L=calibration(rho)
    assert abs(a*L-q*q*(2-q))<1e-12
    # At z=0: r*(r^2-q*r+q^2); at L: (r-q)^2*(r+q).
    # Check strict Schur inequalities at a slightly enlarged disk.
    Q=q+1e-7
    z=np.array([0.,L]);A=a-1-b;B=b-a*(1+g)*z;C=a*g*z
    tests=np.r_[Q**3-C,Q**3+A*Q**2+B*Q+C,Q**3-A*Q**2+B*Q-C,Q**6-B*Q**4+A*C*Q**2-C*C]
    assert min(tests)>0,(rho,tests)
    return dict(rho=rho,alpha=a,beta=b,gamma=g,q=q,L=L,min_enlarged_schur_margin=float(min(tests)))


def grid(side,rho,seed):
    g=np.arange(side**2).reshape(side,side)
    nbr=np.stack([np.roll(g,d,ax).ravel() for ax in (0,1) for d in (-1,1)]+[np.roll(g,(d,-d),(0,1)).ravel() for d in (-1,1)],axis=1)
    w=rho*np.array([.15]*4+[.20]*2); B=np.zeros((side**2,side**2));B[np.arange(side**2)[:,None],nbr]=w
    rng=np.random.default_rng(seed);b=rng.normal(size=side**2);b=.85*b/np.linalg.norm(b)+.15/side
    return nbr,B,b,np.linalg.solve(np.eye(len(b))-B,b)


def run(case,rho,method,delay,seed,cap=30000):
    nbr,B,b,exact=case;n=len(b);s=rho
    eps=(1-rho)/(1+s);a6,b6,g6,_,_=calibration(rho)
    q=(np.sqrt(1+rho)-np.sqrt(1-rho))/(np.sqrt(1+rho)+np.sqrt(1-rho))
    rng=np.random.default_rng(seed);x=np.zeros(n);prev=x.copy();older=x.copy();finish=np.zeros(n)
    arrival=np.zeros(nbr.shape);arrival_prev=arrival.copy();arrival_older=arrival.copy()
    cheb_a=1.;norm=np.linalg.norm(exact)
    for k in range(1,cap+1):
        proc=rng.uniform(.1,1.,n);transit=rng.uniform(0,delay,nbr.shape)
        if method in ('Cheb','FreshCAT'):
            ready=np.maximum(finish,arrival.max(axis=1));r=b-x+B@x
            if method=='FreshCAT':a,m=1+q*q,q*q
            elif k==1:a,m=1.,0.
            else:
                f=rho*rho/(2 if k==2 else 4);a=1/(1-f*cheb_a);m=f*cheb_a*a;cheb_a=a
            new=x+a*r+m*(x-prev)
        else:
            ready=np.maximum(finish,np.maximum(arrival_prev.max(axis=1),arrival_older.max(axis=1)))
            a,m,g=(a6,b6,g6) if method=='A6' else (.5,1-np.sqrt(eps),1.)
            p=(1+g)*prev-g*older
            new=x+m*(x-prev)+a*((b+B@p+s*p)/(1+s)-x)
        finish=ready+proc;sent=np.maximum(arrival,finish[nbr]+transit)
        arrival_older,arrival_prev,arrival=arrival_prev,arrival,sent
        older,prev,x=prev,x,new
        err=float(np.linalg.norm(x-exact)/norm)
        if err<=1e-6 or not np.isfinite(err) or err>1e8:break
    return dict(method=method,rho=rho,delay_max=delay,schedule_seed=seed,rounds=k,row_updates=k*n,scalar_edge_sends=k*n*nbr.shape[1],completion_time=float(max(finish)),relative_error=err,status='converged' if err<=1e-6 else 'failed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('a6_results.json'));args=p.parse_args()
    cert=[certificate(rho) for rho in (.95,.99,.999,.9999,.99999,.999999)]
    rows=[]
    for rho in (.95,.99,.999):
        for seed in (10,11,12,13,14):
            case=grid(8,rho,seed)
            for delay in (1.,8.,32.):
                for method in ('Cheb','FreshCAT','A3','A6'):
                    rows.append(dict(rhs_seed=seed,**run(case,rho,method,delay,1000+seed)))
    summary=[]
    for rho in (.95,.99,.999):
        for delay in (1.,8.,32.):
            group=[r for r in rows if r['rho']==rho and r['delay_max']==delay]
            med={m:{k:float(np.median([r[k] for r in group if r['method']==m])) for k in ('rounds','completion_time')} for m in ('Cheb','FreshCAT','A3','A6')}
            pairs=[]
            for seed in (10,11,12,13,14):
                p={r['method']:r for r in group if r['rhs_seed']==seed};pairs.append(p['A6']['completion_time']/p['Cheb']['completion_time'])
            summary.append(dict(rho=rho,delay_max=delay,medians=med,median_paired_A6_over_Cheb=float(np.median(pairs)),A6_faster=sum(x<1 for x in pairs)))
    data=dict(status='A6_ENDPOINT_CONSTRUCTION_EXECUTED',certificates=cert,rows=rows,summary=summary)
    args.out.write_text(json.dumps(data,indent=2)+'\n');print('runs',len(rows),'converged',sum(r['status']=='converged' for r in rows))
    for s in summary:print(s['rho'],s['delay_max'],s['medians'], 'A6/Cheb',s['median_paired_A6_over_Cheb'], 'wins',s['A6_faster'])
if __name__=='__main__':main()
