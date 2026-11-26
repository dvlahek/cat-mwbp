"""A5: exact switching counterexample and tagged logical-version DAG simulation."""
import argparse, json
from fractions import Fraction as F
import numpy as np


def counterexample():
    a,b,z=F(1,2),F(99,100),F(2,3)
    M0=[[1-a+b+a*z,-b,F(0)],[F(1),F(0),F(0)],[F(0),F(1),F(0)]]
    M1=[[1-a+b,-b+2*a*z,-a*z],[F(1),F(0),F(0)],[F(0),F(1),F(0)]]
    T=[[F(i==j) for j in range(3)] for i in range(3)]
    for M in [M0]*5+[M1]*3:
        T=[[sum(M[i][k]*T[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    A=[[T[i][j]+F(i==j) for j in range(3)] for i in range(3)]
    det=sum(A[0][i]*(A[1][(i+1)%3]*A[2][(i+2)%3]-A[1][(i+2)%3]*A[2][(i+1)%3]) for i in range(3))
    roots=np.linalg.eigvals(np.array(T,float)); assert det<0
    return dict(det_I_plus_T=str(det),period_radius=float(max(abs(roots))),individual_radii=[float(max(abs(np.linalg.eigvals(np.array(M,float))))) for M in (M0,M1)])


def grid(side,rho,seed):
    g=np.arange(side**2).reshape(side,side)
    nbr=np.stack([np.roll(g,d,ax).ravel() for ax in (0,1) for d in (-1,1)]+[np.roll(g,(d,-d),(0,1)).ravel() for d in (-1,1)],axis=1)
    w=rho*np.array([.15]*4+[.20]*2); B=np.zeros((side**2,side**2));B[np.arange(side**2)[:,None],nbr]=w
    rng=np.random.default_rng(seed);b=rng.normal(size=side**2);b=.85*b/np.linalg.norm(b)+.15/side
    return nbr,w,B,b,np.linalg.solve(np.eye(len(b))-B,b)


def run(case,rho,method,delay,seed,cap=30000):
    nbr,w,B,b,exact=case;n=len(b);s=rho
    eps=(1-rho)/(1+s);beta=1-np.sqrt(eps);rng=np.random.default_rng(seed)
    x=np.zeros(n);prev=x.copy();older=x.copy();finish=np.zeros(n)
    arrival=np.zeros(nbr.shape);arrival_prev=arrival.copy();arrival_older=arrival.copy()
    cheb_a=1.; norm=np.linalg.norm(exact)
    for k in range(1,cap+1):
        proc=rng.uniform(.1,1.,n); transit=rng.uniform(0,delay,nbr.shape)
        if method in ('AR','Cheb'):
            ready=np.maximum(finish,arrival.max(axis=1));r=b-x+B@x
            if method=='AR': a,m=1.,0.
            elif k==1:a,m=1.,0.
            else:
                f=rho*rho/(2 if k==2 else 4);a=1/(1-f*cheb_a);m=f*cheb_a*a;cheb_a=a
            new=x+a*r+m*(x-prev)
        else:
            ready=np.maximum(finish,np.maximum(arrival_prev.max(axis=1),arrival_older.max(axis=1)))
            p=2*prev-older; a=.5;m=beta if method=='Predictive' else 0.
            new=x+m*(x-prev)+a*((b+B@p+s*p)/(1+s)-x)
        finish=ready+proc
        sent=np.maximum(arrival,finish[nbr]+transit)
        arrival_older,arrival_prev,arrival=arrival_prev,arrival,sent
        older,prev,x=prev,x,new
        err=float(np.linalg.norm(x-exact)/norm)
        if err<=1e-6 or not np.isfinite(err) or err>1e8:break
    return dict(method=method,rho=rho,delay_max=delay,seed=seed,rounds=k,row_updates=k*n,scalar_edge_sends=k*n*nbr.shape[1],completion_time=float(max(finish)),relative_error=err,status='converged' if err<=1e-6 else 'failed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='a5_tagged_results.json');p.add_argument('--side',type=int,default=6);args=p.parse_args()
    rows=[]
    for rho in (.95,.99,.999):
        for seed in range(2):
            case=grid(args.side,rho,seed)
            for delay in (1.,8.):
                for method in ('AR','Cheb','PredictiveNoMomentum','Predictive'):
                    rows.append(run(case,rho,method,delay,100+seed))
    data=dict(status='A5_TAGGED_DAG_EXECUTED',counterexample=counterexample(),rows=rows)
    with open(args.out,'w') as f:json.dump(data,f,indent=2)
    print(data['counterexample']);print('runs',len(rows),'converged',sum(r['status']=='converged' for r in rows))
    for rho in (.95,.99,.999):
        for method in ('AR','Cheb','PredictiveNoMomentum','Predictive'):
            g=[r for r in rows if r['rho']==rho and r['method']==method];print(rho,method, np.median([r['rounds'] for r in g]),[round(np.median([r['completion_time'] for r in g if r['delay_max']==d]),2) for d in (1.,8.)])
if __name__=='__main__':main()
