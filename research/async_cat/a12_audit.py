"""A12: calibrate exact optimizer, replay common trajectories with matched CG."""
import json
from pathlib import Path
import numpy as np
from a11_train import forward,problem,solve_step,TUNED

def state(S,u,y,t,h=None):
    h=forward(S,u,t,h);d=1-h*h
    A=np.eye(u.shape[1])[None,:,:]-t[0]*S[None,:,:]*d[:,None,:]
    b=(h-y)/h.size;G=d[:,None,:]*np.stack([h@S,u,np.ones_like(u)],axis=1)
    x=np.linalg.solve(A,b[...,None])[...,0];g=np.einsum('bpn,bn->p',G,x)
    return h,d,A,b,G,x,g

def path(data,rho,lr,save=False):
    S,u,y,uv,yv=data;t=np.array([.95*rho,1.,0.]);h=None;loss=[];seq=[]
    for _ in range(60):
        h,d,A,b,G,x,g=state(S,u,y,t,h);loss.append(float(np.mean((h-y)**2)/2))
        if save:seq.append((float(t[0]),d,A,b,G,x,g))
        t=np.clip(t-lr*g,[0.,.1,-.1],[rho,2.,.1])
    loss.append(float(np.mean((forward(S,u,t)-y)**2)/2))
    return dict(initial=loss[0],final=loss[-1],increases=sum(b>a+1e-12 for a,b in zip(loss,loss[1:])),validation=float(np.mean((forward(S,uv,t)-yv)**2)/2)),seq

def cg(A,b,d,x,k):
    s=np.sqrt(d);z=s*x;rhs=s*b
    def op(p):return s*np.einsum('bij,bj->bi',A,p/s)
    r=rhs-op(z);p=r.copy();rr=np.sum(r*r,axis=1)
    for _ in range(k-1):
        Hp=op(p);den=np.sum(p*Hp,axis=1);a=np.divide(rr,den,out=np.zeros_like(rr),where=den>1e-30)
        z+=a[:,None]*p;r-=a[:,None]*Hp;nn=np.sum(r*r,axis=1);beta=np.divide(nn,rr,out=np.zeros_like(rr),where=rr>1e-30);p=r+beta[:,None]*p;rr=nn
    return z/s

def replay(seq,rho,budget,method):
    x=np.zeros_like(seq[0][3]);v=x.copy();cnt=0;a=1.;num=den=0.;adj=ad=0.;actions=0
    for c,d,A,b,G,exact,g in seq:
        if method=='CG':x=cg(A,b,d,x,budget);actions+=budget
        else:
            current=max(c,1e-12) if method in ('CAT-current','Cheb-current') else rho
            m={'CAT-current':'CAT','Cheb-current':'Cheb-restart'}.get(method,method)
            x,v,cnt,a,used,_=solve_step(A,b,x,v,budget,m,current,TUNED[rho][budget],cnt,a);actions+=used
        e=np.einsum('bpn,bn->p',G,x)-g;num+=float(e@e);den+=float(g@g);adj+=float(np.sum((x-exact)**2));ad+=float(np.sum(exact**2))
    return dict(method=method,gradient_nrmse=float(np.sqrt(num/den)),adjoint_nrmse=float(np.sqrt(adj/ad)),actions=actions)

def main():
    rows=[];cal=[];paths=[];curv=[]
    for rho in (.95,.99,.999):
        for lr in (.0001,.0003,.001,.003,.01,.03,.15):
            vals=[path(problem(rho,s),rho,lr)[0] for s in (70,71)]
            cal.append(dict(rho=rho,lr=lr,mean_final=float(np.mean([v['final'] for v in vals])),increases=sum(v['increases'] for v in vals)))
        eligible=[r for r in cal if r['rho']==rho and r['increases']==0];assert eligible
        lr=min(eligible,key=lambda r:r['mean_final'])['lr']
        for seed in (72,73,74):
            data=problem(rho,seed);S,u,y,*_=data;t=np.array([.95*rho,1.,0.]);H=np.empty((3,3))
            for j in range(3):
                delta=np.eye(3)[j]*1e-6;H[:,j]=(state(S,u,y,t+delta)[-1]-state(S,u,y,t-delta)[-1])/2e-6
            lam=float(np.linalg.eigvalsh((H+H.T)/2)[-1]);curv.append(dict(rho=rho,seed=seed,lambda_max=lam,local_quadratic_limit=2/lam))
            met,seq=path(data,rho,lr,True);bad,_=path(data,rho,.03);paths.append(dict(rho=rho,seed=seed,lr=lr,**met,old_lr03=bad))
            for budget in (4,8):
                for m in ('AR','CAT','CAT-tuned','CAT-current','Cheb-restart','Cheb-current','Cheb-continuous','SafeCAT','GMRES','CG'):
                    rows.append(dict(rho=rho,seed=seed,budget=budget,**replay(seq,rho,budget,m)))
    result=dict(status='A12_COMMON_PATH_AUDIT',calibration=cal,paths=paths,curvature=curv,rows=rows)
    Path('a12_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('paths',len(paths),'runs',len(rows));print('lr',[(r['rho'],r['lr']) for r in paths[::3]]);print('curvature limits',[round(r['local_quadratic_limit'],5) for r in curv]);print('median stable/old train loss',np.median([r['final'] for r in paths]),np.median([r['old_lr03']['final'] for r in paths]));print('increases stable/old',sum(r['increases'] for r in paths),sum(r['old_lr03']['increases'] for r in paths))
    for b in (4,8):
        for m in dict.fromkeys(r['method'] for r in rows):
            print(b,m,np.median([r['gradient_nrmse'] for r in rows if r['budget']==b and r['method']==m]))
if __name__=='__main__':main()
