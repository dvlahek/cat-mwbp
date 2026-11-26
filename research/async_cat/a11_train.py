#!/usr/bin/env python3
"""Equal-action implicit training pilot; exact reference is offline."""
import argparse,json
from pathlib import Path
import numpy as np
METHODS=('AR','CAT','CAT-tuned','Cheb-restart','Cheb-tuned','Cheb-continuous','GMRES','MR','SafeCAT','Exact')
# A10 tanh/smooth settings frozen before this training pilot.
TUNED={.95:{4:(1.,.25),8:(1.,1.)},.99:{4:(1.,0.),8:(.75,.25)},.999:{4:(.75,0.),8:(.75,0.)}}

def forward(S,u,theta,h=None):
    c,g,bias=theta
    if h is None:h=np.zeros_like(u)
    for _ in range(60000):
        z=np.tanh(c*(h@S)+g*u+bias)
        if np.max(abs(z-h))<1e-12:return z
        h=z
    raise RuntimeError('forward failed')

def problem(rho,seed,n=24):
    rng=np.random.default_rng(seed);S=np.zeros((n,n));i=np.arange(n);S[i,(i-1)%n]=S[i,(i+1)%n]=.5
    u=rng.normal(0,.055,(8,n));y=forward(S,u,[.98*rho,1.3,.006]);y+=rng.normal(0,.003,y.shape)
    return S,u[:4],y[:4],u[4:],y[4:]

def solve_step(A,b,x,v,budget,method,rho,setting,count,a):
    q=(np.sqrt(1+rho)-np.sqrt(1-rho))/(np.sqrt(1+rho)+np.sqrt(1-rho)); beta=q*q
    shape=b.shape;b=b.ravel();x=x.ravel();v=v.ravel();actions=0
    def apply(y):
        nonlocal actions
        actions+=1;return np.einsum('bij,bj->bi',A,y.reshape(shape)).ravel()
    if method=='Exact':return np.linalg.solve(A,b.reshape(shape)[...,None])[...,0],np.zeros(shape),count,a,0,0.
    if method=='GMRES':
        r=b-apply(x);norm=np.linalg.norm(r);base=x.copy()
        if norm>1e-30:
            V=[r/norm];W=[]
            for _ in range(budget-1):
                w=apply(V[-1]);W.append(w);z=w.copy()
                for _ in range(2):
                    for h in V:z-=np.dot(h,z)*h
                nz=np.linalg.norm(z)
                if nz<1e-13:break
                V.append(z/nz)
            if W:x=base+np.stack(V[:len(W)],axis=1)@np.linalg.lstsq(np.stack(W,axis=1),r,rcond=None)[0]
        return x.reshape(shape),(x-base).reshape(shape),count,a,actions,0.
    if method in ('MR','SafeCAT'):
        r=b-apply(x);violation=0.
        for _ in range(budget-1):
            p=r.copy() if method=='MR' else (1+beta)*r+beta*v
            w=apply(p);den=w@w;lam=float((w@r)/den) if den>1e-30 else 0.
            step=lam*p;newr=r-lam*w
            violation=max(violation,float(np.linalg.norm(newr)-np.linalg.norm(r)))
            assert np.linalg.norm(newr)<=min(np.linalg.norm(r),np.linalg.norm(r-w))+1e-12
            x=x+step;v=step;r=newr
        return x.reshape(shape),v.reshape(shape),count,a,actions,violation
    if method in ('Cheb-restart','Cheb-tuned'):count=0;a=1.;v=np.zeros_like(v)
    for _ in range(budget):
        r=b-apply(x);count+=1
        if method=='AR':aa,m=1.,0.
        elif method.startswith('CAT'):m=beta*(setting[0] if method=='CAT-tuned' else 1.);aa=1+m
        elif count==1:aa,m=1.,0.
        else:
            f=rho*rho/(2 if count==2 else 4);aa=1/(1-f*a);m=f*a*aa
        a=aa;v=aa*r+m*v;x=x+v
    return x.reshape(shape),v.reshape(shape),count,a,actions,0.

def run(data,rho,lr,budget,method,steps):
    S,u,y,uv,yv=data;theta=np.array([.95*rho,1.,0.]);x=np.zeros_like(u);v=x.copy();last=x.copy();older=x.copy();h=None;count=0;a=1.;cost=0;num=den=0.;max_violation=0.;losses=[];setting=TUNED[rho][budget]
    for t in range(steps):
        h=forward(S,u,theta,h);d=1-h*h;A=np.eye(u.shape[1])[None,:,:]-theta[0]*S[None,:,:]*d[:,None,:];b=(h-y)/h.size
        G=d[:,None,:]*np.stack([h@S,u,np.ones_like(u)],axis=1);exact=np.linalg.solve(A,b[...,None])[...,0];ge=np.einsum('bpn,bn->p',G,exact)
        if method=='Cheb-tuned' and t>=2:x=last+setting[1]*(last-older)
        x,v,count,a,actions,violation=solve_step(A,b,x,v,budget,method,rho,setting,count,a);cost+=actions;max_violation=max(max_violation,violation)
        older,last=last,x.copy();g=np.einsum('bpn,bn->p',G,x);num+=float(np.sum((g-ge)**2));den+=float(np.sum(ge**2));losses.append(float(np.mean((h-y)**2)/2))
        theta=np.clip(theta-lr*g,[0.,.1,-.1],[rho,2.,.1]);assert np.all(np.isfinite(theta))
    ht=forward(S,u,theta);hv=forward(S,uv,theta)
    return dict(method=method,actions=cost,gradient_nrmse=float(np.sqrt(num/max(den,1e-300))),train_loss=float(np.mean((ht-y)**2)/2),validation_loss=float(np.mean((hv-yv)**2)/2),loss_auc=float(np.mean(losses)),theta=theta.tolist(),max_guard_violation=max_violation)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('a11_results.json'));ap.add_argument('--steps',type=int,default=60);args=ap.parse_args();rows=[]
    for rho in (.95,.99,.999):
        for seed in (60,61,62):
            data=problem(rho,seed)
            for lr in (.03,.15):
                for budget in (4,8):
                    for m in METHODS:rows.append(dict(rho=rho,seed=seed,lr=lr,budget=budget,**run(data,rho,lr,budget,m,args.steps)))
    summary=[]
    for budget in (4,8):
        pairs={}
        for r in rows:
            if r['budget']==budget:pairs.setdefault((r['rho'],r['seed'],r['lr']),{})[r['method']]=r
        for m in METHODS:
            rr=[p[m] for p in pairs.values()];ref=[p['GMRES'] for p in pairs.values()]
            summary.append(dict(budget=budget,method=m,n=len(rr),median_gradient_nrmse=float(np.median([r['gradient_nrmse'] for r in rr])),median_validation_loss=float(np.median([r['validation_loss'] for r in rr])),gradient_wins_vs_gmres=sum(r['gradient_nrmse']<s['gradient_nrmse'] for r,s in zip(rr,ref)),validation_wins_vs_gmres=sum(r['validation_loss']<s['validation_loss'] for r,s in zip(rr,ref))))
    data=dict(status='A11_TRAINING_PILOT_EXECUTED',steps=args.steps,rows=rows,summary=summary);args.out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');print('runs',len(rows));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
