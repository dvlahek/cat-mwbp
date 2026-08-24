#!/usr/bin/env python3
"""Implicit CAT classification benchmark on the manuscript datasets.

The recurrent state is a fixed sparse local ring. Each hidden unit communicates
with four neighbours at offsets +/-1 and +/-2. AR and CAT use the same graph,
initialization, minibatch order, spectral envelope, and stopping tolerance.
Reported solver counts are actual sparse local J^T v calls.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from metric_wave.data import load_dataset_three_way
from metric_wave.data_vision import raw_vision_dataset

CORE=("moons","circles","iris","wine","breast_cancer","anisotropic","digits","synthetic_large")
VISION=("mnist","cifar10")
METHODS=("AR-bound","CAT-bound")


def config(profile, group):
    base=CORE if group=="core" else VISION if group=="vision" else CORE+VISION
    if profile=="quick":
        ds=tuple(d for d in ("moons","digits","mnist") if d in base)
        return ds,(0,),8,5,32,40,48,2500,3
    if profile=="standard":
        return base,(0,1,2),35,20,48,64,64,8000,7
    return base,tuple(range(5)),60,40,64,96,96,8000,10


def split_vision(x,y,seed):
    xd,xt,yd,yt=train_test_split(x,y,test_size=.25,random_state=seed,stratify=y)
    xr,xv,yr,yv=train_test_split(xd,yd,test_size=.20,random_state=seed+104729,stratify=yd)
    sc=StandardScaler().fit(xr)
    return sc.transform(xr),sc.transform(xv),sc.transform(xt),yr.astype(int),yv.astype(int),yt.astype(int)


def load(name,seed,subsample):
    if name in VISION:
        return split_vision(*raw_vision_dataset(name,seed=seed,subsample=subsample),seed)
    return load_dataset_three_way(name,seed=seed)


def recur(v,rho):
    return .30*rho*(np.roll(v,1,axis=-1)+np.roll(v,-1,axis=-1))+.20*rho*(np.roll(v,2,axis=-1)+np.roll(v,-2,axis=-1))


def recur_dense(h,rho): return recur(np.eye(h),rho)

def recur_eigs(h,rho):
    th=2*np.pi*np.arange(h)/h
    return .60*rho*np.cos(th)+.40*rho*np.cos(2*th)


def coeff(mu,L):
    sm,sL=np.sqrt(mu),np.sqrt(L)
    return 2/(L+mu),4/(sL+sm)**2,((sL-sm)/(sL+sm))**2


def softmax(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)


class Model:
    def __init__(self,nin,hidden,nout,rho,seed):
        rng=np.random.default_rng(seed); self.rho=rho; self.hidden=hidden
        self.Win=.10/np.sqrt(max(1,nin))*rng.normal(size=(nin,hidden)); self.bh=np.zeros(hidden)
        lim=np.sqrt(6/(hidden+nout)); self.Wout=rng.uniform(-lim,lim,size=(hidden,nout)); self.bo=np.zeros(nout)
        ev=recur_eigs(hidden,rho); self.mu=1-float(ev.max()); self.L=1-float(ev.min())
    def blocks(self): return [x.copy() for x in (self.Win,self.bh,self.Wout,self.bo)]
    def restore(self,b): self.Win[:]=b[0]; self.bh[:]=b[1]; self.Wout[:]=b[2]; self.bo[:]=b[3]
    def equilibrium(self,x,tol=1e-8,max_steps=1500):
        drive=x@self.Win+self.bh; h=np.zeros((len(x),self.hidden))
        for k in range(1,max_steps+1):
            new=np.tanh(drive+recur(h,self.rho))
            if np.linalg.norm(new-h)/(np.linalg.norm(new)+1e-30)<=tol: return new,k
            h=new
        return h,max_steps
    def evaluate(self,x,y):
        h,k=self.equilibrium(x); z=h@self.Wout+self.bo; p=softmax(z)
        return -np.log(p[np.arange(len(y)),y]+1e-12).mean(), float((z.argmax(1)==y).mean()), k


class Momentum:
    def __init__(self,m,beta=.9): self.beta=beta; self.v=[np.zeros_like(x) for x in m.blocks()]
    def step(self,m,g,lr,wd=1e-4):
        out=[]
        for i,(p,grad) in enumerate(zip(m.blocks(),g)):
            total=grad+(wd*p if p.ndim>1 else 0)
            self.v[i]=self.beta*self.v[i]+(1-self.beta)*total
            out.append(p-lr*self.v[i])
        m.restore(out)


def solve(rhs,d,m,method,tol=1e-5,max_steps=4000):
    ar,ca,cb=coeff(m.mu,m.L); alpha,beta=(ar,0) if method=="AR-bound" else (ca,cb)
    lam=np.zeros_like(rhs); prev=np.zeros_like(rhs); norm=np.linalg.norm(rhs)+1e-30
    updates=0
    for actions in range(1,max_steps+1):
        jtv=recur(lam*d,m.rho); r=rhs-(lam-jtv); rel=np.linalg.norm(r)/norm
        if rel<=tol: return lam,updates,actions,rel
        new=lam+alpha*r+beta*(lam-prev); prev,lam=lam,new; updates+=1
    return lam,updates,max_steps,rel


def loss_grad(m,x,y,method):
    h,fs=m.equilibrium(x); z=h@m.Wout+m.bo; p=softmax(z); n=len(y)
    loss=-np.log(p[np.arange(n),y]+1e-12).mean(); dz=p.copy(); dz[np.arange(n),y]-=1; dz/=n
    rhs=dz@m.Wout.T; d=1-h*h; t=time.perf_counter(); lam,upd,acts,res=solve(rhs,d,m,method); secs=time.perf_counter()-t
    force=lam*d
    return loss,(x.T@force,force.sum(0),h.T@dz,dz.sum(0)),{"jtv":acts,"backward_s":secs,"forward":fs,"residual":res}


def train(name,seed,rho,method,profile,core_epochs,vision_epochs,hcore,hscaling,hvision,subsample,patience):
    xr,xv,xt,yr,yv,yt=load(name,seed,subsample); classes=int(max(yr.max(),yv.max(),yt.max())+1)
    hidden=hvision if name in VISION else hscaling if name=="synthetic_large" else hcore; epochs=vision_epochs if name in VISION else core_epochs; lr=.02 if name in VISION else .04
    m=Model(xr.shape[1],hidden,classes,rho,seed+1000); opt=Momentum(m); rng=np.random.default_rng(seed+3000)
    best=float("inf"); bestb=m.blocks(); bestep=0; stale=0; counts=[]; times=[]
    for ep in range(1,epochs+1):
        order=rng.permutation(len(xr))
        for start in range(0,len(xr),64):
            ii=order[start:start+64]; _,g,d=loss_grad(m,xr[ii],yr[ii],method); opt.step(m,g,lr); counts.append(d["jtv"]); times.append(d["backward_s"])
        vl,_,_=m.evaluate(xv,yv)
        if vl<best-1e-4: best=vl; bestb=m.blocks(); bestep=ep; stale=0
        else:
            stale+=1
            if stale>=patience: break
    m.restore(bestb); tl,ta,_=m.evaluate(xt,yt)
    row={"dataset":name,"seed":seed,"rho_cap":rho,"method":method,"best_epoch":bestep,"test_loss":tl,"test_accuracy":ta,
         "mean_backward_jtv_actions":float(np.mean(counts)),"total_backward_seconds":float(np.sum(times))}
    return row,m,xt,yt


def audit(m,x,y,count=8):
    x=x[:count]; y=y[:count]; h,_=m.equilibrium(x); z=h@m.Wout+m.bo; p=softmax(z); p[np.arange(len(y)),y]-=1
    rhs_all=p@m.Wout.T; W=recur_dense(m.hidden,m.rho); rows=[]
    for i,(rhs,hi) in enumerate(zip(rhs_all,h)):
        d=1-hi*hi; A=np.eye(m.hidden)-W@np.diag(d); exact=np.linalg.solve(A,rhs); eg=d*exact; eg_norm=np.linalg.norm(eg)+1e-30
        sd=np.sqrt(np.maximum(d,1e-15)); S=sd[:,None]*W*sd[None,:]; ev=np.linalg.eigvalsh(S); mu,L=1-ev.max(),1-ev.min(); ar,ca,cb=coeff(mu,L)
        methods={"AR-oracle":(ar,0),"CAT-oracle":(ca,cb),"CAT-bound":coeff(m.mu,m.L)[1:]}
        for method,(alpha,beta) in methods.items():
            lam=np.zeros_like(rhs); prev=np.zeros_like(rhs)
            for k in range(1,4001):
                r=rhs-(lam-recur(d*lam,m.rho)); new=lam+alpha*r+beta*(lam-prev); prev,lam=lam,new
                if np.linalg.norm(d*lam-eg)/eg_norm<=1e-6: break
            g=d*lam
            rows.append({"sample":i,"method":method,"jtv_actions":k,"rho_actual":float(max(abs(ev.min()),abs(ev.max()))),
                         "gradient_error":np.linalg.norm(g-eg)/eg_norm,
                         "gradient_cosine":float(np.dot(g,eg)/(np.linalg.norm(g)*np.linalg.norm(eg)+1e-30))})
        center,half=(L+mu)/2,(L-mu)/2; lam=np.zeros_like(rhs); prev=np.zeros_like(rhs); alpha=1/center
        for k in range(1,4001):
            r=rhs-(lam-recur(d*lam,m.rho))
            if k==1: new=lam+alpha*r
            else:
                beta=(half*alpha/2)**2; na=1/(center-beta/alpha); new=lam+na*r+beta*(lam-prev); alpha=na
            prev,lam=lam,new
            if np.linalg.norm(d*lam-eg)/eg_norm<=1e-6: break
        g=d*lam; rows.append({"sample":i,"method":"Cheb-oracle","jtv_actions":k,"rho_actual":float(max(abs(ev.min()),abs(ev.max()))),
                              "gradient_error":np.linalg.norm(g-eg)/eg_norm,
                              "gradient_cosine":float(np.dot(g,eg)/(np.linalg.norm(g)*np.linalg.norm(eg)+1e-30))})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--profile",choices=("quick","standard","full"),default="quick"); p.add_argument("--group",choices=("core","vision","all"),default="core")
    p.add_argument("--output",type=Path,default=Path("results/implicit_dataset")); p.add_argument("--rho",type=float,default=.95); args=p.parse_args()
    datasets,seeds,ce,ve,hc,hs,hv,sub,patience=config(args.profile,args.group); args.output.mkdir(parents=True,exist_ok=True)
    runrows=[]; auditrows=[]
    for name in datasets:
        for seed in seeds:
            for method in METHODS:
                row,m,xt,yt=train(name,seed,args.rho,method,args.profile,ce,ve,hc,hs,hv,sub,patience); runrows.append(row)
                a=audit(m,xt,yt); a["dataset"]=name; a["seed"]=seed; a["rho_cap"]=args.rho; a["trained_with"]=method; auditrows.extend(a.to_dict("records"))
                print(name,seed,method,row["test_accuracy"],row["mean_backward_jtv_actions"])
    runs=pd.DataFrame(runrows); aud=pd.DataFrame(auditrows); runs.to_csv(args.output/"runs.csv",index=False); aud.to_csv(args.output/"solver_audit.csv",index=False)
    pairs=[]
    for (name,seed,rho),g in runs.groupby(["dataset","seed","rho_cap"]):
        m=g.set_index("method"); ar=m.loc["AR-bound"]; cat=m.loc["CAT-bound"]
        aa=aud[(aud.dataset==name)&(aud.seed==seed)&(aud.rho_cap==rho)&(aud.trained_with=="CAT-bound")].pivot(index="sample",columns="method",values="jtv_actions")
        pairs.append({"dataset":name,"seed":seed,"rho_cap":rho,"ar_test_accuracy":ar.test_accuracy,"cat_test_accuracy":cat.test_accuracy,
                      "delta_accuracy":cat.test_accuracy-ar.test_accuracy,"training_jtv_speedup":ar.mean_backward_jtv_actions/cat.mean_backward_jtv_actions,
                      "training_time_speedup":ar.total_backward_seconds/cat.total_backward_seconds,
                      "audit_rho_actual":float(aud[(aud.dataset==name)&(aud.seed==seed)&(aud.trained_with=="CAT-bound")].rho_actual.median()),
                      "audit_cat_oracle_vs_ar_oracle":float(np.median(aa["AR-oracle"]/aa["CAT-oracle"])),
                      "audit_cat_bound_vs_ar_oracle":float(np.median(aa["AR-oracle"]/aa["CAT-bound"])),
                      "audit_cat_oracle_vs_cheb":float(np.median(aa["Cheb-oracle"]/aa["CAT-oracle"]))})
    pairs=pd.DataFrame(pairs); pairs.to_csv(args.output/"paired_training.csv",index=False)
    summary=pairs.groupby(["dataset","rho_cap"],as_index=False).mean(numeric_only=True); summary.to_csv(args.output/"dataset_summary.csv",index=False); print(summary.to_string(index=False))

if __name__=="__main__": main()
