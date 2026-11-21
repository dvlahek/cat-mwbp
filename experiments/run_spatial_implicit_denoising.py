#!/usr/bin/env python3
"""Controlled spatial equilibrium denoising with real held-out images.

Predeclared image-disjoint train/validation/test split of local scikit-image data.
An implicit spatial recurrent layer h=tanh(U(noisy features)+rho*S(h)) has
||J_h||_2 <= rho < 1, for a symmetric, doubly stochastic 5-point stencil S.
Training uses CAT implicit adjoint. All four adjoint solvers run at the SAME
frozen checkpoint and RHS, on held-out images; the inference prediction does
not depend on which adjoint solver is used.

Scope: this is a deliberately controlled spatial equilibrium denoiser, not a
reproduction of a published full image restoration DEQ architecture. CPU-only.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import scipy
import skimage
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage import data, color
from skimage.transform import resize
from skimage.restoration import denoise_bilateral
from run_ignn_ogbn_arxiv import stationary_budget, restarted_gmres_budget, reference_solve

TRAIN_NAMES=('camera','coins','moon','page','text','brick','grass','gravel')
VALID_NAMES=('clock','chelsea')
TEST_NAMES=('retina','immunohistochemistry','hubble_deep_field','cell')
GAUSSIAN_SIGMAS=(0.7,1.2,1.8)
FEATURE_COUNT=6
OUTPUT_SCALE=.35


def load_image(name,side):
    raw=np.asarray(getattr(data,name)())
    if raw.ndim==3:
        raw=color.rgb2gray(raw[:,:,:3])
    raw=raw.astype(np.float32)
    if raw.max()>1.0: raw/=255.
    im=resize(raw,(side,side),order=1,anti_aliasing=True,preserve_range=True).astype(np.float32)
    return np.clip(im,0,1)


def crop_patches(name,side,patch,rng):
    im=load_image(name,side)
    if patch>side:raise ValueError('Patch larger than resized image')
    yy=int(rng.integers(0,side-patch+1));xx=int(rng.integers(0,side-patch+1))
    crop=im[yy:yy+patch,xx:xx+patch]
    if rng.random()<.5:crop=np.flip(crop,axis=1)
    if rng.random()<.5:crop=np.flip(crop,axis=0)
    if rng.random()<.5:crop=np.rot90(crop,k=int(rng.integers(1,4)))
    return np.ascontiguousarray(crop)


def prepare(clean,sigma,rng,dtype=torch.float32):
    noisy=np.clip(clean+rng.normal(0,sigma,clean.shape),0,1).astype(np.float32)
    g07=gaussian_filter(noisy,.7,mode='reflect')
    g12=gaussian_filter(noisy,1.2,mode='reflect')
    g18=gaussian_filter(noisy,1.8,mode='reflect')
    residual=noisy-g12
    localvar=np.sqrt(gaussian_filter(residual*residual,1.2,mode='reflect')+1e-8)
    feat=np.stack((noisy,g07,g12,g18,residual,localvar),axis=0).copy()
    return (torch.from_numpy(feat[None]).to(dtype),
            torch.from_numpy(g07[None,None].copy()).to(dtype),
            torch.from_numpy(clean[None,None].copy()).to(dtype),
            noisy)


def stencil(x,center_weight):
    k=(1-center_weight)/4
    return center_weight*x+k*(torch.roll(x,1,-1)+torch.roll(x,-1,-1)+
                              torch.roll(x,1,-2)+torch.roll(x,-1,-2))


class SpatialEquilibrium(nn.Module):
    def __init__(self,width,rho,center):
        super().__init__()
        self.width=width;self.rho=rho;self.center=center
        self.inproj=nn.Sequential(
            nn.Conv2d(FEATURE_COUNT,width,3,padding=1,padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(width,width,3,padding=1,padding_mode="circular"))
        self.readout=nn.Sequential(
            nn.Conv2d(width,width,3,padding=1,padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(width,1,3,padding=1,padding_mode="circular"))
        nn.init.normal_(self.inproj[2].weight,0,.015)
        nn.init.zeros_(self.inproj[2].bias)
        nn.init.normal_(self.readout[-1].weight,0,.015)
        nn.init.zeros_(self.readout[-1].bias)

    def fix(self,feat,max_steps,tol):
        with torch.no_grad():
            drive=self.inproj(feat)
            x=torch.zeros_like(drive)
            for k in range(1,max_steps+1):
                new=torch.tanh(drive+self.rho*stencil(x,self.center))
                delta=float(torch.linalg.vector_norm(new-x)/(torch.linalg.vector_norm(new)+1e-30))
                x=new
                if delta<tol:return x,k,delta
        raise RuntimeError(f'Forward did not converge within {max_steps} iterations. Last delta={delta:.3g}')

    def prediction(self,h,base):
        return base+OUTPUT_SCALE*torch.tanh(self.readout(h))


def cat_implicit_adjoint(b,d,rho,center,tol,max_steps):
    mu,L=1-rho,1+rho
    alpha=4/(math.sqrt(mu)+math.sqrt(L))**2
    beta=((math.sqrt(L)-math.sqrt(mu))/(math.sqrt(L)+math.sqrt(mu)))**2
    x=torch.zeros_like(b);old=torch.zeros_like(b)
    bn=float(torch.linalg.vector_norm(b))+1e-30
    for it in range(max_steps+1):
        r=b-(x-rho*stencil(d*x,center))
        if float(torch.linalg.vector_norm(r))/bn<tol:
            return x,it
        if it==max_steps:break
        nxt=x+alpha*r+beta*(x-old)
        old,x=x,nxt
    raise RuntimeError(f'CAT training backward did not reach relative residual {tol}')


def train_step(model,feat,base,clean,args,optim):
    optim.zero_grad(set_to_none=True)
    h,_,_=model.fix(feat,args.max_forward,args.forward_tol)
    d=1-h*h
    z=h.detach()
    f=torch.tanh(model.inproj(feat)+model.rho*stencil(z,model.center))
    # Return the implicit adjoint in place of the ordinary incoming gradient
    # at the equilibrium correction output, not through unrolled iterations.
    def implicit_backward(b):
        lam,_=cat_implicit_adjoint(b.detach(),d.detach(),model.rho,model.center,
                                    args.backward_tol,args.max_backward)
        return lam
    f.register_hook(implicit_backward)
    pred=model.prediction(f,base)
    loss=F.mse_loss(pred,clean)
    loss.backward()
    optim.step()
    return float(loss.detach())


def psnr(pred,clean):
    a=np.asarray(pred);b=np.asarray(clean)
    mse=float(np.mean((a-b)**2))
    return 10*math.log10(1/(mse+1e-15))


def eval_all(model,images,args,seed,typ):
    rng=np.random.default_rng(100_000+seed+(0 if typ=='valid' else 1000))
    rows=[];vis=[]
    model.eval()
    with torch.no_grad():
        for name in images:
            clean=load_image(name,args.eval_side)
            feat,base,target,noisy=prepare(clean,args.sigma,rng)
            h,nf,_=model.fix(feat,args.max_forward,args.forward_tol)
            pred=model.prediction(h,base).clamp(0,1).numpy().reshape(clean.shape)
            target_np=target.numpy().reshape(clean.shape)
            bclean=base.numpy().reshape(clean.shape)
            methods={'noisy':noisy,'gaussian_0.7':feat.numpy()[0,1],
                     'gaussian_1.2':feat.numpy()[0,2],
                     'gaussian_1.8':feat.numpy()[0,3],
                     'bilateral':denoise_bilateral(noisy,sigma_color=args.sigma,
                                           sigma_spatial=2,channel_axis=None),
                     'spatial_equilibrium':pred}
            for method,im in methods.items():
                rows.append(dict(seed=seed,split=typ,image=name,method=method,
                                 psnr_db=psnr(im,clean),forward_steps=nf if method=='spatial_equilibrium' else 0))
            vis.append((name,clean,noisy,pred,bclean))
    return rows,vis


def split_gaussian_choice(valid_rows):
    df=pd.DataFrame(valid_rows)
    sm=df[df.method.str.startswith('gaussian_')].groupby('method').psnr_db.mean()
    return str(sm.idxmax())


def save_reconstruction(vis,out):
    """Simple deterministic PNG contact sheet, avoids GUI/font backends."""
    from PIL import Image, ImageDraw
    scale=3
    hrow=vis[0][1].shape[0]*scale+34
    wcol=vis[0][1].shape[1]*scale
    sheet=Image.new('RGB',(4*wcol,len(vis)*hrow),'white')
    draw=ImageDraw.Draw(sheet)
    for i,(name,clean,noisy,denoised,gauss) in enumerate(vis):
        for j,(im,label) in enumerate(zip((clean,noisy,gauss,denoised),
                               (name+' | clean','Noisy','Gaussian 0.7','Spatial implicit'))):
            arr=np.uint8(np.round(np.clip(im,0,1)*255))
            patch=Image.fromarray(arr,mode='L').convert('RGB')
            patch=patch.resize((wcol,hrow-34),resample=Image.Resampling.NEAREST)
            sheet.paste(patch,(j*wcol,i*hrow+30))
            draw.text((j*wcol+4,i*hrow+7),label,fill='black')
    sheet.save(out/'reconstructions.png')


def solver_audit(args,model,name,seed,out):
    rng=np.random.default_rng(500_000+seed)
    clean=load_image(name,args.audit_side)
    feat,base,target,_=prepare(clean,args.sigma,rng,torch.float64)
    model=model.double().eval()
    with torch.no_grad():
        h,nf,_=model.fix(feat,args.max_forward,args.reference_tol*.2)
        d=1-h*h
        readout=model.readout(h)
        outpred=base+OUTPUT_SCALE*torch.tanh(readout)
        # Independent autograd derivative of frozen decoder loss wrt h.
        # Does not backpropagate through the equilibrium solver.
        with torch.enable_grad():
            hleaf=h.detach().requires_grad_(True)
            audit_pred=model.prediction(hleaf,base)
            audit_loss=F.mse_loss(audit_pred,target)
            b=torch.autograd.grad(audit_loss,hleaf)[0].detach()
        def jtv(v):return model.rho*stencil(d*v,model.center)
        ref,ri,refbound=reference_solve(b,jtv,model.rho,args.reference_tol,args.max_reference)
        weighted_ref=d*ref
        nref=float(torch.linalg.vector_norm(weighted_ref))
        rows=[];budgets=sorted(set(int(s) for s in args.budgets.split(',')))
        passed=set()
        for budget in budgets:
            for method in ('AR','CAT','Chebyshev','GMRES'):
                if method in passed:continue
                # Every run starts from zero and receives the SAME frozen b.
                times=[]
                for rep in range(args.repeats):
                    t0=time.perf_counter()
                    if method=='GMRES':
                        x,used=restarted_gmres_budget(b,jtv,budget,args.gmres_restart)
                    else:
                        x,used=stationary_budget(b,jtv,model.rho,method,budget)
                    times.append(time.perf_counter()-t0)
                st=time.perf_counter()
                r=b-(x-jtv(x))
                nr=float(torch.linalg.vector_norm(r))
                E=nr/(1-model.rho)
                nx=float(torch.linalg.vector_norm(x))
                bound=E/(nx-E) if nx>E else math.inf
                chk=time.perf_counter()-st
                measured=float(torch.linalg.vector_norm(d*(x-ref)))/(nref+1e-30)
                N=b.numel()
                estimated_bytes=(3*N*8 if method in ('CAT','Chebyshev') else (2*N*8 if method=='AR' else (args.gmres_restart+4)*N*8))
                rows.append(dict(seed=seed,audit_image=name,method=method,budget=budget,
                          solver_jtv=used,check_jtv=1,charged_jtv=used+1,
                          estimated_solver_state_bytes=estimated_bytes,
                          solver_time_median_s=float(np.median(times)),
                          solver_time_min_s=float(min(times)),
                          solver_time_max_s=float(max(times)),
                          terminal_check_s=chk,
                          charged_time_s=float(np.median(times))+chk,
                          certified_relative_adjoint_bound=bound,
                          certified_pass=int(bound<args.certified_tol),
                          weighted_adjoint_error_vs_reference=measured,
                          reference_bound=refbound,reference_steps=ri,
                          forward_steps=nf))
                if bound<args.certified_tol:passed.add(method)
                print('AUDIT',seed,name,method,budget,'JTv',used+1,
                      'cert',f'{bound:.2e}','time_ms',f'{1000*(np.median(times)+chk):.2f}',flush=True)
            if len(passed)==4:break
        frame=pd.DataFrame(rows)
        frame.to_csv(out/'solver_budgets.csv',index=False)
        first=(frame[frame.certified_pass==1].sort_values(['charged_jtv','method'])
               .groupby('method',as_index=False).first())
        first.to_csv(out/'first_certified.csv',index=False)
        return dict(audit_image=name,forward_steps=nf,
                    reference_steps=ri,reference_relative_bound=refbound,
                    first_pass_methods=first.method.tolist(),
                    all_methods_certified=len(first)==4)


def gradient_selftest():
    """Independent finite difference of implicit derivative wrt inproj.weight."""
    torch.manual_seed(413)
    model=SpatialEquilibrium(4,.60,.35).double()
    with torch.no_grad():model.readout[-1].weight.normal_(0,.03)
    generator=np.random.default_rng(118)
    clean=np.clip(generator.uniform(.05,.95,(8,8)),0,1).astype(np.float32)
    feat,base,target,_=prepare(clean,.12,generator,torch.float64)
    h,_,_=model.fix(feat,500,1e-13)
    d=1-h*h
    f=torch.tanh(model.inproj(feat)+model.rho*stencil(h.detach(),model.center))
    f.register_hook(lambda b: cat_implicit_adjoint(b.detach(),d.detach(),model.rho,model.center,1e-13,1000)[0])
    loss=F.mse_loss(model.prediction(f,base),target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad=model.inproj[0].weight.grad.detach().clone()
    idx=np.unravel_index(int(grad.abs().argmax()),grad.shape)
    value=float(grad[idx]); eps=2e-5
    with torch.no_grad():
        orig=float(model.inproj[0].weight[idx])
        model.inproj[0].weight[idx]=orig+eps
        hp,_,_=model.fix(feat,500,1e-13)
        lp=float(F.mse_loss(model.prediction(hp,base),target))
        model.inproj[0].weight[idx]=orig-eps
        hm,_,_=model.fix(feat,500,1e-13)
        lm=float(F.mse_loss(model.prediction(hm,base),target))
        model.inproj[0].weight[idx]=orig
    fd=(lp-lm)/(2*eps)
    err=abs(value-fd)/(abs(fd)+1e-12)
    if not err<1e-5:raise RuntimeError(f'Implicit derivative verification failed: grad={value}, finite_difference={fd}, rel={err}')
    print('SPATIAL_IMPLICIT_GRADIENT_SELFTEST_PASS',f'{err:.3e}',flush=True)


def run(args):
    if args.smoke:
        args.steps=6;args.side=64;args.patch=20;args.eval_side=24
        args.audit_side=20;args.width=6;args.budgets='5,10,15'
        args.train_names=TRAIN_NAMES[:2]
        args.valid_names=VALID_NAMES[:1]
        args.test_names=TEST_NAMES[:1]
    else:
        args.train_names=TRAIN_NAMES
        args.valid_names=VALID_NAMES
        args.test_names=TEST_NAMES
    assert not (set(args.train_names)&set(args.test_names) or set(args.valid_names)&set(args.test_names) or set(args.train_names)&set(args.valid_names))
    if not 0<args.rho<1 or not 0<args.center_weight<1:raise ValueError('Require rho, center_weight in (0,1)')
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    gradient_selftest()
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    model=SpatialEquilibrium(args.width,args.rho,args.center_weight)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    history=[]
    bestval=-math.inf; beststate=None; beststep=None
    for step in range(1,args.steps+1):
        name=args.train_names[int(rng.integers(0,len(args.train_names)))]
        clean=crop_patches(name,args.side,args.patch,rng)
        feat,base,target,_=prepare(clean,args.sigma,rng)
        model.train()
        loss=train_step(model,feat,base,target,args,opt)
        if step==1 or step%args.eval_every==0 or step==args.steps:
            vrows,_=eval_all(model,args.valid_names,args,args.seed,'valid')
            v=pd.DataFrame(vrows)
            score=float(v[v.method=='spatial_equilibrium'].psnr_db.mean())
            history.append(dict(step=step,loss=loss,validation_psnr=score))
            if score>bestval:
                bestval=score;beststep=step
                beststate=copy.deepcopy(model.state_dict())
            print('TRAIN',args.seed,step,'loss',f'{loss:.4g}','valPSNR',f'{score:.3f}',flush=True)
    pd.DataFrame(history).to_csv(out/'train_history.csv',index=False)
    model.load_state_dict(beststate)
    validrows,_=eval_all(model,args.valid_names,args,args.seed,'valid')
    chosen_gaussian=split_gaussian_choice(validrows)
    testrows,vis=eval_all(model,args.test_names,args,args.seed,'test')
    pd.DataFrame(validrows+testrows).to_csv(out/'image_metrics.csv',index=False)
    test=pd.DataFrame(testrows)
    med=test.groupby('method').psnr_db.mean().to_dict()
    holdout_effect=float(med['spatial_equilibrium']-med[chosen_gaussian])
    model.eval()
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(dict(state_dict=state,width=args.width,rho=args.rho,center_weight=args.center_weight,
                    seed=args.seed,best_step=beststep),out/'trained_model.pt')
    print('CHECKPOINT_SAVED',args.seed,flush=True)
    save_reconstruction(vis,out)
    print('RECONSTRUCTION_SAVED',args.seed,flush=True)
    audit=solver_audit(args,model,args.test_names[0],args.seed,out)
    provenance=dict(dataset='skimage.data built-in real photographs/textures',
        model='controlled contractive symmetric five-point spatial equilibrium denoiser',
        original_architecture_claim=False,downloaded_external_data=False,
        train_images=args.train_names,validation_images=args.valid_names,test_images=args.test_names,
        split_by_source_image=True,noise_sigma=args.sigma,
        validation_selected_gaussian=chosen_gaussian,
        best_validation_psnr=bestval,best_step=beststep,
        test_psnr_mean_by_method=med,
        model_minus_validation_selected_gaussian_db=holdout_effect,
        model_surpasses_validation_selected_gaussian=holdout_effect>0,
        test_best_gaussian_diagnostic=max((float(med['gaussian_0.7']),float(med['gaussian_1.2']),float(med['gaussian_1.8']))),
        model_minus_test_best_gaussian_diagnostic=float(med['spatial_equilibrium']-max(float(med['gaussian_0.7']),float(med['gaussian_1.2']),float(med['gaussian_1.8']))),
        split_scope='Pilot development used separate astronaut/coffee/rocket images. Final test names here were never used during pilot tuning.',
        memory_note='State bytes are algorithmic estimates, not observed process peak RAM.',
        stopping_note='Training and terminal certificate compute global norms, so no fully autonomous local stopping claim.',
        solver_audit=audit,model_params=sum(p.numel() for p in model.parameters()),
        learned_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        device='cpu',torch_version=torch.__version__,scipy_version=scipy.__version__,
        scikit_image_version=skimage.__version__,numpy_version=np.__version__,platform=platform.platform(),
        checkpoint_sha256=hashlib.sha256((out/'trained_model.pt').read_bytes()).hexdigest(),
        config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()})
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print('DONE',str(out.resolve()),'test_model',f"{med['spatial_equilibrium']:.3f}",
          'selected_gaussian',f'{med[chosen_gaussian]:.3f}','model_minus_gaussian',f'{holdout_effect:+.3f}',flush=True)


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/spatial_denoising_s0'))
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--side',type=int,default=80,help='Train source image resolution prior to random crop')
    p.add_argument('--patch',type=int,default=48)
    p.add_argument('--eval-side',type=int,default=64)
    p.add_argument('--audit-side',type=int,default=48)
    p.add_argument('--width',type=int,default=24)
    p.add_argument('--rho',type=float,default=.95)
    p.add_argument('--center-weight',type=float,default=.35)
    p.add_argument('--sigma',type=float,default=.12)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--eval-every',type=int,default=25)
    p.add_argument('--max-forward',type=int,default=1000)
    p.add_argument('--forward-tol',type=float,default=2e-5)
    p.add_argument('--max-backward',type=int,default=500)
    p.add_argument('--backward-tol',type=float,default=1e-5)
    p.add_argument('--max-reference',type=int,default=2000)
    p.add_argument('--reference-tol',type=float,default=2e-9)
    p.add_argument('--certified-tol',type=float,default=1e-5)
    p.add_argument('--gmres-restart',type=int,default=15)
    p.add_argument('--budgets',default='10,15,20,25,30,35,40,45,50,60,80,100,120,160,200')
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if a.patch>a.side or a.steps<1 or a.width<1 or a.eval_every<1 or a.threads<1 or a.repeats<1:
        p.error('Invalid dimensions/iterations')
    return a


if __name__=='__main__':
    run(parse())