#!/usr/bin/env python3
"""Controlled, contractive implicit graph network / adjoint solver benchmark.

Features:
  * OGB ogbn-arxiv data with official temporal split (no PyG required)
  * symmetric normalized undirected graph (self-loops, coalesced)
  * implicit layer h=tanh(X W_in + b_in + rho S h)
  * end-to-end training of W_in/b_in and readout with CAT implicit adjoint
  * freeze the SAME learned network for AR, CAT, Chebyshev and restarted GMRES
  * equal budgets of sparse J^T v actions, separate offline reference error
  * mathematically justified bound ||J^T||_2 <= rho < 1
  * synthetic --smoke test WITHOUT any data download

Not an unmodified TorchDEQ implementation or a benchmark for arbitrary DEQs.
This is a controlled graph implicit layer designed to test the local CAT regime.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def device_sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def spmm(sparse: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(sparse, x)


def make_normalized_graph(edge_index: torch.Tensor, n: int, fmt: str) -> torch.Tensor:
    """Symmetrize, eliminate parallel edges, add self-loops and D^-1/2 A D^-1/2."""
    edges = torch.as_tensor(edge_index, dtype=torch.int64, device='cpu')
    if edges.shape[0] != 2:
        raise ValueError('edge_index must have shape [2, num_edges]')
    loops = torch.arange(n, dtype=torch.int64)
    row = torch.cat((edges[0], edges[1], loops))
    col = torch.cat((edges[1], edges[0], loops))
    adj = torch.sparse_coo_tensor(torch.stack((row, col)),
                                  torch.ones(row.numel(), dtype=torch.float32),
                                  (n,n)).coalesce()
    ij = adj.indices()
    # Each unique undirected edge has weight 1, even if cited in both directions.
    deg = torch.bincount(ij[0], minlength=n).to(torch.float64)
    vals = (deg[ij[0]] * deg[ij[1]]).rsqrt()  # float64 normalization
    S = torch.sparse_coo_tensor(ij, vals, (n,n)).coalesce()
    if fmt == 'csr':
        S = S.to_sparse_csr()
    return S


def synthetic_smoke(seed: int):
    g = np.random.default_rng(seed)
    n, f, c = 350, 24, 4
    x = torch.from_numpy(g.normal(size=(n,f)).astype(np.float32))
    src = np.arange(n, dtype=np.int64)
    edges = np.stack((np.tile(src, 3),
                      np.concatenate(((src+1)%n, (src+2)%n, g.integers(0,n,n)))))
    teacher = torch.from_numpy(g.normal(size=(f,c)).astype(np.float32))
    y = (x @ teacher).argmax(dim=1)
    order = g.permutation(n)
    splits = {'train':torch.as_tensor(order[:210],dtype=torch.int64),
              'valid':torch.as_tensor(order[210:280],dtype=torch.int64),
              'test':torch.as_tensor(order[280:],dtype=torch.int64)}
    return x, y, torch.from_numpy(edges), splits, c


def real_data(root: str):
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name='ogbn-arxiv', root=root)
    graph, y = dataset[0]
    split = dataset.get_idx_split()
    x = torch.as_tensor(graph['node_feat'], dtype=torch.float32)
    labels = torch.as_tensor(y.reshape(-1), dtype=torch.long)
    edges = torch.as_tensor(graph['edge_index'], dtype=torch.long)
    indices = {key: torch.as_tensor(split[key], dtype=torch.long).reshape(-1)
               for key in ('train','valid','test')}
    return x, labels, edges, indices, int(dataset.num_classes)


def prepare_data(args, device):
    if args.smoke:
        x,y,edges,idx,nclasses = synthetic_smoke(args.seed)
    else:
        x,y,edges,idx,nclasses = real_data(args.data_root)
    n = x.shape[0]
    # No label-based or val/test-feature-based normalization fitting.
    m = x[idx['train']].mean(0, keepdim=True)
    std = x[idx['train']].std(0, keepdim=True).clamp_min(1e-6)
    x = ((x - m) / std).clamp(-10,10)
    S64 = make_normalized_graph(edges,n,args.sparse_format)
    S32 = S64.to(dtype=torch.float32,device=device)
    device_sync(device)
    return (x.to(device),y.to(device),S32,
            {k:v.to(device) for k,v in idx.items()},nclasses,
            {'num_nodes':int(n),'num_edges_undirected_with_diagonal':int(S64.values().numel()),
             'num_features':int(x.shape[1]),'num_classes':int(nclasses),
             'n_train':int(idx['train'].numel()),'n_valid':int(idx['valid'].numel()),
             'n_test':int(idx['test'].numel())}, S64)


@torch.no_grad()
def equilibrium(drive, S, rho, tol, max_steps):
    h = torch.zeros_like(drive)
    for it in range(1,max_steps+1):
        new = torch.tanh(drive + rho*spmm(S,h))
        change = torch.linalg.vector_norm(new-h)/(torch.linalg.vector_norm(new)+1e-30)
        h = new
        if float(change) <= tol:
            return h,it,float(change)
    raise RuntimeError(f'Forward equilibrium did not converge in {max_steps} steps; '
                       'increase --max-forward or lower --rho.')


def stationary_coeff(rho, method):
    mu,L = 1-rho,1+rho
    if method == 'AR':
        return 2/(mu+L),0.0
    q = (math.sqrt(L)-math.sqrt(mu))/(math.sqrt(L)+math.sqrt(mu))
    return 4/(math.sqrt(L)+math.sqrt(mu))**2,q*q


@torch.no_grad()
def train_backward_cat(rhs, h, S, rho, tol, max_steps):
    d = 1 - h*h
    alpha,beta = stationary_coeff(rho,'CAT')
    x = torch.zeros_like(rhs)
    previous = torch.zeros_like(rhs)
    scale = torch.linalg.vector_norm(rhs) + 1e-30
    for k in range(1,max_steps+1):
        residual = rhs - (x - rho*spmm(S,d*x))
        res = torch.linalg.vector_norm(residual)/scale
        if float(res) <= tol:
            return d*x,k,float(res)
        nxt = x + alpha*residual + beta*(x-previous)
        previous,x = x,nxt
    raise RuntimeError(f'CAT backward residual did not reach {tol} in {max_steps} actions.')


class ImplicitState(torch.autograd.Function):
    last_backward = {}
    @staticmethod
    def forward(ctx, drive, S, rho, forward_tol, max_forward, backward_tol, max_backward):
        h, it, delta = equilibrium(drive,S,rho,forward_tol,max_forward)
        ctx.save_for_backward(h)
        ctx.S,ctx.rho,ctx.backward_tol,ctx.max_backward = S,rho,backward_tol,max_backward
        ctx.forward_it = it
        return h

    @staticmethod
    def backward(ctx, grad_h):
        (h,) = ctx.saved_tensors
        grad_drive, actions, residual = train_backward_cat(
            grad_h,h,ctx.S,ctx.rho,ctx.backward_tol,ctx.max_backward)
        ImplicitState.last_backward = {
            'forward_iterations':ctx.forward_it,
            'backward_actions':actions,
            'backward_relative_residual':residual,
        }
        return grad_drive,None,None,None,None,None,None


class GraphImplicitClassifier(nn.Module):
    def __init__(self, fin, width, nclasses, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.input = nn.Linear(fin,width)
        self.output = nn.Linear(width,nclasses)
        nn.init.normal_(self.input.weight, std=0.35/math.sqrt(fin))
        nn.init.zeros_(self.input.bias)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def drive(self,x):
        return self.input(x)

    def forward(self,x,S,args):
        h = ImplicitState.apply(self.drive(x),S,args.rho,args.forward_tol,
                                 args.max_forward,args.backward_tol,args.max_backward)
        return self.output(h)


def accuracy(logits,y,idx):
    return float((logits[idx].argmax(1) == y[idx]).float().mean())


def train_model(args,x,y,S,indices,nclasses,device,outdir):
    model = GraphImplicitClassifier(x.shape[1],args.width,nclasses,args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_valid = -1.0
    history=[]
    for ep in range(1,args.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        device_sync(device); t0=time.perf_counter()
        logits = model(x,S,args)
        loss = F.cross_entropy(logits[indices['train']],y[indices['train']])
        loss.backward()
        optimizer.step()
        device_sync(device); elapsed=time.perf_counter()-t0
        model.eval()
        with torch.no_grad():
            h, forward_iters, _ = equilibrium(model.drive(x),S,args.rho,
                                              args.forward_tol,args.max_forward)
            predictions = model.output(h)
            valid_acc=accuracy(predictions,y,indices['valid'])
            train_acc=accuracy(predictions,y,indices['train'])
        row=dict(epoch=ep,train_loss=float(loss.detach()),train_accuracy=train_acc,
                 valid_accuracy=valid_acc,epoch_train_seconds=elapsed,
                 eval_forward_iterations=forward_iters,**ImplicitState.last_backward)
        history.append(row)
        print(f'EPOCH {ep:2} train={train_acc:.4f} valid={valid_acc:.4f} '
              f'backward_actions={row["backward_actions"]} sec={elapsed:.2f}',flush=True)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_state = copy.deepcopy(model.state_dict())
        csv_save(outdir/'training_history.csv',history)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        h, forward_iters, _=equilibrium(model.drive(x),S,args.rho,
                                       args.eval_forward_tol,args.max_forward)
        logits=model.output(h)
        scores={k:accuracy(logits,y,v) for k,v in indices.items()}
        # Gradient of classification loss w.r.t. equilibrium hidden state,
        # no autograd through the equilibrium or test labels.
        p = F.softmax(logits[indices['train']],dim=1).clone()
        row_ix=torch.arange(len(indices['train']),device=device)
        p[row_ix,y[indices['train']]] -= 1.0
        p /= len(indices['train'])
        b = torch.zeros_like(h)
        b[indices['train']] = p @ model.output.weight
    state={k:v.cpu() for k,v in model.state_dict().items()}
    torch.save({'state_dict':state,'width':args.width,'rho':args.rho,
                'num_classes':nclasses,'num_features':x.shape[1],
                'seed':args.seed},outdir/'trained_model.pt')
    return model,h,b,scores,forward_iters


def csv_save(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


@torch.no_grad()
def reference_solve(b,jtv,rho,tol,max_steps):
    x=torch.zeros_like(b)
    for it in range(1,max_steps+1):
        new = b + jtv(x)
        delta = torch.linalg.vector_norm(new-x)
        x=new
        # Fixed point residual bound: ||x - x*|| <= ||x - T(x)||/(1-rho).
        # ||x-T(x)|| = ||T(x_prev)-T(x)|| <= rho||x-x_prev||.
        absolute_bound=(rho/(1-rho))*delta
        # Rigorous bound relative to the unknown solution norm:
        # ||x*|| >= ||x|| - ||x-x*|| >= ||x|| - absolute_bound.
        lower_reference_norm=torch.linalg.vector_norm(x)-absolute_bound
        rel_bound=(absolute_bound/lower_reference_norm if float(lower_reference_norm)>0
                   else torch.tensor(float('inf'),device=x.device))
        if float(rel_bound) <= tol:
            return x,it,float(rel_bound)
    raise RuntimeError('Reference iterate failed certified relative bound. '
                       'Increase --max-reference or use lower rho.')


@torch.no_grad()
def stationary_budget(b,jtv,rho,method,budget):
    x = torch.zeros_like(b)
    old = torch.zeros_like(b)
    if method == 'Chebyshev':
        # Matches the experiment's two-vector scheduled Chebyshev control.
        mu,L=1-rho,1+rho
        center,half=(L+mu)/2,(L-mu)/2
        cheb_a=1.0/center
    else:
        alpha,beta=stationary_coeff(rho,method)
    for k in range(budget):
        r=b-(x-jtv(x))
        if method=='Chebyshev':
            if k==0:
                nxt=x+cheb_a*r
            else:
                inertia=(half*cheb_a/2)**2
                new_alpha=1/(center-inertia/cheb_a)
                nxt=x+new_alpha*r+inertia*(x-old)
                cheb_a=new_alpha
        else:
            nxt=x+alpha*r+beta*(x-old)
        old,x=x,nxt
    return x,budget


@torch.no_grad()
def restarted_gmres_budget(b,jtv,budget,restart):
    """Right-preconditioner-free restarted GMRES, counting all A applications.

    Every restart residual counts one Jt action. Inner products, basis vectors,
    and small dense least-squares solves are also performed and timed.
    """
    x=torch.zeros_like(b)
    used=0
    while used < budget:
        r=b-(x-jtv(x));used+=1
        rnorm=torch.linalg.vector_norm(r)
        if float(rnorm)==0.0 or used==budget: break
        V=[r/rnorm]
        kmax=min(restart,budget-used)
        H=torch.zeros((kmax+1,kmax),dtype=torch.float64,device='cpu')
        g=torch.zeros(kmax+1,dtype=torch.float64,device='cpu'); g[0]=float(rnorm)
        x0=x
        candidate=x0
        for j in range(kmax):
            w=V[j]-jtv(V[j]);used+=1
            for i in range(j+1):
                dot=torch.dot(V[i].reshape(-1),w.reshape(-1))
                H[i,j]=float(dot)
                w=w-dot*V[i]
            wnorm=torch.linalg.vector_norm(w)
            H[j+1,j]=float(wnorm)
            least=torch.linalg.lstsq(H[:j+2,:j+1],g[:j+2]).solution.to(device=b.device,dtype=b.dtype)
            candidate=x0+sum(V[i]*least[i] for i in range(j+1))
            if float(wnorm) < 1e-14:
                x=candidate
                return x,used
            if j+1 < kmax:
                V.append(w/wnorm)
        x=candidate
    return x,used


@torch.no_grad()
def residual_error(b,jtv,rho,x,reference,d,ref_error_bound):
    r=b-(x-jtv(x))
    residual_rel=float(torch.linalg.vector_norm(r)/(torch.linalg.vector_norm(b)+1e-30))
    E=float(torch.linalg.vector_norm(r))/(1-rho)
    nx=float(torch.linalg.vector_norm(x))
    certified_rel= E/(nx-E) if nx>E else math.inf
    err_lambda=float(torch.linalg.vector_norm(x-reference)/(torch.linalg.vector_norm(reference)+1e-30))
    err_grad=float(torch.linalg.vector_norm(d*(x-reference))/
                   (torch.linalg.vector_norm(d*reference)+1e-30))
    return residual_rel,certified_rel,err_lambda,err_grad


def experiment(args,S,h,b,device,outdir):
    # Convert graph and fixed-point derivative to float64 for finite-tolerance
    # diagnostics. Training occurs in float32; this is the frozen model audit.
    S=S.to(dtype=torch.float64)
    h=h.double(); b=b.double(); d=1-h*h
    def jtv(v): return args.rho*spmm(S,d*v)
    print('Reference adjoint: contractive fixed-point solve in float64...',flush=True)
    device_sync(device);start=time.perf_counter()
    ref,n_ref,ref_bound=reference_solve(b,jtv,args.rho,args.reference_tol,args.max_reference)
    device_sync(device)
    print(f'Reference: {n_ref} iterations, rel-error upper bound {ref_bound:.3g}, '
          f'{time.perf_counter()-start:.1f} s',flush=True)
    budgets=sorted(set(int(s) for s in args.budgets.split(',')))
    out=[]
    for budget in budgets:
        for method in ('AR','CAT','Chebyshev','GMRES'):
            if device.type == 'cuda':torch.cuda.reset_peak_memory_stats(device)
            device_sync(device);start=time.perf_counter()
            if method=='GMRES':
                x,actions=restarted_gmres_budget(b,jtv,budget,args.gmres_restart)
            else:
                x,actions=stationary_budget(b,jtv,args.rho,method,budget)
            device_sync(device);elapsed=time.perf_counter()-start
            res,cert,err_l,err_g=residual_error(b,jtv,args.rho,x,ref,d,ref_bound)
            mem=(torch.cuda.max_memory_allocated(device)/2**20) if device.type=='cuda' else math.nan
            row=dict(method=method,budget=budget,jtv_actions=actions,
                     relative_residual=res,contraction_relative_error_bound=cert,
                     relative_adjoint_error_against_reference=err_l,
                     relative_gradient_error_against_reference=err_g,
                     solver_seconds=elapsed,peak_cuda_allocated_mib=mem,
                     reference_relative_error_bound=ref_bound,
                     reference_iterations=n_ref,
                     rho_bound=args.rho,hidden_width=args.width,
                     num_nodes=h.shape[0],implicit_state_dimension=h.numel(),
                     gmres_restart=args.gmres_restart)
            out.append(row)
            csv_save(outdir/'solver_budgets.csv',out)
            print(f'BUDGET {budget:3} {method:9} JTv={actions:3} '
                  f'error={err_g:.3g} residual={res:.3g} '
                  f'cert_bound={cert:.3g} time={elapsed:.2f}s',flush=True)
    plot_budgets(out,outdir)
    return out


def plot_budgets(rows,outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d={m:sorted([r for r in rows if r['method']==m],key=lambda x:x['jtv_actions'])
       for m in ('AR','CAT','Chebyshev','GMRES')}
    fig,axs=plt.subplots(1,2,figsize=(9.0,3.5))
    for name, rr in d.items():
        axs[0].plot([r['jtv_actions'] for r in rr],
                    [max(r['relative_gradient_error_against_reference'],1e-15) for r in rr],
                    marker='o',label=name)
        axs[1].plot([r['jtv_actions'] for r in rr],
                    [r['solver_seconds'] for r in rr],marker='o',label=name)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Relative implicit-gradient error')
    axs[1].set_ylabel('Solver wall time (s)')
    for ax,letter in zip(axs,'AB'):
        ax.set_xlabel('Local $J^\\top v$ actions')
        ax.text(-.12,1.03,letter,transform=ax.transAxes,weight='bold')
        ax.legend(frameon=False)
        ax.grid(alpha=.20)
    fig.tight_layout()
    fig.savefig(outdir/'solver_comparison.pdf',bbox_inches='tight')
    fig.savefig(outdir/'solver_comparison.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--smoke',action='store_true',help='Small synthetic graph, no dataset download')
    ap.add_argument('--data-root',default='data/ogb')
    ap.add_argument('--out',type=Path,default=Path('results/ignn_ogbn_arxiv'))
    ap.add_argument('--device',default='auto',help='auto, cpu, or cuda')
    ap.add_argument('--sparse-format',choices=('csr','coo'),default='csr')
    ap.add_argument('--width',type=int,default=32)
    ap.add_argument('--rho',type=float,default=.95)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--lr',type=float,default=0.01)
    ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--forward-tol',type=float,default=2e-5)
    ap.add_argument('--eval-forward-tol',type=float,default=2e-7)
    ap.add_argument('--max-forward',type=int,default=700)
    ap.add_argument('--backward-tol',type=float,default=1e-5)
    ap.add_argument('--max-backward',type=int,default=500)
    ap.add_argument('--reference-tol',type=float,default=2e-8)
    ap.add_argument('--max-reference',type=int,default=2500)
    ap.add_argument('--budgets',default='10,20,40,80')
    ap.add_argument('--gmres-restart',type=int,default=15)
    args=ap.parse_args()
    if not (0<args.rho<1):ap.error('Require 0 < rho < 1')
    if args.width<2 or args.epochs<1 or args.gmres_restart<2:ap.error('Invalid width, epochs or restart')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.device=='auto':
        dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:dev=torch.device(args.device)
    if dev.type=='cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but torch.cuda.is_available() is false')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    if dev.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    print(f'DEVICE {dev} torch={torch.__version__} smoke={args.smoke}',flush=True)
    x,y,S,idx,nclasses,meta,S64=prepare_data(args,dev)
    print('GRAPH',meta,flush=True)
    # Symmetrized normalized graph has spectrum inside [-1,1], including self-loops.
    model,h,b,scores,forward_iters=train_model(args,x,y,S,idx,nclasses,dev,args.out)
    print('FROZEN MODEL ACCURACY',scores,'forward_iters',forward_iters,flush=True)
    rows=experiment(args,S64.to(dev),h,b,dev,args.out)
    payload={'dataset':'synthetic_smoke' if args.smoke else 'ogbn-arxiv',
             'setup':'symmetric normalized contractive implicit graph layer',
             'training':'CAT implicit adjoint, input projection and readout both learned',
             'evaluation':'frozen model, same right-hand side and full graph',
             'spectrum':'guaranteed ||J^T||_2 <= rho via symmetric normalized S and tanh derivative',
             'forward_converged':True,'reference_converged':True,
             'official_split':not args.smoke,'test_accuracy':scores['test'],
             'accuracy':scores,'graph':meta,
             'parameter_count':sum(p.numel() for p in model.parameters()),
             'implicit_state_dimension':int(h.numel()),'device':str(dev),
             'platform':platform.platform(),'torch_version':torch.__version__,
             'seed':args.seed,'config':vars(args),
             'note':'Full-graph transductive benchmark, controlled IGNN-style architecture; no arbitrary DEQ claim.'}
    payload['config']=dict(payload['config'])
    payload['config']['out']=str(args.out)
    (args.out/'run_manifest.json').write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
    print('DONE',args.out.resolve(),flush=True)


if __name__=='__main__':
    main()