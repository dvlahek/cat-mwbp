#!/usr/bin/env python3
"""Audit one terminal residual check per fixed-budget IGNN adjoint solve.

Reloads a frozen checkpoint and original OGB graph. Does not train again.
For each (solver, budget) measures solver time separately from the cost of
ONE residual evaluation (one extra J^T v and global vector norms).
This is a fixed-budget + terminal certificate diagnostic, NOT autonomous local
convergence monitoring and NOT an intermittently checked online stop run.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from run_ignn_ogbn_arxiv import (
    GraphImplicitClassifier, accuracy, csv_save, device_sync, equilibrium,
    prepare_data, reference_solve, restarted_gmres_budget, spmm,
    stationary_budget,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data-root', default='data/ogb')
    ap.add_argument('--out', type=Path, default=Path('results/ignn_postcheck'))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--trust-ogb-cache', action='store_true',
                    help='Only for trusted official OGB caches (legacy PyTorch pickle load)')
    ap.add_argument('--sparse-format', choices=['csr', 'coo'], default='csr')
    ap.add_argument('--eval-forward-tol', type=float, default=2e-7)
    ap.add_argument('--max-forward', type=int, default=700)
    ap.add_argument('--reference-tol', type=float, default=2e-8)
    ap.add_argument('--max-reference', type=int, default=2500)
    ap.add_argument('--budgets', default='20,30,40,50,60,70,80')
    ap.add_argument('--gmres-restart', type=int, default=15)
    ap.add_argument('--certified-tol', type=float, default=1e-5)
    args = ap.parse_args()
    if not args.checkpoint.is_file():
        ap.error('Checkpoint file does not exist: '+str(args.checkpoint))
    # Our own checkpoint contains a tensor-only state_dict and scalar metadata.
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    for key in ('state_dict', 'width', 'seed', 'rho', 'num_features', 'num_classes'):
        if key not in ckpt:
            ap.error('Missing checkpoint field '+key)
    args.seed = int(ckpt['seed'])
    args.width = int(ckpt['width'])
    args.rho = float(ckpt['rho'])
    if not (0 < args.rho < 1):
        ap.error('Checkpoint requires 0<rho<1')
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but unavailable')
    x,y,S,split,nclasses,graph_meta,S64 = prepare_data(args, device)
    if int(x.shape[1]) != int(ckpt['num_features']) or nclasses != int(ckpt['num_classes']):
        ap.error('Checkpoint and graph feature/class dimensions disagree')
    model = GraphImplicitClassifier(int(x.shape[1]), args.width, nclasses, args.seed).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        h, nforward, _ = equilibrium(model.drive(x),S,args.rho,
                                     args.eval_forward_tol,args.max_forward)
        logits = model.output(h)
        scores = {key: accuracy(logits,y,index) for key,index in split.items()}
        p = F.softmax(logits[split['train']],dim=1)
        p[torch.arange(len(split['train']), device=device),y[split['train']]] -= 1.0
        p /= len(split['train'])
        b = torch.zeros_like(h)
        b[split['train']] = p @ model.output.weight
        h,b = h.double(),b.double()
        S64 = S64.to(dtype=torch.float64,device=device)
        d = 1-h*h
        def jtv(v):
            return args.rho*spmm(S64,d*v)
        ref, nit_ref, ref_cert = reference_solve(
            b,jtv,args.rho,args.reference_tol,args.max_reference)
        br = float(torch.linalg.vector_norm(b))
        gref = d*ref
        gnorm = float(torch.linalg.vector_norm(gref))
        lamnorm = float(torch.linalg.vector_norm(ref))
        rows=[]
        budgets = sorted(set(int(x) for x in args.budgets.split(',')))
        for budget in budgets:
            for method in ('AR','CAT','Chebyshev','GMRES'):
                device_sync(device); t0 = time.perf_counter()
                if method == 'GMRES':
                    lam,actions = restarted_gmres_budget(
                        b,jtv,budget,args.gmres_restart)
                else:
                    lam,actions = stationary_budget(b,jtv,args.rho,method,budget)
                device_sync(device); solver_seconds = time.perf_counter()-t0
                # One terminal check: 1 additional J^Tv, norms + reductions.
                # No offline exact-reference operations inside timed check.
                device_sync(device); t0 = time.perf_counter()
                resid = b-(lam-jtv(lam))
                rn=float(torch.linalg.vector_norm(resid))
                residual_ratio = rn/(br+1e-30)
                absolute_bound=rn/(1-args.rho)
                estimate_norm=float(torch.linalg.vector_norm(lam))
                certified_relative=(absolute_bound/(estimate_norm-absolute_bound)
                                    if estimate_norm>absolute_bound else math.inf)
                device_sync(device); check_seconds=time.perf_counter()-t0
                # Offline reference comparison, NEVER part of runtime metric.
                measured_gradient = (float(torch.linalg.vector_norm(d*(lam-ref))) /
                                     (gnorm+1e-30))
                measured_lambda = (float(torch.linalg.vector_norm(lam-ref)) /
                                   (lamnorm+1e-30))
                row=dict(seed=args.seed,method=method,budget=budget,
                    solver_jtv_actions=actions,terminal_check_jtv_actions=1,
                    total_charged_jtv_actions=actions+1,
                    solver_seconds=solver_seconds,
                    terminal_check_seconds=check_seconds,
                    charged_seconds=solver_seconds+check_seconds,
                    certified_relative_adjoint_bound=certified_relative,
                    certified_pass=int(certified_relative<args.certified_tol),
                    relative_residual=residual_ratio,
                    offline_relative_adjoint_error=measured_lambda,
                    offline_relative_gradient_error=measured_gradient,
                    reference_bound=ref_cert,reference_iterations=nit_ref)
                rows.append(row)
                csv_save(args.out/'solver_budgets_charged.csv',rows)
                print('AUDIT',args.seed,method,budget,
                      f'actions_charged={actions+1}',
                      f'cert_bound={certified_relative:.3g}',
                      f'solver_s={solver_seconds:.3f}',
                      f'check_s={check_seconds:.3f}',flush=True)
    file_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest=dict(dataset='synthetic_smoke' if args.smoke else 'ogbn-arxiv',
                  seed=args.seed,graph=graph_meta,scores=scores,
                  checkpoint_sha256=file_sha256,checkpoint=str(args.checkpoint),
                  device=str(device),torch_version=torch.__version__,
                  reference_iterations=nit_ref,reference_relative_error_bound=ref_cert,
                  forward_iterations=nforward,
                  charge_policy='fixed budget, one residual JTv plus global norms; timed terminal check',
                  no_online_stop_claim=True,
                  config={k: str(v) if isinstance(v,Path) else v
                          for k,v in vars(args).items()})
    (args.out/'postcheck_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('DONE',args.out.resolve(),flush=True)

if __name__=='__main__':
    main()