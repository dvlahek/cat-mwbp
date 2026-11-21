#!/usr/bin/env python3
"""Measure CPU time and state of the published CAT numerical solver controls.

Imports the original paper's solver routines and *frozen selected parameters*.
Does not rerun hyperparameter tuning or change the benchmark's stopping target.
Timing excludes case generation, exact reference solution, and offline error checks.
CPU RSS sampling and Python allocation tracking are secondary, not a GPU or
hardware-independent memory measurement. Numerical state budgets are estimates.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import statistics
import sys
import time
import tracemalloc
import threading
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from run_nonlocal_solver_reference import (  # released algorithm and RHS
    make_case, coeff, stationary, gmres_exact, anderson, broyden, TOL
)


def memory_sample(fn, interval_s: float=0.003):
    try:
        import psutil
        process=psutil.Process()
    except ImportError:
        process=None
    baseline=process.memory_info().rss if process else None
    peak=[baseline] if baseline is not None else []
    stop=threading.Event()
    def sample():
        while not stop.is_set():
            try:
                peak.append(process.memory_info().rss)
            except Exception:
                return
            stop.wait(interval_s)
    tracemalloc.start()
    thread=threading.Thread(target=sample,daemon=True) if process else None
    if thread: thread.start()
    try:
        fn()
    finally:
        stop.set()
        if thread: thread.join()
        _,py_peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return py_peak, (max(peak)-baseline if peak else float('nan'))



def anderson_bounded(rhs,jtv,error,m,beta,reg,max_actions=12000):
    """Numerically equivalent to released Anderson-m, with only last m histories.

    The reference script retains *all* lists even though it computes updates
    from only the most recent m columns. The bounded variant avoids attributing
    a removable storage implementation artifact to Anderson itself.
    """
    from collections import deque
    x=np.zeros_like(rhs)
    xs=deque(maxlen=m);fs=deque(maxlen=m);gs=deque(maxlen=m)
    for k in range(max_actions+1):
        if error(x)<=TOL: return k,x
        if k==max_actions:break
        f=rhs+jtv(x);g=f-x
        xs.append(x.copy());fs.append(f.copy());gs.append(g.copy())
        keep=len(xs);X=np.column_stack(xs);F=np.column_stack(fs);G=np.column_stack(gs)
        if keep==1:
            x=(1-beta)*X[:,0]+beta*F[:,0]
            continue
        A=G.T@G+reg*np.eye(keep)
        K=np.block([[np.zeros((1,1)),np.ones((1,keep))],[np.ones((keep,1)),A]])
        q=np.zeros(keep+1);q[0]=1
        try:a=np.linalg.solve(K,q)[1:]
        except np.linalg.LinAlgError:a=np.linalg.lstsq(K,q,rcond=None)[0][1:]
        x=(1-beta)*(X@a)+beta*(F@a)
    return max_actions,x

def estimated_state_bytes(method: str,n: int,m:int|None, actions: int):
    # Persistent FLOAT64 arrays only (excludes workspaces allocated by BLAS/LAPACK).
    # The estimate deliberately reports the growth scale, not exact peak memory.
    vec=8*n
    if method=='AR-oracle': return 2*vec
    if method=='CAT-oracle': return 3*vec
    if method=='GMRES': return (n*(n+1)+(n+1)*n)*8
    if method=='Good-Broyden': return 8*(n*n+4*n)
    if method=='Anderson':
        # Bounded-memory Anderson-m: m saved vectors each of x, f, g,
        # plus active matrices and KKT solve. Excludes LAPACK workspaces.
        active=min(m, actions)
        return 8*(6*n*active + active*active + 5*active)
    raise KeyError(method)


def load_config(path:Path):
    if not path.is_file():
        raise FileNotFoundError(f'Required reference configurations unavailable: {path}')
    return pd.read_csv(path).set_index(['problem','rho'])


def config_at(config,problem,rho):
    hit=config.reset_index()
    q=hit[(hit.problem==problem)&(np.abs(hit.rho-rho)<1e-10)]
    if len(q)!=1: raise ValueError(f'Expected one frozen config for {problem} rho={rho}, got {len(q)}')
    return q.iloc[0]


def run(args):
    conf_a=load_config(ROOT/'reference_results/implicit/anderson_best_config.csv')
    conf_b=load_config(ROOT/'reference_results/implicit/broyden_best_config.csv')
    rows=[]
    for problem in ('linear','tanh'):
        for rho in args.rhos:
            a=config_at(conf_a,problem,rho)
            b=config_at(conf_b,problem,rho)
            for seed in range(args.seeds):
                rhs,exact,jtv,mu,L,error=make_case(problem,rho,seed,args.side)
                ar_alpha,cat_alpha,cat_beta=coeff(mu,L)
                M=int(a.memory)
                fns={
                    'AR-oracle': lambda:stationary(rhs,jtv,error,ar_alpha,0,max_actions=args.max_actions),
                    'CAT-oracle':lambda:stationary(rhs,jtv,error,cat_alpha,cat_beta,max_actions=args.max_actions),
                    'GMRES':lambda:gmres_exact(rhs,jtv,error),
                    'Anderson':lambda:anderson_bounded(rhs,jtv,error,M,float(a.mixing),float(a.regularization),max_actions=args.max_actions),
                    'Good-Broyden':lambda:broyden(rhs,jtv,error,float(b.inverse_scale),float(b.damping),max_actions=args.max_actions),
                }
                # Audit that the bounded-memory form gives the same numerical
                # trajectory/stop as the released unbounded-history routine.
                ref_actions,ref_x=anderson(rhs,jtv,error,M,float(a.mixing),
                                            float(a.regularization),max_actions=args.max_actions)
                new_actions,new_x=fns['Anderson']()
                if ref_actions!=new_actions or not np.allclose(ref_x,new_x,rtol=1e-11,atol=1e-11):
                    raise RuntimeError(f'Anderson-m bounded-memory control changed numerical results: {problem} {rho} {seed}')
                for method,fn in fns.items():
                    # Warm up once before timing to avoid first-use BLAS/threadpool overhead.
                    with threadpool_limits(limits=args.blas_threads):
                        actions,x=fn()
                        if not np.isfinite(error(x)) or error(x)>TOL:
                            raise RuntimeError(f'{problem} rho={rho} seed={seed} {method} did not converge')
                        timings=[]
                        counts=[]
                        for rep in range(args.repeats):
                            t0=time.perf_counter_ns()
                            c,v=fn()
                            elapsed=(time.perf_counter_ns()-t0)/1e9
                            if error(v)>TOL or c!=actions:
                                raise RuntimeError('Non-reproducible solver at fixed seed')
                            timings.append(elapsed); counts.append(c)
                        gc.collect()
                        py_peak,rss_delta=memory_sample(fn)
                    row={
                        'problem':problem, 'rho':rho,'seed':seed,'side':args.side,
                        'state_dim':len(rhs),'method':method,'jtv_actions':actions,
                        'relative_target_error':float(error(x)),
                        'time_seconds_median':float(statistics.median(timings)),
                        'time_seconds_min':float(min(timings)),
                        'time_seconds_max':float(max(timings)),
                        'estimated_persistent_state_bytes':estimated_state_bytes(method,len(rhs),M,actions),
                        'python_peak_allocated_bytes':int(py_peak),
                        'process_rss_delta_sampled_bytes':rss_delta,
                        'anderson_memory':M if method=='Anderson' else None,
                        'anderson_mixing':float(a.mixing) if method=='Anderson' else None,
                        'anderson_regularization':float(a.regularization) if method=='Anderson' else None,
                        'broyden_scale':float(b.inverse_scale) if method=='Good-Broyden' else None,
                        'broyden_damping':float(b.damping) if method=='Good-Broyden' else None,
                        'blas_threads':args.blas_threads,
                    }
                    rows.append(row)
                    print(f'{problem:6s} rho={rho:.2f} seed={seed} {method:13s} '
                          f'JTv={actions:5d} time_median={row["time_seconds_median"]:.4f}s',flush=True)
                    pd.DataFrame(rows).to_csv(args.out/'solver_cost_runs.csv',index=False)
    d=pd.DataFrame(rows)
    summary=(d.groupby(['problem','rho','method'],as_index=False)
             .agg(n_seeds=('seed','nunique'),median_actions=('jtv_actions','median'),
                  median_time_s=('time_seconds_median','median'),
                  median_persistent_state_bytes=('estimated_persistent_state_bytes','median'),
                  median_python_peak_bytes=('python_peak_allocated_bytes','median'),
                  median_rss_sampled_delta_bytes=('process_rss_delta_sampled_bytes','median')))
    summary.to_csv(args.out/'solver_cost_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9.0,3.5))
    for method in ('AR-oracle','CAT-oracle','Anderson','GMRES','Good-Broyden'):
        q=summary[(summary.problem=='tanh')&(summary.method==method)].sort_values('rho')
        axs[0].plot(q.rho,q.median_time_s,'o-',label=method)
        axs[1].plot(q.rho,q.median_persistent_state_bytes/1024,'o-',label=method)
    axs[0].set_ylabel('Median solve time (s), CPU')
    axs[1].set_ylabel('Estimated persistent solver state (KiB)')
    axs[1].set_yscale('log')
    for ax,l in zip(axs,'AB'):
        ax.set_xlabel(r'Recurrent radius $\rho$')
        ax.grid(alpha=.2)
        ax.text(-.10,1.04,l,fontweight='bold',transform=ax.transAxes)
    axs[0].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out/'nonlocal_solver_costs.pdf',bbox_inches='tight')
    fig.savefig(args.out/'nonlocal_solver_costs.png',dpi=250,bbox_inches='tight')
    plt.close(fig)
    (args.out/'manifest.json').write_text(json.dumps({
        'scope':'Original 12x12 controlled nonbipartite symmetric grid; frozen config from reference_results',
        'anderson_implementation':'Numerically verified bounded-memory Anderson-m, last m histories; reference code retained all histories unnecessarily',
        'measurement':'Repeated time per solver excluding oracle case generation and offline error audit',
        'n_seeds':args.seeds,'repeats':args.repeats,'blas_threads':args.blas_threads,
        'side':args.side,'rhos':args.rhos,'tolerance':TOL,
        'memory_warning':'Estimated persistent state is not peak RAM; Python peak excludes native buffers; RSS samples are noisy and process-wide.'
    },indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/solver_cost_audit'))
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--blas-threads',type=int,default=1)
    p.add_argument('--side',type=int,default=12)
    p.add_argument('--rhos',type=lambda s:[float(x) for x in s.split(',')],default=[.8,.95,.99])
    p.add_argument('--max-actions',type=int,default=12000)
    a=p.parse_args()
    if a.seeds<1 or a.repeats<1 or a.blas_threads<1 or a.side<4:p.error('Invalid settings')
    a.out.mkdir(parents=True,exist_ok=True)
    run(a)