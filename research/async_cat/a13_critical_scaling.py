"""A13 scalar continuous-learning stability audit, normalized outer gain."""
import argparse
import json
from pathlib import Path
import numpy as np
import sympy as sp


def symbolic():
    s,f,b,c,d,k,t=sp.symbols('s f b c d k t',positive=True)
    M=sp.Matrix([[-f,0,1],[b,-b,0],[0,-k*f,0]])
    N=sp.Matrix([[-f,0,0,1],[0,0,1,0],[c,-c,-d,0],[0,-k*f,0,0]])
    L=sp.Matrix([[-f,0,1],[b*(1-t*f),-b,b*t],[0,-k*f,0]])
    for mat,p in [(M,s*(s+f)*(s+b)+k*f*b),(N,s*(s+f)*(s*s+d*s+c)+k*f*c),(L,s*(s+f)*(s+b)+k*f*b*(1+t*s))]:
        assert sp.expand((s*sp.eye(mat.rows)-mat).det()-p)==0
    lim=d*(c+f*d+f*f)/(d+f)**2
    # positive difference after bringing matched DC-lag comparison to common denominator
    diff=sp.factor((f+c/d-lim)*d*(d+f)**2)
    assert sp.expand(diff-(2*d*f+f*f)*c-f*f*d*(d+f))==0
    assert sp.expand((s*(s+f)*(s+b)+k*f*b*(1+s/f))-(s+f)*(s*s+b*s+k*b))==0
    return str(diff)


def polynomials(f,b,c,d,k,t):
    return [[1,f+b,f*b,k*f*b], [1,f+d,c+f*d,f*c,k*f*c], [1,f+b,f*b*(1+k*t),k*f*b]]


def abscissa(p,scale):
    # use dimensionless time to reduce polynomial coefficient imbalance
    p=np.array(p,dtype=float)/scale**np.arange(len(p))
    return float(np.max(np.roots(p).real)*scale)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('a13_results.json'));args=ap.parse_args()
    diff=symbolic();rows=[];checks=0
    for f in np.logspace(-1,-7,13):
        c=f;d=2*np.sqrt(f);b=c/d
        k1=f+b;k2=d*(c+f*d+f*f)/(d+f)**2
        assert k2<k1
        for i,lim in [(0,k1),(1,k2)]:
            for mult in (.5,.99,1.01,2.):
                z=abscissa(polynomials(f,b,c,d,lim*mult,1/f)[i],np.sqrt(f))
                assert (z<0)==(mult<1),(f,i,mult,z)
                checks+=1
        for mult in (.1,1.,10.,100.):
            z=abscissa(polynomials(f,b,c,d,mult*f,1/f)[2],np.sqrt(f))
            assert z<0;checks+=1
        rows.append(dict(epsilon=float(f),eta_first_fixed=float(2*f**3),eta_first_matched_lag=float(f*f*k1),eta_second_critical=float(f*f*k2),eta_second_over_matched=float(k2/k1),eta_first_matched_decay=float(f*f*(f+np.sqrt(f)))))
    # For b=f, optimize the actual asymptotic decay, not just the gain margin.
    # No-lead double pole at -f/3; exact lead double pole at -f/2.
    for f in (.1,.001,.00001):
        r1=abscissa(polynomials(f,f,f,2*np.sqrt(f),4*f/27,1/f)[0],f)
        r2=abscissa(polynomials(f,f,f,2*np.sqrt(f),f/4,1/f)[2],f)
        assert abs(r1/f+1/3)<1e-7 and abs(r2/f+.5)<1e-7
        checks+=2
    slopes={key:float(np.polyfit(np.log([r['epsilon'] for r in rows[-5:]]),np.log([r[key] for r in rows[-5:]]),1)[0]) for key in ('eta_first_fixed','eta_first_matched_lag','eta_second_critical','eta_first_matched_decay')}
    data=dict(status='A13_SCALAR_STABILITY_PASS',symbolic_positive_difference=diff,root_checks=checks,slopes=slopes,rows=rows)
    args.out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
if __name__=='__main__':main()
