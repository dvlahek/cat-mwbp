#!/usr/bin/env python3
"""Verify committed compact revision evidence (no external datasets required)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'revision_results'

def verify():
    m=json.loads((RESULTS/'MANIFEST.json').read_text(encoding='utf-8'))
    for fname,spec in m['compact_results'].items():
        contents=(RESULTS/fname).read_bytes()
        assert hashlib.sha256(contents).hexdigest()==spec['sha256'],fname
    p=pd.read_csv(RESULTS/'chebyshev_corrected_pairs.csv')
    assert len(p)==240
    assert ((p.Chebyshev<p.CAT).sum(),(p.Chebyshev==p.CAT).sum(),(p.Chebyshev>p.CAT).sum())==(164,42,34)
    assert abs(p.cheb_actions_over_cat_actions.median()-0.9130434782608696)<1e-10
    ig=pd.read_csv(RESULTS/'ignn_first_certified_by_seed.csv')
    # Column names in original aggregate from three independent checkpoint audits.
    assert len(ig)>=12
    igs=pd.read_csv(RESULTS/'ignn_certification_summary.csv').set_index('method')
    assert int(igs.loc['AR','median_charged_actions'])==81
    assert int(igs.loc['CAT','median_charged_actions'])==51
    assert int(igs.loc['Chebyshev','median_charged_actions'])==51
    s=pd.read_csv(RESULTS/'spatial_first_certified_by_seed.csv')
    assert len(s)==12 and set(s.seed)=={0,1,2}
    assert set(s[s.method=='AR'].charged_jtv)=={161}
    assert set(s[s.method=='CAT'].charged_jtv)=={46}
    assert set(s[s.method=='Chebyshev'].charged_jtv)=={46}
    pt=pd.read_csv(RESULTS/'spatial_paired_speedups.csv')
    assert len(pt)==3 and (pt.AR_over_CAT_actions==3.5).all()
    psnr=pd.read_csv(RESULTS/'spatial_mean_test_psnr.csv')
    assert set(psnr.seed)=={0,1,2}
    print('CAT_REVISION_COMPACT_EVIDENCE_PASS',len(m['compact_results']), 'CSV files', flush=True)

if __name__=='__main__': verify()