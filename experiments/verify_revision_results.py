#!/usr/bin/env python3
"""Verify the compact numerical tables retained for the CAT manuscript."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "revision_results"

def main():
    manifest = json.loads((P / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in manifest["compact_results"].items():
        data = (P / name).read_bytes()
        assert len(data) == info["bytes"], (name, "size")
        assert hashlib.sha256(data).hexdigest() == info["sha256"], (name, "SHA-256")
    cheb = pd.read_csv(P / "chebyshev_corrected_pairs.csv")
    assert len(cheb) == 240
    assert (int((cheb.Chebyshev < cheb.CAT).sum()),
            int((cheb.Chebyshev == cheb.CAT).sum()),
            int((cheb.Chebyshev > cheb.CAT).sum())) == (164, 42, 34)
    ignn = pd.read_csv(P / "ignn_certification_summary.csv").set_index("method")
    for method, actions in (("AR",81),("CAT",51),("Chebyshev",51),("GMRES",31)):
        assert int(ignn.loc[method,"median_charged_actions"]) == actions, method
    den = pd.read_csv(P / "spatial_first_certified_by_seed.csv")
    assert set(den.seed) == {0,1,2}
    for method, actions in (("AR",161),("CAT",46),("Chebyshev",46)):
        assert set(den.loc[den.method == method,"charged_jtv"]) == {actions}, method
    assert len(pd.read_csv(P / "spatial_paired_speedups.csv")) == 3
    print("CAT_MINIMAL_REPRODUCIBILITY_PASS",len(manifest["compact_results"]),"tables")

if __name__ == "__main__":
    main()
