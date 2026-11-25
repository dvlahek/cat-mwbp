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
        # Git on Windows may convert LF to CRLF on checkout. The repository
        # has historical CSVs in both formats. Compare only exact original
        # content hashes under the two standard end-of-line representations.
        # No numeric values or other whitespace are changed by this check.
        lf = data.replace(b"\r\n", b"\n")
        variants = (data, lf, lf.replace(b"\n", b"\r\n"))
        matched = any(
            len(candidate) == info["bytes"]
            and hashlib.sha256(candidate).hexdigest() == info["sha256"]
            for candidate in variants
        )
        assert matched, (
            name, "content mismatch beyond LF/CRLF conversion",
            "checkout_size", len(data), "expected_size", info["bytes"]
        )
    cheb = pd.read_csv(P / "chebyshev_corrected_pairs.csv")
    assert len(cheb) == 240
    assert (int((cheb.Chebyshev < cheb.CAT).sum()),
            int((cheb.Chebyshev == cheb.CAT).sum()),
            int((cheb.Chebyshev > cheb.CAT).sum())) == (164, 42, 34)
    ignn = pd.read_csv(P / "ignn_certification_summary.csv").set_index("method")
    for method, actions in (("AR",81),("CAT",51),("Chebyshev",51),("GMRES",31)):
        assert int(ignn.loc[method,"median_charged_actions"]) == actions, method
    den = pd.read_csv(P / "spatial_first_certified_by_seed.csv")
    archived = pd.concat([
        pd.read_csv(P / "audit" / f"spatial_reproduced_s{seed}" / "first_certified.csv")
        for seed in range(3)
    ], ignore_index=True)
    pd.testing.assert_frame_equal(den, archived)
    assert set(den.seed) == {0, 1, 2}
    summary = pd.read_csv(P / "spatial_certified_solver_summary.csv").set_index("method")
    for method, actions in (("AR",161),("CAT",46),("Chebyshev",46),("GMRES",36)):
        subset = den.loc[den.method == method]
        assert len(subset) == 3 and subset.certified_pass.eq(1).all()
        assert int(subset.charged_jtv.median()) == actions
        assert int(summary.loc[method, "median_charged_jtv"]) == actions
        assert abs(summary.loc[method, "median_charged_seconds"] - subset.charged_time_s.median()) < 1e-12
    paired = pd.read_csv(P / "spatial_paired_speedups.csv").set_index("seed")
    for seed in range(3):
        q = den.loc[den.seed == seed].set_index("method")
        assert abs(paired.loc[seed, "AR_over_CAT_actions"] - q.loc["AR", "charged_jtv"] / q.loc["CAT", "charged_jtv"]) < 1e-12
        assert abs(paired.loc[seed, "AR_over_CAT_time"] - q.loc["AR", "charged_time_s"] / q.loc["CAT", "charged_time_s"]) < 1e-12
    psnr = pd.read_csv(P / "spatial_mean_test_psnr.csv")
    for seed in range(3):
        source = json.loads((P / "audit" / f"spatial_reproduced_s{seed}" / "manifest.json").read_text())
        for method, value in source["test_psnr_mean_by_method"].items():
            row = psnr.loc[(psnr.seed == seed) & (psnr.method == method), "psnr_db"]
            assert len(row) == 1 and abs(float(row.iloc[0]) - value) < 1e-10
    # Solver settings for Table 3.
    aconf = pd.read_csv(P / "anderson_best_config_extended.csv")
    assert len(aconf) == 6
    assert list(aconf.median_actions.astype(int)) == [17,21,24,17,25,29]
    assert set(aconf.regularization) == {1e-14}
    cpu = pd.read_csv(P / "solver_cost_v2_per_seed.csv")
    assert len(cpu) == 25 and set(cpu.seed) == set(range(5))
    assert (cpu.relative_error < 1e-6).all()
    csum = pd.read_csv(P / "solver_cost_v2_summary.csv").set_index("method")
    for method, actions in (("AR-oracle",160),("CAT-oracle",39),
                            ("GMRES",24),("Good Broyden",29),("Anderson-20",25)):
        assert int(csum.loc[method,"median_actions"]) == actions
    assert abs(float(csum.loc["GMRES","core_state_KiB"]) - 32.8125) < 1e-9
    assert abs(float(csum.loc["Anderson-20","core_state_KiB"]) - 67.5) < 1e-9
    vis = pd.read_csv(P / "vision_corrected_chebyshev_pairs.csv")
    assert len(vis) == 48
    assert (int((vis["AR-oracle"] > vis["CAT-oracle"]).sum()),
            int((vis["Chebyshev-oracle-corrected"] < vis["CAT-oracle"]).sum()),
            int((vis["Chebyshev-oracle-corrected"] == vis["CAT-oracle"]).sum())) == (48,35,13)
    fine = pd.read_csv(P / "fine_grid_first_pass.csv")
    assert len(fine) == 18
    assert not fine.duplicated(["application", "seed", "method"]).any()
    assert (fine.first_certified_charged_actions == fine.first_certified_iteration + 1).all()
    assert (fine.certified_relative_adjoint_bound < 1e-5).all()
    for app, expected in {
        "graph": {"AR": [76, 73, 75], "CAT": [46, 46, 46], "Chebyshev": [48, 48, 48]},
        "spatial_retrained": {"AR": [147, 156, 122], "CAT": [42, 41, 43], "Chebyshev": [44, 43, 45]},
    }.items():
        q = fine[fine.application == app].pivot(index="seed", columns="method", values="first_certified_charged_actions")
        assert list(q.index) == [0, 1, 2]
        for method, values in expected.items():
            assert q[method].tolist() == values
        assert (q.Chebyshev - q.CAT == 2).all()
    print("CAT_MINIMAL_REPRODUCIBILITY_PASS",len(manifest["compact_results"]),"tables")

if __name__ == "__main__":
    main()
