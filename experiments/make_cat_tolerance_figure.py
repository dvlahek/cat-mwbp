#!/usr/bin/env python3
"""Produce a caption-free vector figure for finite-tolerance CAT/AR analysis.

Input is tolerance_pairs.csv produced by cat_tolerance_study.py, not by the
older run_revision_scaling.py tolerance mode (which has a different protocol).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("results/cat_tolerance/tolerance_pairs.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/cat_tolerance/cat_tolerance_sensitivity"),
                        help="Output filename without extension")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input missing: {args.input}")
    p = pd.read_csv(args.input)
    required = {"calibration", "problem", "rho_cap", "tolerance", "measured_over_theoretical"}
    if not required.issubset(p.columns):
        parser.error("Input is not cat_tolerance_study.py tolerance_pairs.csv")
    p = p[p["calibration"] == "oracle"].copy()
    if p.empty:
        parser.error("No oracle-calibrated observations")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.2, "axes.labelsize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.45), sharey=True)
    for ax, problem, tag in zip(axs, ("linear", "tanh"), ("A", "B")):
        subset = p[p.problem == problem]
        for rho, marker in ((0.80, "o"), (0.95, "s"), (0.99, "^")):
            q = subset[(subset.rho_cap-rho).abs() < 1e-12]
            if q.empty:
                continue
            m = q.groupby("tolerance", as_index=False).agg(
                median=("measured_over_theoretical", "median")
            ).sort_values("tolerance", ascending=False)
            ax.plot(m.tolerance, m["median"], marker=marker, markersize=5,
                    linewidth=1.6, label=rf"$\rho={rho:.2f}$")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.xaxis.set_major_locator(FixedLocator([1e-3, 1e-4, 1e-5, 1e-6, 1e-7]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_xlabel("Relative-error tolerance")
        ax.set_title("Linear" if problem == "linear" else "Tanh equilibrium",
                     weight="medium", fontsize=10.5, pad=8)
        ax.grid(True, alpha=.27, linewidth=.6)
        ax.set_axisbelow(True)
        ax.text(-.11, 1.075, tag, transform=ax.transAxes,
                fontweight="bold", fontsize=11, ha="left", va="top")
        ax.legend(frameon=False, loc="lower right")
    axs[0].set_ylabel("Measured / theoretical ratio")
    fig.subplots_adjust(left=.11, right=.985, top=.86, bottom=.23, wspace=.17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.08)
    fig.savefig(args.output.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
