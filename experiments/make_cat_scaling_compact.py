#!/usr/bin/env python3
"""Make a clean, caption-free, two-panel CAT parameter-scaling figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("results/revision_width/width_pairs.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/revision_width/cat_parameter_scaling_compact"),
                        help="Output filename without extension")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input missing: {args.input}")
    p = pd.read_csv(args.input)
    required = {"dataset", "trainable_parameters", "action_ratio_ar_over_cat",
                "cat_test_accuracy"}
    if not required.issubset(p.columns):
        parser.error(f"Missing expected width_pairs.csv columns: {sorted(required-set(p.columns))}")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelsize": 10, "legend.fontsize": 9.2,
        "xtick.labelsize": 9.2, "ytick.labelsize": 9.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ds, label, marker in (
        ("breast_cancer", "Breast cancer", "o"),
        ("digits", "Digits", "s"),
        ("synthetic_large", "Synthetic large", "^"),
    ):
        d = p[p.dataset == ds].groupby("trainable_parameters").agg(
            ratio=("action_ratio_ar_over_cat", "median"),
            acc=("cat_test_accuracy", "mean")
        ).reset_index().sort_values("trainable_parameters")
        if d.empty:
            continue
        axs[0].plot(d.trainable_parameters, d.ratio, marker=marker,
                    markersize=4.7, linewidth=1.6, label=label)
        axs[1].plot(d.trainable_parameters, d.acc, marker=marker,
                    markersize=4.7, linewidth=1.6, label=label)
    axs[0].axhline(1.0, ls="--", lw=0.8, c="0.65")
    axs[0].set_ylabel("AR / CAT local-action ratio")
    axs[1].set_ylabel("Held-out accuracy")
    for ax, letter in zip(axs, "AB"):
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters")
        ax.grid(alpha=.25, linewidth=.7)
        ax.set_axisbelow(True)
        ax.text(-.10, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.legend(frameon=False, loc="best")
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.subplots_adjust(left=.09, right=.985, top=.92, bottom=.20, wspace=.30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.08)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
