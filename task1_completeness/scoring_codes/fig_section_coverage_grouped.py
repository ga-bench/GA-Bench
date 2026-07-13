#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 3 (grouped) — per-IMRaD-section mean coverage across all 6 runs.
Reads the per-method section_coverage_*.csv files (cols:
method,section,n,mean_coverage,pct_verdict_explicit,pct_verdict_implied,pct_verdict_absent)
and draws grouped bars: x = IMRaD sections, one bar per model x variant.

Usage:
  python3 fig_section_coverage_grouped.py \
    --glob "./ \
    --out  "./
"""
import argparse, glob, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SECTIONS = ["introduction", "methods", "results", "discussion"]
SEC_LABEL = {"introduction": "Intro", "methods": "Methods",
             "results": "Results", "discussion": "Discussion"}
ORDER = ["qwen3_vl_32b_variantA", "qwen3_vl_32b_variantB",
         "gemma_3_27b_variantA", "gemma_3_27b_variantB",
         "mistral_small_24b_variantA", "mistral_small_24b_variantB"]
LABEL = {"qwen3_vl_32b_variantA": "Qwen-A", "qwen3_vl_32b_variantB": "Qwen-B",
         "gemma_3_27b_variantA": "Gemma-A", "gemma_3_27b_variantB": "Gemma-B",
         "mistral_small_24b_variantA": "Mistral-A", "mistral_small_24b_variantB": "Mistral-B"}
COLORS = ["#4C72B0", "#2E4C7E", "#55A868", "#2F6B43", "#DD8452", "#A85B24"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit("no section_coverage_*.csv files matched")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["method"] = df["method"].astype(str).str.strip()
    df["section"] = df["section"].astype(str).str.strip().str.lower()

    methods = [m for m in ORDER if m in set(df["method"])]
    if not methods:  # fallback: whatever is present
        methods = sorted(df["method"].unique())

    x = np.arange(len(SECTIONS))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(methods):
        vals = [df[(df.method == m) & (df.section == s)]["mean_coverage"].mean()
                for s in SECTIONS]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w,
                      label=LABEL.get(m, m), color=COLORS[i % len(COLORS)])
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([SEC_LABEL[s] for s in SECTIONS])
    ax.set_ylabel("Mean coverage")
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("saved", args.out)

if __name__ == "__main__":
    main()