#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 — Human Validation of Relations (R) and Continuous Completeness (C)
==========================================================================
Validates the automatic Task-1 outputs against the 500-paper expert annotations
BEYOND the discrete level L. Specifically:

  * per-relation agreement (human vs. automatic) for the three cross-section
    relations Intro->Methods, Methods->Results, Results->Discussion:
    exact-match %, quadratic-weighted kappa (3-point ordinal), Spearman;
  * pooled relation agreement over all three links;
  * relation score R (mean of the three relations) human-vs-automatic
    Spearman / Pearson;
  * continuous completeness C=(S+R)/2 vs. the human 0-4 level (Spearman / Pearson).

This addresses the reviewer point that only L was human-validated. Section
coverage S is NOT validated here (no per-section human labels were collected) and
remains a machine-only diagnostic.

DETERMINISTIC: no randomness, no seeds, no model calls. Same inputs -> same
numbers on any machine. Pure pandas + numpy + openpyxl. Runnable on the login
node.

Inputs
------
  --human : annotation workbook (.xlsx). One sheet per stratum; the relation and
            level columns are:
              'Relation: Intro->Methods', 'Relation: Methods->Results',
              'Relation: Results->Discussion'   (values: Traceable/Partial/Not)
              'Completeness Level (0-4)'
            joined by 'doi_folder'.
  --auto  : automatic per-paper scores CSV for the primary method
            (default: scores_qwen3_vl_32b_variantB.csv), providing
              'rel_intro_to_methods','rel_methods_to_results',
              'rel_results_to_discussion'  (traceable/partially_traceable/not_traceable)
              'S_section_score','R_relation_score'
            joined by 'doi_safe'.

Outputs (under --out)
---------------------
  relation_agreement.csv     per-relation + pooled exact/QWK/Spearman
  score_validation.csv       R (human vs auto) and C-vs-human-level correlations
  human_validation_report.txt

Usage
-----
python3 task1_human_relation_validation.py \
    --human /path/annotation_sheet.xlsx \
    --auto  /path/scores_qwen3_vl_32b_variantB.csv \
    --out   /path/task1_scores/human_relation_validation
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- label maps -------------------------------------------------------------
HUMAN_REL = {
    "Relation: Intro->Methods": "intro_to_methods",
    "Relation: Methods->Results": "methods_to_results",
    "Relation: Results->Discussion": "results_to_discussion",
}
HUMAN_MAP = {"traceable": 1.0, "partial": 0.5, "not": 0.0}
AUTO_MAP = {"traceable": 1.0, "partially_traceable": 0.5, "not_traceable": 0.0}
HUMAN_LEVEL_COL = "Completeness Level (0-4)"


# --- helpers ----------------------------------------------------------------
def norm_id(s):
    s = str(s or "").strip().lower().replace("https://doi.org/", "").replace("doi:", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def load_human(path):
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    frames = []
    for name, df in sheets.items():
        if name.lower().startswith("01") or "instruction" in name.lower():
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if "doi_folder" not in df.columns:
            continue
        df = df[df["doi_folder"].notna() & (df["doi_folder"].astype(str).str.strip() != "")]
        frames.append(df)
    h = pd.concat(frames, ignore_index=True)
    h["_id"] = h["doi_folder"].map(norm_id)
    return h


def load_auto(path):
    a = pd.read_csv(path, dtype=str)
    a = a[a.get("parse_ok", pd.Series(["True"] * len(a))).astype(str).str.lower() != "false"]
    idc = "doi_safe" if "doi_safe" in a.columns else a.columns[0]
    a["_id"] = a[idc].map(norm_id)
    return a


def quadratic_weighted_kappa(a, b, K=3):
    """QWK over integer codes 0..K-1."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    if n == 0:
        return float("nan")
    O = np.zeros((K, K))
    for i in range(n):
        O[a[i], b[i]] += 1
    w = np.array([[((i - j) ** 2) / ((K - 1) ** 2) for j in range(K)] for i in range(K)])
    r1 = O.sum(axis=1)
    r2 = O.sum(axis=0)
    E = np.outer(r1, r2) / n
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den else float("nan")


def spearman(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    m = x.notna() & y.notna()
    if m.sum() < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x[m].rank(), y[m].rank())[0, 1])


def pearson(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


# --- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Task 1 human validation of relations (R) and C")
    ap.add_argument("--human", required=True, help="annotation workbook .xlsx")
    ap.add_argument("--auto", required=True, help="automatic scores CSV (primary method)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    h = load_human(args.human)
    a = load_auto(args.auto)
    d = h.merge(a, on="_id", how="inner")
    n_join = len(d)

    # ---- per-relation agreement ----
    rel_rows = []
    pooled_h, pooled_a = [], []
    for hcol, acol in HUMAN_REL.items():
        acol_full = "rel_" + acol
        sub = d[[hcol, acol_full]].copy()
        hv = sub[hcol].str.strip().str.lower().map(HUMAN_MAP)
        av = sub[acol_full].str.strip().str.lower().map(AUTO_MAP)
        m = hv.notna() & av.notna()
        hv, av = hv[m], av[m]
        hi = (hv * 2).astype(int).values  # 0,1,2
        ai = (av * 2).astype(int).values
        exact = round(100 * float((hi == ai).mean()), 2)
        qwk = round(quadratic_weighted_kappa(hi, ai, 3), 3)
        sp = round(spearman(hv.values, av.values), 3)
        rel_rows.append({"relation": acol, "n": int(len(hv)),
                         "exact_pct": exact, "qwk": qwk, "spearman": sp})
        pooled_h += list(hv.values)
        pooled_a += list(av.values)

    pooled_h = np.array(pooled_h)
    pooled_a = np.array(pooled_a)
    hi = (pooled_h * 2).astype(int)
    ai = (pooled_a * 2).astype(int)
    rel_rows.append({"relation": "POOLED", "n": int(len(pooled_h)),
                     "exact_pct": round(100 * float((hi == ai).mean()), 2),
                     "qwk": round(quadratic_weighted_kappa(hi, ai, 3), 3),
                     "spearman": round(spearman(pooled_h, pooled_a), 3)})
    REL = pd.DataFrame(rel_rows)
    REL.to_csv(out / "relation_agreement.csv", index=False)

    # ---- R (mean relation score) human vs auto ----
    d["R_human"] = d[list(HUMAN_REL)].apply(
        lambda r: np.nanmean([HUMAN_MAP.get(str(x).strip().lower(), np.nan) for x in r]), axis=1)
    d["R_auto"] = d[["rel_" + v for v in HUMAN_REL.values()]].apply(
        lambda r: np.nanmean([AUTO_MAP.get(str(x).strip().lower(), np.nan) for x in r]), axis=1)
    rr = d[["R_human", "R_auto"]].dropna()
    R_sp = round(spearman(rr.R_human, rr.R_auto), 3)
    R_pe = round(pearson(rr.R_human, rr.R_auto), 3)

    # ---- continuous C vs human level ----
    S = pd.to_numeric(d.get("S_section_score"), errors="coerce")
    Rr = pd.to_numeric(d.get("R_relation_score"), errors="coerce")
    d["C_unw"] = (S + Rr) / 2.0
    d["human_level"] = pd.to_numeric(d.get(HUMAN_LEVEL_COL), errors="coerce")
    cc = d[["C_unw", "human_level"]].dropna()
    C_sp = round(spearman(cc.C_unw, cc.human_level), 3)
    C_pe = round(pearson(cc.C_unw, cc.human_level), 3)

    SV = pd.DataFrame([
        {"measure": "R_relation_score_vs_human", "spearman": R_sp, "pearson": R_pe, "n": int(len(rr))},
        {"measure": "C_completeness_vs_human_level", "spearman": C_sp, "pearson": C_pe, "n": int(len(cc))},
    ])
    SV.to_csv(out / "score_validation.csv", index=False)

    # ---- report ----
    lines = [
        "Task 1 — human validation of relations (R) and continuous C",
        "=" * 58, "",
        f"joined (human ∩ automatic, primary method) n = {n_join}", "",
        "Per-relation agreement (human vs automatic; 3-point ordinal):",
        "-" * 58,
        REL.to_string(index=False), "",
        "Relation score R (mean of 3 relations), human vs automatic:",
        f"  Spearman = {R_sp}   Pearson = {R_pe}   n = {len(rr)}", "",
        "Continuous completeness C = (S+R)/2 vs human 0-4 level:",
        f"  Spearman = {C_sp}   Pearson = {C_pe}   n = {len(cc)}", "",
        "Note: section coverage S is not human-validated (no per-section human",
        "labels were collected) and remains a machine-only diagnostic.",
    ]
    (out / "human_validation_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\n\nSaved -> {out}")


if __name__ == "__main__":
    main()