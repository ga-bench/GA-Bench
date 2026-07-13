#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 / Stage 4b — Cross-Model Agreement (3 models, InternVL EXCLUDED).

Merges the four consolidated Stage-4 CSVs (Qwen / Gemma / InternVL / Mistral) on
`paper_id` (intersection) and computes per-field inter-model agreement:

  * Krippendorff's alpha   - nominal  : nominal/boolean categorical fields
                             ordinal  : ordered categorical fields (clarity, ynp, etc.)
                             interval : 1-5 graded scores + counts (num_panels)
  * Mean pairwise % agreement (categorical/boolean)
  * Mean pairwise Spearman rho (ordinal/scores/counts)
  * Mean pairwise quadratic-weighted kappa (ordinal/scores)
  * ICC(2,1) and ICC(2,k), two-way random, absolute agreement (scores/counts)

Pure stdlib + numpy. CPU-only. Runnable on the anon login node (no PBS needed).

Outputs (under OUTPUT_DIR):
  agreement_per_field.csv   one row per field with all summary metrics
  pairwise_agreement.csv    one row per (field, model_a, model_b, metric)
  agreement_summary.json    everything, plus structural-vs-judgment roll-ups
  report.txt                human-readable report

Reverse-coded scores (ambiguity_1to5, text_dependency_1to5) need NO un-reversing
here: agreement is about cross-model consistency, and all models share the same
coding, so polarity is irrelevant to alpha/ICC/Spearman/QWK.
"""

import os
import csv
import json
import argparse
from collections import OrderedDict

import numpy as np

# --------------------------------------------------------------------------- #
# Paths (override via CLI)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = "./task2_readability"

DEFAULT_MODEL_CSVS = OrderedDict([
    ("qwen3_vl_32b",
     "output/stage4_vlm_structural_interpretation/consolidated/stage4_interpretations.csv"),
    ("gemma_3_27b",
     "output/stage4_vlm_gemma_3_27b/consolidated/stage4_interpretations.csv"),
    ("mistral_small_24b",
     "output/stage4_vlm_mistral_small_24b/consolidated/stage4_interpretations.csv"),
])
DEFAULT_OUTPUT_DIR = "output/stage4b_cross_model_agreement_ex_intern"
ID_COL = "paper_id"

# --------------------------------------------------------------------------- #
# Field taxonomy + ordinal codings
# --------------------------------------------------------------------------- #
CLARITY3 = {"unclear": 0, "partially_clear": 1, "clear": 2}
YNP3     = {"no": 0, "partially": 1, "yes": 2}

ORDINAL_MAPS = {
    "panel_structure":          {"simple": 0, "moderate": 1, "complex": 2},
    "narrative_arc":            {"none": 0, "partial": 1, "complete": 2},
    "flow_clarity":             CLARITY3,
    "entity_clarity":           CLARITY3,
    "relation_clarity":         CLARITY3,
    "main_message_identifiable": YNP3,
    "method_identifiable":      YNP3,
    "result_identifiable":      YNP3,
    "conclusion_identifiable":  YNP3,
    "visual_clutter":           {"low": 0, "medium": 1, "high": 2},
    "semantic_interpretability": {"low": 0, "medium": 1, "high": 2},
}

NOMINAL_FIELDS = ["layout_type", "main_reading_direction"]
BOOLEAN_FIELDS = ["has_start_point", "has_end_point", "has_arrows_or_connectors"]
ORDINAL_FIELDS = list(ORDINAL_MAPS.keys())
SCORE_FIELDS   = ["sequence_clarity_1to5", "key_message_clarity_1to5",
                  "ambiguity_1to5", "text_dependency_1to5",
                  "overall_interpretability_1to5"]
COUNT_FIELDS   = ["num_panels"]

ALL_FIELDS = NOMINAL_FIELDS + BOOLEAN_FIELDS + ORDINAL_FIELDS + SCORE_FIELDS + COUNT_FIELDS

# "structural/layout" (concrete) vs "judgment" (subjective) for roll-ups
STRUCTURAL = set(["layout_type", "main_reading_direction", "panel_structure",
                  "num_panels", "has_start_point", "has_end_point",
                  "has_arrows_or_connectors", "narrative_arc"])
JUDGMENT = set(ALL_FIELDS) - STRUCTURAL

# --------------------------------------------------------------------------- #
# Small stats helpers (numpy only)
# --------------------------------------------------------------------------- #
def rankdata_avg(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0      # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    rx, ry = rankdata_avg(x), rankdata_avg(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def quadratic_weighted_kappa(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    lo = int(min(a.min(), b.min())); hi = int(max(a.max(), b.max()))
    R = hi - lo + 1
    if R < 2:
        return float("nan")
    O = np.zeros((R, R), float)
    for ai, bi in zip(a, b):
        O[ai - lo, bi - lo] += 1
    w = np.zeros((R, R), float)
    for i in range(R):
        for j in range(R):
            w[i, j] = ((i - j) / (R - 1.0)) ** 2
    hist_a = O.sum(axis=1); hist_b = O.sum(axis=0)
    E = np.outer(hist_a, hist_b) / O.sum()
    denom = (w * E).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (w * O).sum() / denom)


def pct_agreement(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return float(np.mean(x == y))


def icc_two_way(M):
    """ICC(2,1) and ICC(2,k): two-way random, absolute agreement. M = subjects x raters."""
    M = np.asarray(M, float)
    n, k = M.shape
    if n < 2 or k < 2:
        return float("nan"), float("nan")
    grand = M.mean()
    ms_subj = M.mean(axis=1); ms_rater = M.mean(axis=0)
    SSR = k * ((ms_subj - grand) ** 2).sum()        # rows / subjects
    SSC = n * ((ms_rater - grand) ** 2).sum()        # cols / raters
    SST = ((M - grand) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom1 = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    icc21 = (MSR - MSE) / denom1 if denom1 != 0 else float("nan")
    denomk = MSR + (MSC - MSE) / n
    icc2k = (MSR - MSE) / denomk if denomk != 0 else float("nan")
    return float(icc21), float(icc2k)


def gwet_ac(codes, q, weighted=False):
    """
    Gwet's AC1 (weighted=False) / AC2 (weighted=True), multi-rater.
    codes: n_subjects x r_raters integer category indices in 0..q-1 (all present).
    Robust to the high-prevalence paradox that collapses Krippendorff's alpha.
    """
    M = np.asarray(codes, int)
    if M.ndim != 2 or q < 2 or M.shape[1] < 2:
        return float("nan")
    n, r = M.shape
    if weighted:
        ix = np.arange(q, dtype=float)
        W = 1.0 - ((ix[:, None] - ix[None, :]) / (q - 1.0)) ** 2
    else:
        W = np.eye(q)
    Tw = W.sum()
    counts = np.zeros((n, q), float)
    for i in range(n):
        for v in M[i]:
            counts[i, v] += 1.0
    ri = counts.sum(axis=1)
    diagW = np.diag(W)
    pa_terms = []
    for i in range(n):
        c = counts[i]
        num = c @ W @ c - (c * diagW).sum()
        if ri[i] > 1:
            pa_terms.append(num / (ri[i] * (ri[i] - 1)))
    if not pa_terms:
        return float("nan")
    pa = float(np.mean(pa_terms))
    pi = (counts / ri[:, None]).mean(axis=0)            # category prevalence
    pe = (Tw / (q * (q - 1.0))) * float(np.sum(pi * (1.0 - pi)))
    if pe >= 1.0:
        return float("nan")
    return float((pa - pe) / (1.0 - pe))


def krippendorff_alpha(units, level="nominal"):
    """
    units: list of per-item rating lists (one entry per coder; None = missing).
    level: 'nominal' | 'ordinal' | 'interval'. Values must be numeric-coded.
    """
    # value domain
    vals = sorted({v for u in units for v in u if v is not None})
    if len(vals) < 2:
        return float("nan")
    idx = {v: i for i, v in enumerate(vals)}
    V = len(vals)
    coinc = np.zeros((V, V), float)
    for u in units:
        present = [v for v in u if v is not None]
        m = len(present)
        if m < 2:
            continue
        # build counts of each value in the unit
        cnt = {}
        for v in present:
            cnt[v] = cnt.get(v, 0) + 1
        for a, ca in cnt.items():
            for b, cb in cnt.items():
                if a == b:
                    pairs = ca * (ca - 1)
                else:
                    pairs = ca * cb
                coinc[idx[a], idx[b]] += pairs / (m - 1)
    n_marg = coinc.sum(axis=1)
    n_total = n_marg.sum()
    if n_total < 2:
        return float("nan")

    def delta2(i, j):
        if level == "nominal":
            return 0.0 if i == j else 1.0
        if level == "interval":
            return (vals[i] - vals[j]) ** 2
        # ordinal
        lo, hi = (i, j) if i <= j else (j, i)
        s = n_marg[lo:hi + 1].sum() - (n_marg[i] + n_marg[j]) / 2.0
        return s * s

    Do = 0.0; De = 0.0
    for i in range(V):
        for j in range(V):
            d = delta2(i, j)
            if d == 0.0:
                continue
            Do += coinc[i, j] * d
            De += n_marg[i] * n_marg[j] * d
    if De == 0:
        return float("nan")
    return float(1.0 - (n_total - 1) * Do / De)

# --------------------------------------------------------------------------- #
# Loading / value normalization
# --------------------------------------------------------------------------- #
def resolve_columns(header, fields):
    """Map each expected field -> actual column name (exact, prefix-stripped, or suffix)."""
    hset = {h: h for h in header}
    low = {h.lower(): h for h in header}
    out = {}
    for f in fields + [ID_COL]:
        if f in hset:
            out[f] = f; continue
        if f.lower() in low:
            out[f] = low[f.lower()]; continue
        cand = [h for h in header if h.lower().endswith(f.lower())]
        if len(cand) == 1:
            out[f] = cand[0]
    return out


def norm_bool(s):
    s = str(s).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return 1
    if s in ("false", "0", "no", "n", "f"):
        return 0
    return None


def norm_int(s):
    try:
        return int(round(float(str(s).strip())))
    except (ValueError, TypeError):
        return None


def norm_cat(s):
    return str(s).strip().lower().replace(" ", "_") if s not in (None, "") else None


def load_model_csv(path, fields):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        colmap = resolve_columns(reader.fieldnames, fields)
        if ID_COL not in colmap:
            raise SystemExit("ERROR: '%s' column not found in %s" % (ID_COL, path))
        rows = {}
        for r in reader:
            pid = r[colmap[ID_COL]].strip()
            if not pid:
                continue
            rec = {}
            for f in fields:
                if f not in colmap:
                    rec[f] = None; continue
                raw = r[colmap[f]]
                if f in BOOLEAN_FIELDS:
                    rec[f] = norm_bool(raw)
                elif f in SCORE_FIELDS or f in COUNT_FIELDS:
                    rec[f] = norm_int(raw)
                else:
                    rec[f] = norm_cat(raw)
            rows[pid] = rec
    return rows, colmap

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=PROJECT_ROOT)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--subset", default=None,
                    help="optional newline-separated paper_id file; agreement also computed on it")
    args = ap.parse_args()

    root = args.project_root
    out_dir = args.output_dir or os.path.join(root, DEFAULT_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # load all models
    model_rows = OrderedDict()
    for mkey, rel in DEFAULT_MODEL_CSVS.items():
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            raise SystemExit("ERROR: missing CSV for %s: %s" % (mkey, path))
        rows, colmap = load_model_csv(path, ALL_FIELDS)
        model_rows[mkey] = rows
        print("[load] %-18s rows=%d  matched_fields=%d/%d"
              % (mkey, len(rows), sum(1 for f in ALL_FIELDS if f in colmap), len(ALL_FIELDS)))

    models = list(model_rows.keys())
    # intersection of paper_ids
    common = set.intersection(*[set(r.keys()) for r in model_rows.values()])
    common = sorted(common)
    print("[merge] intersection paper_ids = %d" % len(common))

    subset_ids = None
    if args.subset and os.path.isfile(args.subset):
        with open(args.subset) as fh:
            subset_ids = sorted(set(l.strip() for l in fh if l.strip()) & set(common))
        print("[merge] subset paper_ids (in intersection) = %d" % len(subset_ids))

    def compute(field, ids):
        # build raters matrix / unit lists
        if field in BOOLEAN_FIELDS:
            ftype, level = "boolean", "nominal"
        elif field in NOMINAL_FIELDS:
            ftype, level = "nominal", "nominal"
        elif field in ORDINAL_FIELDS:
            ftype, level = "ordinal", "ordinal"
        elif field in SCORE_FIELDS:
            ftype, level = "score", "interval"
        else:
            ftype, level = "count", "interval"

        omap = ORDINAL_MAPS.get(field)
        cat_domain = {}  # for nominal string -> code
        units = []
        complete = []   # rows with all models present (for ICC/pairwise)
        for pid in ids:
            row_vals = []
            for m in models:
                v = model_rows[m][pid][field]
                if v is None:
                    row_vals.append(None); continue
                if ftype == "ordinal":
                    row_vals.append(omap.get(v))
                elif ftype == "nominal":
                    if v not in cat_domain:
                        cat_domain[v] = len(cat_domain)
                    row_vals.append(cat_domain[v])
                else:  # boolean/score/count already int
                    row_vals.append(v)
            units.append(row_vals)
            if all(x is not None for x in row_vals):
                complete.append(row_vals)

        n_units = sum(1 for u in units if sum(x is not None for x in u) >= 2)
        alpha = krippendorff_alpha(units, level=level)

        res = {"field": field, "type": ftype, "n_units": n_units,
               "krippendorff_alpha": alpha,
               "gwet_ac1": None, "gwet_ac2": None,
               "mean_pct_agreement": None, "mean_spearman": None,
               "mean_qwk": None, "icc21": None, "icc2k": None}
        pairwise = []

        if not complete:
            return res, pairwise
        Mc = np.array(complete, float)
        k = Mc.shape[1]

        # ----- Gwet's AC1 / AC2 (chance-corrected, prevalence-robust) ----- #
        Mi = Mc.astype(int)
        if ftype == "nominal":
            res["gwet_ac1"] = gwet_ac(Mi, len(cat_domain), weighted=False)
        elif ftype == "boolean":
            res["gwet_ac1"] = gwet_ac(Mi, 2, weighted=False)
        elif ftype == "ordinal":
            qn = len(omap)
            res["gwet_ac1"] = gwet_ac(Mi, qn, weighted=False)
            res["gwet_ac2"] = gwet_ac(Mi, qn, weighted=True)
        else:  # score / count
            mn = int(Mi.min()); mx = int(Mi.max())
            qn = mx - mn + 1
            res["gwet_ac1"] = gwet_ac(Mi - mn, qn, weighted=False)
            res["gwet_ac2"] = gwet_ac(Mi - mn, qn, weighted=True)

        # pairwise metrics
        pcts, rhos, qwks = [], [], []
        for i in range(k):
            for j in range(i + 1, k):
                xi, xj = Mc[:, i], Mc[:, j]
                if ftype in ("boolean", "nominal", "ordinal"):
                    p = pct_agreement(xi, xj); pcts.append(p)
                    pairwise.append((field, models[i], models[j], "pct_agreement", p))
                if ftype in ("ordinal", "score", "count"):
                    r = spearman(xi, xj); rhos.append(r)
                    pairwise.append((field, models[i], models[j], "spearman", r))
                if ftype in ("ordinal", "score"):
                    q = quadratic_weighted_kappa(xi.astype(int), xj.astype(int)); qwks.append(q)
                    pairwise.append((field, models[i], models[j], "qwk", q))

        nanmean = lambda a: float(np.nanmean(a)) if len(a) else None
        res["mean_pct_agreement"] = nanmean(pcts) if pcts else None
        res["mean_spearman"] = nanmean(rhos) if rhos else None
        res["mean_qwk"] = nanmean(qwks) if qwks else None
        if ftype in ("score", "count"):
            icc21, icc2k = icc_two_way(Mc)
            res["icc21"], res["icc2k"] = icc21, icc2k
        return res, pairwise

    def run_set(ids, tag):
        per_field, pairwise_all = [], []
        for f in ALL_FIELDS:
            r, pw = compute(f, ids)
            per_field.append(r); pairwise_all.extend(pw)
        # structural vs judgment roll-ups (mean alpha)
        def grp_alpha(group):
            a = [r["krippendorff_alpha"] for r in per_field
                 if r["field"] in group and r["krippendorff_alpha"] == r["krippendorff_alpha"]]
            return float(np.mean(a)) if a else None
        def grp_gwet(group):
            a = []
            for r in per_field:
                if r["field"] not in group:
                    continue
                g = r["gwet_ac2"] if r["gwet_ac2"] is not None else r["gwet_ac1"]
                if g is not None and g == g:
                    a.append(g)
            return float(np.mean(a)) if a else None
        rollup = {"n_paper_ids": len(ids),
                  "mean_alpha_structural": grp_alpha(STRUCTURAL),
                  "mean_alpha_judgment": grp_alpha(JUDGMENT),
                  "mean_alpha_all": grp_alpha(set(ALL_FIELDS)),
                  "mean_gwet_structural": grp_gwet(STRUCTURAL),
                  "mean_gwet_judgment": grp_gwet(JUDGMENT),
                  "mean_gwet_all": grp_gwet(set(ALL_FIELDS))}
        return {"tag": tag, "per_field": per_field,
                "pairwise": pairwise_all, "rollup": rollup}

    results = {"models": models, "n_intersection": len(common),
               "full": run_set(common, "full")}
    if subset_ids:
        results["subset"] = run_set(subset_ids, "subset")

    # ----- write per-field CSV (full set) ----- #
    pf_path = os.path.join(out_dir, "agreement_per_field.csv")
    with open(pf_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "type", "n_units", "krippendorff_alpha",
                    "gwet_ac1", "gwet_ac2",
                    "mean_pct_agreement", "mean_spearman", "mean_qwk", "icc21", "icc2k"])
        for r in results["full"]["per_field"]:
            w.writerow([r["field"], r["type"], r["n_units"],
                        _f(r["krippendorff_alpha"]),
                        _f(r["gwet_ac1"]), _f(r["gwet_ac2"]),
                        _f(r["mean_pct_agreement"]),
                        _f(r["mean_spearman"]), _f(r["mean_qwk"]),
                        _f(r["icc21"]), _f(r["icc2k"])])

    # ----- write pairwise CSV (full set) ----- #
    pw_path = os.path.join(out_dir, "pairwise_agreement.csv")
    with open(pw_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "model_a", "model_b", "metric", "value"])
        for row in results["full"]["pairwise"]:
            w.writerow([row[0], row[1], row[2], row[3], _f(row[4])])

    # ----- JSON ----- #
    js_path = os.path.join(out_dir, "agreement_summary.json")
    with open(js_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ----- report ----- #
    rp_path = os.path.join(out_dir, "report.txt")
    with open(rp_path, "w", encoding="utf-8") as fh:
        fh.write("Task 2 / Stage 4b - Cross-Model Agreement (InternVL EXCLUDED)\n")
        fh.write("Models: %s\n" % ", ".join(models))
        fh.write("Intersection paper_ids: %d\n\n" % len(common))
        _write_block(fh, "FULL SET", results["full"])
        if subset_ids:
            fh.write("\n")
            _write_block(fh, "STRATIFIED SUBSET", results["subset"])

    print("\n[done] wrote:")
    for p in (pf_path, pw_path, js_path, rp_path):
        print("   " + p)


def _f(x):
    return "" if x is None or (isinstance(x, float) and x != x) else ("%.4f" % x)


def _write_block(fh, title, block):
    ru = block["rollup"]
    fh.write("=== %s (n=%d) ===\n" % (title, ru["n_paper_ids"]))
    fh.write("mean alpha  structural=%s  judgment=%s  all=%s\n"
             % (_f(ru["mean_alpha_structural"]), _f(ru["mean_alpha_judgment"]),
                _f(ru["mean_alpha_all"])))
    fh.write("mean gwet   structural=%s  judgment=%s  all=%s\n\n"
             % (_f(ru["mean_gwet_structural"]), _f(ru["mean_gwet_judgment"]),
                _f(ru["mean_gwet_all"])))
    fh.write("%-28s %-9s %6s %7s %7s %7s %7s %7s %7s %7s %7s\n"
             % ("field", "type", "n", "alpha", "ac1", "ac2", "%agr", "rho", "qwk", "icc1", "iccK"))
    for r in block["per_field"]:
        fh.write("%-28s %-9s %6d %7s %7s %7s %7s %7s %7s %7s %7s\n"
                 % (r["field"], r["type"], r["n_units"],
                    _f(r["krippendorff_alpha"]),
                    _f(r["gwet_ac1"]), _f(r["gwet_ac2"]),
                    _f(r["mean_pct_agreement"]),
                    _f(r["mean_spearman"]), _f(r["mean_qwk"]),
                    _f(r["icc21"]), _f(r["icc2k"])))


if __name__ == "__main__":
    main()