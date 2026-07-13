#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 / Stage 7b — Per-Domain Temporal Trend Analysis (unweighted baseline).

Companion to Stage 7. Instead of one global yearly trend, this runs the SAME yearly
trend logic SEPARATELY WITHIN EACH DOMAIN, so you can see whether readability moves
inside individual fields even when the pooled trend is flat (i.e. whether within-domain
trends are larger and cancel across domains).

For every domain (primary_domain in the Stage-5 CSV), per publication year:
  * mean / SD / 95% CI of R_overall and each component (R_text, R_visual, R_semantic)
  * per-year descriptive slopes (association, not causal) for overall + 3 components
And across domains:
  * a slope summary table ranking domains by |overall slope|
  * one overlay figure of R_overall-by-year for the top-N domains
  * one small-multiples figure of components per domain

Pure numpy (no pandas). matplotlib guarded. DESCRIPTIVE / association language only.
Provisional until human calibration (weights/thresholds).

Outputs (under OUTPUT_DIR/<model>/):
  tables/per_domain_yearly.csv           long: domain,year,n,mean_overall,ci..,components
  tables/per_domain_slopes.csv           one row per domain: slopes + n + n_years
  figures/overall_by_year_top_domains.png
  figures/components_by_domain.png
  summary.json
  report.txt

Usage:
  python3 stage7b_per_domain_trends.py --min-year 2021 \
      --output-dir .../output/stage7b_per_domain_trends_2021_2026
  python3 stage7b_per_domain_trends.py --models qwen3_vl_32b --min-domain-year-n 30
"""

import os
import csv
import json
import math
import argparse
from collections import OrderedDict, defaultdict

import numpy as np

# --------------------------------------------------------------------------- #
# Paths (mirror Stage 7)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = "./task2_readability"
OUTPUT_DIR = "output/stage7b_per_domain_trends"

MODEL_STAGE5 = OrderedDict([
    ("qwen3_vl_32b",
     "output/stage5_readability_scoring/features/stage5_readability_scores.csv"),
    ("gemma_3_27b",
     "output/stage5_readability_scoring_gemma_3_27b/features/stage5_readability_scores.csv"),
    ("mistral_small_24b",
     "output/stage5_readability_scoring_mistral_small_24b/features/stage5_readability_scores.csv"),
])

YEAR_MIN, YEAR_MAX = 1995, 2026

CANDS = {
    "paper_id":   ["paper_id"],
    "year":       ["publication_year", "year", "pub_year"],
    "R_overall":  ["R_overall_unweighted", "R_overall", "readability_overall", "R_G", "r_overall"],
    "R_text":     ["R_text", "r_text", "readability_text"],
    "R_visual":   ["R_visual", "r_visual", "readability_visual"],
    "R_semantic": ["R_semantic", "r_semantic", "readability_semantic"],
    "domain":     ["primary_domain", "domain", "primary_subject_area", "subject_area"],
}


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def read_csv_dicts(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(path, encoding="utf-8") as fh:
        header = csv.DictReader(fh).fieldnames
    return rows, header


def pick(header, names):
    low = {h.lower(): h for h in header}
    for n in names:
        if n in header:
            return n
        if n.lower() in low:
            return low[n.lower()]
    return None


def to_float(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", "").replace("%", "")
    if s == "" or s.lower() in ("nan", "na", "none", "null", "n/a"):
        return None
    try:
        v = float(s)
        return v if v == v else None
    except (ValueError, TypeError):
        return None


def to_year(x):
    try:
        y = int(round(float(str(x).strip())))
        return y if YEAR_MIN <= y <= YEAR_MAX else None
    except (ValueError, TypeError):
        return None


def first_token(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return s.split(";")[0].strip()


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def mean_ci(vals):
    v = np.array([x for x in vals if x is not None and x == x], float)
    n = len(v)
    if n == 0:
        return None
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return dict(n=n, mean=float(v.mean()), sd=sd,
                ci_lo=float(v.mean() - ci), ci_hi=float(v.mean() + ci))


def lin_slope(years, vals):
    x = np.array([years[i] for i in range(len(years)) if vals[i] is not None and vals[i] == vals[i]], float)
    y = np.array([vals[i] for i in range(len(years)) if vals[i] is not None and vals[i] == vals[i]], float)
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=PROJECT_ROOT)
    ap.add_argument("--stage5", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--min-year", type=int, default=None,
                    help="restrict to publication years >= this (e.g. 2021)")
    ap.add_argument("--max-year", type=int, default=None,
                    help="restrict to publication years <= this (e.g. 2026)")
    ap.add_argument("--min-domain-year-n", type=int, default=20,
                    help="within a domain, drop years with fewer than this many GAs")
    ap.add_argument("--min-domain-n", type=int, default=100,
                    help="skip domains with fewer than this many GAs total (after year window)")
    ap.add_argument("--top-domains", type=int, default=8,
                    help="how many domains to overlay in the overall-by-year figure")
    ap.add_argument("--models", default=None,
                    help="comma-separated model keys (default: all). Keys: " + ",".join(MODEL_STAGE5.keys()))
    args = ap.parse_args()
    root = args.project_root

    if args.stage5:
        name = (args.models.split(",")[0].strip() if args.models else "custom")
        targets = [(name, args.stage5)]
    else:
        keys = ([m.strip() for m in args.models.split(",")] if args.models
                else list(MODEL_STAGE5.keys()))
        targets = [(k, os.path.join(root, MODEL_STAGE5[k])) for k in keys if k in MODEL_STAGE5]

    base_out = args.output_dir or os.path.join(root, OUTPUT_DIR)
    ran = 0
    for model_key, s5 in targets:
        out_dir = os.path.join(base_out, model_key)
        if not os.path.isfile(s5):
            print("[skip] %-18s no Stage-5 CSV at %s" % (model_key, s5))
            continue
        print("\n========== MODEL: %s ==========" % model_key)
        run_for_model(s5, out_dir, args)
        ran += 1
    if ran == 0:
        raise SystemExit("ERROR: no model produced output. Check Stage-5 CSV paths.")
    print("\n[all done] models analyzed: %d -> %s/<model>/" % (ran, base_out))


def run_for_model(s5, out_dir, args):
    tbl_dir = os.path.join(out_dir, "tables")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(tbl_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    rows, header = read_csv_dicts(s5)
    col = {k: pick(header, v) for k, v in CANDS.items()}
    if not col["year"] or not col["R_overall"] or not col["domain"]:
        raise SystemExit("ERROR: need year, R_overall, domain in %s (found: %s)" % (s5, col))
    print("[load] rows=%d  cols: year->%s R_overall->%s domain->%s"
          % (len(rows), col["year"], col["R_overall"], col["domain"]))
    if args.min_year is not None or args.max_year is not None:
        print("[window] years %s..%s"
              % (args.min_year if args.min_year is not None else "min",
                 args.max_year if args.max_year is not None else "max"))

    # domain -> year -> lists
    dom_year = defaultdict(lambda: defaultdict(lambda: {"ov": [], "t": [], "v": [], "s": []}))
    dom_total = defaultdict(int)
    n_win_drop = 0
    for r in rows:
        yr = to_year(r.get(col["year"]))
        ov = to_float(r.get(col["R_overall"]))
        if yr is None or ov is None:
            continue
        if args.min_year is not None and yr < args.min_year:
            n_win_drop += 1; continue
        if args.max_year is not None and yr > args.max_year:
            n_win_drop += 1; continue
        dom = first_token(r.get(col["domain"])) or "unknown"
        d = dom_year[dom][yr]
        d["ov"].append(ov)
        d["t"].append(to_float(r.get(col["R_text"])) if col["R_text"] else None)
        d["v"].append(to_float(r.get(col["R_visual"])) if col["R_visual"] else None)
        d["s"].append(to_float(r.get(col["R_semantic"])) if col["R_semantic"] else None)
        dom_total[dom] += 1
    if args.min_year is not None or args.max_year is not None:
        print("[window] dropped %d GAs outside window" % n_win_drop)

    # build per-domain yearly tables + slopes
    long_rows = []
    slope_rows = []
    domain_series = {}   # domain -> (yrs, overall means) for figure
    comp_series = {}     # domain -> dict of component yearly means

    for dom in sorted(dom_total, key=lambda d: -dom_total[d]):
        if dom_total[dom] < args.min_domain_n:
            continue
        yrs_all = sorted(dom_year[dom].keys())
        yrs, ov_m, t_m, v_m, s_m = [], [], [], [], []
        for yr in yrs_all:
            d = dom_year[dom][yr]
            if len(d["ov"]) < args.min_domain_year_n:
                continue
            o = mean_ci(d["ov"]); tt = mean_ci(d["t"]); vv = mean_ci(d["v"]); ss = mean_ci(d["s"])
            yrs.append(yr)
            ov_m.append(o["mean"] if o else float("nan"))
            t_m.append(tt["mean"] if tt else float("nan"))
            v_m.append(vv["mean"] if vv else float("nan"))
            s_m.append(ss["mean"] if ss else float("nan"))
            long_rows.append({
                "domain": dom, "year": yr, "n": o["n"] if o else 0,
                "mean_overall": _r(o), "ci_lo": _r(o, "ci_lo"), "ci_hi": _r(o, "ci_hi"),
                "mean_R_text": _r(tt), "mean_R_visual": _r(vv), "mean_R_semantic": _r(ss),
            })
        if len(yrs) < 2:
            continue
        slope_rows.append({
            "domain": dom, "n_total": dom_total[dom], "n_years": len(yrs),
            "year_min": yrs[0], "year_max": yrs[-1],
            "slope_overall_per_year": round(lin_slope(yrs, ov_m), 6),
            "slope_R_text_per_year": round(lin_slope(yrs, t_m), 6),
            "slope_R_visual_per_year": round(lin_slope(yrs, v_m), 6),
            "slope_R_semantic_per_year": round(lin_slope(yrs, s_m), 6),
        })
        domain_series[dom] = (yrs, ov_m)
        comp_series[dom] = {"yrs": yrs, "t": t_m, "v": v_m, "s": s_m}

    # rank by |overall slope|
    slope_rows.sort(key=lambda r: -abs(r["slope_overall_per_year"])
                    if r["slope_overall_per_year"] == r["slope_overall_per_year"] else 0)

    # ---- write tables ----
    with open(os.path.join(tbl_dir, "per_domain_yearly.csv"), "w", newline="") as fh:
        cols = ["domain", "year", "n", "mean_overall", "ci_lo", "ci_hi",
                "mean_R_text", "mean_R_visual", "mean_R_semantic"]
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(long_rows)

    with open(os.path.join(tbl_dir, "per_domain_slopes.csv"), "w", newline="") as fh:
        cols = ["domain", "n_total", "n_years", "year_min", "year_max",
                "slope_overall_per_year", "slope_R_text_per_year",
                "slope_R_visual_per_year", "slope_R_semantic_per_year"]
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(slope_rows)

    # ---- figures ----
    made = _make_figures(fig_dir, domain_series, comp_series, slope_rows, args.top_domains)

    # ---- summary + report ----
    summary = {
        "n_domains_reported": len(slope_rows),
        "min_year": args.min_year, "max_year": args.max_year,
        "min_domain_year_n": args.min_domain_year_n, "min_domain_n": args.min_domain_n,
        "slopes_by_domain": slope_rows, "figures": made,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write("Task 2 / Stage 7b - Per-Domain Temporal Trends (unweighted baseline)\n")
        if args.min_year is not None or args.max_year is not None:
            fh.write("Year window: %s..%s\n"
                     % (args.min_year if args.min_year is not None else "min",
                        args.max_year if args.max_year is not None else "max"))
        fh.write("Domains reported: %d (min_domain_n=%d, min_domain_year_n=%d)\n\n"
                 % (len(slope_rows), args.min_domain_n, args.min_domain_year_n))
        fh.write("Per-domain overall slope (ASSOCIATION, per year), ranked by |slope|:\n")
        fh.write("%-32s %8s %7s  %10s %10s %10s %10s\n" %
                 ("domain", "n", "years", "overall", "R_text", "R_visual", "R_sem"))
        for s in slope_rows:
            fh.write("%-32s %8d %7d  %10s %10s %10s %10s\n" %
                     (s["domain"][:32], s["n_total"], s["n_years"],
                      _f(s["slope_overall_per_year"]), _f(s["slope_R_text_per_year"]),
                      _f(s["slope_R_visual_per_year"]), _f(s["slope_R_semantic_per_year"])))
        fh.write("\nNote: within-domain slopes may exceed the pooled slope and cancel across\n")
        fh.write("domains. Descriptive only; provisional until human calibration.\n")
        fh.write("\nFigures: %s\n" % fig_dir)
        if not made:
            fh.write("(matplotlib unavailable - tables/JSON written, figures skipped.)\n")

    print("[done] tables -> %s" % tbl_dir)
    print("[done] figures -> %s (%d)" % (fig_dir, len(made)))
    print("[done] report -> %s" % os.path.join(out_dir, "report.txt"))


def _r(d, key="mean"):
    return round(d[key], 6) if d else ""


def _f(x):
    return "" if x is None or (isinstance(x, float) and x != x) else ("%.5f" % x)


def _make_figures(fig_dir, domain_series, comp_series, slope_rows, top_n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[warn] matplotlib unavailable (%s); skipping figures." % e)
        return []
    made = []

    # 1) overall-by-year overlay for top-N domains (by |slope|)
    top = [s["domain"] for s in slope_rows[:top_n] if s["domain"] in domain_series]
    if top:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for dom in top:
            yrs, ov = domain_series[dom]
            ax.plot(yrs, ov, "-o", label=dom[:26])
        ax.set_xlabel("publication year"); ax.set_ylabel("mean R_overall")
        ax.set_title("Overall readability by year, per domain (top by |slope|)")
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "overall_by_year_top_domains.png"), dpi=150)
        plt.close(fig); made.append("overall_by_year_top_domains.png")

    # 2) components small-multiples for top-N domains
    if top:
        k = len(top)
        ncol = 2 if k > 1 else 1
        nrow = int(math.ceil(k / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.0 * nrow), squeeze=False)
        for i, dom in enumerate(top):
            ax = axes[i // ncol][i % ncol]
            c = comp_series[dom]
            ax.plot(c["yrs"], c["t"], "-o", label="R_text", markersize=3)
            ax.plot(c["yrs"], c["v"], "-o", label="R_visual", markersize=3)
            ax.plot(c["yrs"], c["s"], "-o", label="R_semantic", markersize=3)
            ax.set_title(dom[:30], fontsize=9); ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7)
        for j in range(k, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Readability components by year, per domain")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "components_by_domain.png"), dpi=150)
        plt.close(fig); made.append("components_by_domain.png")

    return made


if __name__ == "__main__":
    main()