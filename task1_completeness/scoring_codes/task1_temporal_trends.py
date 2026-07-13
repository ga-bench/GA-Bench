#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 — Temporal Trend Analysis of Completeness (provisional / pre-human-validation).

Mirrors the Task 2 Stage 7 design, but for completeness instead of readability.

Reads the Task 1 scoring output (scores_all_long.csv), joins publication YEAR (and
confounds: domain, publisher, journal) by paper_id from the GA index CSV, and reports,
per publication year and per method (model x variant, plus naive baseline):

  * mean / median / SD / 95% CI of completeness C
  * mean level (0-4) and % at each level band (low: 0-1, mid: 2, high: 3-4)
  * RAW yearly trends and CONFOUND-ADJUSTED yearly trends

ADJUSTED trends use the same approach as Task 2 Stage 7: OLS residualization. C is
regressed on the confounds ONLY (domain + publisher, one-hot; YEAR EXCLUDED), then
adjusted_C = grand_mean + residual. The yearly mean of adjusted_C therefore removes
between-year differences attributable to confound composition. This is DESCRIPTIVE;
any inferential model is a later step. Use ASSOCIATION language only.

Naive baseline has only `level` (no C); it is trended on level alone.

Pure numpy (no pandas). matplotlib is used only for figures (guarded import so the
tables still run on a headless node without matplotlib).

Outputs (under <OUTPUT_DIR>):
  trends/
    task1_temporal_<method>.csv         per-year table for each method
    task1_temporal_all_methods.csv      long table, all methods stacked
    task1_temporal_summary.json         slopes + n per method
    figures/
      C_by_year_<method>.png            raw vs adjusted mean C
      level_by_year_<method>.png        mean level raw vs adjusted
      C_by_year_all_variants.png        A vs B overlay per model (raw)

Usage:
  python3 task1_temporal_trends.py
  python3 task1_temporal_trends.py --min-year 2018 --year-floor-n 30
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Paths (match the Task 1 scoring script)
# ----------------------------------------------------------------------------
PROJECT_BASE = Path("./task1_completeness_awq")
SCORES_LONG = PROJECT_BASE / "task1_scores" / "scores_all_long.csv"
OUTPUT_DIR = PROJECT_BASE / "task1_scores"
TRENDS_DIR = OUTPUT_DIR / "trends"
FIG_DIR = TRENDS_DIR / "figures"

GA_INDEX_CSV = Path(
    "./"
    "task2_readability/output/stage1_preprocessing/index/stage1_ga_index.csv"
)

# paper_id column in both CSVs.
ID_COL = "paper_id"

# Candidate column names to auto-detect in the GA index (first match wins).
YEAR_CANDIDATES = ["year", "pub_year", "publication_year", "pubyear"]
DOMAIN_CANDIDATES = ["domain", "field", "subject", "scimago_domain", "area"]
PUBLISHER_CANDIDATES = ["publisher", "pub"]
JOURNAL_CANDIDATES = ["journal", "journal_name", "publicationName", "container_title"]


# ----------------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------------
def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return (r.fieldnames or []), rows


def pick_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def to_float(x: str) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


def to_year(x: str) -> Optional[int]:
    if not x:
        return None
    s = str(x).strip()
    # accept "2024", "2024-01-01", "2024.0"
    for token in (s[:4],):
        try:
            y = int(token)
            if 1900 <= y <= 2100:
                return y
        except ValueError:
            pass
    return None


# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------
def mean_ci(vals: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (mean, sd, ci_lo, ci_hi) with 95% normal-approx CI."""
    n = len(vals)
    if n == 0:
        return (float("nan"),) * 4
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    return m, sd, m - 1.96 * se, m + 1.96 * se


def ols_residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return grand_mean + residual of y ~ X (X already includes intercept col)."""
    # least squares
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    return float(np.mean(y)) + resid


def onehot(cats: List[str]) -> Tuple[np.ndarray, List[str]]:
    """One-hot encode (drop-first) a list of category strings."""
    uniq = sorted(set(cats))
    if len(uniq) <= 1:
        return np.zeros((len(cats), 0)), []
    drop = uniq[0]
    keep = uniq[1:]
    idx = {c: i for i, c in enumerate(keep)}
    M = np.zeros((len(cats), len(keep)))
    for r, c in enumerate(cats):
        if c in idx:
            M[r, idx[c]] = 1.0
    return M, keep


def linreg_slope(years: np.ndarray, vals: np.ndarray) -> Optional[float]:
    """OLS slope of vals ~ years (per-year change)."""
    if len(years) < 2:
        return None
    x = years.astype(float)
    A = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
    return float(beta[1])


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------
LEVEL_BANDS = [("low_0_1", lambda l: l <= 1),
               ("mid_2", lambda l: l == 2),
               ("high_3_4", lambda l: l >= 3)]


def build_meta_index() -> Dict[str, Dict[str, Any]]:
    """paper_id -> {year, domain, publisher, journal}."""
    fields, rows = read_csv_rows(GA_INDEX_CSV)
    ycol = pick_col(fields, YEAR_CANDIDATES)
    dcol = pick_col(fields, DOMAIN_CANDIDATES)
    pcol = pick_col(fields, PUBLISHER_CANDIDATES)
    jcol = pick_col(fields, JOURNAL_CANDIDATES)
    idc = pick_col(fields, [ID_COL]) or ID_COL
    print(f"  GA index columns -> id:{idc} year:{ycol} domain:{dcol} publisher:{pcol} journal:{jcol}")
    meta: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pid = row.get(idc, "")
        if not pid:
            continue
        meta[pid] = {
            "year": to_year(row.get(ycol, "")) if ycol else None,
            "domain": (row.get(dcol, "") or "NA") if dcol else "NA",
            "publisher": (row.get(pcol, "") or "NA") if pcol else "NA",
            "journal": (row.get(jcol, "") or "NA") if jcol else "NA",
        }
    return meta


def load_scores() -> List[Dict[str, str]]:
    _, rows = read_csv_rows(SCORES_LONG)
    return rows


def method_key(row: Dict[str, str]) -> str:
    m = row.get("method", "")
    model = row.get("model", "")
    variant = row.get("variant", "")
    if m == "naive":
        return f"naive_{model}"
    return f"srp_{model}_variant{variant}"


def analyze(min_year: int, year_floor_n: int) -> None:
    TRENDS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GA index metadata...")
    meta = build_meta_index()
    print(f"  {len(meta)} papers in metadata index")

    print("Loading Task 1 scores (scores_all_long.csv)...")
    rows = load_scores()
    print(f"  {len(rows)} score rows")

    # group by method
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_nojoin = 0
    for row in rows:
        pid = row.get("doi_safe") or row.get(ID_COL) or row.get("doi")
        md = meta.get(pid)
        if md is None or md.get("year") is None:
            n_nojoin += 1
            continue
        C = to_float(row.get("C_completeness", ""))
        lvl = to_float(row.get("level", ""))
        if lvl is None:
            continue
        by_method[method_key(row)].append({
            "pid": pid, "year": md["year"], "domain": md["domain"],
            "publisher": md["publisher"], "journal": md["journal"],
            "C": C, "level": lvl,
        })
    print(f"  joined; {n_nojoin} rows dropped (no year/metadata)")

    all_long_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"min_year": min_year, "year_floor_n": year_floor_n, "methods": {}}

    for method, recs in sorted(by_method.items()):
        recs = [r for r in recs if r["year"] >= min_year]
        if not recs:
            continue
        has_C = any(r["C"] is not None for r in recs)

        years = np.array([r["year"] for r in recs])
        levels = np.array([r["level"] for r in recs], dtype=float)
        Cvals = np.array([r["C"] if r["C"] is not None else np.nan for r in recs], dtype=float)

        # adjusted C: residualize on domain+publisher (year excluded)
        adjC = None
        if has_C and not np.all(np.isnan(Cvals)):
            mask = ~np.isnan(Cvals)
            dom = [recs[i]["domain"] for i in range(len(recs)) if mask[i]]
            pub = [recs[i]["publisher"] for i in range(len(recs)) if mask[i]]
            Dm, _ = onehot(dom)
            Pm, _ = onehot(pub)
            intercept = np.ones((mask.sum(), 1))
            X = np.column_stack([intercept, Dm, Pm]) if (Dm.size or Pm.size) else intercept
            adj = ols_residualize(Cvals[mask], X)
            adjC = np.full_like(Cvals, np.nan)
            adjC[mask] = adj

        # adjusted level similarly
        domL = [r["domain"] for r in recs]
        pubL = [r["publisher"] for r in recs]
        DmL, _ = onehot(domL)
        PmL, _ = onehot(pubL)
        interceptL = np.ones((len(recs), 1))
        XL = np.column_stack([interceptL, DmL, PmL]) if (DmL.size or PmL.size) else interceptL
        adjLevel = ols_residualize(levels, XL)

        # per-year aggregation
        per_year: Dict[int, Dict[str, Any]] = {}
        for y in sorted(set(years.tolist())):
            yi = years == y
            n = int(yi.sum())
            if n < year_floor_n:
                continue
            row_out: Dict[str, Any] = {"method": method, "year": int(y), "n": n}
            # level
            lm, lsd, llo, lhi = mean_ci(levels[yi])
            row_out.update({"mean_level": round(lm, 4), "sd_level": round(lsd, 4),
                            "level_ci_lo": round(llo, 4), "level_ci_hi": round(lhi, 4),
                            "mean_level_adj": round(float(np.mean(adjLevel[yi])), 4)})
            for name, fn in LEVEL_BANDS:
                frac = float(np.mean([1.0 if fn(l) else 0.0 for l in levels[yi]]))
                row_out[f"pct_{name}"] = round(100 * frac, 2)
            # C
            if has_C:
                cvals_y = Cvals[yi]
                cvals_y = cvals_y[~np.isnan(cvals_y)]
                if len(cvals_y):
                    cm, csd, clo, chi = mean_ci(cvals_y)
                    row_out.update({"mean_C": round(cm, 4), "median_C": round(float(np.median(cvals_y)), 4),
                                    "sd_C": round(csd, 4), "C_ci_lo": round(clo, 4), "C_ci_hi": round(chi, 4)})
                if adjC is not None:
                    adj_y = adjC[yi]
                    adj_y = adj_y[~np.isnan(adj_y)]
                    if len(adj_y):
                        row_out["mean_C_adj"] = round(float(np.mean(adj_y)), 4)
            per_year[int(y)] = row_out
            all_long_rows.append(row_out)

        # write per-method csv
        if per_year:
            keys = sorted({k for r in per_year.values() for k in r.keys()})
            # stable ordering
            head = ["method", "year", "n", "mean_C", "median_C", "sd_C", "C_ci_lo", "C_ci_hi",
                    "mean_C_adj", "mean_level", "sd_level", "level_ci_lo", "level_ci_hi",
                    "mean_level_adj", "pct_low_0_1", "pct_mid_2", "pct_high_3_4"]
            head = [h for h in head if h in keys] + [k for k in keys if k not in head]
            out_csv = TRENDS_DIR / f"task1_temporal_{method}.csv"
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=head, extrasaction="ignore")
                w.writeheader()
                for y in sorted(per_year):
                    w.writerow(per_year[y])

            # slopes over year (raw)
            yr_arr = np.array(sorted(per_year.keys()))
            if has_C and all("mean_C" in per_year[y] for y in yr_arr):
                cser = np.array([per_year[y]["mean_C"] for y in yr_arr])
                slope_C = linreg_slope(yr_arr, cser)
            else:
                slope_C = None
            lser = np.array([per_year[y]["mean_level"] for y in yr_arr])
            slope_L = linreg_slope(yr_arr, lser)
            summary["methods"][method] = {
                "n_records": len(recs), "n_years": len(per_year),
                "years": [int(y) for y in yr_arr],
                "slope_mean_C_per_year": (round(slope_C, 5) if slope_C is not None else None),
                "slope_mean_level_per_year": (round(slope_L, 5) if slope_L is not None else None),
                "csv": str(out_csv),
            }

            _plot_method(method, per_year, has_C)

        print(f"  {method}: {len(recs)} recs, {len(per_year)} year-points")

    # all-methods long
    if all_long_rows:
        keys = sorted({k for r in all_long_rows for k in r.keys()})
        head = ["method", "year", "n"] + [k for k in keys if k not in ("method", "year", "n")]
        with open(TRENDS_DIR / "task1_temporal_all_methods.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=head, extrasaction="ignore")
            w.writeheader()
            for r in all_long_rows:
                w.writerow(r)

    with open(TRENDS_DIR / "task1_temporal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_variant_overlay(by_method, min_year, year_floor_n)

    print("\nDone.")
    print(f"Tables + figures under: {TRENDS_DIR}")
    for m, s in summary["methods"].items():
        print(f"  {m}: slope_C/yr={s['slope_mean_C_per_year']} slope_level/yr={s['slope_mean_level_per_year']}")


# ----------------------------------------------------------------------------
# Plotting (guarded)
# ----------------------------------------------------------------------------
def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"  [warn] matplotlib unavailable ({e}); skipping figures.")
        return None


def _plot_method(method: str, per_year: Dict[int, Dict[str, Any]], has_C: bool) -> None:
    plt = _get_plt()
    if plt is None:
        return
    yrs = sorted(per_year.keys())

    if has_C and all("mean_C" in per_year[y] for y in yrs):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        raw = [per_year[y]["mean_C"] for y in yrs]
        ax.plot(yrs, raw, "o-", label="raw mean C", color="#1C7293", linewidth=2)
        if all("mean_C_adj" in per_year[y] for y in yrs):
            adj = [per_year[y]["mean_C_adj"] for y in yrs]
            ax.plot(yrs, adj, "s--", label="adjusted mean C", color="#2A9D8F", linewidth=2)
        lo = [per_year[y].get("C_ci_lo") for y in yrs]
        hi = [per_year[y].get("C_ci_hi") for y in yrs]
        if all(v is not None for v in lo + hi):
            ax.fill_between(yrs, lo, hi, color="#1C7293", alpha=0.12)
        ax.set_title(f"Completeness C by year — {method}")
        ax.set_xlabel("Publication year"); ax.set_ylabel("Mean C")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(FIG_DIR / f"C_by_year_{method}.png", dpi=140); plt.close(fig)

    # level plot
    fig, ax = plt.subplots(figsize=(7, 4.2))
    raw = [per_year[y]["mean_level"] for y in yrs]
    ax.plot(yrs, raw, "o-", label="raw mean level", color="#16243A", linewidth=2)
    if all("mean_level_adj" in per_year[y] for y in yrs):
        adj = [per_year[y]["mean_level_adj"] for y in yrs]
        ax.plot(yrs, adj, "s--", label="adjusted mean level", color="#E9C46A", linewidth=2)
    ax.set_ylim(0, 4)
    ax.set_title(f"Mean completeness level by year — {method}")
    ax.set_xlabel("Publication year"); ax.set_ylabel("Mean level (0-4)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / f"level_by_year_{method}.png", dpi=140); plt.close(fig)


def _plot_variant_overlay(by_method: Dict[str, List[Dict[str, Any]]], min_year: int, floor_n: int) -> None:
    """Overlay Variant A vs B raw yearly means per model, for both C (0-1) and level (0-4)."""
    plt = _get_plt()
    if plt is None:
        return
    models = set()
    for m in by_method:
        if m.startswith("srp_") and "_variant" in m:
            models.add(m.split("_variant")[0])

    # metric configs: (record-field, y-label, filename-stem, y-limits or None)
    metrics = [
        ("C", "Mean C", "C_by_year", None),
        ("level", "Mean level (0-4)", "level_by_year", (0, 4)),
    ]

    for model in sorted(models):
        model_short = model.replace("srp_", "")
        for field, ylabel, stem, ylim in metrics:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            plotted = False
            for variant, color in (("A", "#1C7293"), ("B", "#2A9D8F")):
                key = f"{model}_variant{variant}"
                recs = [r for r in by_method.get(key, [])
                        if r["year"] >= min_year and r.get(field) is not None]
                if not recs:
                    continue
                yr_to_vals = defaultdict(list)
                for r in recs:
                    yr_to_vals[r["year"]].append(r[field])
                yrs = sorted(y for y in yr_to_vals if len(yr_to_vals[y]) >= floor_n)
                if not yrs:
                    continue
                means = [float(np.mean(yr_to_vals[y])) for y in yrs]
                ax.plot(yrs, means, "o-", label=f"Variant {variant}", color=color, linewidth=2)
                plotted = True
            if plotted:
                nice = "Completeness C" if field == "C" else "Completeness level"
                ax.set_title(f"{nice} by year — {model_short}: A vs B")
                ax.set_xlabel("Publication year"); ax.set_ylabel(ylabel)
                if ylim:
                    ax.set_ylim(*ylim)
                ax.grid(True, alpha=0.3); ax.legend()
                fig.tight_layout()
                fig.savefig(FIG_DIR / f"{stem}_{model_short}_A_vs_B.png", dpi=140)
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 1 temporal trend analysis (completeness)")
    ap.add_argument("--min-year", type=int, default=2010, help="ignore years before this")
    ap.add_argument("--year-floor-n", type=int, default=20,
                    help="minimum papers in a year to include that year-point")
    args = ap.parse_args()

    print("=" * 70)
    print("Task 1 temporal trend analysis (completeness) — provisional")
    print(f"Scores:  {SCORES_LONG}")
    print(f"Index:   {GA_INDEX_CSV}")
    print(f"Output:  {TRENDS_DIR}")
    print("=" * 70)
    analyze(args.min_year, args.year_floor_n)


if __name__ == "__main__":
    main()