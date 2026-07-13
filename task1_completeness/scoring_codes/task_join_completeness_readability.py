#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Completeness x Readability Join  (Task 1  x  Task 2)
====================================================

Joins Task 1 completeness scores to Task 2 readability scores by `paper_id`, for one
chosen model on each side (default: Qwen, Task 1 Variant B  x  Qwen Stage 5), and:

  * computes Spearman + Pearson correlation between completeness C and readability
    R_overall (and between level and readability);
  * builds the 2x2 quadrant table (high/low completeness x readable/unreadable) using
    medians as the split (overridable), with counts, percentages, and per-quadrant means;
  * writes a joined per-paper CSV for later mixed-effects modeling.

DESCRIPTIVE / associational only. This is the raw-metric join; human-validated /
calibrated versions come after annotation.

Inputs
------
Task 1 : task1_scores/scores_all_long.csv
         columns: method, model, variant, doi, doi_safe, level, C_completeness, ...
Task 2 : stage5_readability_scoring/features/stage5_readability_scores.csv
         columns: paper_id, model, ..., R_overall_unweighted, readability_category, ...

Join key: paper_id  (Task 1 `doi_safe` == Task 2 `paper_id`)

Outputs (under OUTPUT_DIR)
--------------------------
  joined_completeness_readability.csv     one row per paper (C, level, R, quadrant, meta)
  quadrant_table.csv                      2x2 counts / pct / means
  correlations.json                       Spearman/Pearson + n + split values
  figures/quadrant_scatter.png            C vs R scatter with quadrant lines (if matplotlib)

Usage
-----
python3 task_join_completeness_readability.py
python3 task_join_completeness_readability.py \
    --t1-model qwen3_vl_32b --t1-variant B \
    --t2-model qwen3_vl_32b \
    --c-split median --r-split median
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_BASE = Path("./task1_completeness_awq")
T1_SCORES = PROJECT_BASE / "task1_scores" / "scores_all_long.csv"
OUTPUT_DIR = PROJECT_BASE / "task1_scores" / "completeness_readability_join"
FIG_DIR = OUTPUT_DIR / "figures"

T2_SCORES = Path(
    "./task2_readability/"
    "output/stage5_readability_scoring/features/stage5_readability_scores.csv"
)

# Task 2 readability column (this dataset uses the *_unweighted name).
R_COL = "R_overall_unweighted"
R_CAT_COL = "readability_category"
T2_ID = "paper_id"

# ----------------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------------
def read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: Optional[str]) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------------
# Stats (no scipy): Spearman via rank + Pearson
# ----------------------------------------------------------------------------
def _ranks(vals: List[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank, 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    return pearson(_ranks(x), _ranks(y))


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------
def load_task1(model: str, variant: str) -> Dict[str, Dict[str, Any]]:
    """paper_id -> {C, level} for the chosen SRP model+variant."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in read_rows(T1_SCORES):
        if r.get("method") != "srp":
            continue
        if r.get("model") != model or r.get("variant") != variant:
            continue
        pid = r.get("doi_safe") or r.get("doi")
        if not pid:
            continue
        out[pid] = {
            "C": to_float(r.get("C_completeness")),
            "level": to_float(r.get("level")),
        }
    return out


def load_task2(model: str) -> Dict[str, Dict[str, Any]]:
    """paper_id -> {R, R_cat, year, domain, publisher, journal} for chosen model."""
    rows = read_rows(T2_SCORES)
    # if a model column exists and has multiple models, filter; else take all
    has_model = rows and "model" in rows[0]
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if has_model and model and r.get("model") not in ("", model):
            continue
        pid = r.get(T2_ID)
        if not pid:
            continue
        out[pid] = {
            "R": to_float(r.get(R_COL)),
            "R_cat": r.get(R_CAT_COL, ""),
            "year": r.get("publication_year", ""),
            "domain": r.get("primary_domain") or r.get("domain", ""),
            "publisher": r.get("publisher", ""),
            "journal": r.get("journal", ""),
        }
    return out


def split_value(vals: List[float], mode: str) -> float:
    if mode == "median":
        return statistics.median(vals)
    try:
        return float(mode)  # explicit numeric threshold
    except ValueError:
        return statistics.median(vals)


def analyze(args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Completeness x Readability join")
    print(f"  Task 1: {args.t1_model} variant {args.t1_variant}")
    print(f"  Task 2: {args.t2_model}  ({R_COL})")
    print("=" * 70)

    t1 = load_task1(args.t1_model, args.t1_variant)
    t2 = load_task2(args.t2_model)
    print(f"  Task 1 papers: {len(t1)}")
    print(f"  Task 2 papers: {len(t2)}")

    # inner join
    joined: List[Dict[str, Any]] = []
    for pid, a in t1.items():
        b = t2.get(pid)
        if not b:
            continue
        if a["C"] is None or b["R"] is None:
            continue
        joined.append({
            "paper_id": pid, "C": a["C"], "level": a["level"], "R": b["R"],
            "R_cat": b["R_cat"], "year": b["year"], "domain": b["domain"],
            "publisher": b["publisher"], "journal": b["journal"],
        })
    n = len(joined)
    print(f"  joined (both C and R present): {n}")
    if n == 0:
        print("  [error] no overlap — check model/variant names and paper_id keys.")
        return

    Cs = [r["C"] for r in joined]
    Rs = [r["R"] for r in joined]
    Ls = [r["level"] for r in joined if r["level"] is not None]
    Rs_for_L = [r["R"] for r in joined if r["level"] is not None]

    corr = {
        "n": n,
        "spearman_C_R": spearman(Cs, Rs),
        "pearson_C_R": pearson(Cs, Rs),
        "spearman_level_R": spearman(Ls, Rs_for_L) if Ls else None,
        "mean_C": round(sum(Cs) / n, 4),
        "mean_R": round(sum(Rs) / n, 4),
    }

    # quadrant split
    c_split = split_value(Cs, args.c_split)
    r_split = split_value(Rs, args.r_split)
    corr["c_split"] = round(c_split, 4)
    corr["r_split"] = round(r_split, 4)
    corr["c_split_mode"] = args.c_split
    corr["r_split_mode"] = args.r_split

    def quad(c: float, r: float) -> str:
        hi_c = c >= c_split
        hi_r = r >= r_split
        if hi_c and hi_r:
            return "complete_readable"
        if hi_c and not hi_r:
            return "complete_unreadable"
        if not hi_c and hi_r:
            return "incomplete_readable"
        return "incomplete_unreadable"

    counts: Dict[str, int] = {"complete_readable": 0, "complete_unreadable": 0,
                              "incomplete_readable": 0, "incomplete_unreadable": 0}
    sums: Dict[str, Dict[str, float]] = {k: {"C": 0.0, "R": 0.0} for k in counts}
    for r in joined:
        q = quad(r["C"], r["R"])
        r["quadrant"] = q
        counts[q] += 1
        sums[q]["C"] += r["C"]
        sums[q]["R"] += r["R"]

    # write joined csv
    jfields = ["paper_id", "C", "level", "R", "R_cat", "quadrant",
               "year", "domain", "publisher", "journal"]
    with open(OUTPUT_DIR / "joined_completeness_readability.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=jfields, extrasaction="ignore")
        w.writeheader()
        for r in joined:
            w.writerow(r)

    # quadrant table
    with open(OUTPUT_DIR / "quadrant_table.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quadrant", "n", "pct", "mean_C", "mean_R"])
        for k in ["complete_readable", "complete_unreadable", "incomplete_readable", "incomplete_unreadable"]:
            c = counts[k]
            pct = round(100 * c / n, 2)
            mc = round(sums[k]["C"] / c, 4) if c else ""
            mr = round(sums[k]["R"] / c, 4) if c else ""
            w.writerow([k, c, pct, mc, mr])

    with open(OUTPUT_DIR / "correlations.json", "w", encoding="utf-8") as f:
        json.dump({"correlations": corr, "quadrant_counts": counts}, f, indent=2)

    _plot(joined, c_split, r_split)

    # console summary
    print("\n  Correlations:")
    print(f"    Spearman C vs R      : {corr['spearman_C_R']}")
    print(f"    Pearson  C vs R      : {corr['pearson_C_R']}")
    print(f"    Spearman level vs R  : {corr['spearman_level_R']}")
    print(f"  Split: C>={corr['c_split']}  R>={corr['r_split']}")
    print("  Quadrants:")
    for k in ["complete_readable", "complete_unreadable", "incomplete_readable", "incomplete_unreadable"]:
        print(f"    {k:24s}: {counts[k]:6d}  ({round(100*counts[k]/n,1)}%)")
    print(f"\n  Outputs in: {OUTPUT_DIR}")


def _plot(joined, c_split, r_split) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [warn] matplotlib unavailable ({e}); skipping scatter.")
        return
    colors = {
        "complete_readable": "#2A9D8F", "complete_unreadable": "#E9C46A",
        "incomplete_readable": "#1C7293", "incomplete_unreadable": "#B85042",
    }
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for q, col in colors.items():
        xs = [r["R"] for r in joined if r["quadrant"] == q]
        ys = [r["C"] for r in joined if r["quadrant"] == q]
        ax.scatter(xs, ys, s=6, alpha=0.35, color=col, label=q.replace("_", " "))
    ax.axvline(r_split, color="#333333", linestyle="--", linewidth=1)
    ax.axhline(c_split, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Readability R_overall")
    ax.set_ylabel("Completeness C")
    ax.set_title("Completeness vs Readability")
    ax.legend(markerscale=2, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quadrant_scatter.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Completeness x Readability join")
    ap.add_argument("--t1-model", default="qwen3_vl_32b")
    ap.add_argument("--t1-variant", default="B", choices=["A", "B"])
    ap.add_argument("--t2-model", default="qwen3_vl_32b")
    ap.add_argument("--c-split", default="median", help="'median' or a numeric threshold")
    ap.add_argument("--r-split", default="median", help="'median' or a numeric threshold")
    args = ap.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()