#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 — Section / Entity-Coverage Breakdown  (descriptive, machine-only)
=========================================================================

Reads the per-model x variant scoring CSVs produced by task1_scoring.py and reports
WHERE completeness comes from — which IMRaD sections graphical abstracts cover well,
how much coverage is explicit vs implied vs absent, and how traceable the causal
relations are. This surfaces the entity/section detail that C and level compress away.

Runs for ALL available methods: the 6 SRP CSVs (qwen/gemma/mistral x A/B). The naive
baseline is skipped (it has no per-section entity verdicts).

For each method it computes:
  * Per-section MEAN coverage (introduction/methods/results/discussion), so sections can
    be ranked best->worst covered.
  * Explicit / implied / absent ENTITY proportions per section (from n_explicit/implied/
    absent columns), i.e. of all entities in that section, what fraction were depicted
    explicitly vs implied vs absent.
  * Section VERDICT distribution (how often each section's section-level verdict is
    explicit / implied / absent).
  * Relation TRACEABILITY rates for the three causal links (traceable / partially /
    not_traceable).

All DESCRIPTIVE / associational and PROVISIONAL until human validation (only section-level
verdicts + level are in the human protocol; entity-level stays machine-only).

Inputs  : task1_scores/scores_<model>_variant<A|B>.csv
Outputs : task1_scores/coverage_breakdown/
    section_coverage_<method>.csv          per-section mean coverage + verdict dist
    entity_explicit_implied_absent_<method>.csv
    relation_traceability_<method>.csv
    coverage_breakdown_all_methods.csv     long table (section coverage, all methods)
    coverage_breakdown_summary.json
    figures/section_coverage_<method>.png
    figures/section_coverage_all_A_vs_B_<model>.png

Usage:
  python3 task1_coverage_breakdown.py
  python3 task1_coverage_breakdown.py --models qwen3_vl_32b gemma_3_27b
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_BASE = Path("./task1_completeness_awq")
SCORES_DIR = PROJECT_BASE / "task1_scores"
OUTPUT_DIR = SCORES_DIR / "coverage_breakdown"
FIG_DIR = OUTPUT_DIR / "figures"

MODELS = ["qwen3_vl_32b", "gemma_3_27b", "mistral_small_24b"]
VARIANTS = ["A", "B"]
SECTIONS = ["introduction", "methods", "results", "discussion"]
RELATIONS = ["intro_to_methods", "methods_to_results", "results_to_discussion"]

SECTION_VERDICTS = ["explicit", "implied", "absent"]
RELATION_VERDICTS = ["traceable", "partially_traceable", "not_traceable"]


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


def to_int(x: Optional[str]) -> int:
    try:
        return int(float(x)) if x not in (None, "") else 0
    except (ValueError, TypeError):
        return 0


def analyze_method(model: str, variant: str) -> Optional[Dict[str, Any]]:
    path = SCORES_DIR / f"scores_{model}_variant{variant}.csv"
    if not path.exists():
        print(f"  [skip] missing {path.name}")
        return None
    rows = read_rows(path)
    rows = [r for r in rows if r.get("parse_ok", "True") in ("True", "true", "1", "")]
    n = len(rows)
    if n == 0:
        return None
    method = f"{model}_variant{variant}"

    # accumulators
    cov_sum = {s: 0.0 for s in SECTIONS}
    cov_cnt = {s: 0 for s in SECTIONS}
    ent = {s: {"explicit": 0, "implied": 0, "absent": 0} for s in SECTIONS}
    sec_verdict = {s: {v: 0 for v in SECTION_VERDICTS} for s in SECTIONS}
    rel_verdict = {r: {v: 0 for v in RELATION_VERDICTS} for r in RELATIONS}

    for row in rows:
        for s in SECTIONS:
            c = to_float(row.get(f"coverage_{s}"))
            if c is not None:
                cov_sum[s] += c
                cov_cnt[s] += 1
            ent[s]["explicit"] += to_int(row.get(f"n_explicit_{s}"))
            ent[s]["implied"] += to_int(row.get(f"n_implied_{s}"))
            ent[s]["absent"] += to_int(row.get(f"n_absent_{s}"))
            v = row.get(f"verdict_{s}", "")
            if v in sec_verdict[s]:
                sec_verdict[s][v] += 1
        for r in RELATIONS:
            rv = row.get(f"rel_{r}", "")
            if rv in rel_verdict[r]:
                rel_verdict[r][rv] += 1

    # ---- section coverage csv ----
    sec_rows = []
    for s in SECTIONS:
        mean_cov = round(cov_sum[s] / cov_cnt[s], 4) if cov_cnt[s] else ""
        vt = sec_verdict[s]
        vtot = sum(vt.values()) or 1
        sec_rows.append({
            "method": method, "section": s, "n": cov_cnt[s],
            "mean_coverage": mean_cov,
            "pct_verdict_explicit": round(100 * vt["explicit"] / vtot, 2),
            "pct_verdict_implied": round(100 * vt["implied"] / vtot, 2),
            "pct_verdict_absent": round(100 * vt["absent"] / vtot, 2),
        })
    with open(OUTPUT_DIR / f"section_coverage_{method}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sec_rows[0].keys()))
        w.writeheader(); w.writerows(sec_rows)

    # ---- entity explicit/implied/absent csv ----
    with open(OUTPUT_DIR / f"entity_explicit_implied_absent_{method}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "section", "n_explicit", "n_implied", "n_absent",
                    "pct_explicit", "pct_implied", "pct_absent", "total_entities"])
        for s in SECTIONS:
            e = ent[s]
            tot = e["explicit"] + e["implied"] + e["absent"]
            d = tot or 1
            w.writerow([method, s, e["explicit"], e["implied"], e["absent"],
                        round(100 * e["explicit"] / d, 2), round(100 * e["implied"] / d, 2),
                        round(100 * e["absent"] / d, 2), tot])

    # ---- relation traceability csv ----
    with open(OUTPUT_DIR / f"relation_traceability_{method}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "relation", "pct_traceable", "pct_partial", "pct_not", "n"])
        for r in RELATIONS:
            rv = rel_verdict[r]
            tot = sum(rv.values()) or 1
            w.writerow([method, r,
                        round(100 * rv["traceable"] / tot, 2),
                        round(100 * rv["partially_traceable"] / tot, 2),
                        round(100 * rv["not_traceable"] / tot, 2), sum(rv.values())])

    _plot_section(method, sec_rows)

    return {
        "method": method, "model": model, "variant": variant, "n": n,
        "mean_coverage_by_section": {s: (round(cov_sum[s] / cov_cnt[s], 4) if cov_cnt[s] else None) for s in SECTIONS},
        "entity_counts_by_section": ent,
        "relation_traceability": rel_verdict,
        "section_rows": sec_rows,
    }


def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"  [warn] matplotlib unavailable ({e}); skipping figures.")
        return None


def _plot_section(method: str, sec_rows: List[Dict[str, Any]]) -> None:
    plt = _get_plt()
    if plt is None:
        return
    labels = [r["section"][:4].capitalize() for r in sec_rows]
    covs = [r["mean_coverage"] if r["mean_coverage"] != "" else 0 for r in sec_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, covs, color=["#1C7293", "#2A9D8F", "#16243A", "#E9C46A"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean coverage")
    ax.set_title(f"Section coverage — {method}")
    for b, c in zip(bars, covs):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.01, f"{c:.2f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"section_coverage_{method}.png", dpi=140)
    plt.close(fig)


def _plot_a_vs_b(model: str, per_method: Dict[str, Dict[str, Any]]) -> None:
    plt = _get_plt()
    if plt is None:
        return
    a = per_method.get(f"{model}_variantA")
    b = per_method.get(f"{model}_variantB")
    if not a or not b:
        return
    import numpy as np
    x = np.arange(len(SECTIONS))
    av = [a["mean_coverage_by_section"][s] or 0 for s in SECTIONS]
    bv = [b["mean_coverage_by_section"][s] or 0 for s in SECTIONS]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.2, av, 0.4, label="Variant A", color="#1C7293")
    ax.bar(x + 0.2, bv, 0.4, label="Variant B", color="#2A9D8F")
    ax.set_xticks(x); ax.set_xticklabels([s[:4].capitalize() for s in SECTIONS])
    ax.set_ylim(0, 1); ax.set_ylabel("Mean coverage")
    ax.set_title(f"Section coverage A vs B — {model}")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"section_coverage_all_A_vs_B_{model}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 1 section/entity coverage breakdown")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Task 1 — section / entity coverage breakdown (descriptive, provisional)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    per_method: Dict[str, Dict[str, Any]] = {}
    all_long: List[Dict[str, Any]] = []
    for model in args.models:
        for variant in args.variants:
            res = analyze_method(model, variant)
            if res:
                per_method[res["method"]] = res
                all_long.extend(res["section_rows"])
                print(f"  {res['method']}: n={res['n']}  "
                      f"coverage " + ", ".join(f"{s[:1].upper()}={res['mean_coverage_by_section'][s]}" for s in SECTIONS))

    for model in args.models:
        _plot_a_vs_b(model, per_method)

    if all_long:
        with open(OUTPUT_DIR / "coverage_breakdown_all_methods.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_long[0].keys()))
            w.writeheader(); w.writerows(all_long)

    summary = {m: {"n": r["n"], "mean_coverage_by_section": r["mean_coverage_by_section"]}
               for m, r in per_method.items()}
    with open(OUTPUT_DIR / "coverage_breakdown_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Tables + figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()