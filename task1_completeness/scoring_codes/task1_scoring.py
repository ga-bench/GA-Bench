#!/usr/bin/env python3
"""
Task 1 Downstream Scoring
=========================
Reads the per-DOI Stage 2 evaluation JSONs (3 models x 2 variants) and the naive
VLM judge baseline judgments, and computes the proposal's completeness scores.

This is a CPU-only post-processing step. It does NOT call any model. It runs on
the login node. The models produced only verdict LABELS; all arithmetic
(coverage, S, R, C, level) is done here so weights can be changed without
re-running inference.

Per the proposal (Task 1, Stage 2, Step 4):
  entity verdict  -> explicit=1.0, implied=0.5, absent=0.0
  Coverage(section) = sum(entity scores) / number of entities in that section
  relation verdict -> traceable=1.0, partially_traceable=0.5, not_traceable=0.0
  Relation coverage R = sum(relation scores) / number of relations
  Section score S = (C_I + C_M + C_R + C_D) / 4
  Final completeness C = w_s * S + w_r * R     (w_s=0.7, w_r=0.3 by default)
  Discrete level = number of IMRaD sections with a section-level verdict of
                   explicit or implied (0..4)

Outputs, under <PROJECT_BASE>/task1_scores/ :
  - scores_<model>_variant<A|B>.csv      (one row per DOI; SRP pipeline)
  - scores_naive_<model>.csv             (one row per DOI; naive baseline level)
  - scores_all_long.csv                  (all methods stacked; tidy/long format)
  - scoring_summary.json                 (counts + mean scores per method)

Usage:
  python3 task1_scoring.py
  python3 task1_scoring.py --w-s 0.6 --w-r 0.4
  python3 task1_scoring.py --models qwen3_vl_32b gemma_3_27b --variants A B
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Configuration
# =============================================================================

PROJECT_BASE = Path("./task1_completeness_awq")
OUTPUT_DIR = PROJECT_BASE / "task1_scores"

# GA index used to join publication year / domain for per-year & per-domain comparison.
GA_INDEX_CSV = Path(
    "./"
    "task2_readability/output/stage1_preprocessing/index/stage1_ga_index.csv"
)
GA_INDEX_ID = "paper_id"
YEAR_CANDIDATES = ["year", "pub_year", "publication_year", "pubyear"]
DOMAIN_CANDIDATES = ["primary_domain", "domain", "field", "subject", "area"]

MODELS = ["qwen3_vl_32b", "gemma_3_27b", "mistral_small_24b"]
VARIANTS = ["A", "B"]

SECTIONS = ["introduction", "methods", "results", "discussion"]
RELATION_KEYS = ["intro_to_methods", "methods_to_results", "results_to_discussion"]

ENTITY_SCORE = {"explicit": 1.0, "implied": 0.5, "absent": 0.0}
RELATION_SCORE = {"traceable": 1.0, "partially_traceable": 0.5, "not_traceable": 0.0}

DEFAULT_W_S = 0.7
DEFAULT_W_R = 0.3

# Eval / judgment file locators (must match the inference scripts' suffixes).
def srp_eval_dir(model: str, variant: str) -> Path:
    return PROJECT_BASE / f"variant_{variant}" / "outputs" / "stage2" / model / "evals"

def srp_eval_suffix(model: str) -> str:
    return f"_eval_{model}.json"

def naive_judgment_dir(model: str) -> Path:
    return PROJECT_BASE / "baselines" / "naive_vlm" / model / "judgments"

def naive_judgment_suffix(model: str) -> str:
    return f"_naivejudge_{model}.json"

# =============================================================================
# Scoring core
# =============================================================================

def score_section_entities(entity_coverage: List[Dict[str, Any]]) -> Tuple[float, int, int, int, int]:
    """Return (coverage, n_entities, n_explicit, n_implied, n_absent) for a section.
    Coverage = mean entity score; empty section -> coverage 0.0."""
    n_e = n_i = n_a = 0
    total = 0.0
    n = 0
    for item in entity_coverage:
        if not isinstance(item, dict):
            continue
        v = item.get("verdict")
        if v not in ENTITY_SCORE:
            continue
        total += ENTITY_SCORE[v]
        n += 1
        if v == "explicit":
            n_e += 1
        elif v == "implied":
            n_i += 1
        else:
            n_a += 1
    coverage = (total / n) if n > 0 else 0.0
    return coverage, n, n_e, n_i, n_a


def score_eval(ev: Dict[str, Any], w_s: float, w_r: float) -> Dict[str, Any]:
    """Compute all proposal scores for a single SRP-pipeline eval JSON."""
    ce = ev.get("component_evaluation", {}) or {}
    ri = ev.get("relational_integrity", {}) or {}

    out: Dict[str, Any] = {}
    section_covs: List[float] = []
    component_count = 0

    for sec in SECTIONS:
        sd = ce.get(sec, {}) if isinstance(ce, dict) else {}
        ecov = sd.get("entity_coverage", []) if isinstance(sd, dict) else []
        if not isinstance(ecov, list):
            ecov = []
        coverage, n, n_e, n_i, n_a = score_section_entities(ecov)
        section_verdict = sd.get("verdict", "") if isinstance(sd, dict) else ""

        out[f"coverage_{sec}"] = round(coverage, 4)
        out[f"verdict_{sec}"] = section_verdict
        out[f"n_entities_{sec}"] = n
        out[f"n_explicit_{sec}"] = n_e
        out[f"n_implied_{sec}"] = n_i
        out[f"n_absent_{sec}"] = n_a

        section_covs.append(coverage)
        if section_verdict in ("explicit", "implied"):
            component_count += 1

    # Section score S = mean of the 4 section coverages.
    S = sum(section_covs) / len(SECTIONS)

    # Relation coverage R.
    rel_total = 0.0
    rel_n = 0
    for k in RELATION_KEYS:
        rd = ri.get(k, {}) if isinstance(ri, dict) else {}
        v = rd.get("verdict") if isinstance(rd, dict) else None
        out[f"rel_{k}"] = v if v in RELATION_SCORE else ""
        if v in RELATION_SCORE:
            rel_total += RELATION_SCORE[v]
            rel_n += 1
    R = (rel_total / rel_n) if rel_n > 0 else 0.0

    C = w_s * S + w_r * R

    out["S_section_score"] = round(S, 4)
    out["R_relation_score"] = round(R, 4)
    out["C_completeness"] = round(C, 4)
    out["w_s"] = w_s
    out["w_r"] = w_r
    out["level"] = component_count          # 0..4, derived from section verdicts
    out["n_relations_scored"] = rel_n
    return out

# =============================================================================
# IO
# =============================================================================

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_id_from_filename(name: str, suffix: str) -> str:
    return name[: -len(suffix)] if name.endswith(suffix) else name


def score_srp_pipeline(model: str, variant: str, w_s: float, w_r: float) -> List[Dict[str, Any]]:
    eval_dir = srp_eval_dir(model, variant)
    suffix = srp_eval_suffix(model)
    rows: List[Dict[str, Any]] = []
    if not eval_dir.exists():
        print(f"  [warn] missing eval dir: {eval_dir}")
        return rows
    files = sorted(eval_dir.glob(f"*{suffix}"))
    for fp in files:
        ev = load_json(fp)
        doi_safe = safe_id_from_filename(fp.name, suffix)
        row: Dict[str, Any] = {
            "method": "srp",
            "model": model,
            "variant": variant,
            "doi_safe": doi_safe,
            "doi": (ev or {}).get("doi", doi_safe),
            "eval_file": str(fp),
            "parse_ok": ev is not None,
        }
        if ev is None:
            row["error"] = "json_load_failed"
            rows.append(row)
            continue
        row.update(score_eval(ev, w_s, w_r))
        rows.append(row)
    print(f"  {model} variant_{variant}: scored {len(rows)} evals")
    return rows


def score_naive(model: str) -> List[Dict[str, Any]]:
    jdir = naive_judgment_dir(model)
    suffix = naive_judgment_suffix(model)
    rows: List[Dict[str, Any]] = []
    if not jdir.exists():
        print(f"  [warn] missing naive judgment dir: {jdir}")
        return rows
    for fp in sorted(jdir.glob(f"*{suffix}")):
        j = load_json(fp)
        doi_safe = safe_id_from_filename(fp.name, suffix)
        row: Dict[str, Any] = {
            "method": "naive",
            "model": model,
            "variant": "",
            "doi_safe": doi_safe,
            "doi": (j or {}).get("doi", doi_safe),
            "judgment_file": str(fp),
            "parse_ok": j is not None,
        }
        if j is None:
            row["error"] = "json_load_failed"
            rows.append(row)
            continue
        # naive judge: authoritative level was already stored as count of present components
        row["level"] = j.get("level", "")
        cp = j.get("components_present", {}) if isinstance(j, dict) else {}
        for sec in SECTIONS:
            row[f"present_{sec}"] = bool(cp.get(sec)) if isinstance(cp, dict) else ""
        rows.append(row)
    print(f"  naive {model}: scored {len(rows)} judgments")
    return rows

# =============================================================================
# CSV writers
# =============================================================================

SRP_FIELDS = (
    ["method", "model", "variant", "doi", "doi_safe", "parse_ok", "error",
     "level", "S_section_score", "R_relation_score", "C_completeness", "w_s", "w_r",
     "n_relations_scored"]
    + [f"coverage_{s}" for s in SECTIONS]
    + [f"verdict_{s}" for s in SECTIONS]
    + [f"n_entities_{s}" for s in SECTIONS]
    + [f"n_explicit_{s}" for s in SECTIONS]
    + [f"n_implied_{s}" for s in SECTIONS]
    + [f"n_absent_{s}" for s in SECTIONS]
    + [f"rel_{k}" for k in RELATION_KEYS]
    + ["eval_file"]
)

NAIVE_FIELDS = (
    ["method", "model", "variant", "doi", "doi_safe", "parse_ok", "error", "level"]
    + [f"present_{s}" for s in SECTIONS]
    + ["judgment_file"]
)


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_long(path: Path, all_rows: List[Dict[str, Any]]) -> None:
    """Tidy long format: one row per (method, model, variant, doi) with the
    headline numbers only, for easy cross-method comparison and joins to humans."""
    fields = ["method", "model", "variant", "doi", "doi_safe",
              "level", "C_completeness", "S_section_score", "R_relation_score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})

# =============================================================================
# Main
# =============================================================================

def mean_or_none(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 4) if xs else None


# =============================================================================
# SRP-level vs naive-level comparison (per model, naive vs Variant A and B)
# =============================================================================

def _read_level_map(csv_path: Path, level_field: str = "level") -> Dict[str, int]:
    """paper_id (doi_safe) -> integer level, for parseable rows only."""
    out: Dict[str, int] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("parse_ok", "True")).lower() in ("false", "0"):
                continue
            pid = r.get("doi_safe") or r.get("doi")
            lv = r.get(level_field, "")
            try:
                if pid and lv != "":
                    out[pid] = int(float(lv))
            except (ValueError, TypeError):
                continue
    return out


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return (num / (dx * dy)) if dx and dy else None


def _weighted_kappa(pairs: List[Tuple[int, int]], k: int = 5) -> Optional[float]:
    """Quadratic-weighted Cohen's kappa over levels 0..k-1."""
    n = len(pairs)
    if n == 0:
        return None
    O = [[0.0] * k for _ in range(k)]
    r1 = [0.0] * k
    r2 = [0.0] * k
    for a, b in pairs:
        if 0 <= a < k and 0 <= b < k:
            O[a][b] += 1
            r1[a] += 1
            r2[b] += 1
    W = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    E = [[r1[i] * r2[j] / n for j in range(k)] for i in range(k)]
    num = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(W[i][j] * E[i][j] for i in range(k) for j in range(k))
    return (1 - num / den) if den else None


def _load_meta_for_compare() -> Dict[str, Dict[str, Any]]:
    """paper_id -> {year:int|None, domain:str}. Empty dict if index missing."""
    meta: Dict[str, Dict[str, Any]] = {}
    if not GA_INDEX_CSV.exists():
        print(f"  [warn] GA index not found ({GA_INDEX_CSV}); per-year/domain comparison skipped.")
        return meta
    with open(GA_INDEX_CSV, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        low = {c.lower(): c for c in fields}
        ycol = next((low[c] for c in YEAR_CANDIDATES if c in low), None)
        dcol = next((low[c] for c in DOMAIN_CANDIDATES if c in low), None)
        idc = low.get(GA_INDEX_ID.lower(), GA_INDEX_ID)
        print(f"  meta join -> id:{idc} year:{ycol} domain:{dcol}")
        for row in r:
            pid = row.get(idc, "")
            if not pid:
                continue
            yr = None
            if ycol and row.get(ycol):
                try:
                    yr = int(str(row[ycol]).strip()[:4])
                    if not (1900 <= yr <= 2100):
                        yr = None
                except ValueError:
                    yr = None
            meta[pid] = {"year": yr, "domain": (row.get(dcol, "") or "NA") if dcol else "NA"}
    return meta


def compare_levels(models: List[str], variants: List[str]) -> Dict[str, Any]:
    """For each model, compare naive level vs each SRP variant level."""
    print("Comparing discrete completeness level vs naive VLM judge...")
    cmp_dir = OUTPUT_DIR / "level_comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_meta_for_compare()
    out_summary: Dict[str, Any] = {}
    long_rows: List[Dict[str, Any]] = []

    for model in models:
        naive_map = _read_level_map(OUTPUT_DIR / f"scores_naive_{model}.csv")
        if not naive_map:
            print(f"  [skip] no naive scores for {model}")
            continue
        for variant in variants:
            srp_map = _read_level_map(OUTPUT_DIR / f"scores_{model}_variant{variant}.csv")
            if not srp_map:
                continue
            common = sorted(set(srp_map) & set(naive_map))
            if not common:
                continue
            srp_l = [srp_map[p] for p in common]
            nav_l = [naive_map[p] for p in common]
            pairs = list(zip(srp_l, nav_l))
            n = len(pairs)

            exact = sum(1 for a, b in pairs if a == b) / n
            within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / n
            srp_higher = sum(1 for a, b in pairs if a > b) / n
            naive_higher = sum(1 for a, b in pairs if b > a) / n

            # 5x5 confusion matrix (rows=SRP level, cols=naive level)
            conf = [[0] * 5 for _ in range(5)]
            for a, b in pairs:
                if 0 <= a < 5 and 0 <= b < 5:
                    conf[a][b] += 1

            tag = f"{model}_variant{variant}"
            with open(cmp_dir / f"confusion_{tag}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["srp_level\\naive_level", 0, 1, 2, 3, 4])
                for i in range(5):
                    w.writerow([i] + conf[i])

            # per-level distribution
            with open(cmp_dir / f"level_distribution_{tag}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["level", "srp_count", "srp_pct", "naive_count", "naive_pct"])
                for lv in range(5):
                    sc = srp_l.count(lv)
                    nc = nav_l.count(lv)
                    w.writerow([lv, sc, round(100 * sc / n, 2), nc, round(100 * nc / n, 2)])

            rec = {
                "model": model, "variant": variant, "n": n,
                "mean_level_srp": round(sum(srp_l) / n, 4),
                "mean_level_naive": round(sum(nav_l) / n, 4),
                "mean_diff_srp_minus_naive": round((sum(srp_l) - sum(nav_l)) / n, 4),
                "exact_agreement_pct": round(100 * exact, 2),
                "within_1_agreement_pct": round(100 * within1, 2),
                "srp_higher_pct": round(100 * srp_higher, 2),
                "naive_higher_pct": round(100 * naive_higher, 2),
                "spearman": (round(_spearman([float(a) for a in srp_l], [float(b) for b in nav_l]), 4)
                             if n > 1 else None),
                "quadratic_weighted_kappa": (round(_weighted_kappa(pairs), 4)),
            }
            out_summary[tag] = rec
            long_rows.append(rec)
            _plot_level_comparison(cmp_dir, tag, srp_l, nav_l)

            # ---- disagreement-magnitude distribution (srp - naive, -4..+4) ----
            with open(cmp_dir / f"disagreement_magnitude_{tag}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["diff_srp_minus_naive", "count", "pct"])
                diffs = [a - b for a, b in pairs]
                for d in range(-4, 5):
                    c = diffs.count(d)
                    w.writerow([d, c, round(100 * c / n, 2)])

            # ---- per-year comparison (needs metadata) ----
            if meta:
                yr_agg: Dict[int, Dict[str, float]] = {}
                for pid, a, b in zip(common, srp_l, nav_l):
                    md = meta.get(pid)
                    if not md or md.get("year") is None:
                        continue
                    y = md["year"]
                    d = yr_agg.setdefault(y, {"n": 0, "srp": 0, "nav": 0, "exact": 0, "srp_hi": 0, "nav_hi": 0})
                    d["n"] += 1; d["srp"] += a; d["nav"] += b
                    d["exact"] += (a == b); d["srp_hi"] += (a > b); d["nav_hi"] += (b > a)
                if yr_agg:
                    with open(cmp_dir / f"per_year_{tag}.csv", "w", encoding="utf-8", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["year", "n", "mean_level_srp", "mean_level_naive",
                                    "mean_diff", "exact_agreement_pct", "srp_higher_pct", "naive_higher_pct"])
                        for y in sorted(yr_agg):
                            d = yr_agg[y]; nn = d["n"]
                            w.writerow([y, nn, round(d["srp"] / nn, 4), round(d["nav"] / nn, 4),
                                        round((d["srp"] - d["nav"]) / nn, 4),
                                        round(100 * d["exact"] / nn, 2),
                                        round(100 * d["srp_hi"] / nn, 2),
                                        round(100 * d["nav_hi"] / nn, 2)])
                    _plot_per_year(cmp_dir, tag, yr_agg)

                # ---- per-domain comparison ----
                dom_agg: Dict[str, Dict[str, float]] = {}
                for pid, a, b in zip(common, srp_l, nav_l):
                    md = meta.get(pid)
                    dom = md.get("domain", "NA") if md else "NA"
                    d = dom_agg.setdefault(dom, {"n": 0, "srp": 0, "nav": 0, "exact": 0})
                    d["n"] += 1; d["srp"] += a; d["nav"] += b; d["exact"] += (a == b)
                if dom_agg:
                    with open(cmp_dir / f"per_domain_{tag}.csv", "w", encoding="utf-8", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["domain", "n", "mean_level_srp", "mean_level_naive", "mean_diff", "exact_agreement_pct"])
                        for dom in sorted(dom_agg, key=lambda k: -dom_agg[k]["n"]):
                            d = dom_agg[dom]; nn = d["n"]
                            w.writerow([dom, nn, round(d["srp"] / nn, 4), round(d["nav"] / nn, 4),
                                        round((d["srp"] - d["nav"]) / nn, 4), round(100 * d["exact"] / nn, 2)])

            print(f"  {tag}: n={n} meanSRP={rec['mean_level_srp']} meanNaive={rec['mean_level_naive']} "
                  f"exact={rec['exact_agreement_pct']}% wk={rec['quadratic_weighted_kappa']}")

    if long_rows:
        fields = ["model", "variant", "n", "mean_level_srp", "mean_level_naive",
                  "mean_diff_srp_minus_naive", "exact_agreement_pct", "within_1_agreement_pct",
                  "srp_higher_pct", "naive_higher_pct", "spearman", "quadratic_weighted_kappa"]
        with open(cmp_dir / "level_comparison_summary.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(long_rows)
    with open(cmp_dir / "level_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)
    return out_summary


def _plot_level_comparison(cmp_dir: Path, tag: str, srp_l: List[int], nav_l: List[int]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    import numpy as np
    x = np.arange(5)
    srp_c = [srp_l.count(i) for i in range(5)]
    nav_c = [nav_l.count(i) for i in range(5)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.2, srp_c, 0.4, label="SRP level", color="#1C7293")
    ax.bar(x + 0.2, nav_c, 0.4, label="Naive judge", color="#AAB7C0")
    ax.set_xticks(x); ax.set_xlabel("Completeness level (0-4)"); ax.set_ylabel("Papers")
    ax.set_title(f"Level distribution: SRP vs naive — {tag}")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    (cmp_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(cmp_dir / "figures" / f"level_dist_{tag}.png", dpi=140)
    plt.close(fig)


def _plot_per_year(cmp_dir: Path, tag: str, yr_agg: Dict[int, Dict[str, float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    yrs = sorted(yr_agg)
    srp = [yr_agg[y]["srp"] / yr_agg[y]["n"] for y in yrs]
    nav = [yr_agg[y]["nav"] / yr_agg[y]["n"] for y in yrs]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(yrs, srp, "o-", label="SRP level", color="#1C7293", linewidth=2)
    ax.plot(yrs, nav, "s--", label="Naive judge", color="#AAB7C0", linewidth=2)
    ax.set_ylim(0, 4)
    ax.set_xlabel("Publication year"); ax.set_ylabel("Mean level (0-4)")
    ax.set_title(f"SRP vs naive level by year — {tag}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    (cmp_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(cmp_dir / "figures" / f"per_year_{tag}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 1 downstream scoring")
    ap.add_argument("--w-s", type=float, default=DEFAULT_W_S, help="weight on section score S")
    ap.add_argument("--w-r", type=float, default=DEFAULT_W_R, help="weight on relation score R")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--no-naive", action="store_true", help="skip naive baseline scoring")
    ap.add_argument("--no-compare", action="store_true", help="skip SRP-vs-naive level comparison")
    args = ap.parse_args()

    if abs((args.w_s + args.w_r) - 1.0) > 1e-6:
        print(f"[warn] w_s + w_r = {args.w_s + args.w_r} (not 1.0); proceeding anyway.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"Task 1 scoring | w_s={args.w_s} w_r={args.w_r}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    all_long: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"w_s": args.w_s, "w_r": args.w_r, "methods": {}}

    # SRP pipeline
    print("Scoring SRP pipeline evals...")
    for model in args.models:
        for variant in args.variants:
            rows = score_srp_pipeline(model, variant, args.w_s, args.w_r)
            if not rows:
                continue
            out_csv = OUTPUT_DIR / f"scores_{model}_variant{variant}.csv"
            write_csv(out_csv, rows, SRP_FIELDS)
            scored = [r for r in rows if r.get("parse_ok")]
            summary["methods"][f"srp_{model}_variant{variant}"] = {
                "n_total": len(rows),
                "n_scored": len(scored),
                "mean_C": mean_or_none([r.get("C_completeness") for r in scored]),
                "mean_S": mean_or_none([r.get("S_section_score") for r in scored]),
                "mean_R": mean_or_none([r.get("R_relation_score") for r in scored]),
                "mean_level": mean_or_none([r.get("level") for r in scored]),
                "csv": str(out_csv),
            }
            all_long.extend(rows)

    # Naive baseline
    if not args.no_naive:
        print("Scoring naive VLM judge baseline...")
        for model in args.models:
            rows = score_naive(model)
            if not rows:
                continue
            out_csv = OUTPUT_DIR / f"scores_naive_{model}.csv"
            write_csv(out_csv, rows, NAIVE_FIELDS)
            scored = [r for r in rows if r.get("parse_ok")]
            summary["methods"][f"naive_{model}"] = {
                "n_total": len(rows),
                "n_scored": len(scored),
                "mean_level": mean_or_none([r.get("level") for r in scored]),
                "csv": str(out_csv),
            }
            all_long.extend(rows)

    # SRP-level vs naive-level comparison (naive vs both variants, per model)
    if not args.no_naive and not args.no_compare:
        summary["level_comparison"] = compare_levels(args.models, args.variants)

    # Combined long table
    long_csv = OUTPUT_DIR / "scores_all_long.csv"
    write_long(long_csv, all_long)
    summary["long_csv"] = str(long_csv)

    with open(OUTPUT_DIR / "scoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Per-method CSVs + scores_all_long.csv in: {OUTPUT_DIR}")
    print("Summary:")
    for name, s in summary["methods"].items():
        extra = f"mean_C={s.get('mean_C')}" if "mean_C" in s else ""
        print(f"  {name}: n_scored={s['n_scored']}/{s['n_total']} mean_level={s.get('mean_level')} {extra}")


if __name__ == "__main__":
    main()