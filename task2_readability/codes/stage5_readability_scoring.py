#!/usr/bin/env python3
"""
Task 2 / Stage 5 — Human-Calibrated Readability Scoring (UNWEIGHTED BASELINE)
============================================================================

Computes the three readability component scores and the overall readability score
for every GA, by merging:

  Stage 2 OCR text features      -> R_text     (text legibility)
  Stage 3 visual/layout features -> R_visual   (visual clarity)   [+ VLM clutter label]
  Stage 4 VLM interpretation     -> R_semantic (semantic interpretability)

This is the UNWEIGHTED baseline from the proposal:
    R(G) = mean(R_text, R_visual, R_semantic)

The CALIBRATED version  R(G) = a*R_text + b*R_visual + c*R_semantic  is deferred to
Stage 6, where a/b/c are fit against human ratings. The three component columns are
written out here so Stage 6 can regress on them directly.

MULTI-MODEL (this version)
--------------------------
Only TWO things are model-dependent: the Stage-4 consolidated CSV (input) and the
output directory. Stage 2/3 inputs are shared. The script runs one or more models via
a registry and writes each to its own output tree:

  qwen3_vl_32b       -> output/stage5_readability_scoring/                  (original path)
  gemma_3_27b        -> output/stage5_readability_scoring_gemma_3_27b/
  mistral_small_24b  -> output/stage5_readability_scoring_mistral_small_24b/

Comparability note: the text features and the 9 Stage-3 visual features are fit over
identical inputs for every model, so their normalization percentiles come out identical
by construction. Only `num_panels` (Stage 4) is fit per model, and only R_semantic is
fully model-specific. A `model` column is added to every output row.

(InternVL is intentionally excluded.)

Design choices (documented for the paper):
  - Each numeric feature is robust min-max normalized to [0,1] using dataset 5th/95th
    percentiles (clipped), then sign-oriented so 1 = MORE readable.
  - Reverse-coded VLM scores (ambiguity, text_dependency) are inverted before use.
  - Text features are normalized over TEXT-BEARING GAs only; GAs with no detected text
    get R_text = NaN and R(G) = mean(R_visual, R_semantic).
  - Low/medium/high categories are assigned by TERTILES of R(G) (per model), provisional
    until replaced by human-calibrated thresholds in Stage 6.

Run (CPU only):
  python3 stage5_readability_scoring.py                       # all configured models
  python3 stage5_readability_scoring.py --models gemma_3_27b  # one model
"""

import csv
import json
import math
import argparse
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("./task2_readability")

# Shared (model-independent) inputs
STAGE2_CSV = PROJECT_ROOT / "output" / "stage2_ocr_text_features" / "features" / "stage2_ocr_text_features.csv"
STAGE3_CSV = PROJECT_ROOT / "output" / "stage3_visual_complexity_features" / "features" / "stage3_visual_complexity_features.csv"

# Per-model registry: Stage-4 consolidated CSV (input) + Stage-5 output subdir.
# Paths are relative to PROJECT_ROOT/output.
MODELS: "OrderedDict[str, Dict[str, str]]" = OrderedDict([
    ("qwen3_vl_32b", {
        "stage4_csv": "stage4_vlm_structural_interpretation/consolidated/stage4_interpretations.csv",
        "out_subdir": "stage5_readability_scoring",
    }),
    ("gemma_3_27b", {
        "stage4_csv": "stage4_vlm_gemma_3_27b/consolidated/stage4_interpretations.csv",
        "out_subdir": "stage5_readability_scoring_gemma_3_27b",
    }),
    ("mistral_small_24b", {
        "stage4_csv": "stage4_vlm_mistral_small_24b/consolidated/stage4_interpretations.csv",
        "out_subdir": "stage5_readability_scoring_mistral_small_24b",
    }),
])

# ---------------------------------------------------------------------------
# Feature configuration
# orientation: "pos" = higher is more readable; "neg" = higher is less readable (inverted)
# ---------------------------------------------------------------------------

# R_text features (from Stage 2). Normalized over text-bearing GAs only.
TEXT_FEATURES = {
    "mean_ocr_confidence": "pos",
    "low_confidence_box_ratio": "neg",
    "tiny_text_box_ratio": "neg",
    "total_text_area_ratio": "neg",
    "acronym_density": "neg",
    "scientific_term_density": "neg",
    "num_text_boxes": "neg",
}

# R_visual numeric features (from Stage 3, except num_panels from Stage 4).
VISUAL_NUM_FEATURES = {
    "edge_density": "neg",
    "whitespace_ratio": "pos",
    "quantized_color_count": "neg",
    "color_entropy": "neg",
    "connected_component_count": "neg",
    "contour_count": "neg",
    "non_background_ratio": "neg",
    "visual_clutter_score": "neg",
    "layout_imbalance_score": "neg",
    "num_panels": "neg",   # from Stage 4 structural; folded into visual per proposal
}

# Categorical maps -> [0,1] (1 = more readable)
CLARITY3 = {"clear": 1.0, "partially_clear": 0.5, "unclear": 0.0}
YNP = {"yes": 1.0, "partially": 0.5, "no": 0.0}
HML = {"high": 1.0, "medium": 0.5, "low": 0.0}
ARC = {"complete": 1.0, "partial": 0.5, "none": 0.0}
CLUTTER = {"low": 1.0, "medium": 0.5, "high": 0.0}  # low clutter = more readable

# R_semantic categorical fields:
SEM_CLARITY3 = ["flow_clarity", "entity_clarity", "relation_clarity"]
SEM_YNP = ["main_message_identifiable", "method_identifiable", "result_identifiable",
           "conclusion_identifiable"]
SEM_SCORE_POS = ["key_message_clarity_1to5", "sequence_clarity_1to5", "overall_interpretability_1to5"]
SEM_SCORE_REV = ["ambiguity_1to5", "text_dependency_1to5"]  # higher = worse -> invert

METADATA_COLS = ["publication_year", "journal", "publisher", "domain", "subject_area",
                 "subject_categories"]

P_LOW = 5.0
P_HIGH = 95.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = (row.get("paper_id") or "").strip()
            if pid:
                out[pid] = row
    return out


def to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def to_int(v: Any) -> Optional[int]:
    f = to_float(v)
    return int(round(f)) if f is not None else None


def robust_minmax(x: float, p5: float, p95: float, orientation: str) -> float:
    if p95 <= p5:
        v = 0.5
    else:
        v = (x - p5) / (p95 - p5)
        v = max(0.0, min(1.0, v))
    return 1.0 - v if orientation == "neg" else v


def nanmean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def primary_domain(domain: str) -> str:
    if not domain:
        return "unknown"
    return domain.split(";")[0].strip() or "unknown"


# ---------------------------------------------------------------------------
# Per-model scoring
# ---------------------------------------------------------------------------

def run_for_model(model_key: str, stage4_csv: Path, out_dir: Path,
                  s2: Dict[str, Dict[str, str]], s3: Dict[str, Dict[str, str]]) -> None:
    features_dir = out_dir / "features"
    params_dir = out_dir / "params"
    summary_dir = out_dir / "summaries"
    report_dir = out_dir / "reports"
    failure_dir = out_dir / "failures"
    for d in [features_dir, params_dir, summary_dir, report_dir, failure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    scores_csv = features_dir / "stage5_readability_scores.csv"
    scores_jsonl = features_dir / "stage5_readability_scores.jsonl"
    params_json = params_dir / "stage5_normalization_params.json"
    report_txt = report_dir / "stage5_readability_report.txt"
    failed_csv = failure_dir / "stage5_unscored.csv"
    year_summary_csv = summary_dir / "stage5_year_summary.csv"
    domain_summary_csv = summary_dir / "stage5_domain_summary.csv"
    journal_summary_csv = summary_dir / "stage5_journal_summary.csv"

    s4 = read_csv(stage4_csv)
    print(f"[{model_key}] Stage2 rows: {len(s2)} | Stage3 rows: {len(s3)} | Stage4 rows: {len(s4)}")

    # Master key set = all GAs known to Stage 3 (one row per GA, has metadata + visual).
    paper_ids = sorted(s3.keys())

    # ---- Pass 1: collect feature arrays for percentile fitting ----
    text_arrays: Dict[str, List[float]] = defaultdict(list)
    visual_arrays: Dict[str, List[float]] = defaultdict(list)

    for pid in paper_ids:
        r2 = s2.get(pid, {})
        r3 = s3.get(pid, {})
        s4r = s4.get(pid, {})
        has_text = (to_int(r2.get("num_text_boxes")) or 0) > 0

        if has_text:
            for feat in TEXT_FEATURES:
                val = to_float(r2.get(feat))
                if val is not None:
                    text_arrays[feat].append(val)

        for feat in VISUAL_NUM_FEATURES:
            src = s4r if feat == "num_panels" else r3
            val = to_float(src.get(feat))
            if val is not None:
                visual_arrays[feat].append(val)

    def fit_params(arrays: Dict[str, List[float]], cfg: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        params: Dict[str, Dict[str, float]] = {}
        for feat, orient in cfg.items():
            arr = np.asarray(arrays.get(feat, []), dtype=np.float64)
            if arr.size == 0:
                params[feat] = {"p5": 0.0, "p95": 1.0, "orientation": orient}
            else:
                params[feat] = {
                    "p5": float(np.percentile(arr, P_LOW)),
                    "p95": float(np.percentile(arr, P_HIGH)),
                    "orientation": orient,
                }
        return params

    text_params = fit_params(text_arrays, TEXT_FEATURES)
    visual_params = fit_params(visual_arrays, VISUAL_NUM_FEATURES)

    # ---- Pass 2: score every GA ----
    scored_rows: List[Dict[str, Any]] = []
    unscored_rows: List[Dict[str, Any]] = []

    for pid in paper_ids:
        r2 = s2.get(pid, {})
        r3 = s3.get(pid, {})
        s4r = s4.get(pid, {})

        meta = {c: (r3.get(c) or s2.get(pid, {}).get(c) or s4r.get(c) or "") for c in METADATA_COLS}
        ga_path = r3.get("ga_path") or r2.get("ga_path") or s4r.get("ga_path") or ""
        has_stage4 = bool(s4r)
        has_text = (to_int(r2.get("num_text_boxes")) or 0) > 0

        # ---- R_text ----
        if has_text:
            tvals = []
            for feat, orient in TEXT_FEATURES.items():
                val = to_float(r2.get(feat))
                if val is not None:
                    p = text_params[feat]
                    tvals.append(robust_minmax(val, p["p5"], p["p95"], orient))
            r_text = nanmean(tvals)
        else:
            r_text = None  # no text component

        # ---- R_visual ----
        vvals = []
        for feat, orient in VISUAL_NUM_FEATURES.items():
            src = s4r if feat == "num_panels" else r3
            val = to_float(src.get(feat))
            if val is not None:
                p = visual_params[feat]
                vvals.append(robust_minmax(val, p["p5"], p["p95"], orient))
        # VLM clutter label
        clutter_lbl = (s4r.get("visual_clutter") or "").strip().lower()
        if clutter_lbl in CLUTTER:
            vvals.append(CLUTTER[clutter_lbl])
        r_visual = nanmean(vvals)

        # ---- R_semantic ----
        svals: List[Optional[float]] = []
        if has_stage4:
            for f in SEM_CLARITY3:
                svals.append(CLARITY3.get((s4r.get(f) or "").strip().lower()))
            for f in SEM_YNP:
                svals.append(YNP.get((s4r.get(f) or "").strip().lower()))
            svals.append(HML.get((s4r.get("semantic_interpretability") or "").strip().lower()))
            svals.append(ARC.get((s4r.get("narrative_arc") or "").strip().lower()))
            for f in SEM_SCORE_POS:
                iv = to_int(s4r.get(f))
                if iv is not None:
                    svals.append((iv - 1) / 4.0)
            for f in SEM_SCORE_REV:
                iv = to_int(s4r.get(f))
                if iv is not None:
                    svals.append((5 - iv) / 4.0)
            r_semantic = nanmean(svals)
        else:
            r_semantic = None

        # ---- R(G) unweighted = mean of available components ----
        comps = [c for c in (r_text, r_visual, r_semantic) if c is not None]
        r_overall = sum(comps) / len(comps) if comps else None

        out = {
            "paper_id": pid,
            "model": model_key,
            "ga_path": ga_path,
            **meta,
            "primary_domain": primary_domain(meta.get("domain", "")),
            "has_text_component": int(has_text),
            "has_stage4": int(has_stage4),
            "R_text": round(r_text, 6) if r_text is not None else "",
            "R_visual": round(r_visual, 6) if r_visual is not None else "",
            "R_semantic": round(r_semantic, 6) if r_semantic is not None else "",
            "R_overall_unweighted": round(r_overall, 6) if r_overall is not None else "",
            "readability_category": "",  # filled after quantiles
        }

        if r_overall is None or r_semantic is None:
            out["unscored_reason"] = "missing_stage4" if not has_stage4 else "no_components"
            unscored_rows.append(out)
        else:
            scored_rows.append(out)

    # ---- Provisional tertile categories on R(G) ----
    overall_vals = np.asarray([r["R_overall_unweighted"] for r in scored_rows], dtype=np.float64)
    t1 = float(np.percentile(overall_vals, 100.0 / 3.0)) if overall_vals.size else 0.0
    t2 = float(np.percentile(overall_vals, 200.0 / 3.0)) if overall_vals.size else 0.0
    for r in scored_rows:
        v = r["R_overall_unweighted"]
        r["readability_category"] = "low" if v < t1 else ("medium" if v < t2 else "high")

    # ---- Write outputs ----
    fieldnames = ["paper_id", "model", "ga_path", *METADATA_COLS, "primary_domain",
                  "has_text_component", "has_stage4",
                  "R_text", "R_visual", "R_semantic", "R_overall_unweighted",
                  "readability_category"]

    with scores_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored_rows)

    with scores_jsonl.open("w", encoding="utf-8") as f:
        for r in scored_rows:
            f.write(json.dumps({k: r.get(k) for k in fieldnames}, ensure_ascii=False) + "\n")

    with failed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["unscored_reason"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unscored_rows)

    params = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": model_key,
        "method": "robust_minmax_p5_p95_unweighted_mean",
        "percentiles": {"low": P_LOW, "high": P_HIGH},
        "text_params": text_params,
        "visual_params": visual_params,
        "categorical_maps": {
            "clarity3": CLARITY3, "ynp": YNP, "hml": HML, "arc": ARC, "clutter": CLUTTER,
        },
        "semantic_score_fields_positive": SEM_SCORE_POS,
        "semantic_score_fields_reverse_coded": SEM_SCORE_REV,
        "category_thresholds_tertile": {"t1_low_medium": t1, "t2_medium_high": t2},
        "note": "Unweighted baseline. Text + Stage-3 visual params are identical across "
                "models by construction (shared inputs); only num_panels and R_semantic "
                "are model-specific. Calibrated weights/thresholds are fit in Stage 6.",
    }
    with params_json.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # ---- Group summaries ----
    def write_group_summary(rows: List[Dict[str, Any]], key: str, out_csv: Path) -> None:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[str(r.get(key) or "unknown").strip() or "unknown"].append(r)

        def col_mean(items, col):
            xs = [to_float(i.get(col)) for i in items]
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 6) if xs else ""

        out = []
        for k, items in sorted(groups.items(), key=lambda x: x[0]):
            cats = [i.get("readability_category") for i in items]
            n = len(items)
            out.append({
                key: k, "n": n,
                "mean_R_overall": col_mean(items, "R_overall_unweighted"),
                "mean_R_text": col_mean(items, "R_text"),
                "mean_R_visual": col_mean(items, "R_visual"),
                "mean_R_semantic": col_mean(items, "R_semantic"),
                "pct_low": round(100 * cats.count("low") / n, 2) if n else "",
                "pct_medium": round(100 * cats.count("medium") / n, 2) if n else "",
                "pct_high": round(100 * cats.count("high") / n, 2) if n else "",
            })
        cols = [key, "n", "mean_R_overall", "mean_R_text", "mean_R_visual", "mean_R_semantic",
                "pct_low", "pct_medium", "pct_high"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(out)

    write_group_summary(scored_rows, "publication_year", year_summary_csv)
    write_group_summary(scored_rows, "primary_domain", domain_summary_csv)
    write_group_summary(scored_rows, "journal", journal_summary_csv)

    # ---- Report ----
    def stat(col):
        xs = [to_float(r.get(col)) for r in scored_rows]
        xs = [x for x in xs if x is not None]
        if not xs:
            return "n/a"
        a = np.asarray(xs)
        return f"mean={a.mean():.4f} median={np.median(a):.4f} sd={a.std():.4f} min={a.min():.4f} max={a.max():.4f}"

    lines = [
        f"Task 2 Stage 5 Readability Scoring Report (UNWEIGHTED BASELINE) - {model_key}",
        "==============================================================",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Model: {model_key}",
        f"Stage4 CSV: {stage4_csv}",
        "",
        f"Stage2 rows: {len(s2)} | Stage3 rows: {len(s3)} | Stage4 rows: {len(s4)}",
        f"Scored GAs: {len(scored_rows)}",
        f"Unscored GAs (no Stage 4 / no components): {len(unscored_rows)}",
        f"GAs with no text component (R_text=NaN): {sum(1 for r in scored_rows if not r['has_text_component'])}",
        "",
        "Score distributions (scored GAs)",
        "--------------------------------",
        f"R_text:    {stat('R_text')}",
        f"R_visual:  {stat('R_visual')}",
        f"R_semantic:{stat('R_semantic')}",
        f"R_overall: {stat('R_overall_unweighted')}",
        "",
        "Provisional tertile thresholds (R_overall_unweighted)",
        "-----------------------------------------------------",
        f"low  < {t1:.4f} <= medium < {t2:.4f} <= high",
        f"low:    {sum(1 for r in scored_rows if r['readability_category']=='low')}",
        f"medium: {sum(1 for r in scored_rows if r['readability_category']=='medium')}",
        f"high:   {sum(1 for r in scored_rows if r['readability_category']=='high')}",
        "",
        "Outputs",
        "-------",
        f"Scores CSV: {scores_csv}",
        f"Params JSON: {params_json}",
        f"Year summary: {year_summary_csv}",
        f"Domain summary: {domain_summary_csv}",
        f"Journal summary: {journal_summary_csv}",
        f"Unscored CSV: {failed_csv}",
        "",
        "NOTE: This is the unweighted baseline. Stage 6 fits a*R_text+b*R_visual+c*R_semantic",
        "against human ratings and replaces the tertile category thresholds with",
        "human-aligned ones. The component columns here are the calibration inputs.",
    ]
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[{model_key}] Scored {len(scored_rows)} | unscored {len(unscored_rows)} -> {scores_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated model keys (default: all). Keys: " + ",".join(MODELS.keys()))
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    root = Path(args.project_root)
    out_base = root / "output"

    keys = ([m.strip() for m in args.models.split(",")] if args.models else list(MODELS.keys()))
    keys = [k for k in keys if k in MODELS]
    if not keys:
        raise SystemExit("ERROR: no valid model keys. Choose from: " + ", ".join(MODELS.keys()))

    print("Reading shared inputs (Stage 2, Stage 3)...")
    s2 = read_csv(root / "output" / "stage2_ocr_text_features" / "features" / "stage2_ocr_text_features.csv")
    s3 = read_csv(root / "output" / "stage3_visual_complexity_features" / "features" / "stage3_visual_complexity_features.csv")

    ran = 0
    for k in keys:
        cfg = MODELS[k]
        stage4_csv = out_base / cfg["stage4_csv"]
        out_dir = out_base / cfg["out_subdir"]
        if not stage4_csv.exists():
            print(f"[skip] {k}: Stage-4 CSV not found at {stage4_csv}")
            continue
        print(f"\n========== MODEL: {k} ==========")
        run_for_model(k, stage4_csv, out_dir, s2, s3)
        ran += 1

    if ran == 0:
        raise SystemExit("ERROR: no model scored (Stage-4 CSVs missing).")
    print(f"\n[all done] models scored: {ran}")


if __name__ == "__main__":
    main()