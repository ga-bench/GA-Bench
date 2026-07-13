#!/usr/bin/env python3
"""
Task 2 / Stage 4 Consolidation
==============================

The full Stage 4 run was resume-sharded, so the per-half results CSVs only reflect
the LAST batch (mostly 'skipped' rows). The authoritative output is the set of
per-GA interpretation JSONs in:

    output/stage4_vlm_structural_interpretation/interpretations/*.json

This script scans every interpretation JSON, flattens it into ONE master CSV (one
row per paper_id) and regenerates the field-distribution calibration report over
the FULL set. The master CSV is the clean merge input for Stage 5.

Run (CPU only, login node or a small interactive job):
    python3 stage4_consolidate_interpretations.py
"""

import csv
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path("./task2_readability")
S4_ROOT = PROJECT_ROOT / "output" / "stage4_vlm_structural_interpretation"
INTERP_DIR = S4_ROOT / "interpretations"

OUT_DIR = S4_ROOT / "consolidated"
CSV_OUT = OUT_DIR / "stage4_interpretations.csv"
DIST_OUT = OUT_DIR / "stage4_field_distributions_full.json"
REPORT_OUT = OUT_DIR / "stage4_consolidation_report.txt"

ENUM_FIELDS = [
    "panel_structure", "layout_type", "main_reading_direction", "narrative_arc",
    "flow_clarity", "entity_clarity", "relation_clarity", "main_message_identifiable",
    "visual_clutter", "semantic_interpretability", "method_identifiable",
    "result_identifiable", "conclusion_identifiable",
]
SCORE_FIELDS = ["sequence_clarity_1to5", "key_message_clarity_1to5", "ambiguity_1to5",
                "text_dependency_1to5", "overall_interpretability_1to5"]
BOOL_FIELDS = ["has_start_point", "has_end_point", "has_arrows_or_connectors"]
METADATA_COLS = ["publication_year", "journal", "publisher", "domain", "subject_area",
                 "subject_categories"]

CSV_FIELDS = (
    ["paper_id", "ga_path", "model"] + METADATA_COLS +
    ["image_orig_width", "image_orig_height", "image_sent_width", "image_sent_height",
     "num_panels"] + ENUM_FIELDS + BOOL_FIELDS +
    ["num_main_entities", "num_process_steps", "num_relationships",
     "main_outcome_present", "main_outcome"] + SCORE_FIELDS +
    ["num_unclear_elements", "num_missing_links", "uncertainty_notes_present"]
)


def count_valid_entities(items: Any) -> int:
    if not isinstance(items, list):
        return 0
    return sum(1 for it in items if isinstance(it, dict) and str(it.get("name", "")).strip())


def count_valid_relationships(items: Any) -> int:
    if not isinstance(items, list):
        return 0
    return sum(1 for it in items if isinstance(it, dict)
               and str(it.get("from", "")).strip() and str(it.get("to", "")).strip())


def count_nonempty(items: Any) -> int:
    if not isinstance(items, list):
        return 0
    return sum(1 for s in items if str(s).strip())


def flatten(record: Dict[str, Any]) -> Dict[str, Any]:
    interp = record.get("interpretation", {})
    row: Dict[str, Any] = {k: "" for k in CSV_FIELDS}
    row["paper_id"] = record.get("paper_id", "")
    row["ga_path"] = record.get("ga_path", "")
    row["model"] = record.get("model", "")
    for c in METADATA_COLS:
        row[c] = record.get(c, "")
    for c in ["image_orig_width", "image_orig_height", "image_sent_width", "image_sent_height"]:
        row[c] = record.get(c, "")
    row["num_panels"] = interp.get("num_panels", "")
    for k in ENUM_FIELDS:
        row[k] = interp.get(k, "")
    for k in BOOL_FIELDS:
        v = interp.get(k, "")
        row[k] = int(bool(v)) if isinstance(v, bool) else v
    for k in SCORE_FIELDS:
        row[k] = interp.get(k, "")
    row["num_main_entities"] = count_valid_entities(interp.get("main_entities"))
    row["num_process_steps"] = count_nonempty(interp.get("process_steps"))
    row["num_relationships"] = count_valid_relationships(interp.get("relationships"))
    row["num_unclear_elements"] = count_nonempty(interp.get("unclear_elements"))
    row["num_missing_links"] = count_nonempty(interp.get("missing_links"))
    mo = str(interp.get("main_outcome", "")).strip()
    un = str(interp.get("uncertainty_notes", "")).strip()
    row["main_outcome"] = mo
    row["main_outcome_present"] = int(bool(mo))
    row["uncertainty_notes_present"] = int(bool(un))
    return row


def compute_distributions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    dist: Dict[str, Any] = {"n_total": len(rows)}
    for k in ENUM_FIELDS:
        dist[k] = dict(Counter(r.get(k) for r in rows))
    for k in BOOL_FIELDS:
        dist[k] = dict(Counter(r.get(k) for r in rows))
    for k in SCORE_FIELDS:
        vals = []
        for r in rows:
            try:
                vals.append(int(r.get(k)))
            except (TypeError, ValueError):
                pass
        dist[k] = {
            "histogram": dict(Counter(vals)),
            "mean": round(statistics.mean(vals), 3) if vals else 0.0,
            "median": round(statistics.median(vals), 3) if vals else 0.0,
        }
    for k in ["num_panels", "num_main_entities", "num_process_steps", "num_relationships",
              "num_unclear_elements", "num_missing_links"]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(k)))
            except (TypeError, ValueError):
                pass
        dist[k] = {
            "mean": round(statistics.mean(vals), 3) if vals else 0.0,
            "median": round(statistics.median(vals), 3) if vals else 0.0,
            "min": min(vals) if vals else 0, "max": max(vals) if vals else 0,
        }
    dist["main_outcome_present_rate_pct"] = round(
        100 * sum(1 for r in rows if r.get("main_outcome_present")) / len(rows), 2) if rows else 0.0
    return dist


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INTERP_DIR.exists():
        raise FileNotFoundError(f"Interpretations dir not found: {INTERP_DIR}")

    paths = sorted(INTERP_DIR.glob("*.json"))
    print(f"Found {len(paths)} interpretation JSON files")

    rows: List[Dict[str, Any]] = []
    bad = 0
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                record = json.load(f)
            rows.append(flatten(record))
        except Exception as e:
            bad += 1
            print(f"[WARN] could not read {p.name}: {e}")

    rows.sort(key=lambda r: r.get("paper_id", ""))

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    dist = compute_distributions(rows)
    with DIST_OUT.open("w", encoding="utf-8") as f:
        json.dump(dist, f, indent=2, ensure_ascii=False)

    lines = [
        "Task 2 Stage 4 Consolidation Report",
        "===================================",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Interpretations dir: {INTERP_DIR}",
        f"Interpretation JSONs found: {len(paths)}",
        f"Successfully consolidated: {len(rows)}",
        f"Unreadable files: {bad}",
        "",
        f"Master CSV: {CSV_OUT}",
        f"Full field distributions: {DIST_OUT}",
        "",
        "Field distributions (FULL SET — calibration check)",
        "--------------------------------------------------",
        json.dumps(dist, indent=2, ensure_ascii=False),
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Consolidated {len(rows)} rows -> {CSV_OUT}")
    print(f"Distributions -> {DIST_OUT}")
    print(f"Report -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
