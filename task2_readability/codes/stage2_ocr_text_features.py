#!/usr/bin/env python3

import os
import re
import csv
import json
import sys
import math
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from statistics import mean, median

from PIL import Image

# ---------------------------------------------------------------------
# Stable temp/cache settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("./task2_readability")

DEFAULT_TMP = str(PROJECT_ROOT / "tmp")
DEFAULT_CACHE = str(PROJECT_ROOT / "dependencies" / "paddleocr_models")

Path(os.environ.get("TMPDIR", DEFAULT_TMP)).mkdir(parents=True, exist_ok=True)
Path(os.environ.get("HF_HOME", DEFAULT_CACHE)).mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TMPDIR", DEFAULT_TMP)
os.environ.setdefault("TMP", os.environ["TMPDIR"])
os.environ.setdefault("TEMP", os.environ["TMPDIR"])

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HOME", DEFAULT_CACHE)
os.environ.setdefault("PADDLE_HOME", DEFAULT_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", DEFAULT_CACHE)

tempfile.tempdir = os.environ["TMPDIR"]

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("ERROR: paddleocr is not installed in this Python environment.")
    sys.exit(1)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

STAGE1_INDEX = Path(
    "./task2_readability/output/stage1_preprocessing/index/stage1_ga_index.csv"
)

OUTPUT_DIR = Path(
    "./task2_readability/output/stage2_ocr_text_features"
)

FEATURES_DIR = OUTPUT_DIR / "features"
OCR_BOX_DIR = OUTPUT_DIR / "ocr_boxes_json"
FAILURE_DIR = OUTPUT_DIR / "failures"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
REPORT_DIR = OUTPUT_DIR / "reports"
TMP_RAW_JSON_DIR = OUTPUT_DIR / "tmp_raw_paddleocr_json"

FEATURES_CSV = FEATURES_DIR / "stage2_ocr_text_features.csv"
FEATURES_JSONL = FEATURES_DIR / "stage2_ocr_text_features.jsonl"
FAILED_CSV = FAILURE_DIR / "stage2_ocr_failed_images.csv"
REPORT_TXT = REPORT_DIR / "stage2_ocr_text_features_report.txt"

YEAR_SUMMARY_CSV = SUMMARY_DIR / "stage2_ocr_year_summary.csv"
DOMAIN_SUMMARY_CSV = SUMMARY_DIR / "stage2_ocr_domain_summary.csv"
JOURNAL_SUMMARY_CSV = SUMMARY_DIR / "stage2_ocr_journal_summary.csv"
HEALTH_SUMMARY_CSV = SUMMARY_DIR / "stage2_ocr_health_summary.csv"

FORCE_RESTART = os.environ.get("FORCE_RESTART", "0") == "1"

LOW_CONFIDENCE_THRESHOLD = 0.60
VERY_LOW_CONFIDENCE_THRESHOLD = 0.40
TINY_BOX_HEIGHT_PX = 12
TINY_BOX_AREA_RATIO = 0.0002
VERY_DENSE_TEXT_AREA_RATIO = 0.30
HIGH_BOX_COUNT_THRESHOLD = 80


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def reset_outputs_if_needed():
    if FORCE_RESTART and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def make_dirs():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    OCR_BOX_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(value):
    value = str(value).strip()
    value = re.sub(r"[^\w.\-]+", "_", value)
    value = value.strip("_")
    if not value:
        value = "unknown"
    return value[:180]


def safe_float(value):
    try:
        if value is None or value == "":
            return 0.0
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    except Exception:
        return 0.0


def safe_int(value):
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except Exception:
        return 0


def pick(row, names, default=""):
    for name in names:
        if name in row and row[name] not in [None, ""]:
            return row[name]
    return default


def safe_len_tokens(text):
    if not text:
        return 0
    return len(str(text).split())


def to_plain_python(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_python(v) for v in obj]
    return obj


def normalize_box_to_polygon(box):
    box = to_plain_python(box)

    if not box:
        return []

    try:
        if (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(v, (int, float)) for v in box)
        ):
            x1, y1, x2, y2 = [float(v) for v in box]
            return [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]

        if (
            isinstance(box, list)
            and len(box) >= 4
            and isinstance(box[0], (list, tuple))
            and len(box[0]) >= 2
        ):
            return [[float(p[0]), float(p[1])] for p in box]

    except Exception:
        return []

    return []


def get_box_width_height(box):
    polygon = normalize_box_to_polygon(box)

    if not polygon:
        return 0.0, 0.0

    try:
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        return max(xs) - min(xs), max(ys) - min(ys)
    except Exception:
        return 0.0, 0.0


def polygon_area_ratio(box, image_width, image_height):
    polygon = normalize_box_to_polygon(box)

    if not polygon or not image_width or not image_height:
        return 0.0

    try:
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        image_area = float(image_width) * float(image_height)
        box_area = width * height

        if image_area <= 0:
            return 0.0

        return box_area / image_area

    except Exception:
        return 0.0


def get_image_dimensions_from_file(image_path):
    try:
        with Image.open(image_path) as img:
            return float(img.width), float(img.height)
    except Exception:
        return 0.0, 0.0


def is_acronym(token):
    token = str(token).strip()
    return bool(re.fullmatch(r"[A-Z]{2,}", token))


def is_numeric_token(token):
    token = str(token).strip()
    return bool(re.search(r"\d", token))


def is_scientific_like_token(token):
    token = str(token).strip()

    if len(token) < 4:
        return False

    common_science_parts = [
        "cell", "gene", "rna", "dna", "protein", "enzyme", "virus",
        "bacteria", "tumor", "cancer", "immune", "pathway", "assay",
        "model", "sample", "therapy", "molecule", "receptor",
        "expression", "analysis", "carbon", "nitrogen", "micro",
        "nano", "bio", "clinical", "plasma", "serum", "method",
        "treatment", "disease", "signal", "culture", "species",
        "soil", "water", "plant", "drug", "dose", "mutation",
        "infection", "metabolism", "synthesis", "surface"
    ]

    lower = token.lower()

    for part in common_science_parts:
        if part in lower:
            return True

    if re.search(r"[A-Za-z]+[-_/][A-Za-z0-9]+", token):
        return True

    if re.search(r"[A-Za-z]+[0-9]+", token):
        return True

    return False


def list_mean(values):
    values = [safe_float(v) for v in values if v is not None]
    if not values:
        return 0.0
    return mean(values)


def list_median(values):
    values = [safe_float(v) for v in values if v is not None]
    if not values:
        return 0.0
    return median(values)


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

def read_stage1_index():
    rows = []

    with open(STAGE1_INDEX, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            paper_id = pick(row, ["paper_id", "doi", "doi_folder", "folder_name"])
            ga_path = pick(row, ["ga_path", "graphical_abstract_path"])

            if paper_id and ga_path:
                row["paper_id"] = paper_id
                row["ga_path"] = ga_path
                rows.append(row)

    return rows


# ---------------------------------------------------------------------
# PaddleOCR parsing
# ---------------------------------------------------------------------

def extract_result_dict(result_item, paper_id, result_index):
    raw_data = {}

    safe_id = safe_filename(paper_id)
    raw_json_path = TMP_RAW_JSON_DIR / f"{safe_id}_raw_{result_index}.json"

    try:
        if hasattr(result_item, "save_to_json"):
            result_item.save_to_json(str(raw_json_path))

            if raw_json_path.exists():
                with open(raw_json_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                try:
                    raw_json_path.unlink()
                except Exception:
                    pass

        elif hasattr(result_item, "json"):
            raw_data = result_item.json

        elif hasattr(result_item, "to_dict"):
            raw_data = result_item.to_dict()

        elif isinstance(result_item, dict):
            raw_data = result_item

    except Exception as e:
        raw_data = {"parser_error": str(e)}

    raw_data = to_plain_python(raw_data)

    if isinstance(raw_data, dict) and "res" in raw_data and isinstance(raw_data["res"], dict):
        return raw_data["res"]

    if isinstance(raw_data, dict):
        return raw_data

    return {}


def run_ocr_on_image(ocr, paper_id, ga_path, image_width, image_height):
    result = ocr.predict(str(ga_path))

    boxes = []

    if not result:
        return boxes

    for result_index, result_item in enumerate(result):
        data = extract_result_dict(
            result_item=result_item,
            paper_id=paper_id,
            result_index=result_index
        )

        if not isinstance(data, dict):
            continue

        rec_texts = to_plain_python(data.get("rec_texts", []))
        rec_scores = to_plain_python(data.get("rec_scores", []))

        rec_polys = to_plain_python(data.get("rec_polys", []))
        rec_boxes = to_plain_python(data.get("rec_boxes", []))

        if rec_polys:
            box_list = rec_polys
            box_source = "rec_polys"
        elif rec_boxes:
            box_list = rec_boxes
            box_source = "rec_boxes"
        else:
            box_list = []
            box_source = ""

        for idx, text in enumerate(rec_texts):
            try:
                text = str(text).strip()

                if not text:
                    continue

                confidence = safe_float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
                raw_box = box_list[idx] if idx < len(box_list) else []

                polygon = normalize_box_to_polygon(raw_box)
                width, height = get_box_width_height(polygon)
                area_ratio = polygon_area_ratio(polygon, image_width, image_height)

                boxes.append({
                    "paper_id": paper_id,
                    "ga_path": str(ga_path),
                    "text": text,
                    "confidence": confidence,
                    "box": polygon,
                    "box_source": box_source,
                    "box_width": width,
                    "box_height": height,
                    "box_area_ratio": area_ratio,
                    "token_count": safe_len_tokens(text),
                    "character_count": len(text),
                })

            except Exception:
                continue

    return boxes


# ---------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------

def get_image_width_height(row):
    ga_path = row["ga_path"]

    image_width = safe_float(pick(row, ["image_width", "width", "ga_width"]))
    image_height = safe_float(pick(row, ["image_height", "height", "ga_height"]))

    if not image_width or not image_height:
        image_width, image_height = get_image_dimensions_from_file(ga_path)

    return image_width, image_height


def compute_feature_row(row, boxes, box_json_path):
    paper_id = row["paper_id"]
    ga_path = row["ga_path"]

    publication_year = pick(row, ["publication_year", "year", "pub_year"])
    journal = pick(row, ["journal", "journal_name", "source_title"])
    publisher = pick(row, ["publisher", "publisher_name"])
    domain = pick(row, ["domain", "matched_domain"])
    subject_area = pick(row, ["subject_area", "subject_areas"])
    subject_categories = pick(row, ["subject_categories", "categories"])

    image_width, image_height = get_image_width_height(row)

    num_boxes = len(boxes)
    total_tokens = sum(b["token_count"] for b in boxes)
    total_characters = sum(b["character_count"] for b in boxes)

    confidences = [b["confidence"] for b in boxes]
    box_widths = [b["box_width"] for b in boxes]
    box_heights = [b["box_height"] for b in boxes]
    box_area_ratios = [b["box_area_ratio"] for b in boxes]

    mean_confidence = list_mean(confidences)
    median_confidence = list_median(confidences)
    min_confidence = min(confidences) if confidences else 0.0
    max_confidence = max(confidences) if confidences else 0.0

    low_confidence_boxes = sum(1 for b in boxes if b["confidence"] < LOW_CONFIDENCE_THRESHOLD)
    very_low_confidence_boxes = sum(1 for b in boxes if b["confidence"] < VERY_LOW_CONFIDENCE_THRESHOLD)

    low_confidence_box_ratio = low_confidence_boxes / num_boxes if num_boxes > 0 else 0.0
    very_low_confidence_box_ratio = very_low_confidence_boxes / num_boxes if num_boxes > 0 else 0.0

    total_text_area_ratio = sum(box_area_ratios)
    mean_box_width = list_mean(box_widths)
    median_box_width = list_median(box_widths)
    mean_box_height = list_mean(box_heights)
    median_box_height = list_median(box_heights)
    mean_box_area_ratio = list_mean(box_area_ratios)
    median_box_area_ratio = list_median(box_area_ratios)

    tiny_text_boxes = sum(
        1 for b in boxes
        if b["box_height"] < TINY_BOX_HEIGHT_PX or b["box_area_ratio"] < TINY_BOX_AREA_RATIO
    )
    tiny_text_box_ratio = tiny_text_boxes / num_boxes if num_boxes > 0 else 0.0

    average_words_per_box = total_tokens / num_boxes if num_boxes > 0 else 0.0
    average_chars_per_box = total_characters / num_boxes if num_boxes > 0 else 0.0

    all_tokens = []
    for b in boxes:
        all_tokens.extend(str(b["text"]).split())

    acronym_count = sum(1 for t in all_tokens if is_acronym(t))
    numeric_token_count = sum(1 for t in all_tokens if is_numeric_token(t))
    scientific_like_token_count = sum(1 for t in all_tokens if is_scientific_like_token(t))

    acronym_density = acronym_count / total_tokens if total_tokens > 0 else 0.0
    numeric_token_density = numeric_token_count / total_tokens if total_tokens > 0 else 0.0
    scientific_term_density = scientific_like_token_count / total_tokens if total_tokens > 0 else 0.0

    detected_text_joined = " ".join([b["text"] for b in boxes])

    if num_boxes == 0:
        ocr_status = "success_no_text_detected"
    else:
        ocr_status = "success"

    flag_no_text = int(num_boxes == 0)
    flag_low_mean_confidence = int(mean_confidence < LOW_CONFIDENCE_THRESHOLD) if num_boxes > 0 else 0
    flag_many_low_confidence_boxes = int(low_confidence_box_ratio >= 0.30) if num_boxes > 0 else 0
    flag_dense_text_area = int(total_text_area_ratio >= VERY_DENSE_TEXT_AREA_RATIO)
    flag_many_text_boxes = int(num_boxes >= HIGH_BOX_COUNT_THRESHOLD)
    flag_many_tiny_text_boxes = int(tiny_text_box_ratio >= 0.30) if num_boxes > 0 else 0

    return {
        "paper_id": paper_id,
        "ga_path": ga_path,
        "publication_year": publication_year,
        "journal": journal,
        "publisher": publisher,
        "domain": domain,
        "subject_area": subject_area,
        "subject_categories": subject_categories,
        "ocr_status": ocr_status,
        "image_width": round(image_width, 4),
        "image_height": round(image_height, 4),
        "num_text_boxes": num_boxes,
        "total_ocr_tokens": total_tokens,
        "total_ocr_characters": total_characters,
        "mean_ocr_confidence": round(mean_confidence, 6),
        "median_ocr_confidence": round(median_confidence, 6),
        "min_ocr_confidence": round(min_confidence, 6),
        "max_ocr_confidence": round(max_confidence, 6),
        "low_confidence_boxes": low_confidence_boxes,
        "very_low_confidence_boxes": very_low_confidence_boxes,
        "low_confidence_box_ratio": round(low_confidence_box_ratio, 6),
        "very_low_confidence_box_ratio": round(very_low_confidence_box_ratio, 6),
        "total_text_area_ratio": round(total_text_area_ratio, 6),
        "mean_box_width": round(mean_box_width, 6),
        "median_box_width": round(median_box_width, 6),
        "mean_box_height": round(mean_box_height, 6),
        "median_box_height": round(median_box_height, 6),
        "mean_box_area_ratio": round(mean_box_area_ratio, 8),
        "median_box_area_ratio": round(median_box_area_ratio, 8),
        "tiny_text_boxes": tiny_text_boxes,
        "tiny_text_box_ratio": round(tiny_text_box_ratio, 6),
        "average_words_per_box": round(average_words_per_box, 6),
        "average_chars_per_box": round(average_chars_per_box, 6),
        "acronym_count": acronym_count,
        "acronym_density": round(acronym_density, 6),
        "numeric_token_count": numeric_token_count,
        "numeric_token_density": round(numeric_token_density, 6),
        "scientific_like_token_count": scientific_like_token_count,
        "scientific_term_density": round(scientific_term_density, 6),
        "flag_no_text": flag_no_text,
        "flag_low_mean_confidence": flag_low_mean_confidence,
        "flag_many_low_confidence_boxes": flag_many_low_confidence_boxes,
        "flag_dense_text_area": flag_dense_text_area,
        "flag_many_text_boxes": flag_many_text_boxes,
        "flag_many_tiny_text_boxes": flag_many_tiny_text_boxes,
        "detected_text_joined": detected_text_joined,
        "ocr_json_path": str(box_json_path),
        "error_message": "",
    }


def failure_feature_row(row, error_message):
    return {
        "paper_id": row.get("paper_id", ""),
        "ga_path": row.get("ga_path", ""),
        "publication_year": pick(row, ["publication_year", "year", "pub_year"]),
        "journal": pick(row, ["journal", "journal_name", "source_title"]),
        "publisher": pick(row, ["publisher", "publisher_name"]),
        "domain": pick(row, ["domain", "matched_domain"]),
        "subject_area": pick(row, ["subject_area", "subject_areas"]),
        "subject_categories": pick(row, ["subject_categories", "categories"]),
        "ocr_status": "failed",
        "image_width": 0,
        "image_height": 0,
        "num_text_boxes": 0,
        "total_ocr_tokens": 0,
        "total_ocr_characters": 0,
        "mean_ocr_confidence": 0,
        "median_ocr_confidence": 0,
        "min_ocr_confidence": 0,
        "max_ocr_confidence": 0,
        "low_confidence_boxes": 0,
        "very_low_confidence_boxes": 0,
        "low_confidence_box_ratio": 0,
        "very_low_confidence_box_ratio": 0,
        "total_text_area_ratio": 0,
        "mean_box_width": 0,
        "median_box_width": 0,
        "mean_box_height": 0,
        "median_box_height": 0,
        "mean_box_area_ratio": 0,
        "median_box_area_ratio": 0,
        "tiny_text_boxes": 0,
        "tiny_text_box_ratio": 0,
        "average_words_per_box": 0,
        "average_chars_per_box": 0,
        "acronym_count": 0,
        "acronym_density": 0,
        "numeric_token_count": 0,
        "numeric_token_density": 0,
        "scientific_like_token_count": 0,
        "scientific_term_density": 0,
        "flag_no_text": 0,
        "flag_low_mean_confidence": 0,
        "flag_many_low_confidence_boxes": 0,
        "flag_dense_text_area": 0,
        "flag_many_text_boxes": 0,
        "flag_many_tiny_text_boxes": 0,
        "detected_text_joined": "",
        "ocr_json_path": "",
        "error_message": error_message,
    }


FIELDNAMES = [
    "paper_id",
    "ga_path",
    "publication_year",
    "journal",
    "publisher",
    "domain",
    "subject_area",
    "subject_categories",
    "ocr_status",
    "image_width",
    "image_height",
    "num_text_boxes",
    "total_ocr_tokens",
    "total_ocr_characters",
    "mean_ocr_confidence",
    "median_ocr_confidence",
    "min_ocr_confidence",
    "max_ocr_confidence",
    "low_confidence_boxes",
    "very_low_confidence_boxes",
    "low_confidence_box_ratio",
    "very_low_confidence_box_ratio",
    "total_text_area_ratio",
    "mean_box_width",
    "median_box_width",
    "mean_box_height",
    "median_box_height",
    "mean_box_area_ratio",
    "median_box_area_ratio",
    "tiny_text_boxes",
    "tiny_text_box_ratio",
    "average_words_per_box",
    "average_chars_per_box",
    "acronym_count",
    "acronym_density",
    "numeric_token_count",
    "numeric_token_density",
    "scientific_like_token_count",
    "scientific_term_density",
    "flag_no_text",
    "flag_low_mean_confidence",
    "flag_many_low_confidence_boxes",
    "flag_dense_text_area",
    "flag_many_text_boxes",
    "flag_many_tiny_text_boxes",
    "detected_text_joined",
    "ocr_json_path",
    "error_message",
]


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def write_group_summary(rows, group_field, output_csv):
    groups = {}

    for row in rows:
        key = str(row.get(group_field, "") or "UNKNOWN").strip() or "UNKNOWN"

        if key not in groups:
            groups[key] = []

        groups[key].append(row)

    out_rows = []

    for key, items in sorted(groups.items(), key=lambda x: x[0]):
        total = len(items)
        failed = sum(1 for r in items if r["ocr_status"] == "failed")
        no_text = sum(1 for r in items if r["ocr_status"] == "success_no_text_detected")
        success = sum(1 for r in items if r["ocr_status"] == "success")

        out_rows.append({
            group_field: key,
            "total_images": total,
            "ocr_success": success,
            "ocr_failed": failed,
            "success_no_text_detected": no_text,
            "mean_num_text_boxes": round(list_mean([r["num_text_boxes"] for r in items]), 6),
            "mean_total_ocr_tokens": round(list_mean([r["total_ocr_tokens"] for r in items]), 6),
            "mean_ocr_confidence": round(list_mean([r["mean_ocr_confidence"] for r in items]), 6),
            "mean_text_area_ratio": round(list_mean([r["total_text_area_ratio"] for r in items]), 6),
            "mean_low_confidence_box_ratio": round(list_mean([r["low_confidence_box_ratio"] for r in items]), 6),
            "mean_acronym_density": round(list_mean([r["acronym_density"] for r in items]), 6),
            "mean_numeric_token_density": round(list_mean([r["numeric_token_density"] for r in items]), 6),
            "mean_scientific_term_density": round(list_mean([r["scientific_term_density"] for r in items]), 6),
        })

    fieldnames = [
        group_field,
        "total_images",
        "ocr_success",
        "ocr_failed",
        "success_no_text_detected",
        "mean_num_text_boxes",
        "mean_total_ocr_tokens",
        "mean_ocr_confidence",
        "mean_text_area_ratio",
        "mean_low_confidence_box_ratio",
        "mean_acronym_density",
        "mean_numeric_token_density",
        "mean_scientific_term_density",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def write_health_summary(rows):
    total = len(rows)
    success = sum(1 for r in rows if r["ocr_status"] == "success")
    failed = sum(1 for r in rows if r["ocr_status"] == "failed")
    no_text = sum(1 for r in rows if r["ocr_status"] == "success_no_text_detected")

    health_rows = [
        {"metric": "total_images", "value": total},
        {"metric": "ocr_success", "value": success},
        {"metric": "ocr_failed", "value": failed},
        {"metric": "success_no_text_detected", "value": no_text},
        {"metric": "total_text_boxes", "value": sum(safe_int(r["num_text_boxes"]) for r in rows)},
        {"metric": "mean_text_boxes_per_image", "value": round(list_mean([r["num_text_boxes"] for r in rows]), 6)},
        {"metric": "mean_ocr_confidence", "value": round(list_mean([r["mean_ocr_confidence"] for r in rows]), 6)},
        {"metric": "mean_text_area_ratio", "value": round(list_mean([r["total_text_area_ratio"] for r in rows]), 6)},
        {"metric": "images_flag_low_mean_confidence", "value": sum(safe_int(r["flag_low_mean_confidence"]) for r in rows)},
        {"metric": "images_flag_many_low_confidence_boxes", "value": sum(safe_int(r["flag_many_low_confidence_boxes"]) for r in rows)},
        {"metric": "images_flag_dense_text_area", "value": sum(safe_int(r["flag_dense_text_area"]) for r in rows)},
        {"metric": "images_flag_many_text_boxes", "value": sum(safe_int(r["flag_many_text_boxes"]) for r in rows)},
        {"metric": "images_flag_many_tiny_text_boxes", "value": sum(safe_int(r["flag_many_tiny_text_boxes"]) for r in rows)},
    ]

    with open(HEALTH_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(health_rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    reset_outputs_if_needed()
    make_dirs()

    print("============================================================")
    print("Stage 2 OCR Text Feature Extraction started")
    print(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    print("============================================================")

    if not STAGE1_INDEX.exists():
        print(f"ERROR: Stage 1 index not found: {STAGE1_INDEX}")
        sys.exit(1)

    rows = read_stage1_index()

    print(f"Images selected for full Stage 2 OCR: {len(rows)}")

    print("Initializing PaddleOCR...")

    ocr = PaddleOCR(
        lang="en",
        device="gpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )

    feature_rows = []
    failed_rows = []

    success_count = 0
    fail_count = 0
    no_text_count = 0
    total_boxes = 0

    for i, row in enumerate(rows, start=1):
        paper_id = row["paper_id"]
        ga_path = row["ga_path"]
        safe_id = safe_filename(paper_id)
        box_json_path = OCR_BOX_DIR / f"{safe_id}.json"

        try:
            image_width, image_height = get_image_width_height(row)

            boxes = run_ocr_on_image(
                ocr=ocr,
                paper_id=paper_id,
                ga_path=ga_path,
                image_width=image_width,
                image_height=image_height
            )

            with open(box_json_path, "w", encoding="utf-8") as f:
                json.dump(boxes, f, ensure_ascii=False, indent=2)

            feature_row = compute_feature_row(
                row=row,
                boxes=boxes,
                box_json_path=box_json_path
            )

            feature_rows.append(feature_row)

            if feature_row["ocr_status"] == "success":
                success_count += 1
            elif feature_row["ocr_status"] == "success_no_text_detected":
                success_count += 1
                no_text_count += 1

            total_boxes += len(boxes)

        except Exception as e:
            fail_count += 1

            failed_rows.append({
                "paper_id": paper_id,
                "ga_path": ga_path,
                "error_message": str(e),
            })

            feature_rows.append(failure_feature_row(row, str(e)))

        if i % 25 == 0:
            print(f"Processed {i}/{len(rows)} images")

    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(feature_rows)

    with open(FEATURES_JSONL, "w", encoding="utf-8") as f:
        for row in feature_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(FAILED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_id", "ga_path", "error_message"])
        writer.writeheader()
        writer.writerows(failed_rows)

    write_group_summary(feature_rows, "publication_year", YEAR_SUMMARY_CSV)
    write_group_summary(feature_rows, "domain", DOMAIN_SUMMARY_CSV)
    write_group_summary(feature_rows, "journal", JOURNAL_SUMMARY_CSV)
    write_health_summary(feature_rows)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Stage 2 OCR Text Feature Extraction Report\n")
        f.write("==========================================\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Input Stage 1 index: {STAGE1_INDEX}\n")
        f.write(f"Images processed: {len(rows)}\n")
        f.write(f"OCR success: {success_count}\n")
        f.write(f"OCR failed: {fail_count}\n")
        f.write(f"Images with no detected text: {no_text_count}\n")
        f.write(f"Total OCR boxes detected: {total_boxes}\n\n")

        f.write("Main outputs:\n")
        f.write(f"- Feature CSV: {FEATURES_CSV}\n")
        f.write(f"- Feature JSONL: {FEATURES_JSONL}\n")
        f.write(f"- OCR boxes folder: {OCR_BOX_DIR}\n")
        f.write(f"- Failed OCR CSV: {FAILED_CSV}\n")
        f.write(f"- Year summary: {YEAR_SUMMARY_CSV}\n")
        f.write(f"- Domain summary: {DOMAIN_SUMMARY_CSV}\n")
        f.write(f"- Journal summary: {JOURNAL_SUMMARY_CSV}\n")
        f.write(f"- Health summary: {HEALTH_SUMMARY_CSV}\n")

    print("============================================================")
    print("Stage 2 OCR Text Feature Extraction completed")
    print(f"OCR success: {success_count}")
    print(f"OCR failed: {fail_count}")
    print(f"Images with no detected text: {no_text_count}")
    print(f"Total OCR boxes detected: {total_boxes}")
    print(f"Report: {REPORT_TXT}")
    print("============================================================")


if __name__ == "__main__":
    main()