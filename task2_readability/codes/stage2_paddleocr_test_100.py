#!/usr/bin/env python3

import os
import re
import csv
import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime
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
    print("Install PaddleOCR in your environment before running this script.")
    sys.exit(1)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

STAGE1_INDEX = Path(
    "./task2_readability/output/stage1_preprocessing/index/stage1_ga_index.csv"
)

OUTPUT_DIR = Path(
    "./task2_readability/output/stage2_ocr_test_100"
)

OCR_BOX_DIR = OUTPUT_DIR / "ocr_boxes_json"
RAW_JSON_DIR = OUTPUT_DIR / "raw_paddleocr_json"

OCR_SUMMARY_CSV = OUTPUT_DIR / "stage2_paddleocr_test_100_summary.csv"
OCR_FAILED_CSV = OUTPUT_DIR / "stage2_paddleocr_test_100_failed.csv"
OCR_REPORT_TXT = OUTPUT_DIR / "stage2_paddleocr_test_100_report.txt"

MAX_IMAGES = 100


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def make_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OCR_BOX_DIR.mkdir(parents=True, exist_ok=True)
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(value):
    value = str(value)
    value = value.strip()
    value = re.sub(r"[^\w.\-]+", "_", value)
    value = value.strip("_")
    if not value:
        value = "unknown"
    return value[:180]


def safe_float(value):
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


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
        # Format: [x_min, y_min, x_max, y_max]
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

        # Format: [[x,y], [x,y], [x,y], [x,y]]
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

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        return width, height

    except Exception:
        return 0.0, 0.0


def get_image_dimensions_from_file(image_path):
    try:
        with Image.open(image_path) as img:
            return float(img.width), float(img.height)
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
        "nano", "bio", "clinical", "plasma", "serum"
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


# ---------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------

def read_stage1_index():
    rows = []

    with open(STAGE1_INDEX, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            paper_id = (
                row.get("paper_id")
                or row.get("doi")
                or row.get("doi_folder")
                or row.get("folder_name")
                or ""
            )

            ga_path = row.get("ga_path") or row.get("graphical_abstract_path") or ""

            if paper_id and ga_path:
                row["paper_id"] = paper_id
                row["ga_path"] = ga_path
                rows.append(row)

    return rows


# ---------------------------------------------------------------------
# PaddleOCR 3.x parsing
# ---------------------------------------------------------------------

def extract_result_dict(result_item, paper_id, result_index):
    raw_data = {}

    safe_id = safe_filename(paper_id)
    raw_json_path = RAW_JSON_DIR / f"{safe_id}_raw_{result_index}.json"

    try:
        if hasattr(result_item, "save_to_json"):
            result_item.save_to_json(str(raw_json_path))

            if raw_json_path.exists():
                with open(raw_json_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

        elif hasattr(result_item, "json"):
            raw_data = result_item.json

        elif hasattr(result_item, "to_dict"):
            raw_data = result_item.to_dict()

        elif isinstance(result_item, dict):
            raw_data = result_item

    except Exception as e:
        raw_data = {
            "parser_error": str(e)
        }

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

        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])

        rec_polys = data.get("rec_polys", [])
        rec_boxes = data.get("rec_boxes", [])

        if rec_polys:
            box_list = rec_polys
            box_source = "rec_polys"
        elif rec_boxes:
            box_list = rec_boxes
            box_source = "rec_boxes"
        else:
            box_list = []
            box_source = ""

        rec_texts = to_plain_python(rec_texts)
        rec_scores = to_plain_python(rec_scores)
        box_list = to_plain_python(box_list)

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
# Main
# ---------------------------------------------------------------------

def main():
    make_dirs()

    print("============================================================")
    print("Stage 2 PaddleOCR test started")
    print(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    print("============================================================")

    if not STAGE1_INDEX.exists():
        print(f"ERROR: Stage 1 index not found: {STAGE1_INDEX}")
        sys.exit(1)

    rows = read_stage1_index()
    rows = rows[:MAX_IMAGES]

    print(f"Images selected for test: {len(rows)}")

    print("Initializing PaddleOCR...")

    ocr = PaddleOCR(
        lang="en",
        device="gpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )

    summary_rows = []
    failed_rows = []

    success_count = 0
    fail_count = 0
    total_boxes = 0

    for i, row in enumerate(rows, start=1):
        paper_id = row["paper_id"]
        ga_path = row["ga_path"]

        image_width = safe_float(row.get("image_width", 0))
        image_height = safe_float(row.get("image_height", 0))

        if not image_width or not image_height:
            image_width = safe_float(row.get("width", 0))
            image_height = safe_float(row.get("height", 0))

        
        if not image_width or not image_height:
            image_width = safe_float(row.get("ga_width", 0))
            image_height = safe_float(row.get("ga_height", 0))


        if not image_width or not image_height:
            image_width, image_height = get_image_dimensions_from_file(ga_path)


        safe_id = safe_filename(paper_id)
        box_json_path = OCR_BOX_DIR / f"{safe_id}.json"

        try:
            boxes = run_ocr_on_image(
                ocr=ocr,
                paper_id=paper_id,
                ga_path=ga_path,
                image_width=image_width,
                image_height=image_height
            )

            with open(box_json_path, "w", encoding="utf-8") as f:
                json.dump(boxes, f, ensure_ascii=False, indent=2)

            num_boxes = len(boxes)
            total_tokens = sum(b["token_count"] for b in boxes)
            total_characters = sum(b["character_count"] for b in boxes)

            mean_confidence = (
                sum(b["confidence"] for b in boxes) / num_boxes
                if num_boxes > 0 else 0.0
            )

            low_confidence_boxes = sum(1 for b in boxes if b["confidence"] < 0.60)
            low_confidence_box_ratio = (
                low_confidence_boxes / num_boxes if num_boxes > 0 else 0.0
            )

            total_text_area_ratio = sum(b["box_area_ratio"] for b in boxes)
            mean_box_width = (
                sum(b["box_width"] for b in boxes) / num_boxes
                if num_boxes > 0 else 0.0
            )
            mean_box_height = (
                sum(b["box_height"] for b in boxes) / num_boxes
                if num_boxes > 0 else 0.0
            )
            average_words_per_box = (
                total_tokens / num_boxes if num_boxes > 0 else 0.0
            )

            all_tokens = []
            for b in boxes:
                all_tokens.extend(str(b["text"]).split())

            acronym_count = sum(1 for t in all_tokens if is_acronym(t))
            numeric_token_count = sum(1 for t in all_tokens if is_numeric_token(t))
            scientific_like_token_count = sum(1 for t in all_tokens if is_scientific_like_token(t))

            acronym_density = acronym_count / total_tokens if total_tokens > 0 else 0.0
            numeric_token_density = numeric_token_count / total_tokens if total_tokens > 0 else 0.0
            scientific_term_density = scientific_like_token_count / total_tokens if total_tokens > 0 else 0.0

            if num_boxes == 0:
                ocr_status = "success_no_text_detected"
            else:
                ocr_status = "success"

            summary_rows.append({
                "paper_id": paper_id,
                "ga_path": ga_path,
                "ocr_status": ocr_status,
                "num_text_boxes": num_boxes,
                "total_ocr_tokens": total_tokens,
                "total_ocr_characters": total_characters,
                "mean_ocr_confidence": round(mean_confidence, 4),
                "low_confidence_boxes": low_confidence_boxes,
                "low_confidence_box_ratio": round(low_confidence_box_ratio, 6),
                "total_text_area_ratio": round(total_text_area_ratio, 6),
                "mean_box_width": round(mean_box_width, 4),
                "mean_box_height": round(mean_box_height, 4),
                "average_words_per_box": round(average_words_per_box, 4),
                "acronym_count": acronym_count,
                "acronym_density": round(acronym_density, 6),
                "numeric_token_count": numeric_token_count,
                "numeric_token_density": round(numeric_token_density, 6),
                "scientific_like_token_count": scientific_like_token_count,
                "scientific_term_density": round(scientific_term_density, 6),
                "ocr_json_path": str(box_json_path),
                "error_message": "",
            })

            success_count += 1
            total_boxes += num_boxes

        except Exception as e:
            fail_count += 1

            failed_rows.append({
                "paper_id": paper_id,
                "ga_path": ga_path,
                "error_message": str(e),
            })

            summary_rows.append({
                "paper_id": paper_id,
                "ga_path": ga_path,
                "ocr_status": "failed",
                "num_text_boxes": 0,
                "total_ocr_tokens": 0,
                "total_ocr_characters": 0,
                "mean_ocr_confidence": 0,
                "low_confidence_boxes": 0,
                "low_confidence_box_ratio": 0,
                "total_text_area_ratio": 0,
                "mean_box_width": 0,
                "mean_box_height": 0,
                "average_words_per_box": 0,
                "acronym_count": 0,
                "acronym_density": 0,
                "numeric_token_count": 0,
                "numeric_token_density": 0,
                "scientific_like_token_count": 0,
                "scientific_term_density": 0,
                "ocr_json_path": "",
                "error_message": str(e),
            })

        if i % 10 == 0:
            print(f"Processed {i}/{len(rows)} images")

    fieldnames = [
        "paper_id",
        "ga_path",
        "ocr_status",
        "num_text_boxes",
        "total_ocr_tokens",
        "total_ocr_characters",
        "mean_ocr_confidence",
        "low_confidence_boxes",
        "low_confidence_box_ratio",
        "total_text_area_ratio",
        "mean_box_width",
        "mean_box_height",
        "average_words_per_box",
        "acronym_count",
        "acronym_density",
        "numeric_token_count",
        "numeric_token_density",
        "scientific_like_token_count",
        "scientific_term_density",
        "ocr_json_path",
        "error_message",
    ]

    with open(OCR_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with open(OCR_FAILED_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames_failed = ["paper_id", "ga_path", "error_message"]
        writer = csv.DictWriter(f, fieldnames=fieldnames_failed)
        writer.writeheader()
        for row in failed_rows:
            writer.writerow(row)

    images_with_text = sum(
        1 for row in summary_rows
        if row["ocr_status"] == "success" and int(row["num_text_boxes"]) > 0
    )

    images_with_no_text = sum(
        1 for row in summary_rows
        if row["ocr_status"] == "success_no_text_detected"
    )

    with open(OCR_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Stage 2 PaddleOCR Test Report\n")
        f.write("=============================\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Input Stage 1 index: {STAGE1_INDEX}\n")
        f.write(f"Images tested: {len(rows)}\n")
        f.write(f"OCR success: {success_count}\n")
        f.write(f"OCR failed: {fail_count}\n")
        f.write(f"Images with detected text: {images_with_text}\n")
        f.write(f"Images with no detected text: {images_with_no_text}\n")
        f.write(f"Total OCR boxes detected: {total_boxes}\n")
        f.write(f"Summary CSV: {OCR_SUMMARY_CSV}\n")
        f.write(f"Failed CSV: {OCR_FAILED_CSV}\n")
        f.write(f"OCR boxes folder: {OCR_BOX_DIR}\n")
        f.write(f"Raw PaddleOCR JSON folder: {RAW_JSON_DIR}\n")

    print("============================================================")
    print("Stage 2 PaddleOCR test completed")
    print(f"OCR success: {success_count}")
    print(f"OCR failed: {fail_count}")
    print(f"Images with detected text: {images_with_text}")
    print(f"Images with no detected text: {images_with_no_text}")
    print(f"Total OCR boxes detected: {total_boxes}")
    print(f"Report: {OCR_REPORT_TXT}")
    print("============================================================")


if __name__ == "__main__":
    main()