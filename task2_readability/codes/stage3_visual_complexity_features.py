#!/usr/bin/env python3

import os
import re
import csv
import json
import sys
import math
import shutil
from pathlib import Path
from datetime import datetime
from statistics import mean

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV/cv2 is not installed in this Python environment.")
    sys.exit(1)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("./task2_readability")

STAGE1_INDEX = PROJECT_ROOT / "output/stage1_preprocessing/index/stage1_ga_index.csv"

STAGE2_FEATURES = PROJECT_ROOT / "output/stage2_ocr_text_features/features/stage2_ocr_text_features.csv"

OUTPUT_DIR = PROJECT_ROOT / "output/stage3_visual_complexity_features"

FEATURES_DIR = OUTPUT_DIR / "features"
PER_IMAGE_JSON_DIR = OUTPUT_DIR / "per_image_json"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
REPORT_DIR = OUTPUT_DIR / "reports"
FAILURE_DIR = OUTPUT_DIR / "failures"

FEATURES_CSV = FEATURES_DIR / "stage3_visual_complexity_features.csv"
FEATURES_JSONL = FEATURES_DIR / "stage3_visual_complexity_features.jsonl"

FAILED_CSV = FAILURE_DIR / "stage3_failed_images.csv"

YEAR_SUMMARY_CSV = SUMMARY_DIR / "stage3_year_summary.csv"
DOMAIN_SUMMARY_CSV = SUMMARY_DIR / "stage3_domain_summary.csv"
JOURNAL_SUMMARY_CSV = SUMMARY_DIR / "stage3_journal_summary.csv"
HEALTH_SUMMARY_CSV = SUMMARY_DIR / "stage3_visual_health_summary.csv"

REPORT_TXT = REPORT_DIR / "stage3_visual_complexity_features_report.txt"

FORCE_RESTART = os.environ.get("FORCE_RESTART", "0") == "1"

MAX_ANALYSIS_DIM = 1200
GRID_SIZE = 3


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def reset_outputs_if_needed():
    if FORCE_RESTART and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def make_dirs():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PER_IMAGE_JSON_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)


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


def list_mean(values):
    values = [safe_float(v) for v in values]
    if not values:
        return 0.0
    return mean(values)


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()

    if total <= 0:
        return 0.0

    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def normalized_entropy_from_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()

    if total <= 0:
        return 0.0

    p = weights[weights > 0] / total
    entropy = float(-(p * np.log2(p)).sum())
    max_entropy = math.log2(len(weights)) if len(weights) > 1 else 1.0

    if max_entropy <= 0:
        return 0.0

    return entropy / max_entropy


def read_csv_rows(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

def load_image_for_analysis(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        original_width, original_height = img.size

        scale = 1.0
        max_dim = max(original_width, original_height)

        if max_dim > MAX_ANALYSIS_DIM:
            scale = MAX_ANALYSIS_DIM / float(max_dim)
            new_width = max(1, int(round(original_width * scale)))
            new_height = max(1, int(round(original_height * scale)))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        arr = np.array(img)

    analysis_height, analysis_width = arr.shape[:2]

    return arr, original_width, original_height, analysis_width, analysis_height, scale


# ---------------------------------------------------------------------
# OCR box helpers
# ---------------------------------------------------------------------

def load_ocr_boxes(row):
    ocr_json_path = row.get("ocr_json_path", "")

    if not ocr_json_path:
        return []

    path = Path(ocr_json_path)

    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            boxes = json.load(f)

        if isinstance(boxes, list):
            return boxes

    except Exception:
        return []

    return []


def polygon_to_rect(polygon, image_width, image_height):
    if not polygon:
        return None

    try:
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]

        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(float(image_width), max(xs))
        y2 = min(float(image_height), max(ys))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    except Exception:
        return None


def rect_area(rect):
    if not rect:
        return 0.0

    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def rect_intersection_area(a, b):
    if not a or not b:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def approximate_union_area_ratio(rects, image_width, image_height, grid_size=300):
    if not rects or image_width <= 0 or image_height <= 0:
        return 0.0

    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for rect in rects:
        x1, y1, x2, y2 = rect

        gx1 = int(max(0, min(grid_size - 1, math.floor((x1 / image_width) * grid_size))))
        gy1 = int(max(0, min(grid_size - 1, math.floor((y1 / image_height) * grid_size))))
        gx2 = int(max(0, min(grid_size, math.ceil((x2 / image_width) * grid_size))))
        gy2 = int(max(0, min(grid_size, math.ceil((y2 / image_height) * grid_size))))

        if gx2 > gx1 and gy2 > gy1:
            mask[gy1:gy2, gx1:gx2] = 1

    return float(mask.mean())


# ---------------------------------------------------------------------
# Visual feature extraction
# ---------------------------------------------------------------------

def get_grid_weights(mask, grid_size=3):
    h, w = mask.shape[:2]
    weights = []

    for gy in range(grid_size):
        for gx in range(grid_size):
            y1 = int(round(gy * h / grid_size))
            y2 = int(round((gy + 1) * h / grid_size))
            x1 = int(round(gx * w / grid_size))
            x2 = int(round((gx + 1) * w / grid_size))

            cell = mask[y1:y2, x1:x2]

            if cell.size == 0:
                weights.append(0.0)
            else:
                weights.append(float(cell.sum()))

    return weights


def compute_image_visual_features(arr):
    h, w = arr.shape[:2]
    pixel_count = h * w

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    brightness_mean = float(gray.mean()) / 255.0
    brightness_std = float(gray.std()) / 255.0

    saturation = hsv[:, :, 1]
    saturation_mean = float(saturation.mean()) / 255.0
    saturation_std = float(saturation.std()) / 255.0

    gray_counts = np.bincount(gray.ravel(), minlength=256)
    grayscale_entropy = entropy_from_counts(gray_counts)

    quantized = (arr // 16).astype(np.int32)
    color_index = (
        quantized[:, :, 0] * 256
        + quantized[:, :, 1] * 16
        + quantized[:, :, 2]
    )
    color_counts = np.bincount(color_index.ravel(), minlength=4096)

    quantized_color_count = int(np.count_nonzero(color_counts))
    color_entropy = entropy_from_counts(color_counts)
    dominant_color_ratio = float(color_counts.max()) / float(pixel_count) if pixel_count > 0 else 0.0

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / float(pixel_count) if pixel_count > 0 else 0.0

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    gradient_mean = float(gradient_magnitude.mean()) / 255.0
    gradient_std = float(gradient_magnitude.std()) / 255.0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = float(laplacian.var())

    background_mask = (gray > 245) & (saturation < 25)
    whitespace_ratio = float(background_mask.mean())
    non_background_mask = ~background_mask
    non_background_ratio = float(non_background_mask.mean())

    component_mask = non_background_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(component_mask, connectivity=8)

    component_areas = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= 30:
            component_areas.append(area)

    connected_component_count = len(component_areas)
    mean_component_area_ratio = (
        float(np.mean(component_areas)) / float(pixel_count)
        if component_areas else 0.0
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    edge_grid_weights = get_grid_weights(edges > 0, GRID_SIZE)
    non_bg_grid_weights = get_grid_weights(non_background_mask, GRID_SIZE)

    edge_grid_entropy = normalized_entropy_from_weights(edge_grid_weights)
    non_bg_grid_entropy = normalized_entropy_from_weights(non_bg_grid_weights)

    edge_grid_std = float(np.std(edge_grid_weights)) if edge_grid_weights else 0.0
    non_bg_grid_std = float(np.std(non_bg_grid_weights)) if non_bg_grid_weights else 0.0

    total_non_bg = float(non_background_mask.sum())

    if total_non_bg > 0:
        ys, xs = np.nonzero(non_background_mask)
        visual_center_x = float(xs.mean()) / float(w)
        visual_center_y = float(ys.mean()) / float(h)
    else:
        visual_center_x = 0.5
        visual_center_y = 0.5

    left_weight = float(non_background_mask[:, :w // 2].sum())
    right_weight = float(non_background_mask[:, w // 2:].sum())
    top_weight = float(non_background_mask[:h // 2, :].sum())
    bottom_weight = float(non_background_mask[h // 2:, :].sum())

    horizontal_balance_difference = (
        abs(left_weight - right_weight) / total_non_bg
        if total_non_bg > 0 else 0.0
    )

    vertical_balance_difference = (
        abs(top_weight - bottom_weight) / total_non_bg
        if total_non_bg > 0 else 0.0
    )

    return {
        "analysis_width": w,
        "analysis_height": h,
        "brightness_mean": round(brightness_mean, 6),
        "brightness_std": round(brightness_std, 6),
        "saturation_mean": round(saturation_mean, 6),
        "saturation_std": round(saturation_std, 6),
        "grayscale_entropy": round(grayscale_entropy, 6),
        "quantized_color_count": quantized_color_count,
        "color_entropy": round(color_entropy, 6),
        "dominant_color_ratio": round(dominant_color_ratio, 6),
        "edge_density": round(edge_density, 6),
        "gradient_mean": round(gradient_mean, 6),
        "gradient_std": round(gradient_std, 6),
        "laplacian_variance": round(laplacian_variance, 6),
        "whitespace_ratio": round(whitespace_ratio, 6),
        "non_background_ratio": round(non_background_ratio, 6),
        "connected_component_count": connected_component_count,
        "mean_component_area_ratio": round(mean_component_area_ratio, 8),
        "contour_count": contour_count,
        "edge_grid_entropy_3x3": round(edge_grid_entropy, 6),
        "non_background_grid_entropy_3x3": round(non_bg_grid_entropy, 6),
        "edge_grid_std_3x3": round(edge_grid_std, 6),
        "non_background_grid_std_3x3": round(non_bg_grid_std, 6),
        "visual_center_x_norm": round(visual_center_x, 6),
        "visual_center_y_norm": round(visual_center_y, 6),
        "horizontal_balance_difference": round(horizontal_balance_difference, 6),
        "vertical_balance_difference": round(vertical_balance_difference, 6),
    }


# ---------------------------------------------------------------------
# Text-layout feature extraction
# ---------------------------------------------------------------------

def compute_text_layout_features(ocr_boxes, image_width, image_height):
    rects = []

    for box in ocr_boxes:
        polygon = box.get("box", [])
        rect = polygon_to_rect(polygon, image_width, image_height)

        if rect:
            rects.append(rect)

    n = len(rects)

    if n == 0 or image_width <= 0 or image_height <= 0:
        return {
            "text_layout_box_count": 0,
            "text_layout_sum_area_ratio": 0.0,
            "text_layout_union_area_ratio": 0.0,
            "text_grid_entropy_3x3": 0.0,
            "text_grid_nonempty_cells_3x3": 0,
            "text_grid_max_cell_ratio_3x3": 0.0,
            "text_center_x_norm": 0.0,
            "text_center_y_norm": 0.0,
            "text_spread_x_norm": 0.0,
            "text_spread_y_norm": 0.0,
            "mean_nearest_text_box_distance_norm": 0.0,
            "overlapping_text_box_pairs": 0,
            "mean_positive_text_box_iou": 0.0,
            "text_overlap_area_ratio": 0.0,
        }

    image_area = image_width * image_height
    diag = math.sqrt(image_width ** 2 + image_height ** 2)

    areas = [rect_area(r) for r in rects]
    text_sum_area_ratio = sum(areas) / image_area if image_area > 0 else 0.0
    text_union_area_ratio = approximate_union_area_ratio(rects, image_width, image_height)

    centers = []
    for r in rects:
        x1, y1, x2, y2 = r
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]

    text_center_x = np.mean(xs) / image_width
    text_center_y = np.mean(ys) / image_height

    text_spread_x = np.std(xs) / image_width
    text_spread_y = np.std(ys) / image_height

    grid_weights = [0.0 for _ in range(GRID_SIZE * GRID_SIZE)]

    for rect, area in zip(rects, areas):
        x1, y1, x2, y2 = rect
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        gx = min(GRID_SIZE - 1, max(0, int(cx / image_width * GRID_SIZE)))
        gy = min(GRID_SIZE - 1, max(0, int(cy / image_height * GRID_SIZE)))

        grid_weights[gy * GRID_SIZE + gx] += area

    text_grid_entropy = normalized_entropy_from_weights(grid_weights)
    total_grid_area = sum(grid_weights)

    text_grid_nonempty_cells = sum(1 for w in grid_weights if w > 0)
    text_grid_max_cell_ratio = max(grid_weights) / total_grid_area if total_grid_area > 0 else 0.0

    nearest_distances = []

    if n > 1:
        for i in range(n):
            cx1, cy1 = centers[i]
            min_dist = None

            for j in range(n):
                if i == j:
                    continue

                cx2, cy2 = centers[j]
                dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

                if min_dist is None or dist < min_dist:
                    min_dist = dist

            if min_dist is not None:
                nearest_distances.append(min_dist / diag if diag > 0 else 0.0)

    mean_nearest_distance = list_mean(nearest_distances)

    overlapping_pairs = 0
    positive_ious = []
    overlap_area_sum = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            inter = rect_intersection_area(rects[i], rects[j])

            if inter > 0:
                overlapping_pairs += 1
                overlap_area_sum += inter

                union = rect_area(rects[i]) + rect_area(rects[j]) - inter
                if union > 0:
                    positive_ious.append(inter / union)

    mean_positive_iou = list_mean(positive_ious)
    text_overlap_area_ratio = overlap_area_sum / image_area if image_area > 0 else 0.0

    return {
        "text_layout_box_count": n,
        "text_layout_sum_area_ratio": round(text_sum_area_ratio, 6),
        "text_layout_union_area_ratio": round(text_union_area_ratio, 6),
        "text_grid_entropy_3x3": round(text_grid_entropy, 6),
        "text_grid_nonempty_cells_3x3": text_grid_nonempty_cells,
        "text_grid_max_cell_ratio_3x3": round(text_grid_max_cell_ratio, 6),
        "text_center_x_norm": round(text_center_x, 6),
        "text_center_y_norm": round(text_center_y, 6),
        "text_spread_x_norm": round(text_spread_x, 6),
        "text_spread_y_norm": round(text_spread_y, 6),
        "mean_nearest_text_box_distance_norm": round(mean_nearest_distance, 6),
        "overlapping_text_box_pairs": overlapping_pairs,
        "mean_positive_text_box_iou": round(mean_positive_iou, 6),
        "text_overlap_area_ratio": round(text_overlap_area_ratio, 6),
    }


# ---------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------

def process_row(row):
    paper_id = row["paper_id"]
    ga_path = row["ga_path"]

    arr, original_width, original_height, analysis_width, analysis_height, scale = load_image_for_analysis(ga_path)

    visual_features = compute_image_visual_features(arr)

    ocr_boxes = load_ocr_boxes(row)
    text_layout_features = compute_text_layout_features(
        ocr_boxes=ocr_boxes,
        image_width=float(original_width),
        image_height=float(original_height),
    )

    pixel_count = original_width * original_height
    aspect_ratio = original_width / original_height if original_height > 0 else 0.0

    visual_clutter_score = (
        safe_float(visual_features["edge_density"])
        + safe_float(visual_features["non_background_ratio"])
        + min(safe_float(visual_features["connected_component_count"]) / 1000.0, 1.0)
        + min(safe_float(row.get("num_text_boxes", 0)) / 100.0, 1.0)
    ) / 4.0

    layout_imbalance_score = (
        safe_float(visual_features["horizontal_balance_difference"])
        + safe_float(visual_features["vertical_balance_difference"])
        + safe_float(text_layout_features["text_grid_max_cell_ratio_3x3"])
    ) / 3.0

    feature_row = {
        "paper_id": paper_id,
        "ga_path": ga_path,
        "publication_year": pick(row, ["publication_year", "year", "pub_year"]),
        "journal": pick(row, ["journal", "journal_name", "source_title"]),
        "publisher": pick(row, ["publisher", "publisher_name"]),
        "domain": pick(row, ["domain", "matched_domain"]),
        "subject_area": pick(row, ["subject_area", "subject_areas"]),
        "subject_categories": pick(row, ["subject_categories", "categories"]),

        "image_width": original_width,
        "image_height": original_height,
        "pixel_count": pixel_count,
        "aspect_ratio": round(aspect_ratio, 6),
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "analysis_scale": round(scale, 6),

        "stage2_ocr_status": row.get("ocr_status", ""),
        "stage2_num_text_boxes": safe_int(row.get("num_text_boxes", 0)),
        "stage2_total_ocr_tokens": safe_int(row.get("total_ocr_tokens", 0)),
        "stage2_total_ocr_characters": safe_int(row.get("total_ocr_characters", 0)),
        "stage2_mean_ocr_confidence": safe_float(row.get("mean_ocr_confidence", 0)),
        "stage2_total_text_area_ratio": safe_float(row.get("total_text_area_ratio", 0)),
    }

    feature_row.update(visual_features)
    feature_row.update(text_layout_features)

    feature_row.update({
        "visual_clutter_score": round(visual_clutter_score, 6),
        "layout_imbalance_score": round(layout_imbalance_score, 6),
        "flag_high_edge_density": int(safe_float(visual_features["edge_density"]) >= 0.15),
        "flag_low_whitespace": int(safe_float(visual_features["whitespace_ratio"]) <= 0.20),
        "flag_high_component_count": int(safe_int(visual_features["connected_component_count"]) >= 500),
        "flag_high_visual_clutter": int(visual_clutter_score >= 0.50),
        "flag_layout_imbalance": int(layout_imbalance_score >= 0.50),
        "error_message": "",
    })

    per_image_json_path = PER_IMAGE_JSON_DIR / f"{safe_filename(paper_id)}.json"

    with open(per_image_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_row, f, ensure_ascii=False, indent=2)

    return feature_row


def failure_row(row, error_message):
    return {
        "paper_id": row.get("paper_id", ""),
        "ga_path": row.get("ga_path", ""),
        "publication_year": pick(row, ["publication_year", "year", "pub_year"]),
        "journal": pick(row, ["journal", "journal_name", "source_title"]),
        "publisher": pick(row, ["publisher", "publisher_name"]),
        "domain": pick(row, ["domain", "matched_domain"]),
        "subject_area": pick(row, ["subject_area", "subject_areas"]),
        "subject_categories": pick(row, ["subject_categories", "categories"]),
        "image_width": 0,
        "image_height": 0,
        "pixel_count": 0,
        "aspect_ratio": 0,
        "analysis_width": 0,
        "analysis_height": 0,
        "analysis_scale": 0,
        "stage2_ocr_status": row.get("ocr_status", ""),
        "stage2_num_text_boxes": safe_int(row.get("num_text_boxes", 0)),
        "stage2_total_ocr_tokens": safe_int(row.get("total_ocr_tokens", 0)),
        "stage2_total_ocr_characters": safe_int(row.get("total_ocr_characters", 0)),
        "stage2_mean_ocr_confidence": safe_float(row.get("mean_ocr_confidence", 0)),
        "stage2_total_text_area_ratio": safe_float(row.get("total_text_area_ratio", 0)),
        "brightness_mean": 0,
        "brightness_std": 0,
        "saturation_mean": 0,
        "saturation_std": 0,
        "grayscale_entropy": 0,
        "quantized_color_count": 0,
        "color_entropy": 0,
        "dominant_color_ratio": 0,
        "edge_density": 0,
        "gradient_mean": 0,
        "gradient_std": 0,
        "laplacian_variance": 0,
        "whitespace_ratio": 0,
        "non_background_ratio": 0,
        "connected_component_count": 0,
        "mean_component_area_ratio": 0,
        "contour_count": 0,
        "edge_grid_entropy_3x3": 0,
        "non_background_grid_entropy_3x3": 0,
        "edge_grid_std_3x3": 0,
        "non_background_grid_std_3x3": 0,
        "visual_center_x_norm": 0,
        "visual_center_y_norm": 0,
        "horizontal_balance_difference": 0,
        "vertical_balance_difference": 0,
        "text_layout_box_count": 0,
        "text_layout_sum_area_ratio": 0,
        "text_layout_union_area_ratio": 0,
        "text_grid_entropy_3x3": 0,
        "text_grid_nonempty_cells_3x3": 0,
        "text_grid_max_cell_ratio_3x3": 0,
        "text_center_x_norm": 0,
        "text_center_y_norm": 0,
        "text_spread_x_norm": 0,
        "text_spread_y_norm": 0,
        "mean_nearest_text_box_distance_norm": 0,
        "overlapping_text_box_pairs": 0,
        "mean_positive_text_box_iou": 0,
        "text_overlap_area_ratio": 0,
        "visual_clutter_score": 0,
        "layout_imbalance_score": 0,
        "flag_high_edge_density": 0,
        "flag_low_whitespace": 0,
        "flag_high_component_count": 0,
        "flag_high_visual_clutter": 0,
        "flag_layout_imbalance": 0,
        "error_message": error_message,
    }


FIELDNAMES = list(failure_row({}, "").keys())


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

SUMMARY_NUMERIC_FIELDS = [
    "stage2_num_text_boxes",
    "stage2_total_ocr_tokens",
    "stage2_mean_ocr_confidence",
    "stage2_total_text_area_ratio",
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "grayscale_entropy",
    "color_entropy",
    "quantized_color_count",
    "dominant_color_ratio",
    "edge_density",
    "gradient_mean",
    "laplacian_variance",
    "whitespace_ratio",
    "non_background_ratio",
    "connected_component_count",
    "contour_count",
    "edge_grid_entropy_3x3",
    "non_background_grid_entropy_3x3",
    "horizontal_balance_difference",
    "vertical_balance_difference",
    "text_layout_union_area_ratio",
    "text_grid_entropy_3x3",
    "overlapping_text_box_pairs",
    "visual_clutter_score",
    "layout_imbalance_score",
]


def write_group_summary(rows, group_field, output_csv):
    groups = {}

    for row in rows:
        key = str(row.get(group_field, "") or "UNKNOWN").strip() or "UNKNOWN"
        groups.setdefault(key, []).append(row)

    fieldnames = [group_field, "total_images", "failed_images"]

    for field in SUMMARY_NUMERIC_FIELDS:
        fieldnames.append(f"mean_{field}")

    out_rows = []

    for key, items in sorted(groups.items(), key=lambda x: x[0]):
        out = {
            group_field: key,
            "total_images": len(items),
            "failed_images": sum(1 for r in items if r.get("error_message")),
        }

        for field in SUMMARY_NUMERIC_FIELDS:
            out[f"mean_{field}"] = round(list_mean([r.get(field, 0) for r in items]), 6)

        out_rows.append(out)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def write_health_summary(rows):
    health_rows = [
        {"metric": "total_images", "value": len(rows)},
        {"metric": "failed_images", "value": sum(1 for r in rows if r.get("error_message"))},
        {"metric": "mean_edge_density", "value": round(list_mean([r["edge_density"] for r in rows]), 6)},
        {"metric": "mean_color_entropy", "value": round(list_mean([r["color_entropy"] for r in rows]), 6)},
        {"metric": "mean_whitespace_ratio", "value": round(list_mean([r["whitespace_ratio"] for r in rows]), 6)},
        {"metric": "mean_connected_component_count", "value": round(list_mean([r["connected_component_count"] for r in rows]), 6)},
        {"metric": "mean_visual_clutter_score", "value": round(list_mean([r["visual_clutter_score"] for r in rows]), 6)},
        {"metric": "mean_layout_imbalance_score", "value": round(list_mean([r["layout_imbalance_score"] for r in rows]), 6)},
        {"metric": "images_flag_high_edge_density", "value": sum(safe_int(r["flag_high_edge_density"]) for r in rows)},
        {"metric": "images_flag_low_whitespace", "value": sum(safe_int(r["flag_low_whitespace"]) for r in rows)},
        {"metric": "images_flag_high_component_count", "value": sum(safe_int(r["flag_high_component_count"]) for r in rows)},
        {"metric": "images_flag_high_visual_clutter", "value": sum(safe_int(r["flag_high_visual_clutter"]) for r in rows)},
        {"metric": "images_flag_layout_imbalance", "value": sum(safe_int(r["flag_layout_imbalance"]) for r in rows)},
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
    print("Stage 3 Visual Complexity and Layout Analysis started")
    print(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    print("============================================================")

    if not STAGE2_FEATURES.exists():
        print(f"ERROR: Stage 2 features file not found: {STAGE2_FEATURES}")
        sys.exit(1)

    rows = read_csv_rows(STAGE2_FEATURES)

    print(f"Images selected for Stage 3: {len(rows)}")

    feature_rows = []
    failed_rows = []

    for i, row in enumerate(rows, start=1):
        try:
            feature_row = process_row(row)
            feature_rows.append(feature_row)

        except Exception as e:
            fr = failure_row(row, str(e))
            feature_rows.append(fr)
            failed_rows.append({
                "paper_id": row.get("paper_id", ""),
                "ga_path": row.get("ga_path", ""),
                "error_message": str(e),
            })

        if i % 100 == 0:
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
        f.write("Stage 3 Visual Complexity and Layout Analysis Report\n")
        f.write("====================================================\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Input Stage 2 features: {STAGE2_FEATURES}\n")
        f.write(f"Images processed: {len(rows)}\n")
        f.write(f"Failed images: {len(failed_rows)}\n\n")
        f.write("Main outputs:\n")
        f.write(f"- Feature CSV: {FEATURES_CSV}\n")
        f.write(f"- Feature JSONL: {FEATURES_JSONL}\n")
        f.write(f"- Per-image JSON folder: {PER_IMAGE_JSON_DIR}\n")
        f.write(f"- Failed images CSV: {FAILED_CSV}\n")
        f.write(f"- Year summary: {YEAR_SUMMARY_CSV}\n")
        f.write(f"- Domain summary: {DOMAIN_SUMMARY_CSV}\n")
        f.write(f"- Journal summary: {JOURNAL_SUMMARY_CSV}\n")
        f.write(f"- Health summary: {HEALTH_SUMMARY_CSV}\n")

    print("============================================================")
    print("Stage 3 Visual Complexity and Layout Analysis completed")
    print(f"Images processed: {len(rows)}")
    print(f"Failed images: {len(failed_rows)}")
    print(f"Report: {REPORT_TXT}")
    print("============================================================")


if __name__ == "__main__":
    main()