#!/usr/bin/env python3
"""
Task 1 / Stage 2 GA Evaluation Inference
Model: Qwen3-VL-32B-Instruct-AWQ, multimodal (GA image + SRP JSON text)

For each DOI in the complete-case list, the script:
  1) loads the GA image (single image, sent as base64),
  2) loads the Stage 1 SRP JSON for the chosen variant (A or B),
  3) sends [system prompt] + [user: SRP JSON text + GA image] to a local vLLM
     OpenAI-compatible server,
  4) parses/validates the Stage 2 evaluation JSON (visual_description,
     component_evaluation, relational_integrity),
  5) saves one eval JSON per DOI-safe id.

The model returns ONLY verdict labels. No scores (S, R, C, level) are produced
here; those are computed by a separate downstream scoring script from these evals.

ONE script serves BOTH variants. --srp-variant {A,B} selects which Stage 1 SRP
tree to read from and which Stage 2 output tree to write to. The GA image, system
prompt, and complete-DOIs list are SHARED across variants.

The --run-tag flag suffixes ONLY the reports + checkpoint (so two halves running
in parallel never clobber each other). Evals/raw_outputs/errors are per-DOI and
safely shared between halves.

Examples:
  python3 task1_stage2_qwen3_vl_32b_inference.py --srp-variant A --limit 1
  python3 task1_stage2_qwen3_vl_32b_inference.py --srp-variant A --start-index 0 --end-index 500 --run-tag h1
  python3 task1_stage2_qwen3_vl_32b_inference.py --srp-variant B --limit 1 --force
"""

import argparse
import asyncio
import base64
import csv
import json
import mimetypes
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "qwen3_vl_32b"
SERVED_MODEL_NAME = "qwen3_vl_32b"
MODEL_PATH = "./models/awq/Qwen3-VL-32B-Instruct-AWQ"
VLLM_URL = "http://localhost:8000"

PROJECT_BASE = Path("./task1_completeness_awq")

# ---- SHARED inputs (same for both variants and all models) ------------------
SHARED_DIR = PROJECT_BASE / "stage2_shared"
SYSTEM_PROMPT_FILE = SHARED_DIR / "system_prompts" / "task1_stage2_system_prompt.txt"
COMPLETE_DOIS_FILE = PROJECT_BASE / "complete_dois.txt"

# GA image index CSV. Must contain a DOI column and a GA-image-path column.
# >>> If your column names differ, change GA_CSV_DOI_COL / GA_CSV_PATH_COL below. <<<
GA_IMAGE_CSV = Path("./task2_readability/output/stage1_preprocessing/index/stage1_ga_index.csv")
GA_CSV_DOI_COL = "paper_id"
GA_CSV_PATH_COL = "ga_path"

# ---- Per-variant trees (resolved in main() from --srp-variant) --------------
# SRP input  : PROJECT_BASE / variant_{A,B} / outputs/stage1/<model>/srps/<doi>_srp_<model>.json
# Eval output: PROJECT_BASE / variant_{A,B} / outputs/stage2/<model>/...
SRP_VARIANT = "A"  # reassigned in main()
SRP_INPUT_DIR: Path = Path("/dev/null")
OUTPUT_ROOT: Path = Path("/dev/null")
EVAL_DIR: Path = Path("/dev/null")
RAW_OUTPUT_DIR: Path = Path("/dev/null")
ERROR_DIR: Path = Path("/dev/null")
LOG_DIR: Path = Path("/dev/null")
REPORT_DIR: Path = Path("/dev/null")
CHECKPOINT_DIR: Path = Path("/dev/null")

SRP_SUFFIX = "_srp_qwen3_vl_32b.json"
EVAL_SUFFIX = "_eval_qwen3_vl_32b.json"
RAW_OUTPUT_SUFFIX = "_raw_output_qwen3_vl_32b.txt"
CLEANED_OUTPUT_SUFFIX = "_cleaned_output_qwen3_vl_32b.txt"
INVALID_EVAL_SUFFIX = "_invalid_eval_qwen3_vl_32b.json"

# Reassigned in main() when --run-tag is provided.
RESULTS_CSV: Path = Path("/dev/null")
SUMMARY_JSON: Path = Path("/dev/null")
SUMMARY_TXT: Path = Path("/dev/null")
CHECKPOINT_JSONL: Path = Path("/dev/null")

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 6144  # Stage 2 output is per-entity verdicts + descriptions
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 600.0

REPETITION_PENALTY = 1.05
MAX_MODEL_LEN = 57344  # must match the vLLM server --max-model-len

SECTIONS = ["introduction", "methods", "results", "discussion"]
RELATION_KEYS = ["intro_to_methods", "methods_to_results", "results_to_discussion"]
ENTITY_VERDICTS = {"explicit", "implied", "absent"}
REL_VERDICTS = {"traceable", "partially_traceable", "not_traceable"}

CSV_FIELDS = [
    "model", "served_model_name", "srp_variant", "doi", "doi_safe",
    "srp_file", "ga_image_path", "eval_file", "status", "error_message",
    "ga_image_found", "ga_image_bytes",
    "input_tokens", "output_tokens", "inference_time", "tokens_per_second",
    "raw_output_length", "had_think_tags", "had_code_fences", "parse_mode",
    "verdict_introduction", "verdict_methods", "verdict_results", "verdict_discussion",
    "n_entities_introduction", "n_entities_methods", "n_entities_results", "n_entities_discussion",
    "n_explicit_total", "n_implied_total", "n_absent_total",
    "rel_intro_to_methods", "rel_methods_to_results", "rel_results_to_discussion",
    "component_count_observed",
]

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class WorkItem:
    doi: str
    doi_safe: str
    ga_image_path: str
    srp_path: Path

# =============================================================================
# Robust JSON handling
# =============================================================================

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


def _input_tokens_from_error(msg: str) -> Optional[int]:
    m = re.search(r"has (\d+) input tokens", msg or "")
    return int(m.group(1)) if m else None


async def create_with_context_retry(client, *, model, messages, temperature, max_tokens, extra_body):
    """Call chat.completions.create; if the server rejects because max_tokens +
    input exceeds the context window, retry once with a reduced max_tokens."""
    try:
        return await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, extra_body=extra_body,
        )
    except Exception as exc:
        in_tok = _input_tokens_from_error(str(exc))
        if in_tok is None:
            raise
        retry_max = max(256, MAX_MODEL_LEN - in_tok - 64)
        if retry_max >= max_tokens:
            raise
        return await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=retry_max, extra_body=extra_body,
        )

# =============================================================================
# Path resolution (variant + run-tag aware)
# =============================================================================

def apply_variant(srp_variant: str) -> None:
    """Resolve all per-variant input/output paths from --srp-variant."""
    global SRP_VARIANT, SRP_INPUT_DIR, OUTPUT_ROOT
    global EVAL_DIR, RAW_OUTPUT_DIR, ERROR_DIR, LOG_DIR, REPORT_DIR, CHECKPOINT_DIR
    global RESULTS_CSV, SUMMARY_JSON, SUMMARY_TXT, CHECKPOINT_JSONL

    SRP_VARIANT = srp_variant
    variant_dir = PROJECT_BASE / f"variant_{srp_variant}"
    SRP_INPUT_DIR = variant_dir / "outputs" / "stage1" / MODEL_NAME / "srps"
    OUTPUT_ROOT = variant_dir / "outputs" / "stage2" / MODEL_NAME
    EVAL_DIR = OUTPUT_ROOT / "evals"
    RAW_OUTPUT_DIR = OUTPUT_ROOT / "raw_outputs"
    ERROR_DIR = OUTPUT_ROOT / "errors"
    LOG_DIR = OUTPUT_ROOT / "logs"
    REPORT_DIR = OUTPUT_ROOT / "reports"
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

    RESULTS_CSV = REPORT_DIR / f"task1_stage2_{MODEL_NAME}_variant{srp_variant}_results.csv"
    SUMMARY_JSON = REPORT_DIR / f"task1_stage2_{MODEL_NAME}_variant{srp_variant}_summary.json"
    SUMMARY_TXT = REPORT_DIR / f"task1_stage2_{MODEL_NAME}_variant{srp_variant}_summary.txt"
    CHECKPOINT_JSONL = CHECKPOINT_DIR / "processed_prompts.jsonl"


def apply_run_tag(run_tag: Optional[str]) -> None:
    """Suffix ONLY report + checkpoint paths so parallel halves never clobber."""
    global RESULTS_CSV, SUMMARY_JSON, SUMMARY_TXT, CHECKPOINT_JSONL
    if not run_tag:
        return
    tag = f"_{run_tag}"
    RESULTS_CSV = REPORT_DIR / f"task1_stage2_{MODEL_NAME}_variant{SRP_VARIANT}_results{tag}.csv"
    SUMMARY_JSON = REPORT_DIR / f"task1_stage2_{MODEL_NAME}_variant{SRP_VARIANT}_summary{tag}.json"
    SUMMARY_TXT = REPORT_DIR / f"task1_stage2_{MODEL_NAME}_variant{SRP_VARIANT}_summary{tag}.txt"
    CHECKPOINT_JSONL = CHECKPOINT_DIR / f"processed_prompts{tag}.jsonl"

# =============================================================================
# Basic utilities
# =============================================================================

def make_dirs() -> None:
    for path in [EVAL_DIR, RAW_OUTPUT_DIR, ERROR_DIR, LOG_DIR, REPORT_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def doi_to_safe_id(doi: str) -> str:
    safe = doi.strip()
    safe = re.sub(r"[^A-Za-z0-9]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def get_gpu_info() -> Dict[str, Any]:
    info = {"gpu_model_name": "unknown", "gpu_count": 0, "gpu_vram_total_mb": 0, "gpu_vram_allocated_mb": 0}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().splitlines()
            names, total, used = [], 0, 0
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    names.append(parts[0]); total += int(float(parts[1])); used += int(float(parts[2]))
            info["gpu_model_name"] = names[0] if names else "unknown"
            info["gpu_count"] = len(lines)
            info["gpu_vram_total_mb"] = total
            info["gpu_vram_allocated_mb"] = used
    except Exception as exc:
        info["gpu_error"] = str(exc)
    return info

# =============================================================================
# Input loading
# =============================================================================

def load_complete_dois() -> List[str]:
    if not COMPLETE_DOIS_FILE.exists():
        raise FileNotFoundError(f"Complete-DOIs file not found: {COMPLETE_DOIS_FILE}")
    dois = []
    with open(COMPLETE_DOIS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                dois.append(s)
    return dois


def load_ga_image_map() -> Dict[str, str]:
    """doi_safe -> ga_image_path, keyed by SAFE doi so lookups are robust to
    DOI punctuation differences between the complete-DOIs list and the CSV."""
    if not GA_IMAGE_CSV.exists():
        raise FileNotFoundError(f"GA image index CSV not found: {GA_IMAGE_CSV}")
    mapping: Dict[str, str] = {}
    with open(GA_IMAGE_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        for need in (GA_CSV_DOI_COL, GA_CSV_PATH_COL):
            if need not in cols:
                raise ValueError(
                    f"GA image CSV missing column '{need}'. Found columns: {sorted(cols)}"
                )
        for row in reader:
            doi = (row.get(GA_CSV_DOI_COL) or "").strip()
            path = (row.get(GA_CSV_PATH_COL) or "").strip()
            if not doi or not path:
                continue
            mapping[doi_to_safe_id(doi)] = path
    return mapping


def load_work_items(limit: Optional[int], start_index: Optional[int], end_index: Optional[int]) -> List[WorkItem]:
    dois = load_complete_dois()
    ga_map = load_ga_image_map()

    items: List[WorkItem] = []
    for doi in dois:
        safe = doi_to_safe_id(doi)
        items.append(
            WorkItem(
                doi=doi,
                doi_safe=safe,
                ga_image_path=ga_map.get(safe, ""),   # "" => recorded as ga_image_missing
                srp_path=SRP_INPUT_DIR / f"{safe}{SRP_SUFFIX}",
            )
        )

    # Deterministic order by safe DOI; index slicing must match the PBS half-split,
    # which derives TOTAL from the line count of complete_dois.txt.
    items.sort(key=lambda x: x.doi_safe)

    if start_index is not None or end_index is not None:
        start = start_index or 0
        end = end_index if end_index is not None else len(items)
        items = items[start:end]
    if limit is not None:
        items = items[:limit]
    return items

# =============================================================================
# Image handling
# =============================================================================

def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def load_ga_image_block(ga_image_path: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Load the single GA image as an OpenAI image_url content block."""
    stats = {"ga_image_found": False, "ga_image_bytes": 0, "error": ""}
    if not ga_image_path:
        stats["error"] = "no GA image path in index CSV"
        return None, stats
    img_path = Path(ga_image_path)
    if not img_path.exists():
        stats["error"] = f"GA image not found: {ga_image_path}"
        return None, stats
    if not img_path.is_file():
        stats["error"] = f"GA image is not a regular file: {ga_image_path}"
        return None, stats
    try:
        data = img_path.read_bytes()
        if not data:
            stats["error"] = "GA image is empty"
            return None, stats
        mime = guess_mime_type(img_path)
        encoded = base64.b64encode(data).decode("ascii")
        block = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
        stats["ga_image_found"] = True
        stats["ga_image_bytes"] = len(data)
        return block, stats
    except Exception as exc:
        stats["error"] = f"GA image load error: {exc}"
        return None, stats

# =============================================================================
# Eval JSON validation + metrics
# =============================================================================

def parse_eval_json(cleaned: str):
    """strict json.loads, then json-repair for syntax only. Repair never invents
    content: a truncated fragment repairs to an incomplete dict that fails validation."""
    try:
        return json.loads(cleaned), "raw"
    except json.JSONDecodeError:
        pass
    if repair_json is not None:
        try:
            obj = repair_json(cleaned, return_objects=True)
        except Exception:
            obj = None
        if isinstance(obj, (dict, list)) and obj:
            return obj, "repaired"
    return None, "failed"


def coerce_eval_schema(ev: Any) -> Any:
    """Fill only the SOFT optional fields the models commonly omit/mistype, so a
    structurally-complete eval is not failed over a single null. Does NOT add or
    alter verdicts, entities, or relations."""
    if not isinstance(ev, dict):
        return ev
    if "visual_description" in ev and not isinstance(ev["visual_description"], str):
        ev["visual_description"] = json.dumps(ev["visual_description"], ensure_ascii=False)

    ce = ev.get("component_evaluation")
    if isinstance(ce, dict):
        for sec in SECTIONS:
            sd = ce.get(sec)
            if isinstance(sd, dict):
                sd.setdefault("visual_evidence", None)
                sd.setdefault("confidence", None)
                if "entity_coverage" not in sd or not isinstance(sd["entity_coverage"], list):
                    sd["entity_coverage"] = []
    ri = ev.get("relational_integrity")
    if isinstance(ri, dict):
        for k in RELATION_KEYS:
            rd = ri.get(k)
            if isinstance(rd, dict):
                rd.setdefault("visual_indicator", None)
    return ev


def validate_eval(ev: Any) -> Tuple[bool, str]:
    if not isinstance(ev, dict):
        return False, "eval root is not a JSON object"
    if "visual_description" not in ev or not isinstance(ev["visual_description"], str):
        return False, "missing or non-string visual_description"

    ce = ev.get("component_evaluation")
    if not isinstance(ce, dict):
        return False, "component_evaluation is not an object"
    for sec in SECTIONS:
        if sec not in ce:
            return False, f"component_evaluation missing section: {sec}"
        sd = ce[sec]
        if not isinstance(sd, dict):
            return False, f"component_evaluation[{sec}] is not an object"
        if sd.get("verdict") not in ENTITY_VERDICTS:
            return False, f"component_evaluation[{sec}].verdict invalid: {sd.get('verdict')}"
        ecov = sd.get("entity_coverage")
        if not isinstance(ecov, list):
            return False, f"component_evaluation[{sec}].entity_coverage is not a list"
        for i, item in enumerate(ecov):
            if not isinstance(item, dict):
                return False, f"component_evaluation[{sec}].entity_coverage[{i}] is not an object"
            if not isinstance(item.get("entity"), str):
                return False, f"component_evaluation[{sec}].entity_coverage[{i}].entity is not a string"
            if item.get("verdict") not in ENTITY_VERDICTS:
                return False, f"component_evaluation[{sec}].entity_coverage[{i}].verdict invalid: {item.get('verdict')}"

    ri = ev.get("relational_integrity")
    if not isinstance(ri, dict):
        return False, "relational_integrity is not an object"
    for k in RELATION_KEYS:
        if k not in ri:
            return False, f"relational_integrity missing key: {k}"
        rd = ri[k]
        if not isinstance(rd, dict):
            return False, f"relational_integrity[{k}] is not an object"
        if rd.get("verdict") not in REL_VERDICTS:
            return False, f"relational_integrity[{k}].verdict invalid: {rd.get('verdict')}"
    return True, ""


def compute_eval_metrics(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Observation counts only (verdict tallies). NO scoring (S/R/C/level) here;
    that is the downstream scoring script's job."""
    out: Dict[str, Any] = {}
    ce = ev.get("component_evaluation", {})
    n_explicit = n_implied = n_absent = component_count = 0
    for sec in SECTIONS:
        sd = ce.get(sec, {}) if isinstance(ce, dict) else {}
        verdict = sd.get("verdict", "")
        out[f"verdict_{sec}"] = verdict
        ecov = sd.get("entity_coverage", []) if isinstance(sd, dict) else []
        out[f"n_entities_{sec}"] = len(ecov) if isinstance(ecov, list) else 0
        if isinstance(ecov, list):
            for item in ecov:
                v = item.get("verdict") if isinstance(item, dict) else None
                if v == "explicit":
                    n_explicit += 1
                elif v == "implied":
                    n_implied += 1
                elif v == "absent":
                    n_absent += 1
        if verdict in ("explicit", "implied"):
            component_count += 1
    out["n_explicit_total"] = n_explicit
    out["n_implied_total"] = n_implied
    out["n_absent_total"] = n_absent
    out["component_count_observed"] = component_count

    ri = ev.get("relational_integrity", {})
    for k in RELATION_KEYS:
        rd = ri.get(k, {}) if isinstance(ri, dict) else {}
        out[f"rel_{k}"] = rd.get("verdict", "") if isinstance(rd, dict) else ""
    return out

# =============================================================================
# Per-item processing
# =============================================================================

def default_result(item: WorkItem) -> Dict[str, Any]:
    base = {
        "model": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "srp_variant": SRP_VARIANT,
        "doi": item.doi,
        "doi_safe": item.doi_safe,
        "srp_file": str(item.srp_path),
        "ga_image_path": item.ga_image_path,
        "eval_file": str(EVAL_DIR / f"{item.doi_safe}{EVAL_SUFFIX}"),
        "status": "",
        "error_message": "",
        "ga_image_found": False,
        "ga_image_bytes": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "inference_time": 0.0,
        "tokens_per_second": 0.0,
        "raw_output_length": 0,
        "had_think_tags": False,
        "had_code_fences": False,
        "parse_mode": "",
        "n_explicit_total": 0,
        "n_implied_total": 0,
        "n_absent_total": 0,
        "component_count_observed": 0,
    }
    for sec in SECTIONS:
        base[f"verdict_{sec}"] = ""
        base[f"n_entities_{sec}"] = 0
    for k in RELATION_KEYS:
        base[f"rel_{k}"] = ""
    return base


def save_checkpoint(result: Dict[str, Any]) -> None:
    CHECKPOINT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


async def process_item(
    item: WorkItem,
    system_prompt: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    force: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    result = default_result(item)
    eval_path = EVAL_DIR / f"{item.doi_safe}{EVAL_SUFFIX}"

    if eval_path.exists() and not force:
        result["status"] = "skipped"
        print(f"[SKIP] {item.doi_safe} | existing eval", flush=True)
        save_checkpoint(result)
        return result

    # --- Load SRP (must exist; we restricted to complete-case DOIs) ----------
    if not item.srp_path.exists():
        result["status"] = "srp_missing_error"
        result["error_message"] = f"SRP not found: {item.srp_path}"
        print(f"[FAIL] {item.doi_safe} | srp_missing_error", flush=True)
        save_checkpoint(result)
        return result
    try:
        srp_text = load_text(item.srp_path)
        json.loads(srp_text)  # sanity: SRP must be valid JSON
    except Exception as exc:
        result["status"] = "srp_read_error"
        result["error_message"] = str(exc)
        print(f"[FAIL] {item.doi_safe} | srp_read_error | {exc}", flush=True)
        save_checkpoint(result)
        return result

    # --- Load GA image -------------------------------------------------------
    image_block, img_stats = load_ga_image_block(item.ga_image_path)
    result["ga_image_found"] = img_stats["ga_image_found"]
    result["ga_image_bytes"] = img_stats["ga_image_bytes"]
    if image_block is None:
        result["status"] = "ga_image_error"
        result["error_message"] = img_stats["error"]
        print(f"[FAIL] {item.doi_safe} | ga_image_error | {img_stats['error']}", flush=True)
        save_checkpoint(result)
        return result

    # --- Build user message: SRP JSON text, then the GA image ----------------
    user_text = "STRUCTURED REFERENCE PROFILE (SRP):\n" + srp_text
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}, image_block]

    async with semaphore:
        start = time.time()
        try:
            response = await create_with_context_retry(
                client,
                model=SERVED_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"repetition_penalty": REPETITION_PENALTY},
            )
        except Exception as exc:
            result["status"] = "api_error"
            result["error_message"] = str(exc)
            print(f"[FAIL] {item.doi_safe} | api_error | {exc}", flush=True)
            save_checkpoint(result)
            return result
        elapsed = time.time() - start

    try:
        raw_content = response.choices[0].message.content or ""
    except Exception:
        raw_content = ""

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    result["input_tokens"] = input_tokens or 0
    result["output_tokens"] = output_tokens or 0
    result["inference_time"] = round(elapsed, 2)
    result["tokens_per_second"] = round((output_tokens or 0) / elapsed, 2) if elapsed > 0 else 0.0
    result["raw_output_length"] = len(raw_content)
    result["had_think_tags"] = bool(re.search(r"<think>", raw_content, flags=re.IGNORECASE))
    result["had_code_fences"] = raw_content.strip().startswith("```")

    if not raw_content.strip():
        result["status"] = "empty_response_error"
        result["error_message"] = "Model returned an empty response"
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        print(f"[FAIL] {item.doi_safe} | empty_response_error", flush=True)
        save_checkpoint(result)
        return result

    cleaned = strip_code_fences(strip_think_tags(raw_content))

    ev, parse_mode = parse_eval_json(cleaned)
    result["parse_mode"] = parse_mode
    if ev is None:
        result["status"] = "json_parse_error"
        result["error_message"] = "JSON parse failed after repair"
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_text(ERROR_DIR / f"{item.doi_safe}{CLEANED_OUTPUT_SUFFIX}", cleaned)
        print(f"[FAIL] {item.doi_safe} | json_parse_error", flush=True)
        save_checkpoint(result)
        return result

    ev = coerce_eval_schema(ev)
    is_valid, validation_error = validate_eval(ev)
    if not is_valid:
        result["status"] = "validation_error"
        result["error_message"] = validation_error
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_json(ERROR_DIR / f"{item.doi_safe}{INVALID_EVAL_SUFFIX}", ev)
        print(f"[FAIL] {item.doi_safe} | validation_error | {validation_error}", flush=True)
        save_checkpoint(result)
        return result

    # Attach identifiers into the saved eval for downstream joins.
    ev_out = {"doi": item.doi, "doi_safe": item.doi_safe, "srp_variant": SRP_VARIANT,
              "model": MODEL_NAME, **ev}
    try:
        save_json(eval_path, ev_out)
        save_text(RAW_OUTPUT_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
    except Exception as exc:
        result["status"] = "save_error"
        result["error_message"] = str(exc)
        print(f"[FAIL] {item.doi_safe} | save_error | {exc}", flush=True)
        save_checkpoint(result)
        return result

    result.update(compute_eval_metrics(ev))
    result["status"] = "success"
    result["eval_file"] = str(eval_path)
    print(
        f"[OK] {item.doi_safe} | comps={result['component_count_observed']}/4 | "
        f"E/I/A={result['n_explicit_total']}/{result['n_implied_total']}/{result['n_absent_total']} | {elapsed:.1f}s",
        flush=True,
    )
    save_checkpoint(result)
    return result

# =============================================================================
# Reports
# =============================================================================

def pct(n: int, d: int) -> float:
    return round(n / d * 100, 2) if d else 0.0


def compute_summary(results: List[Dict[str, Any]], gpu_info: Dict[str, Any], total_wall_time: float, args: argparse.Namespace) -> Dict[str, Any]:
    processed = [r for r in results if r["status"] != "skipped"]
    successful = [r for r in processed if r["status"] == "success"]
    failed = [r for r in processed if r["status"] != "success"]
    skipped = [r for r in results if r["status"] == "skipped"]

    error_breakdown: Dict[str, int] = {}
    for r in failed:
        error_breakdown[r["status"]] = error_breakdown.get(r["status"], 0) + 1

    inference_times = [r["inference_time"] for r in successful]
    tps_values = [r["tokens_per_second"] for r in successful if r["tokens_per_second"] > 0]

    def avg(xs): return round(statistics.mean(xs), 2) if xs else 0.0
    def med(xs): return round(statistics.median(xs), 2) if xs else 0.0

    comp_counts = [r["component_count_observed"] for r in successful]

    summary: Dict[str, Any] = {
        "task": "task1_completeness",
        "stage": "stage2",
        "srp_variant": SRP_VARIANT,
        "model_name": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
        "total_wall_time_sec": round(total_wall_time, 2),
        "system_prompt_file": str(SYSTEM_PROMPT_FILE),
        "ga_image_csv": str(GA_IMAGE_CSV),
        "complete_dois_file": str(COMPLETE_DOIS_FILE),
        "srp_input_dir": str(SRP_INPUT_DIR),
        "output_root": str(OUTPUT_ROOT),
        "args": vars(args),
        **gpu_info,
        "total_rows_selected": len(results),
        "skipped_count": len(skipped),
        "processed_count": len(processed),
        "success_count": len(successful),
        "fail_count": len(failed),
        "success_rate_processed": pct(len(successful), len(processed)),
        "success_rate_all_selected": pct(len(successful), len(results)),
        "error_breakdown": error_breakdown,
        "avg_inference_time": avg(inference_times),
        "median_inference_time": med(inference_times),
        "avg_throughput_tokens_per_sec": avg(tps_values),
        "total_input_tokens": sum(r["input_tokens"] for r in successful),
        "total_output_tokens": sum(r["output_tokens"] for r in successful),
        "avg_component_count_observed": avg(comp_counts),
    }
    return summary


def write_reports(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({field: r.get(field, "") for field in CSV_FIELDS})
    save_json(SUMMARY_JSON, summary)

    lines = [
        f"Task 1 Stage 2 GA Evaluation — {MODEL_NAME} — Variant {SRP_VARIANT}",
        "=" * 60,
        f"Generated: {summary['timestamp']}",
        f"Model path: {summary['model_path']}",
        f"System prompt: {summary['system_prompt_file']}",
        f"GA image CSV: {summary['ga_image_csv']}",
        f"SRP input dir: {summary['srp_input_dir']}",
        f"Output root: {summary['output_root']}",
        "",
        f"Rows selected: {summary['total_rows_selected']}",
        f"Skipped existing evals: {summary['skipped_count']}",
        f"Processed this run: {summary['processed_count']}",
        f"Successful: {summary['success_count']}",
        f"Failed: {summary['fail_count']}",
        f"Success rate processed: {summary['success_rate_processed']}%",
        f"Error breakdown: {summary['error_breakdown']}",
        "",
        f"Avg inference time: {summary['avg_inference_time']} sec",
        f"Median inference time: {summary['median_inference_time']} sec",
        f"Avg throughput: {summary['avg_throughput_tokens_per_sec']} tok/sec",
        f"Total input tokens: {summary['total_input_tokens']:,}",
        f"Total output tokens: {summary['total_output_tokens']:,}",
        f"Avg components observed: {summary['avg_component_count_observed']}/4",
        "",
        f"Results CSV: {RESULTS_CSV}",
        f"Summary JSON: {SUMMARY_JSON}",
    ]
    save_text(SUMMARY_TXT, "\n".join(lines) + "\n")

# =============================================================================
# Main
# =============================================================================

async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Task 1 Stage 2 GA evaluation — Qwen3-VL-32B-AWQ")
    parser.add_argument("--srp-variant", type=str, required=True, choices=["A", "B"],
                        help="Which Stage 1 SRP tree to evaluate against (A=text-only, B=multimodal)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N selected items")
    parser.add_argument("--start-index", type=int, default=None, help="Start index after sorting by safe DOI")
    parser.add_argument("--end-index", type=int, default=None, help="End index after sorting by safe DOI, exclusive")
    parser.add_argument("--run-tag", type=str, default=None, help="Suffix for reports + checkpoint only (e.g. h1, h2)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing evals for selected items")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Async request concurrency")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="OpenAI client timeout in seconds")
    args = parser.parse_args()

    apply_variant(args.srp_variant)
    apply_run_tag(args.run_tag)
    make_dirs()
    if repair_json is None:
        print("ERROR: json-repair not installed. Run: pip install json-repair --break-system-packages", flush=True)
        sys.exit(1)

    if CHECKPOINT_JSONL.exists():
        CHECKPOINT_JSONL.unlink()

    print("=" * 80, flush=True)
    print(f"Task 1 / Stage 2 GA Evaluation — Qwen3-VL-32B-AWQ — Variant {SRP_VARIANT}", flush=True)
    print("=" * 80, flush=True)
    print(f"System prompt:    {SYSTEM_PROMPT_FILE}", flush=True)
    print(f"GA image CSV:     {GA_IMAGE_CSV}", flush=True)
    print(f"Complete DOIs:    {COMPLETE_DOIS_FILE}", flush=True)
    print(f"SRP input dir:    {SRP_INPUT_DIR}", flush=True)
    print(f"Output root:      {OUTPUT_ROOT}", flush=True)
    print(f"Run tag:          {args.run_tag}", flush=True)
    print(f"Index range:      [{args.start_index}, {args.end_index})", flush=True)
    print(f"Force rerun:      {args.force}", flush=True)
    print(f"Max tokens:       {args.max_tokens}", flush=True)
    print(f"Results CSV:      {RESULTS_CSV}", flush=True)

    if not SYSTEM_PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_FILE}")
    if not SRP_INPUT_DIR.exists():
        raise FileNotFoundError(f"SRP input dir not found: {SRP_INPUT_DIR}")
    system_prompt = load_text(SYSTEM_PROMPT_FILE)

    items = load_work_items(limit=args.limit, start_index=args.start_index, end_index=args.end_index)
    print(f"Selected items: {len(items)}", flush=True)
    existing = sum(1 for it in items if (EVAL_DIR / f"{it.doi_safe}{EVAL_SUFFIX}").exists())
    print(f"Existing evals among selected: {existing}", flush=True)
    print(f"Items to process if no --force: {len(items) - existing}", flush=True)

    gpu_info = get_gpu_info()
    print(f"GPU info: {gpu_info}", flush=True)
    print("Connecting to vLLM server...", flush=True)

    client = AsyncOpenAI(base_url=f"{VLLM_URL}/v1", api_key="not-needed", timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    total_start = time.time()
    tasks = [
        process_item(
            item=item, system_prompt=system_prompt, client=client, semaphore=semaphore,
            force=args.force, temperature=args.temperature, max_tokens=args.max_tokens,
        )
        for item in items
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for _item, _r in zip(items, raw_results):
        if isinstance(_r, dict):
            results.append(_r)
        else:
            _row = default_result(_item)
            _row["status"] = "crash_error"
            _row["error_message"] = repr(_r)
            save_checkpoint(_row)
            results.append(_row)
    total_wall_time = time.time() - total_start

    summary = compute_summary(results, gpu_info, total_wall_time, args)
    write_reports(results, summary)

    print("\n" + "=" * 80, flush=True)
    print("STAGE 2 INFERENCE COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Variant:        {SRP_VARIANT}", flush=True)
    print(f"Rows selected:  {summary['total_rows_selected']}", flush=True)
    print(f"Skipped:        {summary['skipped_count']}", flush=True)
    print(f"Processed:      {summary['processed_count']}", flush=True)
    print(f"Successful:     {summary['success_count']}", flush=True)
    print(f"Failed:         {summary['fail_count']}", flush=True)
    print(f"Total wall time:{summary['total_wall_time_sec']} sec", flush=True)
    print(f"Results CSV:    {RESULTS_CSV}", flush=True)
    print(f"Summary TXT:    {SUMMARY_TXT}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    if sys.platform == "linux":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(main_async())
