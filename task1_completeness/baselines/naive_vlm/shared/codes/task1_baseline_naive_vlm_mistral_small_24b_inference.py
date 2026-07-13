#!/usr/bin/env python3
"""
Task 1 Baseline: Naive VLM Judge
Model: Mistral-Small-3.1-24B-Instruct-w4a16, GA image ONLY (no SRP, no paper text).

This is the "no structured reference" control for Task 1. For each DOI in the
complete-case list, the model sees ONLY the GA image and outputs a discrete
completeness level 0-4 (plus which IMRaD components it judged present). No SRP
is loaded; no entity/relation scoring is performed. There is NO variant
dimension. Results are scored later against the same human levels as the SRP
pipeline, making this directly comparable.

The --run-tag flag suffixes ONLY the reports + checkpoint (so two halves running
in parallel never clobber each other). Judgments/raw_outputs/errors are per-DOI
and safely shared between halves.

Examples:
  python3 task1_baseline_naive_vlm_mistral_small_24b_inference.py --limit 1
  python3 task1_baseline_naive_vlm_mistral_small_24b_inference.py --start-index 0 --end-index 4993 --run-tag h1
"""

import argparse
import asyncio
import base64
import csv
import json
import mimetypes
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

MODEL_NAME = "mistral_small_24b"
SERVED_MODEL_NAME = "mistral_small_24b"
MODEL_PATH = "./models/awq/Mistral-Small-3.1-24B-Instruct-w4a16"
VLLM_URL = "http://localhost:8000"

PROJECT_BASE = Path("./task1_completeness_awq")
BASELINE_BASE = PROJECT_BASE / "baselines" / "naive_vlm"

# ---- SHARED inputs ----------------------------------------------------------
SHARED_DIR = BASELINE_BASE / "shared"
SYSTEM_PROMPT_FILE = SHARED_DIR / "system_prompts" / "naive_vlm_system_prompt.txt"
COMPLETE_DOIS_FILE = PROJECT_BASE / "complete_dois.txt"

# GA image index CSV (same source as Stage 2).
GA_IMAGE_CSV = Path("./task2_readability/output/stage1_preprocessing/index/stage1_ga_index.csv")
GA_CSV_DOI_COL = "paper_id"
GA_CSV_PATH_COL = "ga_path"

# ---- Output tree (NO variant dimension) -------------------------------------
OUTPUT_ROOT = BASELINE_BASE / MODEL_NAME
JUDGMENT_DIR = OUTPUT_ROOT / "judgments"
RAW_OUTPUT_DIR = OUTPUT_ROOT / "raw_outputs"
ERROR_DIR = OUTPUT_ROOT / "errors"
LOG_DIR = OUTPUT_ROOT / "logs"
REPORT_DIR = OUTPUT_ROOT / "reports"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

JUDGMENT_SUFFIX = "_naivejudge_mistral_small_24b.json"
RAW_OUTPUT_SUFFIX = "_raw_output_mistral_small_24b.txt"
CLEANED_OUTPUT_SUFFIX = "_cleaned_output_mistral_small_24b.txt"
INVALID_SUFFIX = "_invalid_naivejudge_mistral_small_24b.json"

# Reassigned in main() when --run-tag is provided.
RESULTS_CSV: Path = Path("/dev/null")
SUMMARY_JSON: Path = Path("/dev/null")
SUMMARY_TXT: Path = Path("/dev/null")
CHECKPOINT_JSONL: Path = Path("/dev/null")

DEFAULT_TEMPERATURE = 0.2  # nonzero to avoid AWQ greedy-decode collapse
DEFAULT_MAX_TOKENS = 1024  # raised from 512: 512 truncated JSON mid-value
DEFAULT_TOP_P = 0.9          # with temperature>0, constrains sampling to avoid token-salad
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 600.0

REPETITION_PENALTY = 1.0  # 1.05 triggered AWQ greedy-decode loops
MAX_MODEL_LEN = 49152  # must match the vLLM server --max-model-len

COMPONENTS = ["introduction", "methods", "results", "discussion"]

# Live progress counter (incremented once per finished item, in completion order).
PROGRESS_TOTAL = 0
PROGRESS_DONE = 0


def _progress_prefix() -> str:
    global PROGRESS_DONE
    PROGRESS_DONE += 1
    remaining = PROGRESS_TOTAL - PROGRESS_DONE
    return f"[{PROGRESS_DONE}/{PROGRESS_TOTAL} | {remaining} left]"

CSV_FIELDS = [
    "model", "served_model_name", "doi", "doi_safe",
    "ga_image_path", "judgment_file", "status", "error_message",
    "ga_image_found", "ga_image_bytes",
    "input_tokens", "output_tokens", "inference_time", "tokens_per_second",
    "raw_output_length", "had_think_tags", "had_code_fences", "parse_mode",
    "present_introduction", "present_methods", "present_results", "present_discussion",
    "components_present_count", "level", "level_consistent",
]

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class WorkItem:
    doi: str
    doi_safe: str
    ga_image_path: str

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
    try:
        return await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, extra_body=extra_body,
        )
    except Exception as exc:
        in_tok = _input_tokens_from_error(str(exc))
        if in_tok is None:
            raise
        retry_max = max(128, MAX_MODEL_LEN - in_tok - 64)
        if retry_max >= max_tokens:
            raise
        return await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=retry_max, extra_body=extra_body,
        )

# =============================================================================
# Path resolution (run-tag aware)
# =============================================================================

def apply_run_tag(run_tag: Optional[str]) -> None:
    global RESULTS_CSV, SUMMARY_JSON, SUMMARY_TXT, CHECKPOINT_JSONL
    tag = f"_{run_tag}" if run_tag else ""
    RESULTS_CSV = REPORT_DIR / f"task1_baseline_naive_vlm_{MODEL_NAME}_results{tag}.csv"
    SUMMARY_JSON = REPORT_DIR / f"task1_baseline_naive_vlm_{MODEL_NAME}_summary{tag}.json"
    SUMMARY_TXT = REPORT_DIR / f"task1_baseline_naive_vlm_{MODEL_NAME}_summary{tag}.txt"
    CHECKPOINT_JSONL = CHECKPOINT_DIR / f"processed_prompts{tag}.jsonl"

# =============================================================================
# Basic utilities
# =============================================================================

def make_dirs() -> None:
    for path in [JUDGMENT_DIR, RAW_OUTPUT_DIR, ERROR_DIR, LOG_DIR, REPORT_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def doi_to_safe_id(doi: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", doi.strip())
    return re.sub(r"_+", "_", safe).strip("_")


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
    if not GA_IMAGE_CSV.exists():
        raise FileNotFoundError(f"GA image index CSV not found: {GA_IMAGE_CSV}")
    mapping: Dict[str, str] = {}
    with open(GA_IMAGE_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        for need in (GA_CSV_DOI_COL, GA_CSV_PATH_COL):
            if need not in cols:
                raise ValueError(f"GA image CSV missing column '{need}'. Found: {sorted(cols)}")
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
        items.append(WorkItem(doi=doi, doi_safe=safe, ga_image_path=ga_map.get(safe, "")))
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
    stats = {"ga_image_found": False, "ga_image_bytes": 0, "error": ""}
    if not ga_image_path:
        stats["error"] = "no GA image path in index CSV"
        return None, stats
    img_path = Path(ga_image_path)
    if not img_path.exists() or not img_path.is_file():
        stats["error"] = f"GA image not found: {ga_image_path}"
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
# Judgment JSON validation
# =============================================================================

def parse_judgment_json(cleaned: str):
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


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "y", "1", "present", "explicit", "implied"}:
            return True
        if s in {"false", "no", "n", "0", "absent"}:
            return False
    return None


def validate_judgment(j: Any) -> Tuple[bool, str]:
    if not isinstance(j, dict):
        return False, "judgment root is not a JSON object"
    cp = j.get("components_present")
    if not isinstance(cp, dict):
        return False, "components_present is not an object"
    for c in COMPONENTS:
        if c not in cp:
            return False, f"components_present missing: {c}"
        if _as_bool(cp[c]) is None:
            return False, f"components_present[{c}] not boolean: {cp[c]}"
    lvl = j.get("level")
    if not isinstance(lvl, (int, float)) or int(lvl) not in {0, 1, 2, 3, 4}:
        return False, f"level invalid: {lvl}"
    return True, ""

# =============================================================================
# Per-item processing
# =============================================================================

def default_result(item: WorkItem) -> Dict[str, Any]:
    base = {
        "model": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "doi": item.doi,
        "doi_safe": item.doi_safe,
        "ga_image_path": item.ga_image_path,
        "judgment_file": str(JUDGMENT_DIR / f"{item.doi_safe}{JUDGMENT_SUFFIX}"),
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
        "components_present_count": 0,
        "level": "",
        "level_consistent": "",
    }
    for c in COMPONENTS:
        base[f"present_{c}"] = ""
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
    judgment_path = JUDGMENT_DIR / f"{item.doi_safe}{JUDGMENT_SUFFIX}"

    if judgment_path.exists() and not force:
        result["status"] = "skipped"
        print(f"{_progress_prefix()} [SKIP] {item.doi_safe} | existing judgment", flush=True)
        save_checkpoint(result)
        return result

    image_block, img_stats = load_ga_image_block(item.ga_image_path)
    result["ga_image_found"] = img_stats["ga_image_found"]
    result["ga_image_bytes"] = img_stats["ga_image_bytes"]
    if image_block is None:
        result["status"] = "ga_image_error"
        result["error_message"] = img_stats["error"]
        print(f"{_progress_prefix()} [FAIL] {item.doi_safe} | ga_image_error | {img_stats['error']}", flush=True)
        save_checkpoint(result)
        return result

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": "Graphical abstract image:"},
        image_block,
    ]

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
                extra_body={"repetition_penalty": REPETITION_PENALTY, "top_p": DEFAULT_TOP_P},
            )
        except Exception as exc:
            result["status"] = "api_error"
            result["error_message"] = str(exc)
            print(f"{_progress_prefix()} [FAIL] {item.doi_safe} | api_error | {exc}", flush=True)
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
        print(f"{_progress_prefix()} [FAIL] {item.doi_safe} | empty_response_error", flush=True)
        save_checkpoint(result)
        return result

    cleaned = strip_code_fences(strip_think_tags(raw_content))
    j, parse_mode = parse_judgment_json(cleaned)
    result["parse_mode"] = parse_mode
    if j is None:
        result["status"] = "json_parse_error"
        result["error_message"] = "JSON parse failed after repair"
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_text(ERROR_DIR / f"{item.doi_safe}{CLEANED_OUTPUT_SUFFIX}", cleaned)
        print(f"{_progress_prefix()} [FAIL] {item.doi_safe} | json_parse_error", flush=True)
        save_checkpoint(result)
        return result

    is_valid, validation_error = validate_judgment(j)
    if not is_valid:
        result["status"] = "validation_error"
        result["error_message"] = validation_error
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_json(ERROR_DIR / f"{item.doi_safe}{INVALID_SUFFIX}", j)
        print(f"{_progress_prefix()} [FAIL] {item.doi_safe} | validation_error | {validation_error}", flush=True)
        save_checkpoint(result)
        return result

    # Normalize and record. The authoritative level is the COUNT of present
    # components (we do not trust the model's arithmetic); we also record whether
    # the model's own reported level matched that count.
    cp = j["components_present"]
    present = {c: bool(_as_bool(cp[c])) for c in COMPONENTS}
    count = sum(1 for c in COMPONENTS if present[c])
    model_level = int(j["level"])
    level_consistent = (model_level == count)

    j_out = {
        "doi": item.doi,
        "doi_safe": item.doi_safe,
        "model": MODEL_NAME,
        "baseline": "naive_vlm",
        "components_present": present,
        "components_present_count": count,
        "level": count,                 # authoritative = derived count
        "model_reported_level": model_level,
        "level_consistent": level_consistent,
    }
    try:
        save_json(judgment_path, j_out)
        save_text(RAW_OUTPUT_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
    except Exception as exc:
        result["status"] = "save_error"
        result["error_message"] = str(exc)
        print(f"{_progress_prefix()} [FAIL] {item.doi_safe} | save_error | {exc}", flush=True)
        save_checkpoint(result)
        return result

    for c in COMPONENTS:
        result[f"present_{c}"] = present[c]
    result["components_present_count"] = count
    result["level"] = count
    result["level_consistent"] = level_consistent
    result["status"] = "success"
    result["judgment_file"] = str(judgment_path)
    print(f"{_progress_prefix()} [OK] {item.doi_safe} | level={count}/4 | consistent={level_consistent} | {elapsed:.1f}s", flush=True)
    save_checkpoint(result)
    return result

# =============================================================================
# Reports
# =============================================================================

def pct(n: int, d: int) -> float:
    return round(n / d * 100, 2) if d else 0.0


def compute_summary(results, gpu_info, total_wall_time, args):
    processed = [r for r in results if r["status"] != "skipped"]
    successful = [r for r in processed if r["status"] == "success"]
    failed = [r for r in processed if r["status"] != "success"]
    skipped = [r for r in results if r["status"] == "skipped"]

    error_breakdown: Dict[str, int] = {}
    for r in failed:
        error_breakdown[r["status"]] = error_breakdown.get(r["status"], 0) + 1

    inf_times = [r["inference_time"] for r in successful]
    tps = [r["tokens_per_second"] for r in successful if r["tokens_per_second"] > 0]
    levels = [r["level"] for r in successful if isinstance(r["level"], int)]
    level_hist = {str(k): sum(1 for v in levels if v == k) for k in range(5)}
    inconsistent = sum(1 for r in successful if r["level_consistent"] is False)

    def avg(xs): return round(statistics.mean(xs), 2) if xs else 0.0
    def med(xs): return round(statistics.median(xs), 2) if xs else 0.0

    return {
        "task": "task1_completeness",
        "stage": "baseline_naive_vlm",
        "model_name": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
        "total_wall_time_sec": round(total_wall_time, 2),
        "system_prompt_file": str(SYSTEM_PROMPT_FILE),
        "ga_image_csv": str(GA_IMAGE_CSV),
        "complete_dois_file": str(COMPLETE_DOIS_FILE),
        "output_root": str(OUTPUT_ROOT),
        "args": vars(args),
        **gpu_info,
        "total_rows_selected": len(results),
        "skipped_count": len(skipped),
        "processed_count": len(processed),
        "success_count": len(successful),
        "fail_count": len(failed),
        "success_rate_processed": pct(len(successful), len(processed)),
        "error_breakdown": error_breakdown,
        "avg_inference_time": avg(inf_times),
        "median_inference_time": med(inf_times),
        "avg_throughput_tokens_per_sec": avg(tps),
        "total_input_tokens": sum(r["input_tokens"] for r in successful),
        "total_output_tokens": sum(r["output_tokens"] for r in successful),
        "avg_level": avg(levels),
        "level_histogram": level_hist,
        "model_level_inconsistent_count": inconsistent,
    }


def write_reports(results, summary) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({field: r.get(field, "") for field in CSV_FIELDS})
    save_json(SUMMARY_JSON, summary)

    lines = [
        f"Task 1 Baseline — Naive VLM Judge — {MODEL_NAME}",
        "=" * 60,
        f"Generated: {summary['timestamp']}",
        f"Model path: {summary['model_path']}",
        f"System prompt: {summary['system_prompt_file']}",
        f"GA image CSV: {summary['ga_image_csv']}",
        f"Output root: {summary['output_root']}",
        "",
        f"Rows selected: {summary['total_rows_selected']}",
        f"Skipped existing: {summary['skipped_count']}",
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
        "",
        f"Avg level: {summary['avg_level']}/4",
        f"Level histogram: {summary['level_histogram']}",
        f"Model self-inconsistent levels (reported != count): {summary['model_level_inconsistent_count']}",
        "",
        f"Results CSV: {RESULTS_CSV}",
        f"Summary JSON: {SUMMARY_JSON}",
    ]
    save_text(SUMMARY_TXT, "\n".join(lines) + "\n")

# =============================================================================
# Main
# =============================================================================

async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Task 1 Baseline: Naive VLM Judge — Mistral-Small-24B-AWQ")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--run-tag", type=str, default=None, help="Suffix for reports + checkpoint only (e.g. h1, h2)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    apply_run_tag(args.run_tag)
    make_dirs()
    if repair_json is None:
        print("ERROR: json-repair not installed. Run: pip install json-repair --break-system-packages", flush=True)
        sys.exit(1)
    if CHECKPOINT_JSONL.exists():
        CHECKPOINT_JSONL.unlink()

    print("=" * 80, flush=True)
    print(f"Task 1 Baseline: Naive VLM Judge — Mistral-Small-24B-AWQ", flush=True)
    print("=" * 80, flush=True)
    print(f"System prompt:    {SYSTEM_PROMPT_FILE}", flush=True)
    print(f"GA image CSV:     {GA_IMAGE_CSV}", flush=True)
    print(f"Complete DOIs:    {COMPLETE_DOIS_FILE}", flush=True)
    print(f"Output root:      {OUTPUT_ROOT}", flush=True)
    print(f"Run tag:          {args.run_tag}", flush=True)
    print(f"Index range:      [{args.start_index}, {args.end_index})", flush=True)
    print(f"Force rerun:      {args.force}", flush=True)
    print(f"Results CSV:      {RESULTS_CSV}", flush=True)

    if not SYSTEM_PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_FILE}")
    system_prompt = load_text(SYSTEM_PROMPT_FILE)

    items = load_work_items(limit=args.limit, start_index=args.start_index, end_index=args.end_index)
    print(f"Selected items: {len(items)}", flush=True)
    existing = sum(1 for it in items if (JUDGMENT_DIR / f"{it.doi_safe}{JUDGMENT_SUFFIX}").exists())
    print(f"Existing judgments among selected: {existing}", flush=True)
    print(f"Items to process if no --force: {len(items) - existing}", flush=True)

    gpu_info = get_gpu_info()
    print(f"GPU info: {gpu_info}", flush=True)
    print("Connecting to vLLM server...", flush=True)

    client = AsyncOpenAI(base_url=f"{VLLM_URL}/v1", api_key="not-needed", timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    global PROGRESS_TOTAL, PROGRESS_DONE
    PROGRESS_TOTAL = len(items)
    PROGRESS_DONE = 0

    total_start = time.time()
    tasks = [
        process_item(item=item, system_prompt=system_prompt, client=client, semaphore=semaphore,
                     force=args.force, temperature=args.temperature, max_tokens=args.max_tokens)
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
    print("NAIVE VLM JUDGE BASELINE COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Rows selected:  {summary['total_rows_selected']}", flush=True)
    print(f"Skipped:        {summary['skipped_count']}", flush=True)
    print(f"Processed:      {summary['processed_count']}", flush=True)
    print(f"Successful:     {summary['success_count']}", flush=True)
    print(f"Failed:         {summary['fail_count']}", flush=True)
    print(f"Avg level:      {summary['avg_level']}/4", flush=True)
    print(f"Level hist:     {summary['level_histogram']}", flush=True)
    print(f"Total wall time:{summary['total_wall_time_sec']} sec", flush=True)
    print(f"Results CSV:    {RESULTS_CSV}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    if sys.platform == "linux":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(main_async())