#!/usr/bin/env python3
"""
Task 2 / Stage 4 — VLM Structural Interpretation of Graphical Abstracts
Model: Qwen3-VL-32B-Instruct-AWQ, served by a local vLLM OpenAI-compatible server.

Driver:  output/stage1_preprocessing/index/stage1_ga_index.csv  (canonical per-GA list).
Per GA:  sends the FIXED system prompt + FIXED user prompt + the GA image, parses/validates
         the Stage 4 JSON schema, and saves one interpretation per paper_id.

Runs on EVERY GA with has_ga==1 and a non-empty ga_path, regardless of image quality.

Resume-safe: existing interpretations are skipped; previous failures are retried if no final
JSON exists; existing interpretations are overwritten only with --force.

Sharding for two parallel nodes (mirror of the Task 1 halves):
  --num-splits 2 --split-id 1   (first half)
  --num-splits 2 --split-id 2   (second half)
The --run-tag suffixes ONLY the reports + checkpoint so the two halves never clobber each
other. interpretations/raw_outputs/errors are per-paper and safely shared.

Examples:
  python3 stage4_vlm_structural_interpretation.py --limit 20
  python3 stage4_vlm_structural_interpretation.py --num-splits 2 --split-id 1 --run-tag h1
  python3 stage4_vlm_structural_interpretation.py --limit 5 --force
"""

import argparse
import asyncio
import base64
import csv
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from openai import AsyncOpenAI

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "qwen3_vl_32b"
SERVED_MODEL_NAME = "qwen3_vl_32b"
MODEL_PATH = "./models/awq/Qwen3-VL-32B-Instruct-AWQ"
VLLM_URL = "http://localhost:8000"

PROJECT_ROOT = Path("./task2_readability")

STAGE1_INDEX = PROJECT_ROOT / "output" / "stage1_preprocessing" / "index" / "stage1_ga_index.csv"

PROMPT_DIR = PROJECT_ROOT / "prompts" / "stage4"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "stage4_system_prompt.txt"
USER_PROMPT_FILE = PROMPT_DIR / "stage4_user_prompt.txt"

OUTPUT_ROOT = PROJECT_ROOT / "output" / "stage4_vlm_structural_interpretation"
INTERP_DIR = OUTPUT_ROOT / "interpretations"
RAW_OUTPUT_DIR = OUTPUT_ROOT / "raw_outputs"
ERROR_DIR = OUTPUT_ROOT / "errors"
LOG_DIR = OUTPUT_ROOT / "logs"
REPORT_DIR = OUTPUT_ROOT / "reports"
SUMMARY_DIR = OUTPUT_ROOT / "summaries"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

INTERP_SUFFIX = "_stage4_qwen3_vl_32b.json"
RAW_SUFFIX = "_raw_qwen3_vl_32b.txt"
CLEANED_SUFFIX = "_cleaned_qwen3_vl_32b.txt"
INVALID_SUFFIX = "_invalid_qwen3_vl_32b.json"

# Reassigned in main() when --run-tag is provided.
RESULTS_CSV = REPORT_DIR / "stage4_qwen3_vl_32b_results.csv"
SUMMARY_JSON = REPORT_DIR / "stage4_qwen3_vl_32b_summary.json"
SUMMARY_TXT = REPORT_DIR / "stage4_qwen3_vl_32b_summary.txt"
DISTRIBUTION_JSON = REPORT_DIR / "stage4_qwen3_vl_32b_field_distributions.json"
CHECKPOINT_JSONL = CHECKPOINT_DIR / "processed_gas.jsonl"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT = 900.0
DEFAULT_MAX_IMAGE_DIM = 1280  # longest side; 0 disables resize
REPETITION_PENALTY = 1.05
MAX_MODEL_LEN = 40960  # must match the vLLM server --max-model-len

# -----------------------------------------------------------------------------
# Schema definition
# -----------------------------------------------------------------------------

ENUM_FIELDS = {
    "panel_structure": {"simple", "moderate", "complex"},
    "layout_type": {"single_panel", "linear_flow", "cyclic_flow", "grid_panels",
                    "central_hub", "comparison", "freeform", "vertical_stack"},
    "main_reading_direction": {"left_to_right", "top_to_bottom", "bottom_to_top",
                               "radial", "circular", "mixed", "none"},
    "narrative_arc": {"complete", "partial", "none"},
    "flow_clarity": {"clear", "partially_clear", "unclear"},
    "entity_clarity": {"clear", "partially_clear", "unclear"},
    "relation_clarity": {"clear", "partially_clear", "unclear"},
    "main_message_identifiable": {"yes", "partially", "no"},
    "visual_clutter": {"low", "medium", "high"},
    "semantic_interpretability": {"high", "medium", "low"},
    "method_identifiable": {"yes", "partially", "no"},
    "result_identifiable": {"yes", "partially", "no"},
    "conclusion_identifiable": {"yes", "partially", "no"},
}
SCORE_FIELDS = ["sequence_clarity_1to5", "key_message_clarity_1to5", "ambiguity_1to5",
                "text_dependency_1to5", "overall_interpretability_1to5"]
BOOL_FIELDS = ["has_start_point", "has_end_point", "has_arrows_or_connectors"]
LIST_FIELDS = ["main_entities", "process_steps", "relationships", "unclear_elements",
               "missing_links"]
STR_FIELDS = ["main_outcome", "uncertainty_notes"]
REL_TYPES = {"input_output", "cause_effect", "activation", "inhibition", "transformation",
             "measurement", "comparison", "association", "other"}

METADATA_COLS = ["publication_year", "journal", "publisher", "domain", "subject_area",
                 "subject_categories"]

CSV_FIELDS = [
    "model", "served_model_name", "paper_id", "ga_path",
    "publication_year", "journal", "publisher", "domain", "subject_area", "subject_categories",
    "status", "error_message", "parse_mode", "fields_coerced",
    "input_tokens", "output_tokens", "inference_time", "tokens_per_second",
    "raw_output_length", "had_think_tags", "had_code_fences",
    "image_orig_width", "image_orig_height", "image_sent_width", "image_sent_height",
    # scalar interpretation fields
    "num_panels", "panel_structure", "layout_type", "main_reading_direction",
    "has_start_point", "has_end_point", "has_arrows_or_connectors", "narrative_arc",
    "num_main_entities", "num_process_steps", "num_relationships",
    "main_outcome_present", "main_outcome",
    "flow_clarity", "entity_clarity", "relation_clarity", "main_message_identifiable",
    "visual_clutter", "semantic_interpretability",
    "method_identifiable", "result_identifiable", "conclusion_identifiable",
    "sequence_clarity_1to5", "key_message_clarity_1to5", "ambiguity_1to5",
    "text_dependency_1to5", "overall_interpretability_1to5",
    "num_unclear_elements", "num_missing_links", "uncertainty_notes_present",
    "interpretation_file",
]


@dataclass(frozen=True)
class GAItem:
    paper_id: str
    ga_path: str
    metadata: Dict[str, str]
    interp_path: Path


class Progress:
    """Live running counter, rendered after every completed sample."""

    def __init__(self, total: int):
        self.total = total
        self.start = time.time()
        self.done = 0
        self.success = 0
        self.skipped = 0
        self.failed = 0
        self.status_counts: Dict[str, int] = {}

    def update(self, status: str) -> None:
        self.done += 1
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        if status == "success":
            self.success += 1
        elif status == "skipped":
            self.skipped += 1
        else:
            self.failed += 1

    def render(self, last_id: str, last_status: str) -> str:
        elapsed = time.time() - self.start
        remaining = self.total - self.done
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = remaining / rate if rate > 0 else 0.0
        done_pct = (self.done / self.total * 100) if self.total else 0.0
        return (
            f"[PROGRESS] {self.done}/{self.total} ({done_pct:.1f}%) | "
            f"success={self.success} failed={self.failed} skipped={self.skipped} | "
            f"remaining={remaining} | last={last_id}:{last_status} | "
            f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m | "
            f"avg={elapsed / self.done:.1f}s/sample" if self.done else
            f"[PROGRESS] 0/{self.total}"
        )


# =============================================================================
# Run-tag aware paths
# =============================================================================

def apply_run_tag(run_tag: Optional[str]) -> None:
    global RESULTS_CSV, SUMMARY_JSON, SUMMARY_TXT, DISTRIBUTION_JSON, CHECKPOINT_JSONL
    if not run_tag:
        return
    t = f"_{run_tag}"
    RESULTS_CSV = REPORT_DIR / f"stage4_qwen3_vl_32b_results{t}.csv"
    SUMMARY_JSON = REPORT_DIR / f"stage4_qwen3_vl_32b_summary{t}.json"
    SUMMARY_TXT = REPORT_DIR / f"stage4_qwen3_vl_32b_summary{t}.txt"
    DISTRIBUTION_JSON = REPORT_DIR / f"stage4_qwen3_vl_32b_field_distributions{t}.json"
    CHECKPOINT_JSONL = CHECKPOINT_DIR / f"processed_gas{t}.jsonl"


# =============================================================================
# Files and setup
# =============================================================================

def ensure_directories() -> None:
    for d in [INTERP_DIR, RAW_OUTPUT_DIR, ERROR_DIR, LOG_DIR, REPORT_DIR, SUMMARY_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def load_prompt(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    text = load_text(path)
    if not text:
        raise ValueError(f"{label} is empty: {path}")
    return text


def safe_filename(value: str) -> str:
    value = re.sub(r"[^\w.\-]+", "_", str(value).strip()).strip("_")
    return (value or "unknown")[:180]


def pick(row: Dict[str, Any], names: List[str], default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name]).strip()
    return default


def discover_items(num_splits: Optional[int], split_id: Optional[int],
                   start_index: Optional[int], end_index: Optional[int],
                   limit: Optional[int]) -> List[GAItem]:
    if not STAGE1_INDEX.exists():
        raise FileNotFoundError(f"Stage 1 index not found: {STAGE1_INDEX}")

    rows: List[Dict[str, str]] = []
    with STAGE1_INDEX.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paper_id = pick(row, ["paper_id", "doi_folder_name"])
            ga_path = pick(row, ["ga_path", "graphical_abstract_path"])
            has_ga = pick(row, ["has_ga"], "1")
            if not paper_id or not ga_path:
                continue
            if str(has_ga).strip() not in ("1", "1.0", "true", "True", "yes"):
                continue
            rows.append(row)

    # Deterministic ordering so splits are stable and reproducible.
    rows.sort(key=lambda r: pick(r, ["paper_id", "doi_folder_name"]))

    # Contiguous split for parallel nodes.
    if num_splits is not None:
        if split_id is None or not (1 <= split_id <= num_splits):
            raise ValueError("--split-id must be in [1, num_splits]")
        n = len(rows)
        per = (n + num_splits - 1) // num_splits
        s = (split_id - 1) * per
        e = min(n, split_id * per)
        rows = rows[s:e]
    elif start_index is not None or end_index is not None:
        s = 0 if start_index is None else start_index
        e = len(rows) if end_index is None else end_index
        if s < 0 or e < 0 or e < s:
            raise ValueError(f"Invalid start/end index: start={s}, end={e}")
        rows = rows[s:e]

    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        rows = rows[:limit]

    items: List[GAItem] = []
    for row in rows:
        paper_id = pick(row, ["paper_id", "doi_folder_name"])
        ga_path = pick(row, ["ga_path", "graphical_abstract_path"])
        metadata = {c: pick(row, [c]) for c in METADATA_COLS}
        interp_path = INTERP_DIR / f"{safe_filename(paper_id)}{INTERP_SUFFIX}"
        items.append(GAItem(paper_id=paper_id, ga_path=ga_path, metadata=metadata, interp_path=interp_path))
    return items


# =============================================================================
# Image encoding
# =============================================================================

def encode_image(path: str, max_dim: int) -> Tuple[str, int, int, int, int]:
    """Return (base64_png, orig_w, orig_h, sent_w, sent_h). Longest-side resize, in-memory."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        ow, oh = img.size
        m = max(ow, oh)
        if max_dim and m > max_dim:
            scale = max_dim / float(m)
            nw = max(1, int(round(ow * scale)))
            nh = max(1, int(round(oh * scale)))
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        sw, sh = img.size
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, ow, oh, sw, sh


# =============================================================================
# Output cleaning, parsing, coercion, validation
# =============================================================================

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-len("```")].strip()
    return text.strip()


def extract_json_candidate(text: str) -> str:
    cleaned = strip_code_fences(strip_think_tags(text))
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        return cleaned[first:last + 1].strip()
    return cleaned


try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


def parse_json(cleaned: str):
    """Return (obj_or_None, parse_mode) in {raw, repaired, failed}. Repair fixes SYNTAX only."""
    try:
        return json.loads(cleaned), "raw"
    except json.JSONDecodeError:
        pass
    if repair_json is not None:
        try:
            obj = repair_json(cleaned, return_objects=True)
        except Exception:
            obj = None
        if isinstance(obj, dict) and obj:
            return obj, "repaired"
    return None, "failed"


def _norm_enum(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _to_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        return int(round(float(value)))
    except Exception:
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "yes", "1", "y"):
        return True
    if s in ("false", "no", "0", "n"):
        return False
    return None


def coerce_schema(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Fill safe optional fields and normalize types WITHOUT inventing enum/score content.
    Returns (obj, n_coerced)."""
    n = 0
    if not isinstance(obj, dict):
        return obj, 0

    # Optional lists -> [] if missing/not a list.
    for k in LIST_FIELDS:
        if k not in obj or not isinstance(obj[k], list):
            obj[k] = []
            n += 1
    # Optional strings -> "" if missing.
    for k in STR_FIELDS:
        if k not in obj or not isinstance(obj[k], str):
            obj[k] = "" if obj.get(k) is None else str(obj.get(k, ""))
            n += 1
    # Enum normalization (does not create a missing key).
    for k in ENUM_FIELDS:
        if k in obj and isinstance(obj[k], str):
            ne = _norm_enum(obj[k])
            if ne != obj[k]:
                obj[k] = ne
                n += 1
    # num_panels: coerce to int, clamp to >=1.
    if "num_panels" in obj:
        iv = _to_int(obj["num_panels"])
        if iv is None:
            pass  # leave for validation to fail
        else:
            if iv < 1:
                iv = 1
            if iv != obj["num_panels"]:
                n += 1
            obj["num_panels"] = iv
    # scores: coerce numeric -> int (range checked in validation).
    for k in SCORE_FIELDS:
        if k in obj:
            iv = _to_int(obj[k])
            if iv is not None and iv != obj[k]:
                obj[k] = iv
                n += 1
    # booleans: coerce truthy strings.
    for k in BOOL_FIELDS:
        if k in obj and not isinstance(obj[k], bool):
            bv = _to_bool(obj[k])
            if bv is not None:
                obj[k] = bv
                n += 1
    return obj, n


def validate_schema(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "root is not a JSON object"

    if "num_panels" not in obj:
        return False, "missing num_panels"
    if not isinstance(obj["num_panels"], int) or obj["num_panels"] < 1:
        return False, "num_panels not an int >= 1"

    for k, allowed in ENUM_FIELDS.items():
        if k not in obj:
            return False, f"missing {k}"
        if not isinstance(obj[k], str) or obj[k] not in allowed:
            return False, f"{k} invalid value: {obj.get(k)!r}"

    for k in SCORE_FIELDS:
        if k not in obj:
            return False, f"missing {k}"
        if not isinstance(obj[k], int) or not (1 <= obj[k] <= 5):
            return False, f"{k} not an int in [1,5]: {obj.get(k)!r}"

    for k in BOOL_FIELDS:
        if k not in obj:
            return False, f"missing {k}"
        if not isinstance(obj[k], bool):
            return False, f"{k} not a boolean"

    for k in LIST_FIELDS:
        if not isinstance(obj.get(k), list):
            return False, f"{k} not a list"

    for k in STR_FIELDS:
        if not isinstance(obj.get(k), str):
            return False, f"{k} not a string"

    return True, ""


def count_valid_entities(items: List[Any]) -> int:
    c = 0
    for it in items:
        if isinstance(it, dict) and str(it.get("name", "")).strip():
            c += 1
    return c


def count_valid_relationships(items: List[Any]) -> int:
    c = 0
    for it in items:
        if (isinstance(it, dict) and str(it.get("from", "")).strip()
                and str(it.get("to", "")).strip()):
            c += 1
    return c


# =============================================================================
# Result rows
# =============================================================================

def empty_result(item: GAItem) -> Dict[str, Any]:
    r = {k: "" for k in CSV_FIELDS}
    r.update({
        "model": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "paper_id": item.paper_id,
        "ga_path": item.ga_path,
        "interpretation_file": str(item.interp_path),
        "status": "",
        "error_message": "",
        "parse_mode": "",
        "fields_coerced": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "inference_time": 0.0,
        "tokens_per_second": 0.0,
        "raw_output_length": 0,
        "had_think_tags": False,
        "had_code_fences": False,
        "image_orig_width": 0,
        "image_orig_height": 0,
        "image_sent_width": 0,
        "image_sent_height": 0,
    })
    for c in METADATA_COLS:
        r[c] = item.metadata.get(c, "")
    return r


def flatten_interp(result: Dict[str, Any], interp: Dict[str, Any]) -> None:
    scalar_keys = (["num_panels"] + list(ENUM_FIELDS.keys()) + SCORE_FIELDS + BOOL_FIELDS)
    for k in scalar_keys:
        result[k] = interp.get(k)
    result["num_main_entities"] = count_valid_entities(interp.get("main_entities", []))
    result["num_process_steps"] = len([s for s in interp.get("process_steps", []) if str(s).strip()])
    result["num_relationships"] = count_valid_relationships(interp.get("relationships", []))
    result["num_unclear_elements"] = len([s for s in interp.get("unclear_elements", []) if str(s).strip()])
    result["num_missing_links"] = len([s for s in interp.get("missing_links", []) if str(s).strip()])
    mo = str(interp.get("main_outcome", "")).strip()
    un = str(interp.get("uncertainty_notes", "")).strip()
    result["main_outcome"] = mo
    result["main_outcome_present"] = int(bool(mo))
    result["uncertainty_notes_present"] = int(bool(un))


# =============================================================================
# Save helpers
# =============================================================================

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))


def save_checkpoint(row: Dict[str, Any]) -> None:
    CHECKPOINT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    light = {k: row.get(k) for k in ("paper_id", "status", "error_message", "parse_mode",
                                     "inference_time", "output_tokens")}
    with CHECKPOINT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(), **light}, ensure_ascii=False) + "\n")


def get_gpu_info() -> Dict[str, Any]:
    info = {"gpu_model_name": "unknown", "gpu_count": 0, "gpu_vram_total_mb": 0, "gpu_vram_used_mb": 0}
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            return info
        names, total, used = [], 0, 0
        for line in [x.strip() for x in res.stdout.splitlines() if x.strip()]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                names.append(parts[0]); total += int(float(parts[1])); used += int(float(parts[2]))
        info.update({"gpu_model_name": names[0] if names else "unknown", "gpu_count": len(names),
                     "gpu_vram_total_mb": total, "gpu_vram_used_mb": used})
    except Exception:
        pass
    return info


# =============================================================================
# Context-retry wrapper (mirror of Task 1)
# =============================================================================

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
        retry_max = max(256, MAX_MODEL_LEN - in_tok - 64)
        if retry_max >= max_tokens:
            raise
        return await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=retry_max, extra_body=extra_body,
        )


# =============================================================================
# Inference
# =============================================================================

async def process_one(item: GAItem, system_prompt: str, user_prompt: str,
                      client: AsyncOpenAI, semaphore: asyncio.Semaphore,
                      temperature: float, max_tokens: int, max_image_dim: int,
                      force: bool) -> Dict[str, Any]:
    result = empty_result(item)

    if item.interp_path.exists() and not force:
        result["status"] = "skipped"
        print(f"[SKIP] {item.paper_id} | interpretation exists", flush=True)
        return result

    if not item.ga_path or not Path(item.ga_path).exists():
        result["status"] = "image_missing"
        result["error_message"] = f"GA image not found: {item.ga_path}"
        print(f"[FAIL] {item.paper_id} | image_missing", flush=True)
        save_checkpoint(result)
        return result

    try:
        b64, ow, oh, sw, sh = await asyncio.to_thread(encode_image, item.ga_path, max_image_dim)
        result.update({"image_orig_width": ow, "image_orig_height": oh,
                       "image_sent_width": sw, "image_sent_height": sh})
    except Exception as e:
        result["status"] = "image_read_error"
        result["error_message"] = str(e)
        print(f"[FAIL] {item.paper_id} | image_read_error | {e}", flush=True)
        save_checkpoint(result)
        return result

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]},
    ]

    async with semaphore:
        start = time.time()
        try:
            response = await create_with_context_retry(
                client, model=SERVED_MODEL_NAME, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                extra_body={"repetition_penalty": REPETITION_PENALTY},
            )
        except Exception as e:
            result["status"] = "api_error"
            result["error_message"] = str(e)
            print(f"[FAIL] {item.paper_id} | api_error | {e}", flush=True)
            save_checkpoint(result)
            return result
        elapsed = time.time() - start

    try:
        raw = response.choices[0].message.content or ""
    except Exception:
        raw = ""

    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
    result.update({
        "input_tokens": in_tok or 0,
        "output_tokens": out_tok or 0,
        "inference_time": round(elapsed, 2),
        "tokens_per_second": round((out_tok or 0) / elapsed, 2) if elapsed > 0 else 0.0,
        "raw_output_length": len(raw),
        "had_think_tags": bool(re.search(r"<think>", raw, flags=re.IGNORECASE)),
        "had_code_fences": raw.strip().startswith("```"),
    })

    if not raw.strip():
        result["status"] = "empty_response_error"
        result["error_message"] = "Model returned an empty response"
        save_text(ERROR_DIR / f"{safe_filename(item.paper_id)}{RAW_SUFFIX}", raw)
        print(f"[FAIL] {item.paper_id} | empty_response_error", flush=True)
        save_checkpoint(result)
        return result

    cleaned = extract_json_candidate(raw)
    obj, parse_mode = parse_json(cleaned)
    result["parse_mode"] = parse_mode
    if obj is None:
        result["status"] = "json_parse_error"
        result["error_message"] = "JSON parse failed after repair"
        save_text(ERROR_DIR / f"{safe_filename(item.paper_id)}{RAW_SUFFIX}", raw)
        save_text(ERROR_DIR / f"{safe_filename(item.paper_id)}{CLEANED_SUFFIX}", cleaned)
        print(f"[FAIL] {item.paper_id} | json_parse_error", flush=True)
        save_checkpoint(result)
        return result

    obj, n_coerced = coerce_schema(obj)
    result["fields_coerced"] = n_coerced

    valid, verr = validate_schema(obj)
    if not valid:
        result["status"] = "validation_error"
        result["error_message"] = verr
        save_text(ERROR_DIR / f"{safe_filename(item.paper_id)}{RAW_SUFFIX}", raw)
        save_json(ERROR_DIR / f"{safe_filename(item.paper_id)}{INVALID_SUFFIX}", obj)
        print(f"[FAIL] {item.paper_id} | validation_error | {verr}", flush=True)
        save_checkpoint(result)
        return result

    record = {
        "paper_id": item.paper_id,
        "ga_path": item.ga_path,
        "model": MODEL_NAME,
        "image_orig_width": result["image_orig_width"],
        "image_orig_height": result["image_orig_height"],
        "image_sent_width": result["image_sent_width"],
        "image_sent_height": result["image_sent_height"],
        **{c: item.metadata.get(c, "") for c in METADATA_COLS},
        "interpretation": obj,
    }
    try:
        save_json(item.interp_path, record)
        save_text(RAW_OUTPUT_DIR / f"{safe_filename(item.paper_id)}{RAW_SUFFIX}", raw)
    except Exception as e:
        result["status"] = "save_error"
        result["error_message"] = str(e)
        print(f"[FAIL] {item.paper_id} | save_error | {e}", flush=True)
        save_checkpoint(result)
        return result

    flatten_interp(result, obj)
    result["status"] = "success"
    print(f"[OK] {item.paper_id} | panels={result['num_panels']} "
          f"interp={result['semantic_interpretability']} "
          f"tokens={result['input_tokens']}+{result['output_tokens']} | {result['inference_time']}s",
          flush=True)
    save_checkpoint(result)
    return result


async def process_one_tracked(item: GAItem, system_prompt: str, user_prompt: str,
                              client: AsyncOpenAI, semaphore: asyncio.Semaphore,
                              temperature: float, max_tokens: int, max_image_dim: int,
                              force: bool, progress: "Progress",
                              progress_lock: asyncio.Lock) -> Dict[str, Any]:
    result = await process_one(item, system_prompt, user_prompt, client, semaphore,
                               temperature, max_tokens, max_image_dim, force)
    async with progress_lock:
        progress.update(result.get("status", ""))
        line = progress.render(item.paper_id, result.get("status", ""))
        print(line, flush=True)
    return result

def pct(n: int, d: int) -> float:
    return round(n / d * 100, 2) if d else 0.0


def mean(values: List[float]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def median(values: List[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


def compute_field_distributions(successes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-field value distributions: the calibration check (catches value-collapse)."""
    dist: Dict[str, Any] = {"n_success": len(successes)}
    for k in ENUM_FIELDS:
        dist[k] = dict(Counter(r.get(k) for r in successes))
    for k in BOOL_FIELDS:
        dist[k] = dict(Counter(bool(r.get(k)) for r in successes))
    for k in SCORE_FIELDS:
        vals = [r.get(k) for r in successes if isinstance(r.get(k), int)]
        dist[k] = {
            "histogram": dict(Counter(vals)),
            "mean": mean([float(v) for v in vals]),
            "median": median([float(v) for v in vals]),
        }
    for k in ["num_panels", "num_main_entities", "num_process_steps", "num_relationships",
              "num_unclear_elements", "num_missing_links"]:
        vals = [r.get(k) for r in successes if isinstance(r.get(k), (int, float))]
        dist[k] = {"mean": mean([float(v) for v in vals]), "median": median([float(v) for v in vals]),
                   "min": min(vals) if vals else 0, "max": max(vals) if vals else 0}
    dist["main_outcome_present_rate_pct"] = pct(sum(1 for r in successes if r.get("main_outcome_present")), len(successes))
    return dist


def compute_summary(results: List[Dict[str, Any]], gpu_info: Dict[str, Any],
                    total_wall_time: float, args: argparse.Namespace,
                    distributions: Dict[str, Any]) -> Dict[str, Any]:
    total = len(results)
    skipped = [r for r in results if r["status"] == "skipped"]
    processed = [r for r in results if r["status"] != "skipped"]
    successes = [r for r in processed if r["status"] == "success"]
    failures = [r for r in processed if r["status"] not in ("success", "skipped")]

    error_breakdown: Dict[str, int] = {}
    for r in failures:
        error_breakdown[r["status"]] = error_breakdown.get(r["status"], 0) + 1

    return {
        "model": MODEL_NAME, "served_model_name": SERVED_MODEL_NAME, "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
        "stage1_index": str(STAGE1_INDEX),
        "system_prompt_file": str(SYSTEM_PROMPT_FILE),
        "user_prompt_file": str(USER_PROMPT_FILE),
        "output_root": str(OUTPUT_ROOT),
        "args": vars(args), "gpu_info": gpu_info,
        "total_selected": total,
        "skipped_existing": len(skipped),
        "processed_count": len(processed),
        "success_count": len(successes),
        "fail_count": len(failures),
        "success_rate_processed_pct": pct(len(successes), len(processed)),
        "success_rate_selected_pct": pct(len(successes), total),
        "error_breakdown": error_breakdown,
        "parse_mode_breakdown": dict(Counter(r.get("parse_mode") for r in successes)),
        "total_coerced_fields": sum(int(r.get("fields_coerced") or 0) for r in successes),
        "avg_inference_time_sec": mean([r["inference_time"] for r in successes]),
        "median_inference_time_sec": median([r["inference_time"] for r in successes]),
        "avg_tokens_per_second": mean([r["tokens_per_second"] for r in successes if r["tokens_per_second"] > 0]),
        "total_input_tokens": sum(int(r["input_tokens"]) for r in successes),
        "total_output_tokens": sum(int(r["output_tokens"]) for r in successes),
        "avg_output_tokens": mean([r["output_tokens"] for r in successes]),
        "field_distributions": distributions,
        "total_wall_time_sec": round(total_wall_time, 2),
        "total_wall_time_min": round(total_wall_time / 60, 2),
    }


def write_reports(results: List[Dict[str, Any]], summary: Dict[str, Any],
                  distributions: Dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    save_json(SUMMARY_JSON, summary)
    save_json(DISTRIBUTION_JSON, distributions)

    lines = [
        "Task 2 Stage 4 VLM Structural Interpretation Summary",
        "====================================================",
        f"Generated: {summary['timestamp']}",
        f"Model: {summary['model']}  ({summary['model_path']})",
        f"Stage 1 index: {summary['stage1_index']}",
        f"System prompt: {summary['system_prompt_file']}",
        f"User prompt: {summary['user_prompt_file']}",
        f"Output root: {summary['output_root']}",
        "",
        "Run counts",
        "----------",
        f"Total selected: {summary['total_selected']}",
        f"Skipped existing: {summary['skipped_existing']}",
        f"Processed: {summary['processed_count']}",
        f"Success: {summary['success_count']}",
        f"Failed: {summary['fail_count']}",
        f"Success rate (processed): {summary['success_rate_processed_pct']}%",
        f"Success rate (selected): {summary['success_rate_selected_pct']}%",
        f"Error breakdown: {summary['error_breakdown']}",
        f"Parse-mode breakdown: {summary['parse_mode_breakdown']}",
        f"Total coerced fields: {summary['total_coerced_fields']}",
        "",
        "Timing and tokens",
        "-----------------",
        f"Average inference time: {summary['avg_inference_time_sec']} sec",
        f"Median inference time: {summary['median_inference_time_sec']} sec",
        f"Average tokens/sec: {summary['avg_tokens_per_second']}",
        f"Total output tokens: {summary['total_output_tokens']}",
        f"Average output tokens: {summary['avg_output_tokens']}",
        "",
        "Field distributions (CALIBRATION CHECK — inspect for value-collapse)",
        "-------------------------------------------------------------------",
        json.dumps(distributions, indent=2, ensure_ascii=False),
        "",
        "GPU info",
        "--------",
        json.dumps(summary.get("gpu_info", {}), indent=2),
        "",
        f"Total wall time: {summary['total_wall_time_sec']} sec ({summary['total_wall_time_min']} min)",
    ]
    save_text(SUMMARY_TXT, "\n".join(lines) + "\n")


# =============================================================================
# Main
# =============================================================================

async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Task 2 Stage 4 VLM structural interpretation (Qwen3-VL-32B-AWQ)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N items after slicing")
    parser.add_argument("--num-splits", type=int, default=None, help="Total number of parallel shards")
    parser.add_argument("--split-id", type=int, default=None, help="This shard's id in [1, num-splits]")
    parser.add_argument("--start-index", type=int, default=None, help="Manual start index in sorted list, inclusive")
    parser.add_argument("--end-index", type=int, default=None, help="Manual end index in sorted list, exclusive")
    parser.add_argument("--run-tag", type=str, default=None, help="Suffix for reports + checkpoint only (e.g. h1, h2)")
    parser.add_argument("--force", action="store_true", help="Rerun even if interpretation exists")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent API requests")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM, help="Longest side cap; 0 disables")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="OpenAI client timeout (sec)")
    parser.add_argument("--vllm-url", type=str, default=VLLM_URL, help="Base vLLM URL, without /v1")
    args = parser.parse_args()

    apply_run_tag(args.run_tag)
    ensure_directories()
    if repair_json is None:
        print("ERROR: json-repair not installed. Run: pip install json-repair --break-system-packages", flush=True)
        sys.exit(1)

    print("=" * 80, flush=True)
    print("Task 2 Stage 4 VLM Structural Interpretation — Qwen3-VL-32B-Instruct-AWQ", flush=True)
    print("=" * 80, flush=True)
    print(f"Stage 1 index:   {STAGE1_INDEX}", flush=True)
    print(f"System prompt:   {SYSTEM_PROMPT_FILE}", flush=True)
    print(f"User prompt:     {USER_PROMPT_FILE}", flush=True)
    print(f"Output root:     {OUTPUT_ROOT}", flush=True)
    print(f"vLLM URL:        {args.vllm_url}", flush=True)
    print(f"Run tag:         {args.run_tag}", flush=True)
    print(f"Splits:          num={args.num_splits} id={args.split_id}", flush=True)
    print(f"Start/End:       {args.start_index}/{args.end_index}", flush=True)
    print(f"Concurrency:     {args.concurrency}", flush=True)
    print(f"Max tokens:      {args.max_tokens}", flush=True)
    print(f"Max image dim:   {args.max_image_dim}", flush=True)
    print(f"Temperature:     {args.temperature}", flush=True)
    print(f"Force:           {args.force}", flush=True)
    print(f"Results CSV:     {RESULTS_CSV}", flush=True)
    print(f"Checkpoint:      {CHECKPOINT_JSONL}", flush=True)
    print("=" * 80, flush=True)

    system_prompt = load_prompt(SYSTEM_PROMPT_FILE, "System prompt")
    user_prompt = load_prompt(USER_PROMPT_FILE, "User prompt")

    items = discover_items(args.num_splits, args.split_id, args.start_index, args.end_index, args.limit)
    already = sum(1 for it in items if it.interp_path.exists())
    print(f"Selected GAs: {len(items)} | existing interpretations: {already} | "
          f"to process: {len(items) if args.force else len(items) - already}", flush=True)

    gpu_info = get_gpu_info()
    print(f"GPU info: {gpu_info}", flush=True)

    client = AsyncOpenAI(base_url=f"{args.vllm_url.rstrip('/')}/v1", api_key="not-needed", timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    progress = Progress(total=len(items))
    progress_lock = asyncio.Lock()

    total_start = time.time()
    tasks = [process_one_tracked(it, system_prompt, user_prompt, client, semaphore,
                                 args.temperature, args.max_tokens, args.max_image_dim, args.force,
                                 progress, progress_lock)
             for it in items]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results: List[Dict[str, Any]] = []
    for it, r in zip(items, raw_results):
        if isinstance(r, dict):
            results.append(r)
        else:
            row = empty_result(it)
            row["status"] = "crash_error"
            row["error_message"] = repr(r)
            save_checkpoint(row)
            results.append(row)
    total_wall_time = time.time() - total_start

    successes = [r for r in results if r["status"] == "success"]
    distributions = compute_field_distributions(successes)
    summary = compute_summary(results, gpu_info, total_wall_time, args, distributions)
    write_reports(results, summary, distributions)

    print("\n" + "=" * 80, flush=True)
    print("STAGE 4 COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Selected: {summary['total_selected']} | Skipped: {summary['skipped_existing']} | "
          f"Processed: {summary['processed_count']} | Success: {summary['success_count']} | "
          f"Failed: {summary['fail_count']}", flush=True)
    print(f"Error breakdown: {summary['error_breakdown']}", flush=True)
    print(f"Results CSV: {RESULTS_CSV}", flush=True)
    print(f"Summary TXT: {SUMMARY_TXT}", flush=True)
    print(f"Field distributions: {DISTRIBUTION_JSON}", flush=True)
    print(f"Wall time: {summary['total_wall_time_min']} min", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(main_async())