#!/usr/bin/env python3
"""
Task 1 / Variant B / Stage 1 SRP Inference
Model: gemma-3-27b-it (W4A16 / compressed-tensors), multimodal text + figures

Reads a Variant B CSV with columns:
  - doi
  - user_prompt_path
  - figure_id_to_path_json

INPUTS are read from the ORIGINAL (non-AWQ) tree; OUTPUTS are written to the new
"_awq" tree, so the original running jobs are never touched. Images are sent as
base64 (no --allowed-local-media-path needed).

Gemma has NO system role: the system prompt is prepended to the text block.

The --run-tag flag suffixes ONLY the reports + checkpoint. SRPs/raw/errors are
per-DOI and safely shared between halves.

The live log prints a running progress line after each prompt:
  [PROGRESS] processed X/N | remaining Y

Examples:
  python3 task1_variantB_stage1_gemma_3_27b_inference.py --limit 1
  python3 task1_variantB_stage1_gemma_3_27b_inference.py --start-index 0 --end-index 500 --run-tag h1
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

MODEL_NAME = "gemma_3_27b"
SERVED_MODEL_NAME = "gemma_3_27b"
MODEL_PATH = "./models/awq/gemma-3-27b-it-w4a16"
VLLM_URL = "http://localhost:8000"

# Inputs come from the ORIGINAL tree (shared, read-only).
INPUT_BASE = Path("./task1_completeness/variant_B")
SYSTEM_PROMPT_FILE = INPUT_BASE / "system_prompts" / "task1_variantB_phase1_system_prompt.txt"
FIGURE_PATHS_CSV = INPUT_BASE / "user_prompts" / "task1_variantB_phase1_figure_paths.csv"

# Outputs go to the NEW "_awq" tree.
OUTPUT_BASE = Path("./task1_completeness_awq/variant_B")
OUTPUT_ROOT = OUTPUT_BASE / "outputs" / "stage1" / MODEL_NAME
SRP_DIR = OUTPUT_ROOT / "srps"
RAW_OUTPUT_DIR = OUTPUT_ROOT / "raw_outputs"
ERROR_DIR = OUTPUT_ROOT / "errors"
LOG_DIR = OUTPUT_ROOT / "logs"
REPORT_DIR = OUTPUT_ROOT / "reports"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

SRP_SUFFIX = "_srp_gemma_3_27b.json"
RAW_OUTPUT_SUFFIX = "_raw_output_gemma_3_27b.txt"
CLEANED_OUTPUT_SUFFIX = "_cleaned_output_gemma_3_27b.txt"
INVALID_SRP_SUFFIX = "_invalid_srp_gemma_3_27b.json"
IMAGE_ERROR_SUFFIX = "_image_errors_gemma_3_27b.json"

# These are reassigned in main() when --run-tag is provided.
RESULTS_CSV = REPORT_DIR / "task1_variantB_stage1_gemma_3_27b_results.csv"
SUMMARY_JSON = REPORT_DIR / "task1_variantB_stage1_gemma_3_27b_summary.json"
SUMMARY_TXT = REPORT_DIR / "task1_variantB_stage1_gemma_3_27b_summary.txt"
CHECKPOINT_JSONL = CHECKPOINT_DIR / "processed_prompts.jsonl"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 6144
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 600.0

# Repetition penalty (vLLM sampling extension) to break Gemma's degenerate
# repeat loops at temperature=0 that otherwise truncate JSON mid-string.
REPETITION_PENALTY = 1.05

SECTIONS = ["introduction", "methods", "results", "discussion"]
RELATION_KEYS = ["intro_to_methods", "methods_to_results", "results_to_discussion"]

MAX_MODEL_LEN = 53248  # must match the vLLM server --max-model-len


def _input_tokens_from_error(msg: str) -> Optional[int]:
    m = re.search(r"has (\d+) input tokens", msg or "")
    return int(m.group(1)) if m else None


async def create_with_context_retry(client, *, model, messages, temperature, max_tokens, extra_body):
    """Call chat.completions.create; if the server rejects the request because
    max_tokens + input exceeds the context window, retry once with a reduced
    max_tokens derived from the reported input length. Comparability is kept:
    behaviour is identical unless the request would otherwise 400."""
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

CSV_FIELDS = [
    "model", "served_model_name", "doi", "doi_safe", "prompt_file", "srp_file",
    "status", "error_message",
    "figure_count_in_csv", "valid_image_count", "missing_image_count", "image_load_error_count",
    "total_image_bytes", "figure_ids_used", "missing_figure_ids", "image_error_figure_ids",
    "input_tokens", "output_tokens", "inference_time", "tokens_per_second",
    "raw_output_length", "had_think_tags", "had_code_fences",
    "entity_count", "entities_introduction", "entities_methods", "entities_results", "entities_discussion",
    "total_visual_proxies", "avg_visual_proxies_per_entity", "entities_with_no_proxies",
    "summary_words_introduction", "summary_words_methods", "summary_words_results", "summary_words_discussion",
    "summary_balance_std", "causal_relations_present", "causal_relation_avg_words",
    "all_summaries_nonempty", "all_relations_nonempty", "all_entities_have_proxies", "stage2_ready",
    "parse_mode", "visual_proxies_coerced",
    "repaired",
]

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class WorkItem:
    doi: str
    doi_safe: str
    user_prompt_path: Path
    figure_id_to_path: Dict[str, str]

# =============================================================================
# Run-tag aware paths
# =============================================================================

def apply_run_tag(run_tag: Optional[str]) -> None:
    """Suffix ONLY the report + checkpoint paths so parallel halves never clobber.
    SRPs / raw_outputs / errors are per-DOI and safely shared."""
    global RESULTS_CSV, SUMMARY_JSON, SUMMARY_TXT, CHECKPOINT_JSONL
    if not run_tag:
        return
    tag = f"_{run_tag}"
    RESULTS_CSV = REPORT_DIR / f"task1_variantB_stage1_gemma_3_27b_results{tag}.csv"
    SUMMARY_JSON = REPORT_DIR / f"task1_variantB_stage1_gemma_3_27b_summary{tag}.json"
    SUMMARY_TXT = REPORT_DIR / f"task1_variantB_stage1_gemma_3_27b_summary{tag}.txt"
    CHECKPOINT_JSONL = CHECKPOINT_DIR / f"processed_prompts{tag}.jsonl"

# =============================================================================
# Basic utilities
# =============================================================================

def make_dirs() -> None:
    for path in [SRP_DIR, RAW_OUTPUT_DIR, ERROR_DIR, LOG_DIR, REPORT_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    # errors="replace" guards against lone surrogates in model output that
    # otherwise raise UnicodeEncodeError and abort the whole run.
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def doi_to_safe_id(doi: str) -> str:
    """Convert DOI into safe filename id matching the dataset style."""
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
    info = {
        "gpu_model_name": "unknown",
        "gpu_count": 0,
        "gpu_vram_total_mb": 0,
        "gpu_vram_allocated_mb": 0,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().splitlines()
            names = []
            total = 0
            used = 0
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    names.append(parts[0])
                    total += int(float(parts[1]))
                    used += int(float(parts[2]))
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

def load_work_items(limit: Optional[int], start_index: Optional[int], end_index: Optional[int]) -> List[WorkItem]:
    if not FIGURE_PATHS_CSV.exists():
        raise FileNotFoundError(f"Figure paths CSV not found: {FIGURE_PATHS_CSV}")

    items: List[WorkItem] = []
    with open(FIGURE_PATHS_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"doi", "user_prompt_path", "figure_id_to_path_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row_num, row in enumerate(reader, start=2):
            doi = (row.get("doi") or "").strip()
            prompt_path = (row.get("user_prompt_path") or "").strip()
            figure_json = (row.get("figure_id_to_path_json") or "").strip()

            if not doi:
                raise ValueError(f"Row {row_num}: empty DOI")
            if not prompt_path:
                raise ValueError(f"Row {row_num}: empty user_prompt_path for DOI {doi}")

            try:
                fig_map = json.loads(figure_json) if figure_json else {}
            except json.JSONDecodeError as exc:
                raise ValueError(f"Row {row_num}: invalid figure_id_to_path_json for DOI {doi}: {exc}") from exc

            if not isinstance(fig_map, dict):
                raise ValueError(f"Row {row_num}: figure_id_to_path_json is not a JSON object for DOI {doi}")

            fig_map_clean = {str(k): str(v) for k, v in fig_map.items() if str(v).strip()}
            items.append(
                WorkItem(
                    doi=doi,
                    doi_safe=doi_to_safe_id(doi),
                    user_prompt_path=Path(prompt_path),
                    figure_id_to_path=fig_map_clean,
                )
            )

    # Deterministic order by safe DOI, matching the CSV rows closely but stable.
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


def sort_figure_items(fig_map: Dict[str, str]) -> List[Tuple[str, str]]:
    def sort_key(kv: Tuple[str, str]) -> Tuple[int, str]:
        fig_id = kv[0]
        m = re.search(r"(\d+)", fig_id)
        return (int(m.group(1)) if m else 10**9, fig_id)
    return sorted(fig_map.items(), key=sort_key)


def load_images_as_content_blocks(item: WorkItem, max_images: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load all image paths for one item as OpenAI image_url content blocks.

    max_images=0 means no cap; use every image path from the CSV row.
    """
    blocks: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    load_errors: List[Dict[str, str]] = []
    used: List[Dict[str, Any]] = []
    total_bytes = 0

    fig_items = sort_figure_items(item.figure_id_to_path)
    if max_images and max_images > 0:
        fig_items = fig_items[:max_images]

    for fig_id, img_path_str in fig_items:
        img_path = Path(img_path_str)
        if not img_path.exists():
            missing.append({"figure_id": fig_id, "path": img_path_str})
            continue
        if not img_path.is_file():
            load_errors.append({"figure_id": fig_id, "path": img_path_str, "error": "not a regular file"})
            continue
        try:
            data = img_path.read_bytes()
            if not data:
                load_errors.append({"figure_id": fig_id, "path": img_path_str, "error": "empty file"})
                continue
            mime = guess_mime_type(img_path)
            encoded = base64.b64encode(data).decode("ascii")
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )
            total_bytes += len(data)
            used.append({"figure_id": fig_id, "path": img_path_str, "bytes": len(data), "mime": mime})
        except Exception as exc:
            load_errors.append({"figure_id": fig_id, "path": img_path_str, "error": str(exc)})

    stats = {
        "figure_count_in_csv": len(item.figure_id_to_path),
        "valid_image_count": len(used),
        "missing_image_count": len(missing),
        "image_load_error_count": len(load_errors),
        "total_image_bytes": total_bytes,
        "figure_ids_used": ";".join(x["figure_id"] for x in used),
        "missing_figure_ids": ";".join(x["figure_id"] for x in missing),
        "image_error_figure_ids": ";".join(x["figure_id"] for x in load_errors),
        "used_images": used,
        "missing_images": missing,
        "image_load_errors": load_errors,
    }
    return blocks, stats

# =============================================================================
# SRP validation and metrics
# =============================================================================

def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def loads_tolerant(text: str) -> Any:
    """Parse messy model JSON without external deps.
    Order: strict -> strict=False (allow literal control chars in strings)
    -> brace-extract -> trailing-comma repair. No-op on already-clean JSON.
    Raises json.JSONDecodeError if every attempt fails."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        last = e
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        last = e
    first = text.find("{")
    lastb = text.rfind("}")
    if first != -1 and lastb != -1 and lastb > first:
        candidate = text[first:lastb + 1]
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as e:
            last = e
        try:
            return json.loads(_strip_trailing_commas(candidate), strict=False)
        except json.JSONDecodeError as e:
            last = e
    try:
        return json.loads(_strip_trailing_commas(text), strict=False)
    except json.JSONDecodeError as e:
        last = e
    raise last


# --- Robust JSON handling for messy model output (uniform across all models) ---
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


def parse_srp_json(cleaned: str):
    """Parse model JSON, recording how it was obtained.
    Returns (obj_or_None, parse_mode) with parse_mode in {raw, repaired, failed}.
    1) strict json.loads  2) json-repair (bad escapes, control chars, unterminated
    strings, missing/trailing commas). Repair only fixes SYNTAX, never invents
    content; a truncated fragment repairs to an incomplete dict that then fails
    validation, so genuine failures stay failures."""
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


# Variant labels models emit for the IMRaD "discussion" section. The model
# (e.g. Gemma) frequently merges discussion and conclusion into one key; the
# content is complete and correct, only the key name differs.
_DISCUSSION_ALIASES = (
    "discussion/conclusion",
    "discussion_conclusion",
    "discussion & conclusion",
    "discussion and conclusion",
    "discussion/conclusions",
    "conclusion",
    "conclusions",
    "discussion_and_conclusion",
)


def normalize_section_keys(srp):
    """Rename merged/aliased section labels (e.g. 'discussion/conclusion') to the
    canonical 'discussion' in both section_summaries and key_entities, but only
    when a canonical 'discussion' key is not already present. Renames data in
    place without altering its content. Returns the number of keys renamed."""
    renamed = 0
    if not isinstance(srp, dict):
        return renamed
    for block_name in ("section_summaries", "key_entities"):
        block = srp.get(block_name)
        if not isinstance(block, dict) or "discussion" in block:
            continue
        for alias in _DISCUSSION_ALIASES:
            for existing in list(block.keys()):
                if existing.strip().lower() == alias and "discussion" not in block:
                    block["discussion"] = block.pop(existing)
                    renamed += 1
                    break
    return renamed


def coerce_srp_schema(srp):
    """Fill only the optional field the models commonly omit (visual_proxies),
    so a structurally-complete SRP is not failed over a single absent list.
    Also normalizes merged section labels to the canonical 'discussion'.
    Does NOT add entities, summaries, or relations. Returns (srp, n_filled)."""
    n = 0
    normalize_section_keys(srp)
    if isinstance(srp, dict):
        ke = srp.get("key_entities")
        if isinstance(ke, dict):
            for sec in SECTIONS:
                ents = ke.get(sec)
                if isinstance(ents, list):
                    for ent in ents:
                        if isinstance(ent, dict) and "visual_proxies" not in ent:
                            ent["visual_proxies"] = []
                            n += 1
                        elif isinstance(ent, dict) and not isinstance(ent.get("visual_proxies"), list):
                            ent["visual_proxies"] = []
                            n += 1
    return srp, n


REPAIR_INSTRUCTION = (
    "Your previous response was not a complete, valid SRP. Specifically: {problem}. "
    "Return ONLY the corrected JSON object, with all required top-level keys present "
    "(doi, title, section_summaries, key_entities, causal_relations). section_summaries "
    "MUST contain all four sections (introduction, methods, results, discussion) as "
    "non-empty strings, and key_entities MUST contain all four sections. Do not include "
    "any text outside the JSON object."
)


async def repair_srp_once(
    client: AsyncOpenAI,
    *,
    base_messages: List[Dict[str, Any]],
    prior_raw: str,
    problem: str,
    temperature: float,
    max_tokens: int,
):
    """One-shot corrective re-ask after a validation failure. Re-sends the
    original turn (text + images) plus the model's prior (invalid) output and a
    correction instruction, then returns the parsed+coerced SRP (or None) and raw."""
    repair_messages = list(base_messages) + [
        {"role": "assistant", "content": prior_raw},
        {"role": "user", "content": REPAIR_INSTRUCTION.format(problem=problem)},
    ]
    response = await create_with_context_retry(
        client,
        model=SERVED_MODEL_NAME,
        messages=repair_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"repetition_penalty": REPETITION_PENALTY},
    )
    raw = ""
    try:
        raw = response.choices[0].message.content or ""
    except Exception:
        raw = ""
    if not raw.strip():
        return None, raw
    cleaned = strip_code_fences(strip_think_tags(raw))
    srp, _ = parse_srp_json(cleaned)
    if srp is None:
        return None, raw
    srp, _ = coerce_srp_schema(srp)
    return srp, raw


def validate_srp(srp: Any) -> Tuple[bool, str]:
    if not isinstance(srp, dict):
        return False, "SRP root is not a JSON object"

    # Coerce a missing/malformed causal_relations into a valid empty object so a
    # paper with no extractable (or truncated) relations validates with empty
    # strings instead of hard-failing. Matches the Qwen pipeline's behaviour.
    if not isinstance(srp.get("causal_relations"), dict):
        srp["causal_relations"] = {k: "" for k in RELATION_KEYS}
    else:
        for k in RELATION_KEYS:
            if not isinstance(srp["causal_relations"].get(k), str):
                srp["causal_relations"][k] = ""

    required_top = ["doi", "title", "section_summaries", "key_entities", "causal_relations"]
    for key in required_top:
        if key not in srp:
            return False, f"Missing top-level key: {key}"

    if not isinstance(srp["section_summaries"], dict):
        return False, "section_summaries is not a dict"
    for sec in SECTIONS:
        if sec not in srp["section_summaries"]:
            return False, f"section_summaries missing section: {sec}"
        if not isinstance(srp["section_summaries"][sec], str):
            return False, f"section_summaries[{sec}] is not a string"

    if not isinstance(srp["key_entities"], dict):
        return False, "key_entities is not a dict"
    for sec in SECTIONS:
        if sec not in srp["key_entities"]:
            return False, f"key_entities missing section: {sec}"
        if not isinstance(srp["key_entities"][sec], list):
            return False, f"key_entities[{sec}] is not a list"
        for idx, entity in enumerate(srp["key_entities"][sec]):
            if not isinstance(entity, dict):
                return False, f"Entity {idx} in {sec} is not an object"
            for key in ["entity", "type", "visual_proxies"]:
                if key not in entity:
                    return False, f"Entity {idx} in {sec} missing key: {key}"
            if not isinstance(entity["visual_proxies"], list):
                return False, f"Entity {idx} in {sec} visual_proxies is not a list"

    if not isinstance(srp["causal_relations"], dict):
        return False, "causal_relations is not a dict"
    for key in RELATION_KEYS:
        if key not in srp["causal_relations"]:
            return False, f"Missing causal relation: {key}"
        if not isinstance(srp["causal_relations"][key], str):
            return False, f"causal_relations[{key}] is not a string"

    return True, ""


def compute_paper_metrics(srp: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    total_entities = 0
    total_proxies = 0
    no_proxy = 0

    for sec in SECTIONS:
        entities = srp.get("key_entities", {}).get(sec, []) or []
        metrics[f"entities_{sec}"] = len(entities)
        total_entities += len(entities)
        for ent in entities:
            proxies = ent.get("visual_proxies", []) or []
            total_proxies += len(proxies)
            if len(proxies) == 0:
                no_proxy += 1

    metrics["entity_count"] = total_entities
    metrics["total_visual_proxies"] = total_proxies
    metrics["avg_visual_proxies_per_entity"] = round(total_proxies / total_entities, 2) if total_entities else 0.0
    metrics["entities_with_no_proxies"] = no_proxy

    word_counts = []
    for sec in SECTIONS:
        summary = srp.get("section_summaries", {}).get(sec, "") or ""
        wc = len(summary.split()) if summary else 0
        metrics[f"summary_words_{sec}"] = wc
        word_counts.append(wc)
    metrics["summary_balance_std"] = round(statistics.pstdev(word_counts), 2) if word_counts else 0.0

    relations = srp.get("causal_relations", {}) or {}
    nonempty_relations = []
    for rk in RELATION_KEYS:
        val = relations.get(rk, "")
        if isinstance(val, str) and val.strip():
            nonempty_relations.append(val.strip())
    metrics["causal_relations_present"] = len(nonempty_relations)
    metrics["causal_relation_avg_words"] = round(
        sum(len(x.split()) for x in nonempty_relations) / len(nonempty_relations), 2
    ) if nonempty_relations else 0.0

    metrics["all_summaries_nonempty"] = all(
        bool((srp.get("section_summaries", {}).get(sec, "") or "").strip()) for sec in SECTIONS
    )
    metrics["all_relations_nonempty"] = len(nonempty_relations) == len(RELATION_KEYS)
    metrics["all_entities_have_proxies"] = total_entities > 0 and no_proxy == 0
    metrics["stage2_ready"] = (
        metrics["all_summaries_nonempty"]
        and metrics["all_relations_nonempty"]
        and metrics["all_entities_have_proxies"]
    )
    return metrics

# =============================================================================
# Processing
# =============================================================================

def default_result(item: WorkItem) -> Dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "doi": item.doi,
        "doi_safe": item.doi_safe,
        "prompt_file": str(item.user_prompt_path),
        "srp_file": str(SRP_DIR / f"{item.doi_safe}{SRP_SUFFIX}"),
        "status": "",
        "error_message": "",
        "figure_count_in_csv": len(item.figure_id_to_path),
        "valid_image_count": 0,
        "missing_image_count": 0,
        "image_load_error_count": 0,
        "total_image_bytes": 0,
        "figure_ids_used": "",
        "missing_figure_ids": "",
        "image_error_figure_ids": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "inference_time": 0.0,
        "tokens_per_second": 0.0,
        "raw_output_length": 0,
        "had_think_tags": False,
        "had_code_fences": False,
        "entity_count": 0,
        "entities_introduction": 0,
        "entities_methods": 0,
        "entities_results": 0,
        "entities_discussion": 0,
        "total_visual_proxies": 0,
        "avg_visual_proxies_per_entity": 0.0,
        "entities_with_no_proxies": 0,
        "summary_words_introduction": 0,
        "summary_words_methods": 0,
        "summary_words_results": 0,
        "summary_words_discussion": 0,
        "summary_balance_std": 0.0,
        "causal_relations_present": 0,
        "causal_relation_avg_words": 0.0,
        "all_summaries_nonempty": False,
        "all_relations_nonempty": False,
        "all_entities_have_proxies": False,
        "stage2_ready": False,
        "parse_mode": "",
        "visual_proxies_coerced": 0,
        "repaired": False,
    }


def save_checkpoint(result: Dict[str, Any]) -> None:
    CHECKPOINT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.lock = asyncio.Lock()

    async def tick(self, doi_safe: str, status: str) -> None:
        async with self.lock:
            self.done += 1
            done = self.done
        remaining = self.total - done
        print(f"[PROGRESS] {doi_safe} | {status} | processed {done}/{self.total} | remaining {remaining}", flush=True)


async def _process_item_core(
    item: WorkItem,
    system_prompt: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    force: bool,
    temperature: float,
    max_tokens: int,
    max_images: int,
) -> Dict[str, Any]:
    result = default_result(item)
    srp_path = SRP_DIR / f"{item.doi_safe}{SRP_SUFFIX}"

    if srp_path.exists() and not force:
        result["status"] = "skipped"
        print(f"[SKIP] {item.doi_safe} | existing SRP", flush=True)
        save_checkpoint(result)
        return result

    try:
        user_prompt = load_text(item.user_prompt_path)
    except Exception as exc:
        result["status"] = "prompt_read_error"
        result["error_message"] = str(exc)
        print(f"[FAIL] {item.doi_safe} | prompt_read_error | {exc}", flush=True)
        save_checkpoint(result)
        return result

    image_blocks, image_stats = load_images_as_content_blocks(item, max_images=max_images)
    for key in [
        "figure_count_in_csv", "valid_image_count", "missing_image_count", "image_load_error_count",
        "total_image_bytes", "figure_ids_used", "missing_figure_ids", "image_error_figure_ids",
    ]:
        result[key] = image_stats[key]

    if image_stats["missing_images"] or image_stats["image_load_errors"]:
        save_json(
            ERROR_DIR / f"{item.doi_safe}{IMAGE_ERROR_SUFFIX}",
            {
                "doi": item.doi,
                "doi_safe": item.doi_safe,
                "missing_images": image_stats["missing_images"],
                "image_load_errors": image_stats["image_load_errors"],
            },
        )

    # Gemma has no system role; prepend the system prompt to the text block,
    # then append every valid paper figure/table image from the CSV.
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"}]
    user_content.extend(image_blocks)

    async with semaphore:
        start = time.time()
        base_messages = [
            {"role": "user", "content": user_content},
        ]
        try:
            response = await create_with_context_retry(
                client,
                model=SERVED_MODEL_NAME,
                messages=base_messages,
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

    srp, parse_mode = parse_srp_json(cleaned)
    result["parse_mode"] = parse_mode
    if srp is None:
        result["status"] = "json_parse_error"
        result["error_message"] = "JSON parse failed after repair"
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_text(ERROR_DIR / f"{item.doi_safe}{CLEANED_OUTPUT_SUFFIX}", cleaned)
        print(f"[FAIL] {item.doi_safe} | json_parse_error", flush=True)
        save_checkpoint(result)
        return result

    srp, _n_coerced = coerce_srp_schema(srp)
    result["visual_proxies_coerced"] = _n_coerced

    is_valid, validation_error = validate_srp(srp)
    if not is_valid:
        # One-shot corrective re-ask: ask the model to fix the specific problem
        # (commonly a missing IMRaD section). Only fail if the repair also fails.
        try:
            repaired_srp, repair_raw = await repair_srp_once(
                client,
                base_messages=base_messages,
                prior_raw=raw_content,
                problem=validation_error,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            repaired_srp, repair_raw = None, ""
        if repaired_srp is not None:
            re_valid, re_error = validate_srp(repaired_srp)
            if re_valid:
                srp = repaired_srp
                raw_content = repair_raw or raw_content
                result["repaired"] = True
                is_valid, validation_error = True, ""
            else:
                validation_error = re_error
    if not is_valid:
        result["status"] = "validation_error"
        result["error_message"] = validation_error
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_json(ERROR_DIR / f"{item.doi_safe}{INVALID_SRP_SUFFIX}", srp)
        print(f"[FAIL] {item.doi_safe} | validation_error | {validation_error}", flush=True)
        save_checkpoint(result)
        return result

    try:
        save_json(srp_path, srp)
        save_text(RAW_OUTPUT_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
    except Exception as exc:
        result["status"] = "save_error"
        result["error_message"] = str(exc)
        print(f"[FAIL] {item.doi_safe} | save_error | {exc}", flush=True)
        save_checkpoint(result)
        return result

    result.update(compute_paper_metrics(srp))
    result["status"] = "success"
    result["srp_file"] = str(srp_path)
    print(
        f"[OK] {item.doi_safe} | images={result['valid_image_count']}/{result['figure_count_in_csv']} | "
        f"entities={result['entity_count']} | {elapsed:.1f}s",
        flush=True,
    )
    save_checkpoint(result)
    return result


async def process_item(
    item: WorkItem,
    system_prompt: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    force: bool,
    temperature: float,
    max_tokens: int,
    max_images: int,
    progress: Progress,
) -> Dict[str, Any]:
    result = await _process_item_core(item, system_prompt, client, semaphore, force, temperature, max_tokens, max_images)
    await progress.tick(item.doi_safe, result.get("status", "unknown"))
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

    summary: Dict[str, Any] = {
        "task": "task1_completeness",
        "variant": "B",
        "stage": "stage1",
        "model_name": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
        "total_wall_time_sec": round(total_wall_time, 2),
        "system_prompt_file": str(SYSTEM_PROMPT_FILE),
        "figure_paths_csv": str(FIGURE_PATHS_CSV),
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
        "total_input_tokens": sum(r["input_tokens"] for r in successful),
        "total_output_tokens": sum(r["output_tokens"] for r in successful),
        "avg_inference_time": round(statistics.mean(inference_times), 2) if inference_times else 0.0,
        "median_inference_time": round(statistics.median(inference_times), 2) if inference_times else 0.0,
        "avg_throughput_tokens_per_sec": round(statistics.mean(tps_values), 2) if tps_values else 0.0,
        "total_figures_in_csv_selected": sum(r["figure_count_in_csv"] for r in results),
        "total_valid_images_sent": sum(r["valid_image_count"] for r in results),
        "total_missing_images": sum(r["missing_image_count"] for r in results),
        "total_image_load_errors": sum(r["image_load_error_count"] for r in results),
        "total_image_bytes_sent": sum(r["total_image_bytes"] for r in results),
    }

    if successful:
        summary.update({
            "avg_entity_count": round(statistics.mean([r["entity_count"] for r in successful]), 2),
            "avg_visual_proxies_per_entity": round(statistics.mean([r["avg_visual_proxies_per_entity"] for r in successful]), 2),
            "avg_causal_relations_present": round(statistics.mean([r["causal_relations_present"] for r in successful]), 2),
            "stage2_ready_count": sum(1 for r in successful if r["stage2_ready"]),
            "stage2_ready_rate": pct(sum(1 for r in successful if r["stage2_ready"]), len(successful)),
            "avg_entities_per_section": {
                sec: round(statistics.mean([r[f"entities_{sec}"] for r in successful]), 2) for sec in SECTIONS
            },
            "avg_summary_words_per_section": {
                sec: round(statistics.mean([r[f"summary_words_{sec}"] for r in successful]), 2) for sec in SECTIONS
            },
        })
    else:
        summary.update({
            "avg_entity_count": 0.0,
            "avg_visual_proxies_per_entity": 0.0,
            "avg_causal_relations_present": 0.0,
            "stage2_ready_count": 0,
            "stage2_ready_rate": 0.0,
            "avg_entities_per_section": {sec: 0.0 for sec in SECTIONS},
            "avg_summary_words_per_section": {sec: 0.0 for sec in SECTIONS},
        })

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
        "Task 1 Variant B Stage 1 Gemma-3-27B (W4A16) Inference Summary",
        "=============================================================",
        f"Generated: {summary['timestamp']}",
        f"Model: {summary['model_name']}",
        f"Model path: {summary['model_path']}",
        f"System prompt: {summary['system_prompt_file']}",
        f"Figure paths CSV: {summary['figure_paths_csv']}",
        f"Output root: {summary['output_root']}",
        "",
        f"Rows selected: {summary['total_rows_selected']}",
        f"Skipped existing SRPs: {summary['skipped_count']}",
        f"Processed this run: {summary['processed_count']}",
        f"Successful: {summary['success_count']}",
        f"Failed: {summary['fail_count']}",
        f"Success rate processed: {summary['success_rate_processed']}%",
        f"Error breakdown: {summary['error_breakdown']}",
        "",
        f"Figures listed in selected CSV rows: {summary['total_figures_in_csv_selected']}",
        f"Valid images sent: {summary['total_valid_images_sent']}",
        f"Missing images: {summary['total_missing_images']}",
        f"Image load errors: {summary['total_image_load_errors']}",
        f"Total image bytes sent: {summary['total_image_bytes_sent']:,}",
        "",
        f"Avg inference time: {summary['avg_inference_time']} sec",
        f"Median inference time: {summary['median_inference_time']} sec",
        f"Avg throughput: {summary['avg_throughput_tokens_per_sec']} tok/sec",
        f"Total input tokens: {summary['total_input_tokens']:,}",
        f"Total output tokens: {summary['total_output_tokens']:,}",
        "",
        f"Avg entity count: {summary['avg_entity_count']}",
        f"Avg proxies/entity: {summary['avg_visual_proxies_per_entity']}",
        f"Avg causal relations present: {summary['avg_causal_relations_present']}/3",
        f"Stage 2 ready: {summary['stage2_ready_count']} ({summary['stage2_ready_rate']}%)",
        "",
        f"Results CSV: {RESULTS_CSV}",
        f"Summary JSON: {SUMMARY_JSON}",
    ]
    save_text(SUMMARY_TXT, "\n".join(lines) + "\n")

# =============================================================================
# Main
# =============================================================================

async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Task 1 Variant B Stage 1 Gemma-3-27B (W4A16) SRP inference")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N selected CSV rows")
    parser.add_argument("--start-index", type=int, default=None, help="Start index after sorting rows by safe DOI")
    parser.add_argument("--end-index", type=int, default=None, help="End index after sorting rows by safe DOI, exclusive")
    parser.add_argument("--run-tag", type=str, default=None, help="Suffix for reports + checkpoint only (e.g. h1, h2). SRPs are shared.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing SRPs for selected rows")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Async request concurrency")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="OpenAI client timeout in seconds")
    parser.add_argument("--max-images", type=int, default=0, help="Debug cap for images per paper; 0 means use all images")
    args = parser.parse_args()

    apply_run_tag(args.run_tag)
    make_dirs()
    if repair_json is None:
        print("ERROR: json-repair not installed in this env. Run: pip install json-repair --break-system-packages", flush=True)
        sys.exit(1)

    # Regenerate this run's checkpoint while preserving SRPs/raw/error artifacts.
    if CHECKPOINT_JSONL.exists():
        CHECKPOINT_JSONL.unlink()

    print("=" * 80, flush=True)
    print("Task 1 / Variant B / Stage 1 SRP Inference — Gemma-3-27B (W4A16)", flush=True)
    print("=" * 80, flush=True)
    print(f"System prompt: {SYSTEM_PROMPT_FILE}", flush=True)
    print(f"Figure paths CSV: {FIGURE_PATHS_CSV}", flush=True)
    print(f"Output root: {OUTPUT_ROOT}", flush=True)
    print(f"Run tag: {args.run_tag}", flush=True)
    print(f"Start index: {args.start_index} | End index: {args.end_index}", flush=True)
    print(f"Force rerun: {args.force}", flush=True)
    print(f"Concurrency: {args.concurrency}", flush=True)
    print(f"Max tokens: {args.max_tokens}", flush=True)
    print(f"Max images per paper: {'ALL' if args.max_images == 0 else args.max_images}", flush=True)
    print(f"Results CSV: {RESULTS_CSV}", flush=True)
    print(f"Checkpoint: {CHECKPOINT_JSONL}", flush=True)

    if not SYSTEM_PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_FILE}")
    system_prompt = load_text(SYSTEM_PROMPT_FILE)
    items = load_work_items(limit=args.limit, start_index=args.start_index, end_index=args.end_index)
    print(f"Selected rows: {len(items)}", flush=True)

    existing = sum(1 for item in items if (SRP_DIR / f"{item.doi_safe}{SRP_SUFFIX}").exists())
    print(f"Existing SRPs among selected rows: {existing}", flush=True)
    print(f"Rows to process if no --force: {len(items) - existing}", flush=True)

    gpu_info = get_gpu_info()
    print(f"GPU info: {gpu_info}", flush=True)
    print("Connecting to vLLM server...", flush=True)

    client = AsyncOpenAI(
        base_url=f"{VLLM_URL}/v1",
        api_key="not-needed",
        timeout=args.timeout,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    progress = Progress(total=len(items))

    total_start = time.time()
    tasks = [
        process_item(
            item=item,
            system_prompt=system_prompt,
            client=client,
            semaphore=semaphore,
            force=args.force,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_images=args.max_images,
            progress=progress,
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
    print("INFERENCE COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Rows selected:          {summary['total_rows_selected']}", flush=True)
    print(f"Skipped:                {summary['skipped_count']}", flush=True)
    print(f"Processed:              {summary['processed_count']}", flush=True)
    print(f"Successful:             {summary['success_count']}", flush=True)
    print(f"Failed:                 {summary['fail_count']}", flush=True)
    print(f"Valid images sent:      {summary['total_valid_images_sent']}", flush=True)
    print(f"Missing images:         {summary['total_missing_images']}", flush=True)
    print(f"Image load errors:      {summary['total_image_load_errors']}", flush=True)
    print(f"Total wall time:        {summary['total_wall_time_sec']} sec", flush=True)
    print(f"Results CSV:            {RESULTS_CSV}", flush=True)
    print(f"Summary TXT:            {SUMMARY_TXT}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    if sys.platform == "linux":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(main_async())