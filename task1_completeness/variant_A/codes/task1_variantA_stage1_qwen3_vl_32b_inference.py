#!/usr/bin/env python3
"""
Task 1 / Variant A / Stage 1 SRP Inference
Model: Qwen3-VL-32B-Instruct-AWQ, used in text-only mode for Variant A

Reads pre-built user prompt files, reads the existing system prompt, sends them
through a local vLLM OpenAI-compatible server, parses/validates SRP JSON, and
saves one SRP per DOI-safe prompt id.

INPUTS are read from the ORIGINAL (non-AWQ) tree; OUTPUTS are written to the new
"_awq" tree, so the original running jobs are never touched.

Default rerun behavior:
  - Resume-safe: existing SRPs are skipped.
  - Previous failures are retried if no final SRP exists.
  - Reports are regenerated each run.
  - Existing SRPs are overwritten only with --force.

The --run-tag flag suffixes ONLY the reports + checkpoint (so two halves running
in parallel do not clobber each other). SRPs/raw_outputs/errors are per-DOI and
safely shared between halves.

Example:
  python3 task1_variantA_stage1_qwen3_vl_32b_inference.py --limit 5
  python3 task1_variantA_stage1_qwen3_vl_32b_inference.py --start-index 0 --end-index 500 --run-tag h1
  python3 task1_variantA_stage1_qwen3_vl_32b_inference.py --limit 5 --force
"""

import argparse
import asyncio
import csv
import json
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

# Inputs come from the ORIGINAL tree (shared, read-only).
INPUT_BASE = Path("./task1_completeness/variant_A")
SYSTEM_PROMPT_FILE = INPUT_BASE / "system_prompts" / "task1_variantA_phase1_system_prompt.txt"
USER_PROMPT_DIR = INPUT_BASE / "user_prompts" / "phase1_shared"

# Outputs go to the NEW "_awq" tree.
OUTPUT_BASE = Path("./task1_completeness_awq/variant_A")
OUTPUT_ROOT = OUTPUT_BASE / "outputs" / "stage1" / MODEL_NAME
SRP_DIR = OUTPUT_ROOT / "srps"
RAW_OUTPUT_DIR = OUTPUT_ROOT / "raw_outputs"
ERROR_DIR = OUTPUT_ROOT / "errors"
LOG_DIR = OUTPUT_ROOT / "logs"
REPORT_DIR = OUTPUT_ROOT / "reports"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

USER_PROMPT_SUFFIX = "_task1_variantA_phase1_user_prompt.txt"
SRP_SUFFIX = "_srp_qwen3_vl_32b.json"
RAW_OUTPUT_SUFFIX = "_raw_output_qwen3_vl_32b.txt"
CLEANED_OUTPUT_SUFFIX = "_cleaned_output_qwen3_vl_32b.txt"
INVALID_SRP_SUFFIX = "_invalid_srp_qwen3_vl_32b.json"

# These are reassigned in main() when --run-tag is provided.
RESULTS_CSV = REPORT_DIR / "task1_variantA_stage1_qwen3_vl_32b_results.csv"
SUMMARY_JSON = REPORT_DIR / "task1_variantA_stage1_qwen3_vl_32b_summary.json"
SUMMARY_TXT = REPORT_DIR / "task1_variantA_stage1_qwen3_vl_32b_summary.txt"
CHECKPOINT_JSONL = CHECKPOINT_DIR / "processed_prompts.jsonl"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 6144
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT = 300.0

# Repetition penalty (vLLM sampling extension), applied uniformly across all
# models for comparability and to break degenerate repeat loops at temperature=0.
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
    "model", "served_model_name", "doi", "prompt_file", "srp_file",
    "status", "error_message",
    "input_tokens", "output_tokens", "inference_time", "tokens_per_second",
    "raw_output_length", "had_think_tags", "had_code_fences",
    "entity_count", "entities_introduction", "entities_methods", "entities_results", "entities_discussion",
    "total_visual_proxies", "avg_visual_proxies_per_entity", "entities_with_no_proxies",
    "summary_words_introduction", "summary_words_methods", "summary_words_results", "summary_words_discussion",
    "summary_balance_std", "causal_relations_present", "causal_relation_avg_words",
    "all_summaries_nonempty", "all_relations_nonempty", "all_entities_have_proxies", "stage2_ready",
    "parse_mode", "visual_proxies_coerced",
]


@dataclass(frozen=True)
class PromptItem:
    doi: str
    prompt_path: Path
    srp_path: Path


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
    RESULTS_CSV = REPORT_DIR / f"task1_variantA_stage1_qwen3_vl_32b_results{tag}.csv"
    SUMMARY_JSON = REPORT_DIR / f"task1_variantA_stage1_qwen3_vl_32b_summary{tag}.json"
    SUMMARY_TXT = REPORT_DIR / f"task1_variantA_stage1_qwen3_vl_32b_summary{tag}.txt"
    CHECKPOINT_JSONL = CHECKPOINT_DIR / f"processed_prompts{tag}.jsonl"


# =============================================================================
# Files and setup
# =============================================================================

def ensure_directories() -> None:
    for d in [SRP_DIR, RAW_OUTPUT_DIR, ERROR_DIR, LOG_DIR, REPORT_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt file not found: {SYSTEM_PROMPT_FILE}")
    prompt = load_text(SYSTEM_PROMPT_FILE)
    if not prompt:
        raise ValueError(f"System prompt file is empty: {SYSTEM_PROMPT_FILE}")
    return prompt


def doi_from_prompt_filename(path: Path) -> str:
    name = path.name
    if not name.endswith(USER_PROMPT_SUFFIX):
        raise ValueError(f"Unexpected user prompt filename: {name}")
    return name[: -len(USER_PROMPT_SUFFIX)]


def discover_prompt_items(start_index: Optional[int] = None, end_index: Optional[int] = None, limit: Optional[int] = None) -> List[PromptItem]:
    if not USER_PROMPT_DIR.exists():
        raise FileNotFoundError(f"User prompt directory not found: {USER_PROMPT_DIR}")

    prompt_paths = sorted(USER_PROMPT_DIR.glob(f"*{USER_PROMPT_SUFFIX}"))
    if not prompt_paths:
        raise FileNotFoundError(f"No user prompts found in {USER_PROMPT_DIR} matching *{USER_PROMPT_SUFFIX}")

    if start_index is not None or end_index is not None:
        start = 0 if start_index is None else start_index
        end = len(prompt_paths) if end_index is None else end_index
        if start < 0 or end < 0 or end < start:
            raise ValueError(f"Invalid start/end index: start={start}, end={end}")
        prompt_paths = prompt_paths[start:end]

    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        prompt_paths = prompt_paths[:limit]

    items: List[PromptItem] = []
    for path in prompt_paths:
        doi = doi_from_prompt_filename(path)
        srp_path = SRP_DIR / f"{doi}{SRP_SUFFIX}"
        items.append(PromptItem(doi=doi, prompt_path=path, srp_path=srp_path))
    return items


# =============================================================================
# Output cleaning and validation
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
    """Return the most likely JSON object string from model output."""
    cleaned = strip_code_fences(strip_think_tags(text))
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        return cleaned[first:last + 1].strip()
    return cleaned


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


def coerce_srp_schema(srp):
    """Fill only the optional field the models commonly omit (visual_proxies),
    so a structurally-complete SRP is not failed over a single absent list.
    Does NOT add entities, summaries, or relations. Returns (srp, n_filled)."""
    n = 0
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
    return srp, n


def validate_srp(srp: Any) -> Tuple[bool, str]:
    if not isinstance(srp, dict):
        return False, "SRP root is not a JSON object"

    # Coerce a missing/malformed causal_relations into a valid empty object so
    # papers with no extractable causal links validate instead of hard-failing.
    if not isinstance(srp.get("causal_relations"), dict):
        srp["causal_relations"] = {k: "" for k in RELATION_KEYS}
    else:
        for k in RELATION_KEYS:
            if not isinstance(srp["causal_relations"].get(k), str):
                srp["causal_relations"][k] = ""

    required_top = ["doi", "title", "section_summaries", "key_entities"]
    for key in required_top:
        if key not in srp:
            return False, f"Missing top-level key: {key}"

    if not isinstance(srp["section_summaries"], dict):
        return False, "section_summaries is not an object"
    for sec in SECTIONS:
        if sec not in srp["section_summaries"]:
            return False, f"section_summaries missing section: {sec}"
        if not isinstance(srp["section_summaries"][sec], str):
            return False, f"section_summaries[{sec}] is not a string"

    if not isinstance(srp["key_entities"], dict):
        return False, "key_entities is not an object"
    for sec in SECTIONS:
        if sec not in srp["key_entities"]:
            return False, f"key_entities missing section: {sec}"
        if not isinstance(srp["key_entities"][sec], list):
            return False, f"key_entities[{sec}] is not a list"
        for i, entity in enumerate(srp["key_entities"][sec]):
            if not isinstance(entity, dict):
                return False, f"key_entities[{sec}][{i}] is not an object"
            for k in ["entity", "type", "visual_proxies"]:
                if k not in entity:
                    return False, f"key_entities[{sec}][{i}] missing key: {k}"
            if not isinstance(entity["entity"], str):
                return False, f"key_entities[{sec}][{i}].entity is not a string"
            if not isinstance(entity["type"], str):
                return False, f"key_entities[{sec}][{i}].type is not a string"
            if not isinstance(entity["visual_proxies"], list):
                return False, f"key_entities[{sec}][{i}].visual_proxies is not a list"
            for j, proxy in enumerate(entity["visual_proxies"]):
                if not isinstance(proxy, str):
                    entity["visual_proxies"][j] = str(proxy)

    if not isinstance(srp["causal_relations"], dict):
        return False, "causal_relations is not an object"
    for key in RELATION_KEYS:
        if key not in srp["causal_relations"]:
            return False, f"causal_relations missing key: {key}"
        if not isinstance(srp["causal_relations"][key], str):
            return False, f"causal_relations[{key}] is not a string"

    return True, ""


# =============================================================================
# Metrics
# =============================================================================

def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) and text.strip() else 0


def compute_paper_metrics(srp: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    total_entities = 0
    total_visual_proxies = 0
    entities_with_no_proxies = 0

    for sec in SECTIONS:
        entities = srp.get("key_entities", {}).get(sec, [])
        metrics[f"entities_{sec}"] = len(entities)
        total_entities += len(entities)
        for ent in entities:
            proxies = ent.get("visual_proxies", []) if isinstance(ent, dict) else []
            total_visual_proxies += len(proxies)
            if len(proxies) == 0:
                entities_with_no_proxies += 1

    metrics["entity_count"] = total_entities
    metrics["total_visual_proxies"] = total_visual_proxies
    metrics["avg_visual_proxies_per_entity"] = round(total_visual_proxies / total_entities, 2) if total_entities else 0.0
    metrics["entities_with_no_proxies"] = entities_with_no_proxies

    summary_word_counts = []
    for sec in SECTIONS:
        wc = word_count(srp.get("section_summaries", {}).get(sec, ""))
        metrics[f"summary_words_{sec}"] = wc
        summary_word_counts.append(wc)
    metrics["summary_balance_std"] = round(statistics.pstdev(summary_word_counts), 2) if summary_word_counts else 0.0

    relations = srp.get("causal_relations", {})
    nonempty = []
    for key in RELATION_KEYS:
        val = relations.get(key, "")
        if isinstance(val, str) and val.strip():
            nonempty.append(val.strip())
    metrics["causal_relations_present"] = len(nonempty)
    metrics["causal_relation_avg_words"] = round(sum(word_count(x) for x in nonempty) / len(nonempty), 2) if nonempty else 0.0

    metrics["all_summaries_nonempty"] = all(
        isinstance(srp.get("section_summaries", {}).get(sec, ""), str)
        and srp.get("section_summaries", {}).get(sec, "").strip()
        for sec in SECTIONS
    )
    metrics["all_relations_nonempty"] = len(nonempty) == len(RELATION_KEYS)
    metrics["all_entities_have_proxies"] = total_entities > 0 and entities_with_no_proxies == 0
    metrics["stage2_ready"] = (
        metrics["all_summaries_nonempty"]
        and metrics["all_relations_nonempty"]
        and metrics["all_entities_have_proxies"]
    )
    return metrics


def empty_result(item: PromptItem) -> Dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "doi": item.doi,
        "prompt_file": str(item.prompt_path),
        "srp_file": str(item.srp_path),
        "status": "",
        "error_message": "",
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
    }


# =============================================================================
# Save helpers
# =============================================================================

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    # errors="replace" guards against lone surrogates in model output that
    # otherwise raise UnicodeEncodeError and abort the whole run.
    with path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def save_checkpoint(row: Dict[str, Any]) -> None:
    CHECKPOINT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), **row}
    with CHECKPOINT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_gpu_info() -> Dict[str, Any]:
    info = {
        "gpu_model_name": "unknown",
        "gpu_count": 0,
        "gpu_vram_total_mb": 0,
        "gpu_vram_used_mb": 0,
    }
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return info
        lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        if not lines:
            return info
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
        info["gpu_count"] = len(names)
        info["gpu_vram_total_mb"] = total
        info["gpu_vram_used_mb"] = used
    except Exception:
        pass
    return info


# =============================================================================
# Inference
# =============================================================================

async def process_one(
    item: PromptItem,
    system_prompt: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    temperature: float,
    max_tokens: int,
    force: bool,
) -> Dict[str, Any]:
    result = empty_result(item)

    if item.srp_path.exists() and not force:
        result["status"] = "skipped"
        print(f"[SKIP] {item.doi} | SRP exists", flush=True)
        save_checkpoint(result)
        return result

    try:
        user_prompt = load_text(item.prompt_path)
        if not user_prompt:
            raise ValueError("User prompt file is empty")
    except Exception as e:
        result["status"] = "prompt_read_error"
        result["error_message"] = str(e)
        print(f"[FAIL] {item.doi} | prompt_read_error | {e}", flush=True)
        save_checkpoint(result)
        return result

    async with semaphore:
        start = time.time()
        try:
            response = await create_with_context_retry(
                client,
                model=SERVED_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"repetition_penalty": REPETITION_PENALTY},
            )
        except Exception as e:
            result["status"] = "api_error"
            result["error_message"] = str(e)
            print(f"[FAIL] {item.doi} | api_error | {e}", flush=True)
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
        save_text(ERROR_DIR / f"{item.doi}{RAW_OUTPUT_SUFFIX}", raw_content)
        print(f"[FAIL] {item.doi} | empty_response_error", flush=True)
        save_checkpoint(result)
        return result

    cleaned = extract_json_candidate(raw_content)

    srp, parse_mode = parse_srp_json(cleaned)
    result["parse_mode"] = parse_mode
    if srp is None:
        result["status"] = "json_parse_error"
        result["error_message"] = "JSON parse failed after repair"
        save_text(ERROR_DIR / f"{item.doi}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_text(ERROR_DIR / f"{item.doi}{CLEANED_OUTPUT_SUFFIX}", cleaned)
        print(f"[FAIL] {item.doi} | json_parse_error", flush=True)
        save_checkpoint(result)
        return result

    srp, _n_coerced = coerce_srp_schema(srp)
    result["visual_proxies_coerced"] = _n_coerced

    valid, validation_error = validate_srp(srp)
    if not valid:
        result["status"] = "validation_error"
        result["error_message"] = validation_error
        save_text(ERROR_DIR / f"{item.doi}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_json(ERROR_DIR / f"{item.doi}{INVALID_SRP_SUFFIX}", srp)
        print(f"[FAIL] {item.doi} | validation_error | {validation_error}", flush=True)
        save_checkpoint(result)
        return result

    try:
        save_json(item.srp_path, srp)
        save_text(RAW_OUTPUT_DIR / f"{item.doi}{RAW_OUTPUT_SUFFIX}", raw_content)
    except Exception as e:
        result["status"] = "save_error"
        result["error_message"] = str(e)
        print(f"[FAIL] {item.doi} | save_error | {e}", flush=True)
        save_checkpoint(result)
        return result

    result.update(compute_paper_metrics(srp))
    result["status"] = "success"
    result["srp_file"] = str(item.srp_path)

    print(
        f"[OK] {item.doi} | entities={result['entity_count']} | "
        f"tokens={result['input_tokens']}+{result['output_tokens']} | {result['inference_time']}s",
        flush=True,
    )
    save_checkpoint(result)
    return result


# =============================================================================
# Reports
# =============================================================================

def pct(n: int, d: int) -> float:
    return round(n / d * 100, 2) if d else 0.0


def mean(values: List[float]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def median(values: List[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


def compute_summary(results: List[Dict[str, Any]], gpu_info: Dict[str, Any], total_wall_time: float, args: argparse.Namespace) -> Dict[str, Any]:
    total = len(results)
    skipped = [r for r in results if r["status"] == "skipped"]
    processed = [r for r in results if r["status"] != "skipped"]
    successes = [r for r in processed if r["status"] == "success"]
    failures = [r for r in processed if r["status"] != "success"]

    error_breakdown: Dict[str, int] = {}
    for r in failures:
        error_breakdown[r["status"]] = error_breakdown.get(r["status"], 0) + 1

    section_entity_avg = {
        sec: mean([r[f"entities_{sec}"] for r in successes])
        for sec in SECTIONS
    }
    section_summary_word_avg = {
        sec: mean([r[f"summary_words_{sec}"] for r in successes])
        for sec in SECTIONS
    }

    stage2_ready_count = sum(1 for r in successes if r.get("stage2_ready"))

    return {
        "model": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
        "system_prompt_file": str(SYSTEM_PROMPT_FILE),
        "user_prompt_dir": str(USER_PROMPT_DIR),
        "output_root": str(OUTPUT_ROOT),
        "args": vars(args),
        "gpu_info": gpu_info,
        "total_selected_prompts": total,
        "skipped_existing_srps": len(skipped),
        "processed_count": len(processed),
        "success_count": len(successes),
        "fail_count": len(failures),
        "success_rate_processed_pct": pct(len(successes), len(processed)),
        "success_rate_selected_pct": pct(len(successes), total),
        "error_breakdown": error_breakdown,
        "avg_inference_time_sec": mean([r["inference_time"] for r in successes]),
        "median_inference_time_sec": median([r["inference_time"] for r in successes]),
        "avg_tokens_per_second": mean([r["tokens_per_second"] for r in successes if r["tokens_per_second"] > 0]),
        "total_input_tokens": sum(int(r["input_tokens"]) for r in successes),
        "total_output_tokens": sum(int(r["output_tokens"]) for r in successes),
        "avg_input_tokens": mean([r["input_tokens"] for r in successes]),
        "avg_output_tokens": mean([r["output_tokens"] for r in successes]),
        "avg_entity_count": mean([r["entity_count"] for r in successes]),
        "avg_entities_by_section": section_entity_avg,
        "avg_visual_proxies_per_entity": mean([r["avg_visual_proxies_per_entity"] for r in successes]),
        "total_entities_with_no_proxies": sum(int(r["entities_with_no_proxies"]) for r in successes),
        "avg_summary_words_by_section": section_summary_word_avg,
        "avg_causal_relations_present": mean([r["causal_relations_present"] for r in successes]),
        "stage2_ready_count": stage2_ready_count,
        "stage2_ready_rate_pct": pct(stage2_ready_count, len(successes)),
        "total_wall_time_sec": round(total_wall_time, 2),
        "total_wall_time_min": round(total_wall_time / 60, 2),
    }


def write_reports(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    save_json(SUMMARY_JSON, summary)

    lines = [
        "Task 1 Variant A Stage 1 SRP Inference Summary",
        "================================================",
        f"Generated: {summary['timestamp']}",
        f"Model: {summary['model']}",
        f"Served model name: {summary['served_model_name']}",
        f"Model path: {summary['model_path']}",
        f"System prompt: {summary['system_prompt_file']}",
        f"User prompts: {summary['user_prompt_dir']}",
        f"Output root: {summary['output_root']}",
        "",
        "Run counts",
        "----------",
        f"Total selected prompts: {summary['total_selected_prompts']}",
        f"Skipped existing SRPs: {summary['skipped_existing_srps']}",
        f"Processed: {summary['processed_count']}",
        f"Success: {summary['success_count']}",
        f"Failed: {summary['fail_count']}",
        f"Success rate among processed: {summary['success_rate_processed_pct']}%",
        f"Success rate among selected: {summary['success_rate_selected_pct']}%",
        f"Error breakdown: {summary['error_breakdown']}",
        "",
        "Timing and tokens",
        "-----------------",
        f"Average inference time: {summary['avg_inference_time_sec']} sec",
        f"Median inference time: {summary['median_inference_time_sec']} sec",
        f"Average tokens/sec: {summary['avg_tokens_per_second']}",
        f"Total input tokens: {summary['total_input_tokens']}",
        f"Total output tokens: {summary['total_output_tokens']}",
        f"Average input tokens: {summary['avg_input_tokens']}",
        f"Average output tokens: {summary['avg_output_tokens']}",
        "",
        "SRP content metrics",
        "-------------------",
        f"Average entity count: {summary['avg_entity_count']}",
        f"Average entities by section: {summary['avg_entities_by_section']}",
        f"Average visual proxies/entity: {summary['avg_visual_proxies_per_entity']}",
        f"Average summary words by section: {summary['avg_summary_words_by_section']}",
        f"Average causal relations present: {summary['avg_causal_relations_present']}/3",
        f"Stage 2 ready: {summary['stage2_ready_count']} ({summary['stage2_ready_rate_pct']}%)",
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
    parser = argparse.ArgumentParser(description="Task 1 Variant A Stage 1 SRP inference with Qwen3-VL-32B-Instruct-AWQ")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N selected prompt files after sorting/slicing")
    parser.add_argument("--start-index", type=int, default=None, help="Start index in sorted prompt list, inclusive")
    parser.add_argument("--end-index", type=int, default=None, help="End index in sorted prompt list, exclusive")
    parser.add_argument("--run-tag", type=str, default=None, help="Suffix for reports + checkpoint only (e.g. h1, h2). SRPs are shared.")
    parser.add_argument("--force", action="store_true", help="Rerun selected prompts even if SRP output already exists")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent API requests")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum output tokens per SRP")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="OpenAI client timeout in seconds")
    parser.add_argument("--vllm-url", type=str, default=VLLM_URL, help="Base vLLM server URL, without /v1")
    args = parser.parse_args()

    apply_run_tag(args.run_tag)
    ensure_directories()
    if repair_json is None:
        print("ERROR: json-repair not installed in this env. Run: pip install json-repair --break-system-packages", flush=True)
        sys.exit(1)

    print("=" * 80, flush=True)
    print("Task 1 Variant A Stage 1 SRP Inference — Qwen3-VL-32B-Instruct-AWQ", flush=True)
    print("=" * 80, flush=True)
    print(f"System prompt: {SYSTEM_PROMPT_FILE}", flush=True)
    print(f"User prompt dir: {USER_PROMPT_DIR}", flush=True)
    print(f"Output root: {OUTPUT_ROOT}", flush=True)
    print(f"Served model: {SERVED_MODEL_NAME}", flush=True)
    print(f"vLLM URL: {args.vllm_url}", flush=True)
    print(f"Run tag: {args.run_tag}", flush=True)
    print(f"Start index: {args.start_index} | End index: {args.end_index}", flush=True)
    print(f"Concurrency: {args.concurrency}", flush=True)
    print(f"Max tokens: {args.max_tokens}", flush=True)
    print(f"Temperature: {args.temperature}", flush=True)
    print(f"Force rerun: {args.force}", flush=True)
    print(f"Results CSV: {RESULTS_CSV}", flush=True)
    print(f"Checkpoint: {CHECKPOINT_JSONL}", flush=True)
    print("=" * 80, flush=True)

    system_prompt = load_system_prompt()
    items = discover_prompt_items(start_index=args.start_index, end_index=args.end_index, limit=args.limit)
    already_done = sum(1 for item in items if item.srp_path.exists())
    to_process = len(items) if args.force else len(items) - already_done

    print(f"Selected prompt files: {len(items)}", flush=True)
    print(f"Existing SRPs in selection: {already_done}", flush=True)
    print(f"Will process now: {to_process}", flush=True)

    gpu_info = get_gpu_info()
    print(f"GPU info: {gpu_info}", flush=True)

    client = AsyncOpenAI(
        base_url=f"{args.vllm_url.rstrip('/')}/v1",
        api_key="not-needed",
        timeout=args.timeout,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    total_start = time.time()
    tasks = [
        process_one(
            item=item,
            system_prompt=system_prompt,
            client=client,
            semaphore=semaphore,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            force=args.force,
        )
        for item in items
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for _item, _r in zip(items, raw_results):
        if isinstance(_r, dict):
            results.append(_r)
        else:
            _row = empty_result(_item)
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
    print(f"Selected prompts: {summary['total_selected_prompts']}", flush=True)
    print(f"Skipped: {summary['skipped_existing_srps']}", flush=True)
    print(f"Processed: {summary['processed_count']}", flush=True)
    print(f"Success: {summary['success_count']}", flush=True)
    print(f"Failed: {summary['fail_count']}", flush=True)
    print(f"Error breakdown: {summary['error_breakdown']}", flush=True)
    print(f"Stage 2 ready: {summary['stage2_ready_count']} ({summary['stage2_ready_rate_pct']}%)", flush=True)
    print(f"Results CSV: {RESULTS_CSV}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON}", flush=True)
    print(f"Summary TXT: {SUMMARY_TXT}", flush=True)
    print(f"Wall time: {summary['total_wall_time_min']} min", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(main_async())