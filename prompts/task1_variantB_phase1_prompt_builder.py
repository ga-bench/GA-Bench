#!/usr/bin/env python3
"""
Task 1 Completeness - Variant B - Phase 1 Prompt Builder
========================================================

Builds Variant B Phase 1 prompts for all papers in dataset_10k.

Key behavior:
- Keeps the same IMRaD grouping, heading recovery, fallback borrowing, logging,
  and rerun behavior as the Variant A Phase 1 prompt builder.
- Uses fulltext_imrad.json as the source for IMRaD section text.
- Uses metadata JSON only as fallback for DOI/title/abstract/journal/publisher.
- Adds paper figure evidence from figures.json.
- Adds paper table evidence from tables.json.
- Saves user prompts to:
  ./task1_completeness/variant_B/user_prompts/phase1_shared
- Saves a minimal figure-path CSV to:
  ./task1_completeness/variant_B/user_prompts/task1_variantB_phase1_figure_paths.csv
- Saves builder logs/skipped/summary to:
  ./prompt_builders/output
- On rerun, deletes only this script's own output files, previous Variant B Phase 1 user prompts,
  and the Variant B figure-path CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path("./")
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset_10k"

PROMPT_BUILDER_DIR = PROJECT_ROOT / "prompt_builders"
PROMPT_BUILDER_OUTPUT_DIR = PROMPT_BUILDER_DIR / "output"

VARIANT_B_ROOT = PROJECT_ROOT / "task1_completeness" / "variant_B"

SYSTEM_PROMPT_DIR = VARIANT_B_ROOT / "system_prompts"
PHASE1_SYSTEM_PROMPT_PATH = SYSTEM_PROMPT_DIR / "task1_variantB_phase1_system_prompt.txt"

VARIANT_B_USER_PROMPT_ROOT = VARIANT_B_ROOT / "user_prompts"
PHASE1_SHARED_USER_PROMPT_DIR = VARIANT_B_USER_PROMPT_ROOT / "phase1_shared"
PHASE1_FIGURE_PATHS_CSV = VARIANT_B_USER_PROMPT_ROOT / "task1_variantB_phase1_figure_paths.csv"

# Kept in prompt_builders/output like Variant A, but renamed for Variant B.
PHASE1_SKIPPED_CSV = PROMPT_BUILDER_OUTPUT_DIR / "task1_variantB_phase1_skipped_papers.csv"
PHASE1_LOG_PATH = PROMPT_BUILDER_OUTPUT_DIR / "task1_variantB_phase1_prompt_builder_log.txt"
PHASE1_SUMMARY_JSON = PROMPT_BUILDER_OUTPUT_DIR / "task1_variantB_phase1_prompt_builder_summary.json"

USER_PROMPT_SUFFIX = "_task1_variantB_phase1_user_prompt.txt"


# =============================================================================
# PROMPTS
# =============================================================================

PHASE1_SYSTEM_PROMPT = """You are a scientific paper analysis assistant. Your task is to construct a Structured Reference Profile (SRP) from a scientific paper.

An SRP is a standardized representation of a paper's conceptual structure, designed to capture what a Graphical Abstract (GA) should visually communicate. The SRP you produce will be used as a reference standard to evaluate whether a GA adequately represents the full scientific narrative of the paper. Because of this, the SRP must be comprehensive, i.e., it should capture all important concepts, entities, and relationships from the paper, not just the most visually obvious ones. A complete SRP enables accurate detection of what a GA might be missing.

You will receive the DOI, title, abstract, four IMRaD sections of the paper, and additional paper evidence from figures and tables. The figure/table evidence may include figure captions, table captions, table contents, and, when available, attached paper figure images. Use the figures and tables as supporting evidence during SRP construction, especially for Methods and Results entities such as workflows, architectures, experimental setups, datasets, metrics, comparisons, and quantitative outcomes.

From these inputs, extract the following:

1. SECTION SUMMARIES
For each IMRaD section, write a concise 3–5 sentence summary describing the main contribution of that section. Focus on what would be visually representable in a GA. If the paper combines Results and Discussion into a single section, the combined text may appear under one of the two fields while the other may be short or partially overlapping. Summarize based on what is actually present.

2. KEY ENTITIES WITH VISUAL PROXIES
Extract concrete entities from each section that could plausibly appear in a GA. Entities must be specific, named things from the paper text, figure/table captions, table contents, or clearly visible attached figure evidence, not vague concepts. Extract whatever entities are relevant to the paper you are analyzing.

Extract at least 2 concrete entities specifically from EACH of the four IMRaD sections whenever the section contains enough supported entities. If there could be more than 2 valid entities for each section, extract them. You must group these entities into a dictionary using the section names as keys: introduction, methods, results, discussion.

If a section genuinely contains fewer than 2 extractable entities, extract only what is present. Do not pad the list with vague or fabricated entries.

If an entity appears in multiple sections, extract it only from the section where it plays the most central role. Do not repeat the same entity across sections.

Examples of good entities: "convolutional neural network", "MCF-7 breast cancer cells", "CIFAR-10 dataset", "gold nanoparticles", "72.3% classification accuracy", "Western blot analysis", "random forest classifier", "HeLa cells", "TEM imaging", "pH 7.4 buffer solution"

Examples of bad entities: "the method", "important results", "novel approach", "significant improvement", "the model", "our technique", "key findings"

For each entity, provide:
- entity: The entity name as written in the paper. Use the most complete and specific form of the name, e.g., "convolutional neural network" rather than "CNN". If an acronym is standard in the field, include it in parentheses, e.g., "convolutional neural network (CNN)".
- type: Classify the entity with a descriptive type in snake_case format. Common examples include: model, algorithm, dataset, metric, compound, material, organism, cell_line, technique, instrument, condition, process, structure, disease, gene_protein, software, framework, evaluation_method, imaging_modality, statistical_test, chemical_reaction, nanostructure, drug, receptor, pathway. These are not exhaustive. If none of them fit the entity, create a descriptive type in the same snake_case format, e.g., sensor_device, clinical_outcome, tissue_type. Prefer reusing an existing type when it fits rather than creating a new one.
- visual_proxies: A list of at least 3 distinct visual elements that could each independently represent this entity in a graphical abstract. Each proxy should be something a graphic designer could draw. Provide diverse representations, not slight rewordings of the same idea. Aim for at least 3 when possible, but if fewer than 3 genuinely distinct options exist for an entity, provide only what is meaningfully different.

3. CAUSAL RELATIONS
Extract three core scientific relations that form the paper's narrative arc:
- intro_to_methods: How the problem or gap in the Introduction motivates the method. Write one natural-language sentence.
- methods_to_results: What outcome the method produced. Write one natural-language sentence.
- results_to_discussion: How the results support or challenge the broader interpretation. Write one natural-language sentence.

VERIFICATION PASS
Before producing the final SRP, internally verify every extracted entity and relation against the provided paper text, figure/table captions, table contents, or attached figure evidence.

Keep an entity only if:
- it is explicitly supported by the provided paper text, figure/table caption, table content, or clearly visible attached figure evidence,
- it is specific and concrete,
- it could plausibly be represented visually,
- it is not a vague generic phrase,
- it is assigned to the IMRaD section where it plays the most central role.

Remove unsupported, duplicated, vague, or weakly grounded entities. However, do not over-filter. Each section should retain at least 2 valid concrete entities whenever the provided evidence contains enough supported entities. If fewer than 2 valid entities are available after verification, keep only the valid entities and do not fabricate replacements.

For causal relations, always produce all three relation fields:
- intro_to_methods
- methods_to_results
- results_to_discussion

Keep a relation only if it is grounded in the provided evidence and reflects the actual scientific narrative of the paper. If the relation is directly supported, write it precisely. If the relation is only indirectly supported, write the best grounded relation using only the provided evidence. Do not fabricate details, but do not leave relation fields blank unless the corresponding IMRaD content is genuinely absent.

IMPORTANT RULES
- Be precise and grounded in the provided paper evidence.
- Do not infer or fabricate entities that are not supported.
- Use figures and tables only as supporting evidence for the source paper
- Every entity must be supported by the paper text, figure/table captions, table contents, or attached figure evidence.
- Visual proxies should be concrete visual elements, e.g., "layered network architecture diagram", "bar chart comparing accuracy values", "mouse silhouette icon", "molecular structure illustration with labeled atoms".
- Provide diverse visual proxies for each entity, not slight rewordings of the same idea.
- Causal relations must reflect the actual narrative of the paper, not generic templates.
- Copy the DOI and title exactly as provided. Do not modify, reformat, or add prefixes to the DOI.
- Replace all placeholder values with actual content from the paper. Do not copy any template text into your output.

EXPECTED JSON STRUCTURE
{
  "doi": "...",
  "title": "...",
  "section_summaries": {
    "introduction": "...",
    "methods": "...",
    "results": "...",
    "discussion": "..."
  },
  "key_entities": {
    "introduction": [
      {
        "entity": "...",
        "type": "...",
        "visual_proxies": ["...", "...", "..."]
      }
    ],
    "methods": [
      {
        "entity": "...",
        "type": "...",
        "visual_proxies": ["...", "...", "..."]
      }
    ],
    "results": [
      {
        "entity": "...",
        "type": "...",
        "visual_proxies": ["...", "...", "..."]
      }
    ],
    "discussion": [
      {
        "entity": "...",
        "type": "...",
        "visual_proxies": ["...", "...", "..."]
      }
    ]
  },
  "causal_relations": {
    "intro_to_methods": "...",
    "methods_to_results": "...",
    "results_to_discussion": "..."
  }
}

OUTPUT FORMAT
Respond with ONLY a valid JSON object. No explanation, no markdown, no code fences, no text before or after the JSON."""

USER_PROMPT_TEMPLATE = """Paper DOI:
{doi}

Paper Title:
{title}

Abstract:
{abstract}

Introduction:
{introduction_text}

Methods:
{methods_text}

Results:
{results_text}

Discussion/Conclusion:
{discussion_text}

Paper Figures:
{figures_block}

Paper Tables:
{tables_block}

Generate the final Structured Reference Profile (SRP) for this paper using the IMRaD text, figure evidence, and table evidence above.
"""


# =============================================================================
# SETTINGS
# =============================================================================

MAX_FIGURES_PER_PAPER = 12
MAX_TABLES_PER_PAPER = 8
MAX_TABLE_CHARS = 6000
MAX_CAPTION_CHARS = 2500

ALLOWED_FIGURE_QUALITIES = {"good", "page_render"}
ALLOWED_IMRAD_SECTIONS = {"introduction", "methods", "results", "discussion"}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_inline(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_heading(value: Any) -> str:
    h = clean_inline(value).lower()
    h = re.sub(r"[^a-z0-9]+", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def ensure_dirs() -> None:
    SYSTEM_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    VARIANT_B_USER_PROMPT_ROOT.mkdir(parents=True, exist_ok=True)
    PHASE1_SHARED_USER_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_BUILDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_safe(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def safe_filename_from_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    doi = doi.strip()

    safe = re.sub(r"[^A-Za-z0-9]+", "_", doi)
    safe = re.sub(r"_+", "_", safe).strip("_")

    if not safe:
        digest = hashlib.sha1(doi.encode("utf-8")).hexdigest()[:12]
        safe = "missing_doi_" + digest

    return safe


def doi_from_folder_name(folder_name: str) -> str:
    s = normalize_text(folder_name)
    if s.startswith("10_"):
        return s.replace("_", "/", 1)
    return s


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[TRUNCATED]"


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


# =============================================================================
# CLEANUP / LOGGING
# =============================================================================

def remove_file_if_exists(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def cleanup_previous_outputs() -> None:
    ensure_dirs()

    own_output_files = [
        PHASE1_FIGURE_PATHS_CSV,
        PHASE1_SKIPPED_CSV,
        PHASE1_LOG_PATH,
        PHASE1_SUMMARY_JSON,
    ]

    for path in own_output_files:
        remove_file_if_exists(path)

    if PHASE1_SHARED_USER_PROMPT_DIR.exists():
        shutil.rmtree(PHASE1_SHARED_USER_PROMPT_DIR)

    PHASE1_SHARED_USER_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    VARIANT_B_USER_PROMPT_ROOT.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    ensure_dirs()

    logger = logging.getLogger("task1_variantB_phase1_prompt_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(PHASE1_LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def get_doi_folders(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError("Dataset directory does not exist: " + str(dataset_dir))
    return sorted([p for p in dataset_dir.iterdir() if p.is_dir()])


def find_json_in_extracted(paper_folder: Path, exact_name: str, glob_pattern: str) -> Optional[Path]:
    extracted = paper_folder / "extracted"

    if not extracted.exists() or not extracted.is_dir():
        return None

    direct = extracted / exact_name
    if direct.exists() and direct.is_file():
        return direct

    matches = sorted(extracted.glob(glob_pattern))
    return matches[0] if matches else None


def find_fulltext_imrad_json(paper_folder: Path) -> Optional[Path]:
    return find_json_in_extracted(paper_folder, "fulltext_imrad.json", "*fulltext_imrad*.json")


def find_figures_json(paper_folder: Path) -> Optional[Path]:
    return find_json_in_extracted(paper_folder, "figures.json", "*figures*.json")


def find_tables_json(paper_folder: Path) -> Optional[Path]:
    return find_json_in_extracted(paper_folder, "tables.json", "*tables*.json")


def find_figures_dir(paper_folder: Path) -> Path:
    return paper_folder / "extracted" / "figures"


def find_metadata_json(paper_folder: Path) -> Optional[Path]:
    expected = paper_folder / (paper_folder.name + "_Metadata.json")
    if expected.exists() and expected.is_file():
        return expected

    matches = sorted(paper_folder.glob("*_Metadata.json"))
    return matches[0] if matches else None


def find_ga_path(paper_folder: Path) -> str:
    patterns = [
        "*Graphical_Abstract*.jpg",
        "*Graphical_Abstract*.jpeg",
        "*Graphical_Abstract*.png",
        "*graphical_abstract*.jpg",
        "*graphical_abstract*.jpeg",
        "*graphical_abstract*.png",
        "*graphical*abstract*.jpg",
        "*graphical*abstract*.jpeg",
        "*graphical*abstract*.png",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(paper_folder.glob(pattern))

    unique = sorted({str(p) for p in candidates})
    return unique[0] if unique else ""


# =============================================================================
# METADATA FALLBACK
# =============================================================================

def get_nested(data: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        cur = data
        ok = True

        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break

        if ok and cur not in (None, ""):
            return cur

    return ""


def recursive_find_by_key(data: Any, target_keys: List[str]) -> str:
    target_set = {k.lower() for k in target_keys}

    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in target_set:
                text = normalize_text(value)
                if text:
                    return text

        for value in data.values():
            found = recursive_find_by_key(value, target_keys)
            if found:
                return found

    elif isinstance(data, list):
        for item in data:
            found = recursive_find_by_key(item, target_keys)
            if found:
                return found

    return ""


def first_nonempty(*values: Any) -> str:
    for value in values:
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(normalize_text(item.get("text") or item.get("abstract") or item.get("$")))
                else:
                    parts.append(normalize_text(item))
            value = "\n".join([p for p in parts if p])

        if isinstance(value, dict):
            value = (
                value.get("text")
                or value.get("abstract")
                or value.get("$")
                or value.get("value")
                or ""
            )

        text = normalize_text(value)
        if text:
            return text

    return ""


def extract_doi(imrad_data: Dict[str, Any], metadata: Dict[str, Any], paper_folder: Path) -> str:
    imrad_doi = get_nested(imrad_data, [
        "doi",
        "DOI",
        "metadata.doi",
        "metadata.DOI",
    ])

    meta_doi = get_nested(metadata, [
        "doi",
        "DOI",
        "prism:doi",
        "coredata.prism:doi",
        "dc:identifier",
        "coredata.dc:identifier",
    ])

    doi = first_nonempty(imrad_doi, meta_doi)

    if doi.lower().startswith("doi:"):
        doi = doi[4:].strip()

    if not doi:
        doi = doi_from_folder_name(paper_folder.name)

    return doi


def extract_title(imrad_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    imrad_title = get_nested(imrad_data, [
        "title",
        "paper_title",
        "metadata.title",
        "metadata.paper_title",
    ])

    meta_title = get_nested(metadata, [
        "title",
        "dc:title",
        "coredata.dc:title",
        "article_title",
        "publicationTitle",
    ])

    recursive_title = recursive_find_by_key(metadata, [
        "title",
        "dc:title",
        "article_title",
    ])

    return first_nonempty(imrad_title, meta_title, recursive_title)


def extract_abstract(imrad_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    imrad_abstract = get_nested(imrad_data, [
        "abstract",
        "paper_abstract",
        "metadata.abstract",
        "metadata.paper_abstract",
    ])

    meta_abstract = get_nested(metadata, [
        "abstract",
        "description",
        "dc:description",
        "coredata.dc:description",
        "article_abstract",
        "abstracts",
        "openaccessArticle.abstract",
    ])

    recursive_abstract = recursive_find_by_key(metadata, [
        "abstract",
        "description",
        "dc:description",
        "article_abstract",
    ])

    return first_nonempty(imrad_abstract, meta_abstract, recursive_abstract)


def extract_journal(imrad_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    imrad_journal = get_nested(imrad_data, [
        "journal",
        "source",
        "metadata.journal",
        "metadata.source",
    ])

    meta_journal = get_nested(metadata, [
        "journal",
        "publicationName",
        "prism:publicationName",
        "coredata.prism:publicationName",
        "container-title",
        "sourceTitle",
    ])

    return first_nonempty(imrad_journal, meta_journal)


def extract_publisher(imrad_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    imrad_publisher = get_nested(imrad_data, [
        "publisher",
        "metadata.publisher",
    ])

    meta_publisher = get_nested(metadata, [
        "publisher",
        "dc:publisher",
        "coredata.dc:publisher",
    ])

    return first_nonempty(imrad_publisher, meta_publisher)


# =============================================================================
# IMRAD GROUPING WITH HEADING OVERRIDE
# =============================================================================

IMRAD_LABELS = {"introduction", "methods", "results", "discussion"}


def normalize_imrad_label(value: Any) -> str:
    label = clean_inline(value).lower()
    label = re.sub(r"[^a-z]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip()

    if label in IMRAD_LABELS:
        return label

    if label in {"intro", "background", "research background", "problem statement"}:
        return "introduction"

    if label in {
        "method",
        "methods",
        "materials",
        "materials methods",
        "materials and methods",
        "methods and materials",
        "material and methods",
        "methodology",
        "experimental",
        "experiment",
        "experimental section",
        "experimental procedure",
        "experimental procedures",
        "study design",
        "data collection",
        "data analysis",
        "analytical methods",
        "statistical analysis",
        "statistical analyses",
        "statistical methods",
    }:
        return "methods"

    if label in {"result", "results", "findings", "results and discussion", "results discussion"}:
        return "results"

    if label in {
        "discussion",
        "conclusion",
        "conclusions",
        "discussion conclusion",
        "discussion and conclusion",
        "discussion conclusions",
        "summary",
        "summary and conclusions",
        "concluding remarks",
        "limitations",
        "study limitations",
    }:
        return "discussion"

    return ""


def section_to_prompt_block(section: Dict[str, Any]) -> str:
    heading = normalize_text(section.get("heading", ""))
    text = normalize_text(section.get("text", ""))

    if heading and text:
        return "Heading: " + heading + "\nText: " + text
    if text:
        return "Text: " + text
    if heading:
        return "Heading: " + heading
    return ""


def is_combined_results_discussion_heading(heading: Any) -> bool:
    h = normalize_heading(heading)
    return bool(re.search(r"\bresults?\s+and\s+discussion\b|\bresults?\s+discussion\b", h))


def heading_category(heading: Any) -> str:
    h = normalize_heading(heading)

    if not h:
        return ""

    if is_combined_results_discussion_heading(h):
        return "results_discussion"

    if re.search(r"\b(introduction|background|research background|problem statement|overview)\b", h):
        return "introduction"

    if re.search(
        r"\b(materials and methods|methods and materials|material and methods|materials methods|methodology|"
        r"methods|method|experimental section|experimental procedure|experimental procedures|experimental|"
        r"study design|study area|study population|patient selection|patient population|ethical approval|ethical statement|"
        r"data collection|data analysis|statistical analysis|statistical analyses|statistical methods|"
        r"sample preparation|sample collection|chemicals and reagents|chemicals and materials|materials and reagents|"
        r"materials|cell culture|cell lines|cell culture and treatment|animal experiments|experimental animals|"
        r"bacterial strains|molecular docking|bioinformatics analysis)\b",
        h,
    ):
        return "methods"

    if re.search(
        r"\b(results|findings|characterization|characterisation|characterizations|characterisations|"
        r"material characterization|materials characterization|catalyst characterization|catalysts characterization|"
        r"physicochemical characterization|electrochemical characterization|structural characterization|"
        r"morphological studies|thermal analysis|mechanical properties|flow cytometry|western blotting|"
        r"western blot analysis|western blot|immunohistochemistry|immunofluorescence staining|"
        r"immunofluorescence|cell viability assay|cell viability|cellular uptake|in vitro cellular uptake|"
        r"in vitro cytotoxicity|hemolysis assay|wound healing assay|elisa|enzyme linked immunosorbent assay|"
        r"quantitative real time pcr|gene expression analysis|drug release|in vitro drug release|"
        r"pharmacokinetic studies|pharmacokinetic|adsorption kinetics|adsorption isotherms|photocatalytic activity|"
        r"catalytic activity|catalytic tests|product analysis|electrochemical measurements|scanning electron microscopy|"
        r"transmission electron microscopy|dynamic light scattering|comparison to rdcs analysis|performance|evaluation|"
        r"validation|application|case study|model performance|sensitivity analysis)\b",
        h,
    ):
        return "results"

    if re.search(
        r"\b(discussion|conclusion|conclusions|summary|summary and conclusions|concluding remarks|"
        r"study limitations|limitations|implications|future perspectives|perspectives)\b",
        h,
    ):
        return "discussion"

    return ""


def get_section_order(section: Dict[str, Any], fallback: int) -> int:
    try:
        return int(section.get("order", fallback))
    except Exception:
        return fallback


def is_substantial_text(block: str, min_chars: int = 120) -> bool:
    return len(normalize_text(block)) >= min_chars


def group_imrad_sections(imrad_data: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, str]]:
    """
    Build the four IMRaD prompt buckets from parser sections.

    Same logic as Variant A:
    - Do NOT let heading override erase parser IMRaD labels.
    - If heading and parser label disagree, keep the section in both buckets.
    - Add non-fabricating fallback that borrows substantial existing sections by paper order.
    """
    grouped = {
        "introduction": [],
        "methods": [],
        "results": [],
        "discussion": [],
    }

    recovery_notes = {
        "introduction": "",
        "methods": "",
        "results": "",
        "discussion": "",
    }

    sections = imrad_data.get("sections", [])
    if not isinstance(sections, list):
        sections = []

    all_sections = []

    def add_section(label: str, order: int, block: str, note: str = "") -> None:
        if label not in grouped:
            return
        grouped[label].append((order, block))
        if note and not recovery_notes.get(label):
            recovery_notes[label] = note

    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue

        block = section_to_prompt_block(section)
        if not block:
            continue

        order = get_section_order(section, idx)
        heading = normalize_text(section.get("heading", ""))
        hcat = heading_category(heading)

        explicit_labels = []
        for key in ["imrad", "imrad_secondary", "section_type", "label"]:
            label = normalize_imrad_label(section.get(key, ""))
            if label and label not in explicit_labels:
                explicit_labels.append(label)

        all_sections.append({
            "order": order,
            "heading": heading,
            "heading_cat": hcat,
            "explicit_labels": explicit_labels,
            "block": block,
        })

        if hcat == "results_discussion":
            add_section("results", order, block)
            add_section("discussion", order, block)
        elif hcat in IMRAD_LABELS:
            add_section(hcat, order, block)

        for label in explicit_labels:
            add_section(label, order, block)
            if hcat in IMRAD_LABELS and hcat != label and not recovery_notes.get(label):
                recovery_notes[label] = "kept_parser_label_despite_heading_override"

    for label in ["introduction", "methods", "results", "discussion"]:
        if grouped[label]:
            continue

        recovered = []
        for item in all_sections:
            if label == "results" and item["heading_cat"] in {"results", "results_discussion"}:
                recovered.append((item["order"], item["block"]))
            elif label == "discussion" and item["heading_cat"] in {"discussion", "results_discussion"}:
                recovered.append((item["order"], item["block"]))
            elif item["heading_cat"] == label:
                recovered.append((item["order"], item["block"]))

        if recovered:
            grouped[label] = recovered
            recovery_notes[label] = "recovered_from_heading"

    if not grouped["results"]:
        result_like = []
        for item in all_sections:
            hcat = item["heading_cat"]
            h = normalize_heading(item["heading"])
            if hcat == "results" or re.search(
                r"\b(characterization|characterisation|analysis|assay|measurements|performance|evaluation|"
                r"validation|activity|properties|uptake|release|cytotoxicity|western blot|flow cytometry|"
                r"microscopy|spectroscopy|kinetics|isotherms|application|case study|simulation|model results|"
                r"experimental results|numerical results|observations|findings)\b",
                h,
            ):
                result_like.append((item["order"], item["block"]))

        if result_like:
            grouped["results"] = result_like
            recovery_notes["results"] = "borrowed_result_like_sections"

    if not grouped["discussion"]:
        discussion_like = []
        for item in all_sections:
            h = normalize_heading(item["heading"])
            if re.search(
                r"\b(conclusion|conclusions|summary|discussion|limitations|implications|perspectives|"
                r"future work|future directions|outlook|concluding remarks)\b",
                h,
            ):
                discussion_like.append((item["order"], item["block"]))

        if discussion_like:
            grouped["discussion"] = discussion_like
            recovery_notes["discussion"] = "borrowed_discussion_like_sections"

    if not grouped["introduction"]:
        intro_like = []
        for item in all_sections:
            h = normalize_heading(item["heading"])
            if re.search(r"\b(introduction|background|overview|problem statement|motivation)\b", h):
                intro_like.append((item["order"], item["block"]))

        if not intro_like:
            substantial = [item for item in all_sections if is_substantial_text(item["block"])]
            if substantial:
                first_item = sorted(substantial, key=lambda x: x["order"])[0]
                intro_like = [(first_item["order"], first_item["block"])]

        if intro_like:
            grouped["introduction"] = intro_like
            recovery_notes["introduction"] = "borrowed_intro_or_first_substantial_section"

    if not grouped["methods"]:
        method_like = []
        for item in all_sections:
            h = normalize_heading(item["heading"])
            if re.search(
                r"\b(method|methods|methodology|materials|experimental|procedure|procedures|study design|"
                r"data collection|data analysis|statistical|sample|sampling|chemicals|reagents|cell culture|"
                r"animal|patient|docking|simulation setup|model setup|computational|numerical model|"
                r"model description|algorithm|framework|architecture|fabrication|synthesis|preparation|"
                r"instrumentation|measurements|experimental setup|protocol|workflow)\b",
                h,
            ):
                method_like.append((item["order"], item["block"]))

        if method_like:
            grouped["methods"] = method_like
            recovery_notes["methods"] = "borrowed_method_like_sections"

    substantial_ordered = sorted(
        [item for item in all_sections if is_substantial_text(item["block"], min_chars=80)],
        key=lambda x: x["order"],
    )

    if substantial_ordered:
        fallback_by_label = {
            "introduction": [substantial_ordered[0]],
            "methods": substantial_ordered[1:2] or substantial_ordered[:1],
            "results": substantial_ordered[-2:-1] or substantial_ordered[-1:],
            "discussion": substantial_ordered[-1:],
        }

        for label in ["introduction", "methods", "results", "discussion"]:
            if not grouped[label]:
                fallback_items = fallback_by_label.get(label, [])
                if fallback_items:
                    grouped[label] = [(item["order"], item["block"]) for item in fallback_items]
                    recovery_notes[label] = "final_order_based_fallback_existing_text"

    grouped_text = {}
    counts = {}

    for label, items in grouped.items():
        seen_blocks = set()
        blocks = []

        for _, block in sorted(items, key=lambda x: x[0]):
            key = normalize_text(block)[:700]
            if key in seen_blocks:
                continue
            seen_blocks.add(key)
            blocks.append(block)

        grouped_text[label] = "\n\n---\n\n".join(blocks).strip()
        counts[label] = len(blocks)

    return grouped_text, counts, recovery_notes


# =============================================================================
# FIGURE / TABLE FORMATTING
# =============================================================================

def normalize_section_label(value: Any) -> str:
    label = normalize_imrad_label(value)
    if label:
        return label
    hcat = heading_category(value)
    if hcat == "results_discussion":
        return "results"
    if hcat in ALLOWED_IMRAD_SECTIONS:
        return hcat
    return ""


def is_main_paper_figure(fig: Dict[str, Any]) -> bool:
    if bool(fig.get("is_supplementary_or_appendix", False)):
        return False
    if bool(fig.get("image_file_missing", False)):
        return False
    if (fig.get("type") or "figure") == "table":
        return False
    section = normalize_section_label(fig.get("assigned_section") or fig.get("section"))
    if not section:
        return False
    quality = clean_inline(fig.get("extraction_quality", "good")).lower()
    if quality and quality not in ALLOWED_FIGURE_QUALITIES:
        # Keep only figures usable as image inputs.
        return False
    if not clean_inline(fig.get("caption", "")) and not clean_inline(fig.get("display_label") or fig.get("label")):
        return False
    return True


def resolve_figure_image_path(paper_folder: Path, fig: Dict[str, Any]) -> Optional[Path]:
    figures_dir = find_figures_dir(paper_folder)
    figure_id = clean_inline(fig.get("figure_id", ""))
    image_file = clean_inline(fig.get("image_file", ""))

    candidates = []
    if image_file:
        candidates.append(figures_dir / image_file)
        candidates.append(paper_folder / "extracted" / image_file)
    if figure_id:
        candidates.append(figures_dir / f"{figure_id}.png")
        candidates.append(figures_dir / f"{figure_id}.jpg")
        candidates.append(figures_dir / f"{figure_id}.jpeg")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def format_figures_block(
    figures_data: Dict[str, Any],
    paper_folder: Path,
) -> Tuple[str, Dict[str, str], int, int]:
    figures = figures_data.get("figures", [])
    if not isinstance(figures, list):
        figures = []

    blocks = []
    figure_id_to_path: Dict[str, str] = {}
    total_count = len(figures)

    for fig in figures:
        if not isinstance(fig, dict):
            continue
        if not is_main_paper_figure(fig):
            continue

        image_path = resolve_figure_image_path(paper_folder, fig)
        if image_path is None:
            continue

        fig_id = clean_inline(fig.get("figure_id", "")) or f"fig_{len(blocks) + 1:02d}"
        label = clean_inline(fig.get("display_label") or fig.get("label") or fig_id)
        section = normalize_section_label(fig.get("assigned_section") or fig.get("section")) or "unassigned"
        caption = truncate_text(fig.get("caption", ""), MAX_CAPTION_CHARS)

        blocks.append(
            f"[Figure {len(blocks) + 1}]\n"
            f"figure_id: {fig_id}\n"
            f"label: {label}\n"
            f"assigned_section: {section}\n"
            f"caption: {caption}\n"
            f"image_input_id: {fig_id}"
        )
        figure_id_to_path[fig_id] = str(image_path)

        if len(blocks) >= MAX_FIGURES_PER_PAPER:
            break

    if not blocks:
        return "No usable main-paper figures were extracted for this paper.", {}, total_count, 0

    return "\n\n".join(blocks), figure_id_to_path, total_count, len(blocks)


def is_main_paper_table(tbl: Dict[str, Any]) -> bool:
    section = normalize_section_label(tbl.get("section") or tbl.get("assigned_section"))
    if not section:
        return False
    if bool(tbl.get("is_supplementary_or_appendix", False)):
        return False
    caption = clean_inline(tbl.get("caption", ""))
    content = normalize_text(tbl.get("markdown") or tbl.get("structured_markdown") or tbl.get("text") or tbl.get("raw_markdown_text") or "")
    return bool(caption or content)


def format_tables_block(tables_data: Dict[str, Any]) -> Tuple[str, int, int]:
    tables = tables_data.get("tables", [])
    if not isinstance(tables, list):
        tables = []

    blocks = []
    total_count = len(tables)

    for tbl in tables:
        if not isinstance(tbl, dict):
            continue
        if not is_main_paper_table(tbl):
            continue

        table_id = clean_inline(tbl.get("table_id", "")) or f"table_{len(blocks) + 1:02d}"
        label = clean_inline(tbl.get("label") or tbl.get("num") or table_id)
        section = normalize_section_label(tbl.get("section") or tbl.get("assigned_section")) or "unassigned"
        caption = truncate_text(tbl.get("caption", ""), MAX_CAPTION_CHARS)
        content = normalize_text(
            tbl.get("markdown")
            or tbl.get("structured_markdown")
            or tbl.get("text")
            or tbl.get("raw_markdown_text")
            or ""
        )
        content = truncate_text(content, MAX_TABLE_CHARS)

        blocks.append(
            f"[Table {len(blocks) + 1}]\n"
            f"table_id: {table_id}\n"
            f"label: {label}\n"
            f"assigned_section: {section}\n"
            f"caption: {caption}\n"
            f"content:\n{content}"
        )

        if len(blocks) >= MAX_TABLES_PER_PAPER:
            break

    if not blocks:
        return "No usable main-paper tables were extracted for this paper.", total_count, 0

    return "\n\n".join(blocks), total_count, len(blocks)


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def validate_prompt_inputs(
    doi: str,
    title: str,
    abstract: str,
    grouped_text: Dict[str, str],
    imrad_path: Optional[Path],
) -> Tuple[bool, str]:
    if imrad_path is None:
        return False, "missing_fulltext_imrad_json"

    if not normalize_text(doi):
        return False, "missing_doi"

    if not normalize_text(title):
        return False, "missing_title"

    if not normalize_text(abstract):
        return False, "missing_abstract"

    for label in ("introduction", "methods", "results", "discussion"):
        if not normalize_text(grouped_text.get(label, "")):
            return False, "missing_" + label + "_text"

    return True, "valid"


def build_user_prompt(
    doi: str,
    title: str,
    abstract: str,
    grouped_text: Dict[str, str],
    figures_block: str,
    tables_block: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        doi=normalize_text(doi),
        title=normalize_text(title),
        abstract=normalize_text(abstract),
        introduction_text=grouped_text["introduction"],
        methods_text=grouped_text["methods"],
        results_text=grouped_text["results"],
        discussion_text=grouped_text["discussion"],
        figures_block=normalize_text(figures_block),
        tables_block=normalize_text(tables_block),
    )


def csv_writer(path: Path, fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def build_phase1_prompts(
    dataset_dir: Path,
    limit: Optional[int],
    logger: logging.Logger,
) -> Dict[str, Any]:
    start_time = datetime.now()

    ensure_dirs()
    write_text(PHASE1_SYSTEM_PROMPT_PATH, PHASE1_SYSTEM_PROMPT.rstrip() + "\n")

    doi_folders_all = get_doi_folders(dataset_dir)
    doi_folders = doi_folders_all[:limit] if limit is not None else doi_folders_all

    logger.info("Dataset dir: " + str(dataset_dir))
    logger.info("Total DOI folders found: " + str(len(doi_folders_all)))

    if limit is not None:
        logger.info("Applying limit: " + str(limit))

    figure_path_fields = [
        "doi",
        "user_prompt_path",
        "figure_id_to_path_json",
    ]

    skipped_fields = [
        "doi_metadata",
        "title",
        "paper_folder_name",
        "paper_folder_path",
        "metadata_json_path",
        "imrad_json_path",
        "figures_json_path",
        "tables_json_path",
        "reason",
        "details",
    ]

    figure_paths_file, figure_paths_writer = csv_writer(PHASE1_FIGURE_PATHS_CSV, figure_path_fields)
    skipped_file, skipped = csv_writer(PHASE1_SKIPPED_CSV, skipped_fields)

    counters = Counter()
    skipped_reasons = Counter()
    used_safe_dois = Counter()
    recovery_counters = Counter()

    try:
        total_to_process = len(doi_folders)

        for i, paper_folder in enumerate(doi_folders, start=1):
            counters["processed"] += 1

            imrad_path = find_fulltext_imrad_json(paper_folder)
            figures_path = find_figures_json(paper_folder)
            tables_path = find_tables_json(paper_folder)
            metadata_path = find_metadata_json(paper_folder)

            imrad_data = load_json_safe(imrad_path)
            figures_data = load_json_safe(figures_path)
            tables_data = load_json_safe(tables_path)
            metadata = load_json_safe(metadata_path)

            doi = extract_doi(imrad_data, metadata, paper_folder)
            title = extract_title(imrad_data, metadata)
            abstract = extract_abstract(imrad_data, metadata)
            abstract_source = "json_or_metadata"

            journal = extract_journal(imrad_data, metadata)
            publisher = extract_publisher(imrad_data, metadata)
            _ = journal, publisher, abstract_source  # retained for parity/debug compatibility

            try:
                grouped_text, section_counts, recovery_notes = group_imrad_sections(imrad_data)
                _ = section_counts

                # Same final abstract fallback as Variant A.
                if not normalize_text(abstract) and normalize_text(grouped_text.get("introduction", "")):
                    intro_text = normalize_text(grouped_text["introduction"])
                    abstract = "Abstract was not available in metadata. Introductory context fallback: " + intro_text[:1500]
                    abstract_source = "introduction_fallback"

                valid, reason = validate_prompt_inputs(
                    doi=doi,
                    title=title,
                    abstract=abstract,
                    grouped_text=grouped_text,
                    imrad_path=imrad_path,
                )

                if not valid:
                    counters["skipped"] += 1
                    skipped_reasons[reason] += 1

                    skipped.writerow({
                        "doi_metadata": doi,
                        "title": title,
                        "paper_folder_name": paper_folder.name,
                        "paper_folder_path": str(paper_folder),
                        "metadata_json_path": str(metadata_path) if metadata_path else "",
                        "imrad_json_path": str(imrad_path) if imrad_path else "",
                        "figures_json_path": str(figures_path) if figures_path else "",
                        "tables_json_path": str(tables_path) if tables_path else "",
                        "reason": reason,
                        "details": "",
                    })

                else:
                    base_safe_doi = safe_filename_from_doi(doi)
                    used_safe_dois[base_safe_doi] += 1

                    if used_safe_dois[base_safe_doi] == 1:
                        safe_doi = base_safe_doi
                    else:
                        safe_doi = base_safe_doi + "_dup" + str(used_safe_dois[base_safe_doi])

                    figures_block, figure_id_to_path, figure_total, figure_used = format_figures_block(
                        figures_data=figures_data,
                        paper_folder=paper_folder,
                    )
                    tables_block, table_total, table_used = format_tables_block(tables_data=tables_data)

                    user_prompt = build_user_prompt(
                        doi=doi,
                        title=title,
                        abstract=abstract,
                        grouped_text=grouped_text,
                        figures_block=figures_block,
                        tables_block=tables_block,
                    )

                    user_prompt_path = PHASE1_SHARED_USER_PROMPT_DIR / (safe_doi + USER_PROMPT_SUFFIX)
                    write_text(user_prompt_path, user_prompt.rstrip() + "\n")

                    for label in ["introduction", "methods", "results", "discussion"]:
                        if recovery_notes.get(label):
                            recovery_counters[label + ":" + recovery_notes[label]] += 1

                    figure_paths_writer.writerow({
                        "doi": doi,
                        "user_prompt_path": str(user_prompt_path),
                        "figure_id_to_path_json": json_dumps_compact(figure_id_to_path),
                    })

                    counters["built"] += 1
                    counters["figures_total"] += figure_total
                    counters["figures_used"] += figure_used
                    counters["tables_total"] += table_total
                    counters["tables_used"] += table_used

            except Exception as e:
                counters["skipped"] += 1
                skipped_reasons["exception"] += 1

                skipped.writerow({
                    "doi_metadata": doi,
                    "title": title,
                    "paper_folder_name": paper_folder.name,
                    "paper_folder_path": str(paper_folder),
                    "metadata_json_path": str(metadata_path) if metadata_path else "",
                    "imrad_json_path": str(imrad_path) if imrad_path else "",
                    "figures_json_path": str(figures_path) if figures_path else "",
                    "tables_json_path": str(tables_path) if tables_path else "",
                    "reason": "exception",
                    "details": repr(e),
                })

                logger.warning("Skipped due to exception: " + str(paper_folder) + " | " + repr(e))

            if i % 100 == 0 or i == total_to_process:
                logger.info(
                    "Processed: {}/{} | built={} | skipped={} | figures_used={} | tables_used={}".format(
                        i,
                        total_to_process,
                        counters["built"],
                        counters["skipped"],
                        counters["figures_used"],
                        counters["tables_used"],
                    )
                )

    finally:
        figure_paths_file.close()
        skipped_file.close()

    elapsed = (datetime.now() - start_time).total_seconds()

    duplicate_bases = {
        safe_doi: count
        for safe_doi, count in used_safe_dois.items()
        if count > 1
    }

    total_doi_folders = len(doi_folders_all)
    processed_after_limit = len(doi_folders)
    expected_10k_complete = (
        total_doi_folders == 10000
        and counters["processed"] == 10000
        and counters["built"] == 10000
        and counters["skipped"] == 0
    )

    summary = {
        "generated_at": utc_now_iso(),
        "script": "task1_variantB_phase1_prompt_builder.py",
        "dataset_dir": str(dataset_dir),
        "system_prompt_path": str(PHASE1_SYSTEM_PROMPT_PATH),
        "phase1_user_prompt_dir": str(PHASE1_SHARED_USER_PROMPT_DIR),
        "figure_paths_csv": str(PHASE1_FIGURE_PATHS_CSV),
        "prompt_builder_output_dir": str(PROMPT_BUILDER_OUTPUT_DIR),
        "skipped_csv": str(PHASE1_SKIPPED_CSV),
        "log_path": str(PHASE1_LOG_PATH),
        "summary_json": str(PHASE1_SUMMARY_JSON),
        "total_doi_folders": total_doi_folders,
        "processed_after_limit": processed_after_limit,
        "processed": counters["processed"],
        "built": counters["built"],
        "skipped": counters["skipped"],
        "figures_total_seen": counters["figures_total"],
        "figures_used_in_prompts": counters["figures_used"],
        "tables_total_seen": counters["tables_total"],
        "tables_used_in_prompts": counters["tables_used"],
        "skipped_reasons": dict(skipped_reasons),
        "recovery_counts": dict(recovery_counters),
        "duplicate_safe_dois": duplicate_bases,
        "expected_10k_complete": expected_10k_complete,
        "elapsed_seconds": round(elapsed, 2),
    }

    write_json(PHASE1_SUMMARY_JSON, summary)

    logger.info("Prompt building complete.")
    logger.info("Total DOI folders: " + str(total_doi_folders))
    logger.info("Processed: " + str(counters["processed"]))
    logger.info("Built: " + str(counters["built"]))
    logger.info("Skipped: " + str(counters["skipped"]))
    logger.info("Figures used in prompts: " + str(counters["figures_used"]))
    logger.info("Tables used in prompts: " + str(counters["tables_used"]))
    logger.info("Skipped reasons: " + json.dumps(dict(skipped_reasons), ensure_ascii=False))
    logger.info("Recovery counts: " + json.dumps(dict(recovery_counters), ensure_ascii=False))
    logger.info("Expected 10k complete: " + str(expected_10k_complete))
    logger.info("Figure paths CSV: " + str(PHASE1_FIGURE_PATHS_CSV))
    logger.info("Skipped CSV: " + str(PHASE1_SKIPPED_CSV))
    logger.info("Summary: " + str(PHASE1_SUMMARY_JSON))

    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Task 1 Variant B Phase 1 multimodal SRP prompts from fulltext_imrad.json, figures.json, and tables.json files."
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Dataset directory to scan. Default: " + str(DEFAULT_DATASET_DIR),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing. Example: --limit 10",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dirs()
    cleanup_previous_outputs()
    logger = setup_logging()

    logger.info("Starting Task 1 Variant B Phase 1 prompt builder")
    logger.info("Script path: " + str(Path(__file__).resolve()))
    logger.info("Dataset dir: " + str(args.dataset_dir))
    logger.info("Output dir: " + str(PROMPT_BUILDER_OUTPUT_DIR))
    logger.info("System prompt path: " + str(PHASE1_SYSTEM_PROMPT_PATH))
    logger.info("User prompt dir: " + str(PHASE1_SHARED_USER_PROMPT_DIR))
    logger.info("Figure paths CSV: " + str(PHASE1_FIGURE_PATHS_CSV))

    build_phase1_prompts(
        dataset_dir=Path(args.dataset_dir),
        limit=args.limit,
        logger=logger,
    )


if __name__ == "__main__":
    main()
