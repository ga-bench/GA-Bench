"""
IMRaD section classifier (v8.1).

v6 changes vs v5.3:
    1. Methods propagation scope limiter. In v5.3, once a "methods" label was
       assigned, it propagated to ALL subsequent unlabeled sections until
       another IMRaD heading was hit. This caused results/discussion subsections
       with technique-specific headings (e.g. "XRD", "Differential pulse
       voltammetry", "Sensing mechanism") to inherit the methods label. v6 adds
       a propagation budget: methods propagation stops after N consecutive
       unlabeled sections (default 4), preventing runaway inheritance.
    2. Expanded SYNONYM_PATTERNS for results: added technique-specific result
       headings commonly found in chemistry/materials/sensor papers: "XRD",
       "XPS", "FESEM", "SEM", "TEM", "EIS", "cyclic voltammetry", "DPV",
       "sensing mechanism", "interference", "selectivity", "reproducibility",
       "stability", "real water sample", "practical implication".
    3. Section role tagging: sections that look like characterization results
       (heading matches a known analytical technique) get tagged as results
       even when they follow a methods section, overriding propagation.
    4. is_characterization_heading() detects technique-specific headings that
       are almost always results, not methods.

Public API unchanged.
"""

from __future__ import annotations

import re
from collections import Counter

IMRAD_CATEGORIES = ["introduction", "methods", "results", "discussion"]

# Max consecutive unlabeled sections that inherit via propagation.
MAX_PROPAGATION_DEPTH = 4

import re as _re_bm
BACK_MATTER_HEADINGS = _re_bm.compile(
    r"\b(?:acknowledge?ments?|references?|bibliography|appendix|appendices|"
    r"funding|declarations?|conflicts? of interest|competing interests?|"
    r"credit authorship|author(?:s'?)? contributions?|data availability|"
    r"supplementary|supporting information|abbreviations|nomenclature|"
    r"ethics|consent|disclosure)\b",
    _re_bm.IGNORECASE,
)

def is_back_matter(heading) -> bool:
    h = (heading or "").strip()
    if not h:
        return False
    return bool(BACK_MATTER_HEADINGS.search(h))


NON_IMRAD_HEADINGS = re.compile(
    r"\b(?:"
    r"related\s+works?|prior\s+works?|previous\s+studies|previous\s+works?|"
    r"literature\s+review|review\s+of\s+literature|background\s+literature|"
    r"state\s+of\s+the\s+art|"
    r"theoretical\s+background|conceptual\s+background|"
    r"preliminaries|preliminary|"
    r"nomenclature|abbreviations?|notation|"
    r"introduction\s+to\b|"
    r"graph\s+theory|transportation\s+theory|"
    r"computational\s+fluid\s+dynamic\s+models?\s+for\b|"
    r"review\s+of\s+software\s+tools?|"
    r"healthcare\s+processes\s+and\s+computer-interpretable\s+guidelines|"
    r"application\s+of\s+bpm\s+techniques\s+in\s+healthcare"
    r")\b",
    re.IGNORECASE,
)

RELATED_WORK_HEADINGS = re.compile(
    r"\b(?:"
    r"related\s+works?|prior\s+works?|previous\s+studies|previous\s+works?|"
    r"literature\s+review|review\s+of\s+literature|background\s+literature|"
    r"state\s+of\s+the\s+art"
    r")\b",
    re.IGNORECASE,
)

def is_related_work(heading: str | None) -> bool:
    h = normalize_heading(heading)
    return bool(h and RELATED_WORK_HEADINGS.search(h))

def is_non_imrad_heading(heading: str | None) -> bool:
    h = normalize_heading(heading)
    if not h:
        return False
    return bool(looks_like_equation_heading(heading) or is_back_matter(h) or NON_IMRAD_HEADINGS.search(h))

def _mark_non_imrad(section: dict, reason: str = "non_imrad_heading_excluded") -> None:
    if is_related_work(section.get("heading", "")):
        section["section_role"] = "related_work"
        reason = "related_work_excluded_from_imrad"
    else:
        section.setdefault("section_role", "body")

    section["use_for_imrad"] = False
    section["imrad"] = None
    section.pop("imrad_secondary", None)
    section.pop("imrad_secondary_source", None)
    section.pop("imrad_secondary_confidence", None)
    section.pop("imrad_secondary_reason", None)
    section["imrad_source"] = "excluded"
    section["imrad_confidence"] = 0.0
    section["imrad_reason"] = reason


# ─────────────────────────────────────────────────────────────────────────────
# v6: Characterization/technique heading detection
# These headings appear in results sections of chemistry/materials papers.
# They describe WHAT was measured (results), not HOW the experiment was set up.
# ─────────────────────────────────────────────────────────────────────────────

CHARACTERIZATION_HEADINGS = re.compile(
    r"\b(?:"
    # Spectroscopy/diffraction techniques as section headings
    r"XRD|WAXS|SAXS|WAXD|XPS|FTIR|NMR|UV[\s-]?vis|Raman|"
    r"(?:FE[\s-]?)?SEM|TEM|AFM|EDX|EDS|EELS|"
    # Electrochemistry result headings
    r"cyclic\s+voltammetr|differential\s+pulse\s+voltammetr|"
    r"electrochemical\s+impedance|EIS\b|DPV\b|CV\b|"
    r"chronoamperometr|potentiometr|"
    # Thermal analysis
    r"DSC|TGA|DTA|"
    # Material characterization as results
    r"structural\s+and\s+morphological|morphological\s+characterization|"
    r"surface\s+(?:topograph|morpholog)|"
    # Sensor/device testing (results, not methods)
    r"sensing\s+mechanism|interference.*(?:selectivity|ions?)|"
    r"selectivity\s+(?:test|stud)|"
    r"reproducibility\s+and\s+stability|stability\s+(?:test|of)|"
    r"real\s+(?:water|sample)\s+(?:analysis|application|test)|"
    r"practical\s+implication|"
    # Performance characterization
    r"material\s+characterization\s+of\b"
    r")\b",
    re.IGNORECASE,
)


def is_characterization_heading(heading: str | None) -> bool:
    """True if heading describes a characterization technique typically found
    in results/discussion sections, not methods."""
    h = normalize_heading(heading)
    if not h:
        return False
    return bool(CHARACTERIZATION_HEADINGS.search(h))


# ─────────────────────────────────────────────────────────────────────────────
# Code-block detection (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────────────

_CODE_LINE_PATTERNS = [
    re.compile(r"^\s*#"),
    re.compile(r"^\s*(?:def|class|import|from|return|elif|else:|for |while |if |try:|except)\b"),
    re.compile(r"\bGPIO\.|\bself\.|\bprint\s*\(|\brange\s*\(|\bimport\b"),
    re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=]"),
    re.compile(r"[;{}]\s*$"),
]


def is_code_block(text: str) -> bool:
    if not text:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    code_hits = 0
    for ln in lines:
        for pat in _CODE_LINE_PATTERNS:
            if pat.search(ln):
                code_hits += 1
                break
    return (code_hits / len(lines)) >= 0.5


PROPAGATING_CATEGORIES = {"methods", "results", "discussion"}

SECTION_NUMBER_PREFIX = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)*\.?"
    r"|[IVXLCDM]+\.?"
    r"|[A-Z]\."
    r"|\([a-zA-Z0-9]+\)"
    r")\s+",
    re.IGNORECASE,
)

TRAILING_PUNCT = re.compile(r"[\s:;.\-–—]+$")

_MATH_HEADING_RE = re.compile(r"[=∫∑∏√λσψ∂ημνκΩ]|\b(?:dα|dmu|dx|dt)\b|[+*/^]", re.IGNORECASE)


def looks_like_equation_heading(heading: str | None) -> bool:
    h = str(heading or "").strip()
    if not h:
        return False
    # Real biomedical/statistical headings may contain "+" (e.g., "N+ rate").
    if re.search(r"\bn\s*\+\s*rate\b", h, re.IGNORECASE):
        return False
    if _MATH_HEADING_RE.search(h) and len(re.findall(r"[A-Za-z]{3,}", h)) < 5:
        return True
    return False


def normalize_heading(heading: str | None) -> str:
    if not heading:
        return ""
    h = str(heading).strip()
    h = SECTION_NUMBER_PREFIX.sub("", h)
    h = h.replace("&", " and ")
    h = re.sub(r"[_/]+", " ", h)
    h = re.sub(r"\s+", " ", h)
    h = TRAILING_PUNCT.sub("", h)
    return h.lower().strip()


def category_propagates(category: str | None) -> bool:
    return category in PROPAGATING_CATEGORIES


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


EXACT_PATTERNS = {
    "introduction": [
        r"^introduction$",
        r"^intro$",
    ],
    "methods": [
        r"^methods?$",
        r"^materials? and methods?$",
        r"^methodology$",
        r"^experimental methods?$",
        r"^experimental section$",
        r"^experimental procedure$",
    ],
    "results": [
        r"^results?$",
        r"^findings?$",
        r"^results? and analysis$",
        r"^results? and performance$",
    ],
    "discussion": [
        r"^discussion$",
        r"^conclusions?$",
        r"^summary and conclusions?$",
        r"^concluding remarks$",
        r"^concluding remarks\b.*",
        r"^conclusions? and (?:future|outlook)\b.*",
        r"^summary$",
    ],
}


SYNONYM_PATTERNS = {
    "introduction": [
        r"\bbackground\b",
        r"\bmotivation\b",
        r"\bobjective\b",
        r"\bobjectives\b",
        r"\baims?\b",
        r"\bpurpose\b",
        r"\bresearch problem\b",
        r"\bresearch gap\b",
    ],
    "methods": [
        r"\bmaterials? and methods?\b",
        r"\bdata and methods?\b",
        r"\bmethodology\b",
        r"\bresearch methods?\b",
        r"\bexperimental setup\b",
        r"\bexperimental design\b",
        r"\bexperimental procedure\b",
        r"\bstudy design\b",
        r"\bdata collection\b",
        r"\bmethod implementation\b",
        r"\bimplementation details\b",
        r"\bcomputational fluid dynamic simulations\b",
        r"\bsimulation setup\b",
        r"\bnumerical method\b",
        r"\bstatistical analysis\b",
        r"\bmaterials? preparation\b",
        r"\bsample preparation\b",
        r"\bmeasurement setup\b",
        r"\binstrumentation\b",
        r"\bproblem setup\b",
        r"\bproblem formulation\b",
        r"\bproblem definition\b",
        r"\bproposed\b.*\bmodel\b",
        r"\bproposed\b.*\bmethod\b",
        r"\bproposed\b.*\bapproach\b",
        r"\bproposed\b.*\bframework\b",
        r"\bproposed\b.*\barchitecture\b",
        r"\bmathematical model\b",
        r"\bsystem model\b",
        r"\bsystem design\b",
        r"\bmodel description\b",
        r"\bmodel architecture\b",
        r"\bnetwork architecture\b",
        r"\bdataset description\b",
        r"\bdata description\b",
        r"\bdata preparation\b",
        r"\bfeature extraction\b",
        r"\btraining procedure\b",
        r"\btraining details\b",
        r"\bevaluation metrics?\b",
        r"\bevaluation setup\b",
        r"\bevaluation protocol\b",
        r"\bevaluation methodology\b",
        r"\bprotocol\s+specification\b",
        r"\bproposed\s+scheme\b",
        r"\bproposed\s+protocol\b",
        r"\bsecurity\s+protocol\b",
        r"\b(?:first|second|third|fourth)\s+part\s+of\b.*\bprotocol\b",
        r"\bprotocol\s+design\b",
        r"\bsecurity\s+analysis\b",
        r"\banalysis\s+of\s+the\s+protocol\b",
        r"\bcase\s+study\b",
        r"\bnetwork\s+model\b",
        r"\bthreats?\s+and\s+adversarial\s+model\b",
        r"\bsecurity\s+requirements\b",
        r"\bcharacterization\b",
        # v6: preparation-specific method headings
        r"\bpreparation of\b.*\bsensor\b",
        r"\bpreparation of\b.*\belectrode\b",
        r"\bpreparation of\b.*\bcomposite\b",
        r"\bapparatus\b",
        r"\banalytical procedure\b",
        r"\banimals?\b",
        r"\bpatients?\b",
        r"\btissue source\b",
        r"\bpreparation\b",
        r"\brecording\b",
        r"\bvideo[- ]?EEG\b",
        r"\belectrode implantation\b",
        r"\bclassification of\b",
        r"\binduction of\b",
        r"\bco-analysis\b",
        r"\bdata analysis\b",
        r"\bethical considerations?\b",
        r"\bpower considerations?\b",
        r"\bstatistical methods?\b",
    ],
    "results": [
        r"\bresults?\b",
        r"\bfindings?\b",
        r"\bexperimental results?\b",
        r"\bevaluation results?\b",
        r"\bempirical results?\b",
        r"\bresults?\s+and\s+analysis\b",
        r"\bresults?\s+and\s+performance\b",
        r"\bperformance\s+evaluation\b",
        r"\bperformance\s+comparison\b",
        r"\bperformance\s+analysis\b",
        r"\bsensitivity\s+analysis\b",
        r"\bsensitivity\s+to\b",
        r"\bablation\s+stud(?:y|ies)\b",
        r"\bcomparative\s+analysis\b",
        r"\bcomparative\s+stud(?:y|ies)\b",
        r"\bcost\b.*\bperformance\b",
        r"\bclassification\s+capabilities\b",
        r"\baccuracy\b.*\banalysis\b",
        # v6: results and discussion combined
        r"\bresults?\s+and\s+discussions?\b",
        # v6: characterization/testing results
        r"\bmaterial\s+characterization\b",
        r"\binterface\s+performance\b",
        r"\bsensing\s+mechanism\b",
        r"\binterference\b.*\bselectivity\b",
        r"\bselectivity\b.*\binterference\b",
        r"\breproducibility\b.*\bstability\b",
        r"\bstability\b.*\breproducibility\b",
        r"\bpractical\s+implication\b",
        r"\breal\s+water\s+sample\b",
        r"\breal\s+sample\s+analysis\b",
        # v6: technique headings that ARE results
        r"\bcyclic\s+voltammetr\b",
        r"\bdifferential\s+pulse\s+voltammetr\b",
        r"\belectrochemical\s+impedance\b",
        r"\bpreliminary study\b",
        r"\bprevalence\b",
        r"\bnumber of\b",
        r"\busv categor(?:y|ies)\b",
        r"\bacoustic properties\b",
        r"\bpostictal\b",
        r"\btypes of vocalizations\b",
        r"\bviscoelastic properties\b",
        r"\bthermogravimetric analysis\b",
        r"\bmapping and quantification\b",
        r"\bhistological staining\b",
        r"\barticulating surface damage\b",
        r"\bsurface roughness\b",
        r"\bquantification of\b",
        r"\bn\s*\+\s*rate\b",
        r"\bdisease[- ]free survival\b",
        r"\boverall survival\b",
        r"\bsurvival analysis\b",
        r"\bpostoperative death\b",
    ],
    "discussion": [
        r"\bdiscussion\b",
        r"\bconclusions?\b",
        r"\bimplications?\b",
        r"\blimitations?\b",
        r"\bfuture work\b",
        r"\bsummary and conclusions?\b",
        r"\bconcluding remarks\b",
        r"\boutlook\b",
        # v6: discussion + implications + limitations combined
        r"\bdiscussion.*implications.*limitations\b",
        r"\bimplications.*limitations\b",
    ],
}


COMBINED_PATTERNS = [
    (
        re.compile(r"\bresults?\b.*\bdiscussion\b|\bdiscussion\b.*\bresults?\b", re.IGNORECASE),
        "results",
        "discussion",
        "combined_results_discussion",
        0.95,
    ),
    (
        re.compile(r"\bmethods?\b.*\bresults?\b|\bresults?\b.*\bmethods?\b", re.IGNORECASE),
        "methods",
        "results",
        "combined_methods_results",
        0.80,
    ),
]


INTRO_CONTENT = [
    r"\bwe (propose|present|introduce|investigate|study|examine)\b",
    r"\bthe aim\b",
    r"\bthe objective\b",
    r"\bchallenge\b",
    r"\bproblem\b",
    r"\bmotivat",
    r"\brecent(ly)?\b",
    r"\bimportant\b",
]

METHODS_CONTENT = [
    r"\bwe (use|used|employ|employed|develop|developed|implement|implemented|train|trained|construct|constructed)\b",
    r"\bdataset\b",
    r"\bdata set\b",
    r"\bmodel\b",
    r"\balgorithm\b",
    r"\bparameter\b",
    r"\bprotocol\b",
    r"\bexperiment\b",
    r"\bsimulation\b",
    r"\bmeasurement\b",
    r"\btraining\b",
    r"\boptimizer\b",
    r"\blearning rate\b",
    r"\bbatch size\b",
    r"\barchitecture\b",
    r"\bbaseline\b",
    r"\bpreprocess",
    r"\bfeature\b",
    r"\bstatistical analysis\b",
    r"\bmaterials?\b",
    r"\bsynthesis\b",
    r"\bethics committee\b",
    r"\bapproved by\b",
    r"\bmeasured using\b",
    r"\bwere measured\b",
    r"\bwere used\b",
]

RESULTS_CONTENT = [
    r"\bresults? (show|shows|showed|demonstrate|demonstrated|indicate|indicated)\b",
    r"\bwe (found|observed|achieved|obtained|report)\b",
    r"\bperformance\b",
    r"\baccuracy\b",
    r"\bprecision\b",
    r"\brecall\b",
    r"\bf1\b",
    r"\baUC\b",
    r"\bsignificant(ly)?\b",
    r"\bincrease(d)?\b",
    r"\bdecrease(d)?\b",
    r"\bimprove(d|ment)?\b",
    r"\boutperform",
    r"\bcompared with\b",
    r"\btable\s+\d+",
    r"\bfig(?:ure)?\.?\s+\d+",
    r"\bp\s*[<=>]\s*0\.\d+",
    r"\bmean\s+±\s+",
    r"\bwas significantly\b",
    r"\bwere significantly\b",
    r"\bshowed that\b",
]

DISCUSSION_CONTENT = [
    r"\bthese results\b",
    r"\bthis finding\b",
    r"\bthis suggests\b",
    r"\bthis indicates\b",
    r"\btherefore\b",
    r"\bin conclusion\b",
    r"\bwe conclude\b",
    r"\bimplication",
    r"\blimitation",
    r"\bfuture work\b",
    r"\bfuture studies\b",
    r"\bmay be due to\b",
    r"\bconsistent with\b",
]


CONTENT_PATTERNS = {
    "introduction": INTRO_CONTENT,
    "methods": METHODS_CONTENT,
    "results": RESULTS_CONTENT,
    "discussion": DISCUSSION_CONTENT,
}


def _content_score(text: str, category: str) -> float:
    if not text:
        return 0.0
    patterns = CONTENT_PATTERNS.get(category, [])
    raw = 0
    for pat in patterns:
        raw += len(re.findall(pat, text, re.IGNORECASE))
    length_factor = max(1.0, len(text) / 1500.0)
    return raw / length_factor


def classify_imrad_detailed(heading: str | None) -> dict:
    h = normalize_heading(heading)

    empty = {
        "label": None,
        "secondary_label": None,
        "source": "none",
        "confidence": 0.0,
        "reason": "empty_heading",
    }

    if not h:
        return empty

    if is_non_imrad_heading(h):
        return {
            "label": None,
            "secondary_label": None,
            "source": "excluded",
            "confidence": 0.0,
            "reason": "non_imrad_heading_excluded",
        }

    for pattern, primary, secondary, reason, conf in COMBINED_PATTERNS:
        if pattern.search(h):
            return {
                "label": primary,
                "secondary_label": secondary,
                "source": "combined_section",
                "confidence": conf,
                "reason": reason,
            }

    for label, patterns in EXACT_PATTERNS.items():
        if _match_any(h, patterns):
            return {
                "label": label,
                "secondary_label": None,
                "source": "heading_exact",
                "confidence": 1.0,
                "reason": f"exact_{label}",
            }

    if re.search(r"\bexperimental results?\b", h, re.IGNORECASE):
        return {
            "label": "results",
            "secondary_label": None,
            "source": "heading_synonym",
            "confidence": 0.90,
            "reason": "experimental_results",
        }

    if re.search(r"\bevaluation setup\b|\bevaluation protocol\b|\bevaluation metric", h, re.IGNORECASE):
        return {
            "label": "methods",
            "secondary_label": None,
            "source": "heading_synonym",
            "confidence": 0.85,
            "reason": "evaluation_setup",
        }

    # v6: characterization headings → results (before general synonym match,
    # so "XRD" or "Cyclic voltammetry" gets results, not methods via the
    # generic "characterization" synonym).
    if is_characterization_heading(h):
        return {
            "label": "results",
            "secondary_label": None,
            "source": "heading_characterization",
            "confidence": 0.80,
            "reason": "characterization_technique_heading",
        }

    for label in ["introduction", "methods", "results", "discussion"]:
        if _match_any(h, SYNONYM_PATTERNS[label]):
            conf = {
                "introduction": 0.75,
                "methods": 0.82,
                "results": 0.82,
                "discussion": 0.82,
            }[label]
            return {
                "label": label,
                "secondary_label": None,
                "source": "heading_synonym",
                "confidence": conf,
                "reason": f"synonym_{label}",
            }

    return empty


def classify_imrad(heading: str | None) -> str | None:
    return classify_imrad_detailed(heading).get("label")


def _section_text(section: dict) -> str:
    return section.get("text_no_tables") or section.get("text") or ""


def _assign_label(
    section: dict,
    label: str,
    source: str,
    confidence: float,
    reason: str,
    *,
    allow_override: bool = False,
) -> bool:
    if is_back_matter(section.get("heading", "")):
        _mark_non_imrad(section, "back_matter_excluded")
        return False

    if is_non_imrad_heading(section.get("heading", "")):
        _mark_non_imrad(section, "non_imrad_heading_excluded")
        return False

    if is_code_block(_section_text(section)):
        section.setdefault("imrad_source", "none")
        section.setdefault("imrad_confidence", 0.0)
        section.setdefault("imrad_reason", "code_block_excluded")
        return False

    current = section.get("imrad")
    if current and not allow_override:
        section.setdefault("imrad_source", "existing")
        section.setdefault("imrad_confidence", 0.70)
        section.setdefault("imrad_reason", "existing_label_from_parser")
        return False

    section["imrad"] = label
    section["imrad_source"] = source
    section["imrad_confidence"] = round(float(confidence), 3)
    section["imrad_reason"] = reason
    return True


def _add_secondary(section: dict, secondary_label: str, source: str, confidence: float, reason: str) -> None:
    if not secondary_label:
        return
    section["imrad_secondary"] = secondary_label
    section["imrad_secondary_source"] = source
    section["imrad_secondary_confidence"] = round(float(confidence), 3)
    section["imrad_secondary_reason"] = reason


def _first_index(sections: list[dict], label: str) -> int | None:
    for i, s in enumerate(sections):
        if s.get("imrad") == label or s.get("imrad_secondary") == label:
            return i
    return None


def _missing_categories(sections: list[dict]) -> set[str]:
    found = set()
    for s in sections:
        if is_related_work(s.get("heading", "")) or s.get("section_role") == "related_work":
            continue
        if s.get("imrad") in IMRAD_CATEGORIES:
            found.add(s["imrad"])
        if s.get("imrad_secondary") in IMRAD_CATEGORIES:
            found.add(s["imrad_secondary"])
    return set(IMRAD_CATEGORIES) - found


def _best_candidate_by_content(
    sections: list[dict],
    category: str,
    candidate_indices: list[int],
    min_score: float,
) -> tuple[int | None, float]:
    best_idx = None
    best_score = 0.0

    for idx in candidate_indices:
        s = sections[idx]
        if is_non_imrad_heading(s.get("heading", "")):
            continue
        if is_related_work(s.get("heading", "")):
            continue
        if s.get("section_role") == "related_work":
            continue
        if s.get("imrad") in IMRAD_CATEGORIES and not _can_add_or_override_low_conf(s):
            continue
        text = _section_text(s)
        if len(text.strip()) < 250:
            continue
        if is_code_block(text):
            continue

        score = _content_score(text, category)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None or best_score < min_score:
        return None, best_score
    return best_idx, best_score


_RESULTS_BOUNDARY_RE = re.compile(r"(?:^|\n|\s)(?:3|4)\.\s*Results?\s*$", re.IGNORECASE)
_DISCUSSION_BOUNDARY_RE = re.compile(r"(?:^|\n|\s)(?:4|5)\.\s*(?:Discussion|Conclusions?)\s*$", re.IGNORECASE)


def _text_sets_next_boundary(section: dict) -> str | None:
    tail = (_section_text(section) or "")[-300:]
    if _RESULTS_BOUNDARY_RE.search(tail):
        return "results"
    if _DISCUSSION_BOUNDARY_RE.search(tail):
        return "discussion"
    return None


def _can_add_or_override_low_conf(section: dict) -> bool:
    src = section.get("imrad_source", "")
    return (
        not section.get("imrad")
        or src in {"none", "unknown", "unclassified", "propagation", "content_position_fallback"}
        or float(section.get("imrad_confidence", 0.0) or 0.0) < 0.6
    )



_SECTION_MAJOR_NUM_RE = re.compile(r"^\s*(\d+)(?:\.|\s)")
_SECTION_DECIMAL_NUM_RE = re.compile(r"^\s*(\d+)\.(\d+)")


def _major_num(section: dict) -> int | None:
    h = section.get("heading", "") or ""
    m = _SECTION_MAJOR_NUM_RE.match(h)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _apply_numbered_parent_inheritance(sections: list[dict]) -> int:
    """Conservative IMRaD inheritance for numbered subsections.

    If a top-level numbered section (e.g. 2 Methods) is labeled methods, then
    2.1/2.2 subsections can inherit methods. This avoids broad propagation and
    prevents Related Work from becoming Methods.
    """
    parent_by_major: dict[int, str] = {}
    for s in sections:
        h = s.get("heading", "") or ""
        m = _SECTION_MAJOR_NUM_RE.match(h)
        if not m:
            continue
        if _SECTION_DECIMAL_NUM_RE.match(h):
            continue
        label = s.get("imrad")
        if label in IMRAD_CATEGORIES and not is_non_imrad_heading(h):
            parent_by_major[int(m.group(1))] = label

    changed = 0
    for s in sections:
        h = s.get("heading", "") or ""
        m = _SECTION_DECIMAL_NUM_RE.match(h)
        if not m:
            continue
        if is_non_imrad_heading(h) or is_related_work(h):
            _mark_non_imrad(s, "non_imrad_numbered_heading_excluded")
            continue
        major = int(m.group(1))
        parent_label = parent_by_major.get(major)
        if parent_label and _can_add_or_override_low_conf(s):
            if _assign_label(s, parent_label, "numbered_parent_inheritance", 0.78,
                             f"inherited_from_numbered_parent_{major}", allow_override=True):
                changed += 1
    return changed

def _apply_implicit_heading_boundaries(sections: list[dict]) -> int:
    changed = 0
    pending = None
    for i, s in enumerate(sections):
        if pending and _can_add_or_override_low_conf(s) and not is_non_imrad_heading(s.get("heading", "")):
            if _assign_label(s, pending, "implicit_numbered_heading", 0.72, f"inherited_after_embedded_{pending}_heading", allow_override=True):
                changed += 1
        boundary = _text_sets_next_boundary(s)
        if boundary:
            pending = boundary
        elif s.get("imrad") in {"discussion"}:
            pending = "discussion"
    return changed


def enrich_imrad_sections(sections: list[dict]) -> dict:
    report = {
        "version": "v8.1",
        "heading_classified": 0,
        "fallback_classified": 0,
        "secondary_labels": 0,
        "source_counts": {},
        "found": [],
        "missing": [],
    }

    if not sections:
        report["missing"] = list(IMRAD_CATEGORIES)
        return report

    excluded_count = 0
    for s in sections:
        if is_non_imrad_heading(s.get("heading", "")):
            _mark_non_imrad(s, "non_imrad_heading_excluded")
            excluded_count += 1

    for s in sections:
        detail = classify_imrad_detailed(s.get("heading", ""))

        if detail["label"]:
            changed = _assign_label(
                s,
                detail["label"],
                detail["source"],
                detail["confidence"],
                detail["reason"],
                allow_override=False,
            )
            if changed:
                report["heading_classified"] += 1

        if detail.get("secondary_label"):
            _add_secondary(
                s,
                detail["secondary_label"],
                detail["source"],
                detail["confidence"],
                detail["reason"],
            )
            report["secondary_labels"] += 1

        if not s.get("imrad"):
            s.setdefault("imrad_source", "none")
            s.setdefault("imrad_confidence", 0.0)
            s.setdefault("imrad_reason", "unclassified")

    report["fallback_classified"] += _apply_implicit_heading_boundaries(sections)
    report["fallback_classified"] += _apply_numbered_parent_inheritance(sections)

    # v6: Apply propagation with depth limit.
    # Walk sections in order. When a section has an explicit IMRaD label,
    # reset the propagation counter. Unlabeled sections inherit from the
    # current label, but only up to MAX_PROPAGATION_DEPTH consecutive times.
    current_propagation_label = None
    propagation_count = 0

    for s in sections:
        if s.get("imrad") in IMRAD_CATEGORIES:
            src = s.get("imrad_source", "")
            # Only labels from heading classification can start propagation.
            if src in ("heading_exact", "heading_synonym", "combined_section",
                       "heading_characterization"):
                if category_propagates(s["imrad"]):
                    current_propagation_label = s["imrad"]
                    propagation_count = 0
                else:
                    current_propagation_label = None
                    propagation_count = 0
            continue

        # Section has no IMRaD label (or was set by parser but not heading).
        existing = s.get("imrad")
        if existing in IMRAD_CATEGORIES:
            # Already labeled (e.g. from tei_parser). Check if it was
            # from propagation in the parser (imrad_source == "existing").
            src = s.get("imrad_source", "")
            if src == "existing":
                # This was propagated by tei_parser. v6: re-evaluate.
                # If we have a characterization heading, override to results.
                if is_characterization_heading(s.get("heading", "")):
                    s["imrad"] = "results"
                    s["imrad_source"] = "heading_characterization"
                    s["imrad_confidence"] = 0.80
                    s["imrad_reason"] = "v6_characterization_override"
                    current_propagation_label = "results"
                    propagation_count = 0
            continue

        if current_propagation_label and propagation_count < MAX_PROPAGATION_DEPTH:
            # Before propagating, check if this section's heading
            # suggests a DIFFERENT category.
            heading = s.get("heading", "")
            if heading:
                detail = classify_imrad_detailed(heading)
                if detail["label"] and detail["label"] != current_propagation_label:
                    # Heading suggests different category; use that instead.
                    _assign_label(s, detail["label"], detail["source"],
                                  detail["confidence"], detail["reason"],
                                  allow_override=True)
                    if category_propagates(detail["label"]):
                        current_propagation_label = detail["label"]
                        propagation_count = 0
                    else:
                        current_propagation_label = None
                    continue

            _assign_label(s, current_propagation_label, "propagation",
                          0.50, f"propagated_from_{current_propagation_label}_depth_{propagation_count+1}",
                          allow_override=True)
            propagation_count += 1
        else:
            # Exhausted propagation budget or no current label.
            current_propagation_label = None
            propagation_count = 0

    n = len(sections)
    missing = _missing_categories(sections)

    intro_idx = _first_index(sections, "introduction")
    methods_idx = _first_index(sections, "methods")
    results_idx = _first_index(sections, "results")
    discussion_idx = _first_index(sections, "discussion")

    if "introduction" in missing:
        candidate_indices = list(range(0, min(2, n)))
        idx, score = _best_candidate_by_content(sections, "introduction", candidate_indices, 1.5)
        if idx is not None:
            if _assign_label(sections[idx], "introduction", "content_position_fallback", 0.62, f"intro_content_score={score:.2f}", allow_override=True):
                report["fallback_classified"] += 1

    intro_idx = _first_index(sections, "introduction")
    missing = _missing_categories(sections)

    if "methods" in missing:
        start = (intro_idx + 1) if intro_idx is not None else 0
        end_candidates = [x for x in [results_idx, discussion_idx] if x is not None and x > start]
        end = min(end_candidates) if end_candidates else min(n, start + 5)
        candidate_indices = list(range(start, end))
        idx, score = _best_candidate_by_content(sections, "methods", candidate_indices, 2.5)
        if idx is not None:
            if _assign_label(sections[idx], "methods", "content_position_fallback", 0.65, f"methods_content_score={score:.2f}", allow_override=True):
                report["fallback_classified"] += 1

    methods_idx = _first_index(sections, "methods")
    missing = _missing_categories(sections)

    if "results" in missing:
        start_candidates = [x for x in [methods_idx, intro_idx] if x is not None]
        start = max(start_candidates) + 1 if start_candidates else 0
        end = discussion_idx if discussion_idx is not None and discussion_idx > start else n
        candidate_indices = list(range(start, end))
        idx, score = _best_candidate_by_content(sections, "results", candidate_indices, 1.2)
        if idx is not None:
            if _assign_label(sections[idx], "results", "content_position_fallback", 0.63, f"results_content_score={score:.2f}", allow_override=True):
                report["fallback_classified"] += 1

    results_idx = _first_index(sections, "results")
    missing = _missing_categories(sections)

    if "discussion" in missing:
        candidate_indices = list(range(max(0, n - 3), n))
        idx, score = _best_candidate_by_content(sections, "discussion", candidate_indices, 1.5)
        if idx is not None:
            if _assign_label(sections[idx], "discussion", "content_position_fallback", 0.60, f"discussion_content_score={score:.2f}", allow_override=True):
                report["fallback_classified"] += 1

    found = []
    source_counts = Counter()
    for s in sections:
        if is_related_work(s.get("heading", "")) or s.get("section_role") == "related_work":
            continue
        if s.get("imrad") in IMRAD_CATEGORIES:
            found.append(s["imrad"])
            source_counts[s.get("imrad_source", "unknown")] += 1
        if s.get("imrad_secondary") in IMRAD_CATEGORIES:
            found.append(s["imrad_secondary"])
            source_counts[s.get("imrad_secondary_source", "unknown")] += 1

    found_unique = [c for c in IMRAD_CATEGORIES if c in set(found)]
    missing_final = [c for c in IMRAD_CATEGORIES if c not in set(found)]

    report["found"] = found_unique
    report["missing"] = missing_final
    report["source_counts"] = dict(source_counts)
    report["complete"] = len(missing_final) == 0
    report["excluded_non_imrad"] = excluded_count

    return report


def infer_methods_from_sections(sections: list[dict]) -> dict:
    before_methods = [
        i for i, s in enumerate(sections)
        if s.get("imrad") == "methods"
    ]

    enrichment = enrich_imrad_sections(sections)

    after_methods = [
        i for i, s in enumerate(sections)
        if s.get("imrad") == "methods"
    ]

    report = {
        "inferred": False,
        "section_index": None,
        "section_heading": None,
        "score": 0.0,
        "candidates_considered": len(sections or []),
        "v4_enrichment": enrichment,
    }

    if after_methods:
        idx = after_methods[0]
        s = sections[idx]
        source = s.get("imrad_source", "")
        inferred = idx not in before_methods or source in {
            "heading_synonym",
            "content_position_fallback",
            "combined_section",
            "heading_characterization",
            "propagation",
        }
        report.update({
            "inferred": bool(inferred),
            "section_index": idx,
            "section_heading": s.get("heading", ""),
            "score": float(s.get("imrad_confidence", 0.0) or 0.0),
            "source": source,
            "reason": s.get("imrad_reason"),
        })

    return report


if __name__ == "__main__":
    tests = [
        ("Introduction", "introduction"),
        ("Background", "introduction"),
        ("Related Work", None),
        ("Previous studies", None),
        ("Theoretical background", None),
        ("Materials and Methods", "methods"),
        ("System Description", None),
        ("Results", "results"),
        ("Experimental Results", "results"),
        ("Results and Discussion", "results"),
        ("Discussion", "discussion"),
        ("Conclusion", "discussion"),
        ("Acknowledgments", None),
        # v6 tests
        ("XRD", "results"),
        ("Cyclic voltammetry", "results"),
        ("Differential pulse voltammetry (DPV)", "results"),
        ("Sensing mechanism for heavy metal ion detection", "results"),
        ("Interference/selectivity with other heavy metal ions", "results"),
        ("Reproducibility and stability of modified electrode", "results"),
        ("Practical implication of electrochemical sensor in real water samples", "results"),
        ("Material characterization of sensor XRD", "results"),
        ("Structural and morphological chemistry", "results"),
        ("XPS", "results"),
        ("Discussion, implications, and limitations", "discussion"),
    ]
    failed = 0
    for heading, expected in tests:
        got = classify_imrad(heading)
        ok = got == expected
        print(f"{'OK' if ok else 'FAIL'}: {heading!r} -> {got!r}, expected {expected!r}")
        failed += 0 if ok else 1

    code_text = "\n".join([
        "import RPi.GPIO as GPIO", "GPIO.setmode(GPIO.BCM)", "def wetting(delay, steps):",
        "    for i in range(steps):", "        GPIO.output(STEP, GPIO.HIGH)",
        "        sleep(delay)", "    return my_Sr", "# Constants", "GAIN = 16",
    ])
    assert is_code_block(code_text), "code block not detected"
    assert not is_code_block("We measured suction with a sensor. The results show a clear trend. " * 5)
    print("OK code-block guard")
    if failed:
        raise SystemExit(1)