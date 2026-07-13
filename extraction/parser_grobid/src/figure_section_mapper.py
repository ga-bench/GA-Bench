"""
Figure-section mapper (v8).

v7 changes vs v6:
    1. Effective-IMRaD resolution. A figure mention found in a non-IMRaD
       subsection heading (e.g. "Database of locations of cocoa") used to be
       bucketed under that raw heading. Now each section is resolved to its
       effective IMRaD label by forward-filling the nearest preceding IMRaD
       label (with a light heading guess for combined headings the parser
       missed, and a reset at back-matter). Figure mentions are bucketed under
       that effective IMRaD label, so assigned_section is one of
       introduction / methods / results / discussion whenever possible.
    2. Page-proximity fallback. Figures whose in-text mention was lost during
       GROBID segmentation (truncated paragraphs, appendix figures) used to end
       up "unassigned". When a figure has zero mentions, it is now assigned to
       the IMRaD section whose start page is the latest at or before the
       figure's page. Requires sections to carry `page` (tei_parser v7.1).
    3. The mapper now sets `assigned_section` and `assignment_method` directly
       on each figure ("mention_based" / "page_proximity" / "unassigned"); the
       output writer honours these instead of recomputing from mentions only.

All label parsing / mention counting (label_utils-based id assignment, mention
counting via anchor finder, range/list expansion, dedup with letter suffix) is
unchanged from v6.
"""

from __future__ import annotations
import re

from label_utils import parse_label, find_caption_anchors

try:
    from imrad_classifier import classify_imrad as _classify_imrad, is_non_imrad_heading as _is_non_imrad_heading
except Exception:  # pragma: no cover - defensive
    def _classify_imrad(_h):
        return None

    def _is_non_imrad_heading(_h):
        return False


_IMRAD = {"introduction", "methods", "results", "discussion"}

# Headings that should stop IMRaD forward-fill (figures after these don't
# inherit the last body label).
_BACKMATTER_RE = re.compile(
    r"\b(?:references?|bibliography|appendix|appendices|acknowledge?ments?|"
    r"supplementary|supporting\s+information|funding|declarations?|"
    r"author(?:s'?)?\s+contributions?|data\s+availability|"
    r"competing\s+interests?|conflicts?\s+of\s+interest)\b",
    re.IGNORECASE,
)

# Light fallback for combined / unusual headings the main classifier misses
# (e.g. "Methods and data", "Data and methods"), used only to drive forward-fill.
_LOCAL_GUESS = [
    (re.compile(r"\b(?:methods?\s+and\s+data|data\s+and\s+methods?|"
                r"materials?\s+and\s+methods?|methodolog|experimental|"
                r"\bprocedure)\b", re.IGNORECASE), "methods"),
    (re.compile(r"\bresults?\b|\bfindings?\b", re.IGNORECASE), "results"),
    (re.compile(r"\bdiscussion\b|\bconclusion", re.IGNORECASE), "discussion"),
    (re.compile(r"\bintroduction\b|\bbackground\b", re.IGNORECASE), "introduction"),
]


def _guess_imrad(heading: str) -> str | None:
    if not heading:
        return None
    lab = _classify_imrad(heading)
    if lab in _IMRAD:
        return lab
    for rx, lab in _LOCAL_GUESS:
        if rx.search(heading):
            return lab
    return None


def _section_key(section: dict) -> str:
    if section.get("imrad"):
        return section["imrad"]
    heading = (section.get("heading") or "").strip().lower()
    heading = re.sub(
        r"^\s*(?:\d+(?:\.\d+)*\.?|[ivxlcdm]+\.?|[a-z]\.|\([a-z0-9]+\))\s+",
        "",
        heading,
    )
    heading = re.sub(r"\s+", " ", heading).strip()
    return heading if heading else "_unlabeled"


def _effective_imrad_per_section(sections: list[dict]) -> list[str | None]:
    """Forward-fill the nearest preceding IMRaD label so subsections inherit
    their parent's bucket. Resets at back-matter so post-body figures don't
    inherit the last body label."""
    eff: list[str | None] = []
    last: str | None = None
    for s in sections:
        heading = s.get("heading", "") or ""
        lab = s.get("imrad")
        if lab not in _IMRAD:
            if heading and (_is_non_imrad_heading(heading) or _BACKMATTER_RE.search(heading)):
                last = None
                eff.append(None)
                continue
            lab = _guess_imrad(heading)
        if lab in _IMRAD:
            last = lab
            eff.append(lab)
        else:
            eff.append(last)
    return eff


def _effective_keys(sections: list[dict], eff_imrad: list[str | None]) -> list[str]:
    keys = []
    for s, ei in zip(sections, eff_imrad):
        if ei in _IMRAD:
            keys.append(ei)
        else:
            keys.append(_section_key(s))
    return keys


def _dominant(mentions: dict) -> str | None:
    if not mentions:
        return None
    return sorted(mentions.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _assign_by_page(fig_page, sections: list[dict], eff_imrad: list[str | None]) -> str | None:
    """Latest IMRaD section that starts at or before the figure's page.
    Returns None if no section carries a usable `page` (e.g. GROBID emitted no
    <head> coords) — the caller then falls back to _assign_by_figure_pages."""
    if fig_page is None:
        return None
    best = None
    best_start = -1
    for i, s in enumerate(sections):
        sp = s.get("page")
        k = eff_imrad[i]
        if sp is None or k not in _IMRAD:
            continue
        if sp <= fig_page and sp > best_start:
            best_start = sp
            best = k
    return best


def _assign_by_figure_pages(fig_page, page_anchors: list[tuple[int, str]]) -> str | None:
    """Fallback used when sections carry no page info. `page_anchors` is a list
    of (page, imrad_bucket) built from the figures that DID get a mention-based
    assignment. The unmentioned figure inherits the bucket of the nearest such
    figure by page number (ties prefer the later page / results-side)."""
    if fig_page is None or not page_anchors:
        return None
    best = None
    best_dist = None
    for pg, bucket in page_anchors:
        if bucket not in _IMRAD:
            continue
        d = abs(pg - fig_page)
        if best_dist is None or d < best_dist or (d == best_dist and pg >= fig_page):
            best_dist = d
            best = bucket
    return best


def _strip_caption_text(section_text: str, captions: list[str]) -> str:
    cleaned = section_text
    for cap in captions:
        if not cap:
            continue
        cap_head = cap[:80].strip()
        if len(cap_head) > 20 and cap_head in cleaned:
            cleaned = cleaned.replace(cap_head, " ")
    return cleaned


_LIST_SEP = r"\s*(?:(?:,|;)\s*(?:and\s+)?|\s+and\s+|\s*&\s*)"
_RANGE_RES = {
    "figure": re.compile(r"\b(?:Fig(?:s|ures?)?)\.?\s*(\d+)\s*(?:-|\u2013|\u2014|to)\s*(\d+)(?![0-9])", re.IGNORECASE),
    "table":  re.compile(r"\b(?:Tab(?:les?)?)\.?\s*(\d+)\s*(?:-|\u2013|\u2014|to)\s*(\d+)(?![0-9])", re.IGNORECASE),
}
_LIST_RES = {
    "figure": re.compile(r"\b(?:Fig(?:s|ures?)?)\.?\s*(\d+(?:" + _LIST_SEP + r"\d+)+)(?![0-9])", re.IGNORECASE),
    "table":  re.compile(r"\b(?:Tab(?:les?)?)\.?\s*(\d+(?:" + _LIST_SEP + r"\d+)+)(?![0-9])", re.IGNORECASE),
}


def _count_mentions_for_key(section_text: str, target_key: str, kind: str) -> int:
    if not section_text or not target_key:
        return 0

    count = 0
    for rec in find_caption_anchors(section_text):
        if rec["kind"] == kind and rec["key"] == target_key:
            count += 1

    parts = target_key.split(":")
    if len(parts) == 3 and parts[1] == "":
        try:
            target_n = int(parts[2])
        except ValueError:
            return count
        rng = _RANGE_RES.get(kind)
        if rng is not None:
            for m in rng.finditer(section_text):
                try:
                    lo, hi = int(m.group(1)), int(m.group(2))
                except ValueError:
                    continue
                if lo <= target_n <= hi:
                    count += 1
        lst = _LIST_RES.get(kind)
        if lst is not None:
            for m in lst.finditer(section_text):
                nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
                if target_n in nums:
                    count += 1
    return count


def _assign_figure_id(fig: dict, fallback_idx: int) -> str:
    # Prefer an id already derived from the real caption label. This preserves
    # supplementary/appendix labels such as fig_S8 and fig_A_1.
    hinted = fig.get("figure_id_hint")
    if hinted:
        return hinted
    rec = parse_label(fig.get("label", "") or fig.get("caption", "") or "")
    if rec:
        return rec["id"]
    fig_type = fig.get("type", "figure")
    prefix = {"figure": "fig", "table": "table", "scheme": "scheme"}.get(fig_type, "fig")
    return f"{prefix}_unlabeled_{fallback_idx:02d}"


def map_figures_to_sections(
    figures: list[dict],
    sections: list[dict],
) -> list[dict]:
    all_captions = [f.get("caption", "") for f in figures]
    cleaned_section_texts = []
    for s in sections:
        body = s.get("text_no_tables") or s.get("text") or ""
        cleaned_section_texts.append(_strip_caption_text(body, all_captions))

    # v7: resolve each section to its effective IMRaD bucket.
    eff_imrad = _effective_imrad_per_section(sections)
    eff_keys = _effective_keys(sections, eff_imrad)

    assigned_ids: list[str] = []
    seen_ids: dict[str, int] = {}
    for i, fig in enumerate(figures):
        base_id = _assign_figure_id(fig, fallback_idx=i + 1)
        if base_id in seen_ids:
            seen_ids[base_id] += 1
            suffix_letter = chr(ord("a") + seen_ids[base_id] - 1)
            unique_id = f"{base_id}{suffix_letter}"
        else:
            seen_ids[base_id] = 1
            unique_id = base_id
        assigned_ids.append(unique_id)

    # Pass 1: mention-based assignment. Also collect (page, bucket) anchors from
    # every figure that got a confident mention-based bucket — these let us place
    # mention-less figures even when sections carry no page info (GROBID emitted
    # no <head> coords, so section["page"] is None everywhere).
    enriched = []
    page_anchors: list[tuple[int, str]] = []
    for fig, figure_id in zip(figures, assigned_ids):
        fig_type = fig.get("type", "figure")
        rec = parse_label(fig.get("label", "") or "")
        target_key = rec["key"] if rec else ""

        mentions: dict = {}
        if target_key:
            for cleaned_text, key in zip(cleaned_section_texts, eff_keys):
                n = _count_mentions_for_key(cleaned_text, target_key, fig_type)
                if n > 0:
                    mentions[key] = mentions.get(key, 0) + n

        new_fig = dict(fig)
        new_fig["figure_id"] = figure_id
        if "caption_source" not in new_fig:
            new_fig["caption_source"] = "pdffigures2"
        new_fig["mentions_by_section"] = mentions
        new_fig["possible_sections"] = list(mentions.keys())
        new_fig["assignment_evidence"] = {
            "mention_counts": mentions,
            "page": fig.get("page"),
            "page_proximity_candidate": None,
            "all_possible_sections_source": "mentions" if mentions else "none",
        }

        if mentions:
            assigned_section = _dominant(mentions)
            new_fig["assigned_section"] = assigned_section
            new_fig["assignment_method"] = "mention_based"
            pg = fig.get("page")
            if isinstance(pg, int) and assigned_section in _IMRAD:
                page_anchors.append((pg, assigned_section))
        else:
            new_fig["assigned_section"] = None
            new_fig["assignment_method"] = "unassigned"
        enriched.append(new_fig)

    # Pass 2: resolve the still-unassigned figures. Prefer section-page proximity
    # (needs section["page"]); if that yields nothing, fall back to the nearest
    # mention-anchored figure by page number.
    for new_fig in enriched:
        if new_fig["assignment_method"] != "unassigned":
            continue
        fig_page = new_fig.get("page")
        key = _assign_by_page(fig_page, sections, eff_imrad)
        if key is not None:
            new_fig["assigned_section"] = key
            new_fig["assignment_method"] = "page_proximity"
            new_fig.setdefault("possible_sections", [])
            if key not in new_fig["possible_sections"]:
                new_fig["possible_sections"].append(key)
            new_fig.setdefault("assignment_evidence", {})["page_proximity_candidate"] = key
            new_fig["assignment_evidence"]["all_possible_sections_source"] = "page_proximity"
            continue
        key = _assign_by_figure_pages(fig_page, page_anchors)
        if key is not None:
            new_fig["assigned_section"] = key
            new_fig["assignment_method"] = "page_proximity"
            new_fig.setdefault("possible_sections", [])
            if key not in new_fig["possible_sections"]:
                new_fig["possible_sections"].append(key)
            new_fig.setdefault("assignment_evidence", {})["page_proximity_candidate"] = key
            new_fig["assignment_evidence"]["all_possible_sections_source"] = "page_proximity"

    return enriched


if __name__ == "__main__":
    sections = [
        {"heading": "Introduction", "imrad": "introduction", "page": 1,
         "text_no_tables": "We refer to Figure 1 for the design."},
        {"heading": "Methods and data", "imrad": None, "page": 2,
         "text_no_tables": "Our pipeline is described here."},
        {"heading": "Database of locations of cocoa", "imrad": None, "page": 3,
         "text_no_tables": "FIGURE 2 shows the validated occurrence pixels."},
        {"heading": "Results", "imrad": "results", "page": 5,
         "text_no_tables": "As shown in Fig. 3 the effect was clear."},
        {"heading": "Discussion", "imrad": "discussion", "page": 8,
         "text_no_tables": "These results suggest a clear trend."},
    ]
    figs = [
        {"label": "Figure 1", "type": "figure", "caption": "FIGURE 1", "page": 1},
        {"label": "Figure 2", "type": "figure", "caption": "FIGURE 2 ...", "page": 3},
        {"label": "Figure 3", "type": "figure", "caption": "FIGURE 3 ...", "page": 5},
        # No mention anywhere; sits on page 8 -> should map to discussion by page.
        {"label": "Figure 4", "type": "figure", "caption": "FIGURE 4 ...", "page": 8},
    ]
    for f in map_figures_to_sections(figs, sections):
        print(f"  {f['figure_id']:8} <- {f['label']:10} "
              f"assigned={str(f['assigned_section']):14} "
              f"method={f['assignment_method']:14} mentions={f['mentions_by_section']}")