"""
GROBID TEI XML parser (v8).

v7.2 changes vs v7.1:
    1. Equations: each <formula> is replaced inline by a stable placeholder
       "[[EQN:<n>]]" (n = document-global counter) instead of its garbled
       text. The formula's `coords` and raw text are recorded per-section under
       `equations`: [{id, coords, raw}]. The output writer OCRs each coord to
       LaTeX and substitutes it back into the placeholder. Formulas appear in
       BOTH the with-tables and no-tables variants (only tables are stripped).
       Inline math inside <p> (no coords) is left as-is.

v7.1 changes vs v7: section `page`, table_count/table_unrecovered_count,
    unrecovered-table placeholder (unchanged here).

v7 changes vs v6: reference extraction; text/text_no_tables split.
"""

from __future__ import annotations
import re
from pathlib import Path

from lxml import etree

from imrad_classifier import classify_imrad, category_propagates, is_code_block, is_non_imrad_heading
from pua_remap import remap_pua_glyphs
from text_cleanup import clean_text_artifacts


TEI_NS = "http://www.tei-c.org/ns/1.0"
NSMAP = {"tei": TEI_NS}


_IMRAD_HEADING_VOCAB = re.compile(
    r"^(?:introduction|intro|background|motivation|overview|objectives?|"
    r"aims?|purpose|methods?|methodology|methodologies|materials?|"
    r"experimental|approach|approaches|procedure|procedures|results?|"
    r"findings?|observations?|discussions?|conclusions?|summary|"
    r"preliminaries|related|references|appendix|appendices|"
    r"acknowledgments?|acknowledgements?|abstract|nomenclature|notation)$",
    re.IGNORECASE,
)

SHORT_HEADING_WHITELIST = {
    "abstract", "intro", "methods", "results", "discussion", "conclusion",
    "references", "appendix", "acknowledgments", "acknowledgements",
}

SUBSECTION_MARKER_PATTERNS = [
    re.compile(r"^[A-Z]\.\s+", re.IGNORECASE),
    re.compile(r"^Case\s+\d", re.IGNORECASE),
    re.compile(r"^Step\s+\d", re.IGNORECASE),
    re.compile(r"^Example\s+\d", re.IGNORECASE),
]

CAPTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:fig(?:ure)?|tab(?:le)?|algorithm|alg|scheme|eq(?:uation)?)\.?\s+\d+",
    re.IGNORECASE,
)

APPENDIX_HEADING_PATTERN = re.compile(
    r"^\s*(?:appendix|appendices|supplementary(?:\s+(?:material|information))?|"
    r"supporting\s+information)\b",
    re.IGNORECASE,
)


def _is_appendix_heading(heading: str) -> bool:
    if not heading:
        return False
    return APPENDIX_HEADING_PATTERN.match(heading.strip()) is not None


def _is_caption_heading(heading: str) -> bool:
    if not heading:
        return False
    return CAPTION_HEADING_PATTERN.match(heading.strip()) is not None


def _is_likely_figure_label(heading: str) -> bool:
    h = heading.strip()
    if not h:
        return False
    if len(h) > 15:
        return False
    if ":" in h or any(c.isdigit() for c in h):
        return False
    if h.lower() in SHORT_HEADING_WHITELIST:
        return False
    if _IMRAD_HEADING_VOCAB.match(h):
        return False
    if " " in h:
        return False
    if h[0].isupper() and h[1:].islower():
        return True
    return False


def _is_subsection_marker(heading: str) -> bool:
    h = heading.strip()
    for pat in SUBSECTION_MARKER_PATTERNS:
        if pat.match(h):
            return True
    return False


_MATH_HEADING_SYMBOL_RE = re.compile(r"[=∫∑∏√≈≠≤≥<>±*/^]|\b(?:lambda|sigma|psi|theta|eta|mu|nu|kappa)\b|[λσψθμηνκΩ∂]", re.IGNORECASE)

def _is_math_like_heading(heading: str) -> bool:
    h = (heading or "").strip()
    if not h:
        return False
    if _MATH_HEADING_SYMBOL_RE.search(h):
        return True
    # Reject very formula-like headings made mostly of short math tokens.
    tokens = re.findall(r"[A-Za-z0-9_]+", h)
    if len(tokens) >= 3 and len(h) < 90:
        short = sum(1 for t in tokens if len(t) <= 3)
        if short / max(1, len(tokens)) > 0.65 and not _IMRAD_HEADING_VOCAB.match(h):
            return True
    return False

def _heading_passes_filter(heading: str, has_tei_head_level: bool) -> bool:
    h = (heading or "").strip()
    if not h:
        return False
    if _is_caption_heading(h):
        return False
    if _is_math_like_heading(h):
        return False
    if has_tei_head_level:
        return True
    if len(h) < 3 and h.lower() not in SHORT_HEADING_WHITELIST:
        return False
    if _is_likely_figure_label(h):
        return False
    return True


CAPTION_SOFT_CAP_CHARS = 400

CAPTION_TRIM_MARKERS = [
    re.compile(r"\bAs shown in\s+(?:Fig|Figure|Table)\b", re.IGNORECASE),
    re.compile(r"\bFrom\s+(?:Fig|Figure|Table)\.?\s+\d", re.IGNORECASE),
    re.compile(r"\bIn\s+(?:Fig|Figure|Table)\.?\s+\d", re.IGNORECASE),
    re.compile(r"\b(?:Fig|Figure|Table|Scheme|Algorithm)\.?\s+\d+\s+[A-Z]"),
]


def _trim_caption(caption: str) -> str:
    if not caption:
        return caption
    c = caption.strip()
    if len(c) <= CAPTION_SOFT_CAP_CHARS:
        return c
    earliest = len(c)
    for pat in CAPTION_TRIM_MARKERS:
        m = pat.search(c, pos=20)
        if m and m.start() < earliest:
            earliest = m.start()
    if earliest < len(c):
        return c[:earliest].rstrip(" .,;:")
    cut = c.rfind(". ", 0, CAPTION_SOFT_CAP_CHARS)
    if cut < CAPTION_SOFT_CAP_CHARS // 2:
        cut = CAPTION_SOFT_CAP_CHARS
    return c[:cut].rstrip(" .,;:") + "."


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


_SPACED_TAGGED = r"T\s+[Aa]\s+[Gg]\s+[Gg]\s+[Ee]\s+[Dd]"


def _spaced_opt(word: str) -> str:
    return r"\s*".join(re.escape(c) for c in word)


_TAG_NAME = (
    r"(?:" + _spaced_opt("End") + r"|" + _spaced_opt("Start") + r"|H\s*\d|P\b|"
    + _spaced_opt("Table") + r"|" + _spaced_opt("Figure") + r"|"
    + _spaced_opt("Box") + r"|" + _spaced_opt("List") + r")"
)

_TAGGED_MARKER_RE = re.compile(
    r"(?:" + _SPACED_TAGGED + r")\s*(?:" + _TAG_NAME + r")?"
    r"|Tagged\s*" + _TAG_NAME,
    re.IGNORECASE,
)


def _strip_tagged_markers(text: str) -> str:
    if not text:
        return text
    out = _TAGGED_MARKER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", out).strip()


def _text_of(element) -> str:
    if element is None:
        return ""
    text = " ".join(element.itertext())
    text = re.sub(r"\s+", " ", text).strip()
    text = remap_pua_glyphs(text)
    text = _strip_tagged_markers(text)
    return clean_text_artifacts(text)


def _coords_first_page(coords) -> int | None:
    if not coords:
        return None
    try:
        first_region = str(coords).split(";")[0]
        return int(first_region.split(",")[0])
    except (ValueError, IndexError, AttributeError):
        return None


def _section_page(div_el, head_el) -> int | None:
    if head_el is not None:
        p = _coords_first_page(head_el.get("coords"))
        if p is not None:
            return p
    for child in div_el:
        p = _coords_first_page(child.get("coords"))
        if p is not None:
            return p
        graphic = child.find("./tei:graphic", NSMAP)
        if graphic is not None:
            p = _coords_first_page(graphic.get("coords"))
            if p is not None:
                return p
    return None


def _extract_title(root) -> str:
    title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title[@type='main']", NSMAP)
    if title_el is None:
        title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", NSMAP)
    return _text_of(title_el)


def _extract_abstract(root) -> str:
    abstract_el = root.find(".//tei:profileDesc/tei:abstract", NSMAP)
    if abstract_el is None:
        return ""
    paragraphs = []
    for p in abstract_el.findall(".//tei:p", NSMAP):
        text = _text_of(p)
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _table_to_markdown(table_el):
    rows = table_el.findall(".//tei:row", NSMAP)
    if not rows:
        raw = _text_of(table_el)
        return raw, raw
    grid = []
    for row in rows:
        cells = row.findall("./tei:cell", NSMAP)
        grid.append([_text_of(c) for c in cells])
    if not grid or not grid[0]:
        raw = _text_of(table_el)
        return raw, raw
    ncols = max(len(r) for r in grid)
    grid = [r + [""] * (ncols - len(r)) for r in grid]
    header = "| " + " | ".join(grid[0]) + " |"
    sep = "| " + " | ".join(["---"] * ncols) + " |"
    body_rows = ["| " + " | ".join(r) + " |" for r in grid[1:]]
    markdown = "\n".join([header, sep] + body_rows)
    raw_text = "\n".join("\t".join(r) for r in grid)
    return markdown, raw_text


TABLE_UNRECOVERED_PLACEHOLDER = "[Table: content not recovered]"


def _table_num_from_text(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\bTable\s+((?:[A-Z]\.?)?\d+[A-Za-z]?|[IVXLC]+)\b", text, re.IGNORECASE)
    return m.group(1) if m else None


def _table_label_from_num(num: str | None, fallback_id: str) -> str:
    return f"Table {num}" if num else fallback_id


def _table_placeholder(label: str) -> str:
    return f"[{label} here]"


def _extract_section_content(div_el, eq_state):
    """Returns (text_with_placeholders, text_without_tables, tables, equations).

    In v8, table content is not inserted into body text. A stable placeholder is
    inserted instead, and the table body is stored separately for tables.json.
    Equations remain in-place through placeholders that output_writer resolves
    into readable [Equation N: ...] blocks.
    """
    tables = []
    equations = []
    parts_with = []
    parts_without = []
    eq_state.setdefault("counter", 0)
    eq_state.setdefault("table_counter", 0)

    for child in div_el:
        tag = _strip_ns(child.tag)
        if tag == "head":
            continue
        elif tag == "div":
            continue
        elif tag == "p":
            text = _text_of(child)
            if text:
                parts_with.append(text)
                parts_without.append(text)
        elif tag == "table":
            md, raw = _table_to_markdown(child)
            raw = clean_text_artifacts(raw)
            md = clean_text_artifacts(md)
            eq_state["table_counter"] += 1
            table_id = f"table{eq_state['table_counter']:02d}"
            num = _table_num_from_text(raw or md)
            label = _table_label_from_num(num, table_id)
            placeholder = _table_placeholder(label)
            tables.append({
                "table_id": table_id,
                "num": num,
                "label": label,
                "caption": "",
                "markdown": md,
                "raw_text": raw,
                "source": "grobid_tei",
                "recovered": bool(raw or md),
                "placeholder": placeholder,
            })
            parts_with.append(placeholder)
            parts_without.append(placeholder)
        elif tag == "formula":
            raw = _text_of(child)
            coords = child.get("coords")
            eq_state["counter"] += 1
            eq_id = eq_state["counter"]
            equations.append({"id": eq_id, "coords": coords, "raw": raw})
            placeholder = f"[[EQN:{eq_id}]]"
            parts_with.append(placeholder)
            parts_without.append(placeholder)
        elif tag == "list":
            text = _text_of(child)
            if text:
                parts_with.append(text)
                parts_without.append(text)
        else:
            text = _text_of(child)
            if text:
                parts_with.append(text)
                parts_without.append(text)

    return ("\n\n".join(parts_with), "\n\n".join(parts_without), tables, equations)


def _table_stats(tables: list[dict]) -> tuple[int, int]:
    total = len(tables)
    unrecovered = sum(1 for t in tables if not (t.get("raw_text") or "").strip())
    return total, unrecovered


_LEADING_NUMBERED_INTRO = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+introduction\b", re.IGNORECASE)


def _walk_sections(body_el):
    sections = []
    if body_el is None:
        return sections
    state = {"current_imrad": None, "order": 0, "eq": {"counter": 0, "table_counter": 0}}
    for div in body_el.findall("./tei:div", NSMAP):
        _walk_div_recursive(div, sections, state)
    return sections


def _walk_div_recursive(div, sections, state):
    head_el = div.find("./tei:head", NSMAP)
    heading = _text_of(head_el) if head_el is not None else ""
    has_level = (head_el is not None) and (head_el.get("level") is not None)

    is_subsection = _is_subsection_marker(heading)
    is_caption = _is_caption_heading(heading)
    is_appendix = _is_appendix_heading(heading)

    if not heading and not has_level:
        effective_imrad = state["current_imrad"]
        kept_heading = ""
    elif is_subsection:
        effective_imrad = state["current_imrad"]
        kept_heading = heading
    elif is_caption:
        effective_imrad = state["current_imrad"]
        kept_heading = ""
    else:
        if not _heading_passes_filter(heading, has_level):
            for child_div in div.findall("./tei:div", NSMAP):
                _walk_div_recursive(child_div, sections, state)
            return
        kept_heading = heading
        if is_non_imrad_heading(heading):
            effective_imrad = None
            state["current_imrad"] = None
        else:
            own_imrad = classify_imrad(heading)
            if own_imrad is not None:
                effective_imrad = own_imrad
                if category_propagates(own_imrad):
                    state["current_imrad"] = own_imrad
                else:
                    state["current_imrad"] = None
            else:
                effective_imrad = state["current_imrad"]

    text_with, text_without, tables, equations = _extract_section_content(div, state["eq"])
    section_page = _section_page(div, head_el)
    table_count, table_unrecovered = _table_stats(tables)

    appendix_flag = bool(is_appendix)
    body_is_code = is_code_block(text_without)
    if is_appendix:
        effective_imrad = None
        state["current_imrad"] = None

    if kept_heading or text_with:
        if appendix_flag or body_is_code:
            section = {
                "order": state["order"],
                "heading": kept_heading,
                "imrad": None,
                "text": text_with,
                "text_no_tables": "" if body_is_code else text_without,
                "tables": [] if body_is_code else tables,
                "equations": [] if body_is_code else equations,
                "appendix": True,
                "appendix_text": text_with,
                "is_code_block": bool(body_is_code),
                "page": section_page,
                "table_count": 0 if body_is_code else table_count,
                "table_unrecovered_count": 0 if body_is_code else table_unrecovered,
            }
            sections.append(section)
            state["order"] += 1
        else:
            sections.append({
                "order": state["order"],
                "heading": kept_heading,
                "imrad": effective_imrad,
                "text": text_with,
                "text_no_tables": text_without,
                "tables": tables,
                "equations": equations,
                "appendix": False,
                "page": section_page,
                "table_count": table_count,
                "table_unrecovered_count": table_unrecovered,
            })
            state["order"] += 1

    for child_div in div.findall("./tei:div", NSMAP):
        _walk_div_recursive(child_div, sections, state)

    return sections


_ORPHAN_INTRO_PREFIXES = [
    re.compile(r"^\s*\[\d+\]"),
    re.compile(r"^\s*With\b", re.IGNORECASE),
    re.compile(r"^\s*In recent years", re.IGNORECASE),
    re.compile(r"^\s*Recent(?:ly)?\b", re.IGNORECASE),
    re.compile(r"^\s*The (?:rapid|growing|increasing)", re.IGNORECASE),
    re.compile(r"^\s*Over the (?:past|last)", re.IGNORECASE),
]


def _looks_like_orphan_introduction(section: dict) -> bool:
    if section.get("heading"):
        return False
    if section.get("imrad") is not None:
        return False
    text = section.get("text") or section.get("text_no_tables") or ""
    if _LEADING_NUMBERED_INTRO.match(text) and len(text) >= 60:
        return True
    if len(text) < 200:
        return False
    for pat in _ORPHAN_INTRO_PREFIXES:
        if pat.match(text):
            return True
    return False


def _rescue_first_section_heading(sections: list[dict]) -> None:
    if not sections:
        return
    if any(s.get("imrad") == "introduction" for s in sections):
        return
    first = sections[0]
    if _looks_like_orphan_introduction(first):
        first["heading"] = "Introduction"
        first["imrad"] = "introduction"


def _figure_page_from_coords(coords):
    if not coords:
        return None
    try:
        first_region = coords.split(";")[0]
        return int(first_region.split(",")[0])
    except (ValueError, IndexError):
        return None


def _extract_figures(root):
    figures = []
    for fig in root.findall(".//tei:figure", NSMAP):
        label_el = fig.find("./tei:head", NSMAP)
        label = _text_of(label_el) if label_el is not None else ""
        if not label:
            label_el = fig.find("./tei:label", NSMAP)
            label = _text_of(label_el) if label_el is not None else ""

        desc_el = fig.find("./tei:figDesc", NSMAP)
        caption = _text_of(desc_el) if desc_el is not None else ""
        caption = _trim_caption(caption)

        graphic_el = fig.find("./tei:graphic", NSMAP)
        coords = graphic_el.get("coords") if graphic_el is not None else None
        page = _figure_page_from_coords(coords)

        figures.append({
            "label": label,
            "caption": caption,
            "page": page,
            "coords": coords,
        })

    return figures


def _format_authors(bs) -> str:
    out = []
    for pers in bs.findall(".//tei:author/tei:persName", NSMAP):
        forenames = [_text_of(f) for f in pers.findall("tei:forename", NSMAP)]
        surname_el = pers.find("tei:surname", NSMAP)
        surname = _text_of(surname_el) if surname_el is not None else ""
        initials = "".join(f"{fn[0]}." for fn in forenames if fn)
        if surname and initials:
            out.append(f"{surname}, {initials}")
        elif surname:
            out.append(surname)
    return ", ".join(out)


def _format_biblstruct(bs) -> str:
    authors = _format_authors(bs)

    analytic_title = bs.find(".//tei:analytic/tei:title", NSMAP)
    monogr_title_m = bs.find(".//tei:monogr/tei:title[@level='m']", NSMAP)
    monogr_title_j = bs.find(".//tei:monogr/tei:title[@level='j']", NSMAP)

    if analytic_title is not None:
        title = _text_of(analytic_title)
        venue = _text_of(monogr_title_j) or _text_of(monogr_title_m)
    elif monogr_title_m is not None:
        title = _text_of(monogr_title_m)
        venue = ""
    else:
        title = _text_of(monogr_title_j)
        venue = ""

    date_el = bs.find(".//tei:imprint/tei:date", NSMAP)
    year = ""
    if date_el is not None:
        year = (date_el.get("when") or _text_of(date_el) or "")[:4]

    vol_el = bs.find(".//tei:imprint/tei:biblScope[@unit='volume']", NSMAP)
    issue_el = bs.find(".//tei:imprint/tei:biblScope[@unit='issue']", NSMAP)
    page_el = bs.find(".//tei:imprint/tei:biblScope[@unit='page']", NSMAP)
    volume = _text_of(vol_el)
    issue = _text_of(issue_el)
    if page_el is not None:
        pfrom = page_el.get("from")
        pto = page_el.get("to")
        if pfrom and pto:
            pages = f"{pfrom}\u2013{pto}"
        elif pfrom:
            pages = pfrom
        else:
            pages = _text_of(page_el)
    else:
        pages = ""

    doi_el = bs.find(".//tei:idno[@type='DOI']", NSMAP)
    doi = _text_of(doi_el)

    parts = []
    if authors:
        parts.append(authors)
    if year:
        parts.append(f"({year})")
    if title:
        parts.append(f"{title}.")

    venue_bits = ""
    if venue:
        venue_bits = venue
        if volume:
            venue_bits += f", {volume}"
            if issue:
                venue_bits += f"({issue})"
        if pages:
            venue_bits += f", {pages}"
        venue_bits += "."
    elif volume or pages:
        vb = [x for x in (volume, pages) if x]
        venue_bits = ", ".join(vb) + "."
    if venue_bits:
        parts.append(venue_bits)

    if doi:
        parts.append(doi if doi.lower().startswith("http") else f"https://doi.org/{doi}")

    citation = " ".join(p for p in parts if p).strip()
    return re.sub(r"\s+", " ", citation)


def _extract_references(root) -> list[str]:
    refs = []
    for bs in root.findall(".//tei:back//tei:listBibl/tei:biblStruct", NSMAP):
        s = _format_biblstruct(bs)
        if s:
            refs.append(s)
    if not refs:
        for bs in root.findall(".//tei:listBibl/tei:biblStruct", NSMAP):
            s = _format_biblstruct(bs)
            if s:
                refs.append(s)
    return refs


def parse_tei(tei_path):
    tei_path = Path(tei_path)
    if not tei_path.exists():
        raise FileNotFoundError(tei_path)

    parser = etree.XMLParser(recover=True, encoding="utf-8")
    tree = etree.parse(str(tei_path), parser)
    root = tree.getroot()

    title = _extract_title(root)
    abstract = _extract_abstract(root)

    body_el = root.find(".//tei:text/tei:body", NSMAP)
    sections = _walk_sections(body_el)

    _rescue_first_section_heading(sections)

    figures = _extract_figures(root)
    references = _extract_references(root)

    return {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "figures": figures,
        "references": references,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 tei_parser.py <path/to/tei.xml>")
        sys.exit(1)
    paper = parse_tei(sys.argv[1])
    print(f"Title: {paper['title'][:120]}")
    print(f"Sections: {len(paper['sections'])}")
    print(f"References: {len(paper['references'])}")
    for s in paper["sections"]:
        marker = f"[{s['imrad']}]" if s["imrad"] else ("[appendix]" if s.get("appendix") else "[-]")
        pg = s.get("page")
        print(f"  {marker:14} pg={str(pg):>4} tbl={s.get('table_count', 0)}"
              f"({s.get('table_unrecovered_count', 0)} empty) eq={len(s.get('equations', []))} {s['heading'][:50]:50}")