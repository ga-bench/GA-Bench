"""
GROBID TEI figure fallback (v6).

v6 changes vs v5.3:
    1. crosscheck_bbox_with_pymupdf now triggers on BOTH null bounding_box AND
       absurdly small bboxes (width < 80px or height < 80px). v5.3 only
       triggered on null bbox, missing cases like Figure 10 in sample 2 where
       pdffigures2 returned a 46px-wide sliver.
    2. recover_table_bbox_pymupdf also triggers on absurdly small table bboxes.
    3. Minor: added bbox area sanity check — recovered bbox must be at least
       2% of the page area to be accepted (prevents icon/logo false matches).

Everything else unchanged from v5.3.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from lxml import etree

from label_utils import (
    parse_label,
    canonical_key,
    split_merged_caption,
    find_caption_anchors,
    strip_tagged_markers,
    is_crossref_caption,
)
from text_cleanup import clean_text_artifacts


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _parse_page_from_coords(coords: str) -> Optional[int]:
    if not coords:
        return None
    first_region = coords.split(";")[0]
    parts = first_region.split(",")
    if len(parts) < 1:
        return None
    try:
        return int(parts[0])
    except (ValueError, TypeError):
        return None


def _classify_tei_figure_type(label: str, caption: str) -> str:
    rec = parse_label(label) or parse_label(caption)
    if rec:
        return rec["kind"]
    text = f"{label or ''} {caption or ''}".lower()
    if re.match(r"\s*table\b", text):
        return "table"
    if re.match(r"\s*scheme\b", text):
        return "scheme"
    return "figure"


def parse_tei_figures(tei_xml_path: str | Path) -> list[dict]:
    tei_xml_path = Path(tei_xml_path)
    if not tei_xml_path.exists():
        return []

    try:
        tree = etree.parse(str(tei_xml_path))
        root = tree.getroot()
    except etree.XMLSyntaxError:
        return []

    figures = []
    for fig_elem in root.iter("{http://www.tei-c.org/ns/1.0}figure"):
        tei_id = fig_elem.get("{http://www.w3.org/XML/1998/namespace}id", "")
        fig_type_attr = (fig_elem.get("type") or "").lower()

        head_elem = fig_elem.find("tei:head", TEI_NS)
        label_text = ""
        if head_elem is not None and head_elem.text:
            label_text = head_elem.text.strip()
        if not label_text:
            label_elem = fig_elem.find("tei:label", TEI_NS)
            if label_elem is not None and label_elem.text:
                label_text = f"Figure {label_elem.text.strip()}"

        figdesc_elem = fig_elem.find("tei:figDesc", TEI_NS)
        caption_text = ""
        if figdesc_elem is not None:
            caption_text = "".join(figdesc_elem.itertext()).strip()
            caption_text = re.sub(r"\s+", " ", caption_text)

        label_text = strip_tagged_markers(label_text)
        caption_text = strip_tagged_markers(caption_text)

        if is_crossref_caption(caption_text) or is_crossref_caption(label_text):
            continue

        page = None
        graphic_elem = fig_elem.find("tei:graphic", TEI_NS)
        if graphic_elem is not None:
            page = _parse_page_from_coords(graphic_elem.get("coords", ""))
        if page is None:
            page = _parse_page_from_coords(fig_elem.get("coords", ""))

        blob = f"{label_text} {caption_text}".strip()
        split_recs = split_merged_caption(blob)

        if len(split_recs) >= 2:
            for rec in split_recs:
                figures.append({
                    "tei_id": tei_id,
                    "label": rec["norm"],
                    "caption": rec["caption"],
                    "page": page,
                    "type": rec["kind"],
                })
            continue

        if fig_type_attr == "table":
            fig_type = "table"
        else:
            fig_type = _classify_tei_figure_type(label_text, caption_text)

        rec = parse_label(label_text) or parse_label(caption_text)
        clean_label = rec["norm"] if rec else label_text

        figures.append({
            "tei_id": tei_id,
            "label": clean_label,
            "caption": caption_text or (rec["norm"] if rec else ""),
            "page": page,
            "type": fig_type,
        })

    return figures


def _render_page_to_png(pdf_path: Path, page_1indexed: int, out_png: Path, dpi: int = 150) -> bool:
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return False
    try:
        page_idx = page_1indexed - 1
        if page_idx < 0 or page_idx >= len(doc):
            return False
        page = doc[page_idx]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_png))
        return True
    except Exception:
        return False
    finally:
        try:
            doc.close()
        except Exception:
            pass


def extract_tei_figures_with_images(
    tei_xml_path: str | Path,
    pdf_path: str | Path,
    output_figures_dir: str | Path,
    dpi: int = 150,
) -> list[dict]:
    pdf_path = Path(pdf_path)
    output_figures_dir = Path(output_figures_dir)

    tei_figures = parse_tei_figures(tei_xml_path)
    enriched = []

    for i, fig in enumerate(tei_figures):
        page = fig.get("page")
        img_basename = None
        if page is not None:
            img_basename = f"grobid_fig_{i:03d}.png"
            out_png = output_figures_dir / img_basename
            if not _render_page_to_png(pdf_path, page, out_png, dpi=dpi):
                img_basename = None

        enriched.append({
            "label": fig["label"],
            "type": fig["type"],
            "caption": fig["caption"],
            "page": page,
            "bounding_box": None,
            "image_file": img_basename,
            "pdffigures2_name": None,
            "extraction_method": "grobid_tei_page_render",
            "extraction_quality": "page_render",
            "quality_reasons": ["sourced from GROBID TEI fallback"],
        })

    return enriched


def _find_caption_page(pdf_path: Path, anchor_match: str, kind: str, number: str, prefix: str) -> Optional[int]:
    kind_word = {"figure": "Figure", "table": "Table", "scheme": "Scheme"}[kind]
    pn = f"{prefix}{number}" if prefix else number
    candidates = [
        f"{kind_word} {pn}",
        f"{kind_word.upper()} {pn}",
    ]
    if prefix == "S":
        candidates += [f"Supplemental {kind_word} {pn}", f"Supplementary {kind_word} {pn}",
                       f"SUPPLEMENTAL {kind_word.upper()} {pn}"]
    if prefix == "A":
        candidates += [f"Appendix {kind_word} {pn}"]
    if prefix == "E":
        candidates += [f"Extended Data {kind_word} {pn}"]

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return None
    try:
        first_anchor = None
        first_mention = None
        for page_idx in range(doc.page_count):
            text = doc[page_idx].get_text("text") or ""
            if not text:
                continue
            for cand in candidates:
                if re.search(r"(?:^|\n)\s*" + re.escape(cand) + r"\b", text, re.IGNORECASE):
                    return page_idx + 1
                if first_mention is None and cand.lower() in text.lower():
                    first_mention = page_idx + 1
        return first_anchor or first_mention
    finally:
        try:
            doc.close()
        except Exception:
            pass


def recover_figures_by_caption_scan(
    pdf_path: str | Path,
    body_text: str,
    raw_text: str,
    output_figures_dir: str | Path,
    already_have_keys: set[str] | None = None,
    dpi: int = 150,
) -> list[dict]:
    pdf_path = Path(pdf_path)
    output_figures_dir = Path(output_figures_dir)
    already = set(already_have_keys or set())

    anchors: dict[str, dict] = {}
    for src in (body_text or "", raw_text or ""):
        for rec in find_caption_anchors(src):
            if rec["kind"] not in ("figure", "scheme"):
                continue
            if rec["key"] in already:
                continue
            anchors.setdefault(rec["key"], rec)

    out = []
    idx = 0
    for key, rec in anchors.items():
        page = _find_caption_page(pdf_path, rec.get("match", ""), rec["kind"], rec["number"], rec["prefix"])
        img_basename = None
        if page is not None:
            img_basename = f"lastresort_fig_{idx:03d}.png"
            out_png = output_figures_dir / img_basename
            if not _render_page_to_png(pdf_path, page, out_png, dpi=dpi):
                img_basename = None
        idx += 1

        out.append({
            "label": rec["norm"],
            "type": rec["kind"],
            "caption": rec["norm"],
            "page": page,
            "bounding_box": None,
            "image_file": img_basename,
            "pdffigures2_name": None,
            "extraction_method": "caption_scan_page_render",
            "extraction_quality": "page_render",
            "quality_reasons": ["last-resort caption-anchor page render"],
        })
    return out


def merge_pdffigures2_and_tei(pf2_figures: list[dict], tei_figures: list[dict]) -> list[dict]:
    pf2_keys = set()
    for f in pf2_figures:
        k = canonical_key(f.get("label", "") or "")
        if k:
            pf2_keys.add(k)

    additions = []
    for tei_fig in tei_figures:
        k = canonical_key(tei_fig.get("label", "") or "")
        if not k:
            continue
        if k in pf2_keys:
            continue
        additions.append(tei_fig)
        pf2_keys.add(k)

    return list(pf2_figures) + additions


_STUB_CAPTION_MAX_LEN = 25


def _caption_badness(caption: str) -> int:
    """Small heuristic: higher score means more obvious PDF/caption corruption."""
    c = caption or ""
    bad = 0
    bad += c.count("�") * 5
    bad += c.count("  ")
    bad += len(re.findall(r"\b(?:s|vs|emove|tep|ntroduced|reen)\b", c, re.IGNORECASE))
    # Many isolated one-letter lowercase words often signal dropped first letters.
    bad += len(re.findall(r"\b[a-z]\b", c)) // 3
    return bad


def _caption_incomplete(caption: str) -> bool:
    c = clean_text_artifacts(caption or "").strip()
    if not c:
        return True
    tail = c[-80:].lower().strip(" .;:,)")
    return bool(
        len(c) < 40
        or re.search(r"(?:for details,? see|see|shown in|described in)\s*$", c, re.IGNORECASE)
        or tail.endswith(("see", "for details", "for details see"))
    )


def enrich_stub_captions(
    figures: list[dict],
    tei_figures: list[dict],
) -> int:
    tei_captions: dict[str, str] = {}
    for tf in tei_figures:
        k = canonical_key(tf.get("label", "") or "")
        cap = clean_text_artifacts((tf.get("caption") or "").strip())
        if k and cap and len(cap) > _STUB_CAPTION_MAX_LEN:
            if k not in tei_captions or len(cap) > len(tei_captions[k]):
                tei_captions[k] = cap

    enriched = 0
    for fig in figures:
        cap = clean_text_artifacts((fig.get("caption") or "").strip())
        k = canonical_key(fig.get("label", "") or "")
        if not k or k not in tei_captions:
            if cap:
                fig["caption"] = cap
            continue
        tei_cap = tei_captions[k]
        replace = False
        if len(cap) <= _STUB_CAPTION_MAX_LEN:
            replace = True
        elif len(tei_cap) > len(cap) * 1.15:
            replace = True
        elif _caption_badness(cap) > _caption_badness(tei_cap):
            replace = True
        if replace and not _caption_incomplete(tei_cap):
            fig["caption"] = tei_cap
            fig["caption_source"] = "tei_enriched"
            enriched += 1
        else:
            fig["caption"] = cap
    return enriched

# ─────────────────────────────────────────────────────────────────────────────
# v6: bbox size threshold for triggering recovery
# ─────────────────────────────────────────────────────────────────────────────

MIN_BBOX_DIMENSION_PT = 80  # minimum width/height in points before bbox is "absurd"


def _bbox_is_absurd(bbox: dict | None) -> bool:
    """True if bbox is None or too small to be a real figure/table."""
    if bbox is None:
        return True
    if not isinstance(bbox, dict):
        return True
    w = bbox.get("w", 0) or 0
    h = bbox.get("h", 0) or 0
    if w < MIN_BBOX_DIMENSION_PT or h < MIN_BBOX_DIMENSION_PT:
        return True
    return False


def crosscheck_bbox_with_pymupdf(
    figures: list[dict],
    pdf_path: str | Path,
    output_figures_dir: str | Path,
    min_image_area_ratio: float = 0.02,
    dpi: int = 150,
) -> int:
    """
    v6: triggers on both null bbox AND absurdly small bboxes.
    """
    pdf_path = Path(pdf_path)
    output_figures_dir = Path(output_figures_dir)

    # v6: check for absurdly small bboxes too, not just null
    needs_fix = [(i, fig) for i, fig in enumerate(figures)
                 if fig.get("page") is not None and _bbox_is_absurd(fig.get("bounding_box"))]
    if not needs_fix:
        return 0

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return 0

    fixed = 0
    try:
        for idx, fig in needs_fix:
            page_num = fig["page"]
            page_idx = page_num - 1
            if page_idx < 0 or page_idx >= len(doc):
                continue

            page = doc[page_idx]
            page_area = page.rect.width * page.rect.height
            if page_area <= 0:
                continue

            image_infos = page.get_image_info(xrefs=True)
            if not image_infos:
                continue

            best_img = None
            best_area = 0
            for img_info in image_infos:
                bbox = img_info.get("bbox")
                if not bbox:
                    continue
                x0, y0, x1, y1 = bbox
                area = (x1 - x0) * (y1 - y0)
                if area / page_area < min_image_area_ratio:
                    continue
                if area > best_area:
                    best_area = area
                    best_img = bbox

            if best_img is None:
                continue

            x0, y0, x1, y1 = best_img
            fig["bounding_box"] = {
                "x": round(x0, 2),
                "y": round(y0, 2),
                "w": round(x1 - x0, 2),
                "h": round(y1 - y0, 2),
            }

            try:
                clip = fitz.Rect(x0, y0, x1, y1)
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                img_file = fig.get("image_file")
                if img_file:
                    out_png = output_figures_dir / img_file
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    pix.save(str(out_png))
                    fig["extraction_quality"] = "good"
                    fig["crop_quality"] = "pymupdf_crosscheck"
                    if "quality_reasons" not in fig:
                        fig["quality_reasons"] = []
                    fig["quality_reasons"].append("bbox recovered via PyMuPDF image cross-check")
                    fixed += 1
            except Exception:
                fixed += 1

    finally:
        try:
            doc.close()
        except Exception:
            pass

    return fixed


def recover_table_bbox_pymupdf(
    tables: list[dict],
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 150,
) -> int:
    """
    v6: triggers on null bbox, suspicious crop quality, AND absurdly small bboxes.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # v6: also trigger on absurdly small bboxes
    needs_fix = [(i, t) for i, t in enumerate(tables)
                 if t.get("page") is not None and (
                     t.get("crop_quality") == "suspicious"
                     or _bbox_is_absurd(t.get("bounding_box"))
                 )]
    if not needs_fix:
        return 0

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return 0

    fixed = 0
    try:
        for idx, tbl in needs_fix:
            page_num = tbl["page"]
            page_idx = page_num - 1
            if page_idx < 0 or page_idx >= len(doc):
                continue

            page = doc[page_idx]
            try:
                pymupdf_tables = page.find_tables()
            except Exception:
                continue

            if not pymupdf_tables or len(pymupdf_tables.tables) == 0:
                continue

            existing_bbox = tbl.get("bounding_box")
            best_table = None

            if existing_bbox and isinstance(existing_bbox, dict) and existing_bbox.get("x") is not None:
                # v6: only use existing bbox as reference if it's not absurd
                if not _bbox_is_absurd(existing_bbox):
                    ex_cx = existing_bbox["x"] + existing_bbox.get("w", 0) / 2
                    ex_cy = existing_bbox["y"] + existing_bbox.get("h", 0) / 2
                    best_dist = float("inf")
                    for pt in pymupdf_tables.tables:
                        r = pt.bbox
                        cx = (r.x0 + r.x1) / 2
                        cy = (r.y0 + r.y1) / 2
                        dist = ((cx - ex_cx) ** 2 + (cy - ex_cy) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_table = pt
                else:
                    # Absurd existing bbox: pick the largest table
                    best_area = 0
                    for pt in pymupdf_tables.tables:
                        r = pt.bbox
                        area = (r.x1 - r.x0) * (r.y1 - r.y0)
                        if area > best_area:
                            best_area = area
                            best_table = pt
            else:
                best_area = 0
                for pt in pymupdf_tables.tables:
                    r = pt.bbox
                    area = (r.x1 - r.x0) * (r.y1 - r.y0)
                    if area > best_area:
                        best_area = area
                        best_table = pt

            if best_table is None:
                continue

            r = best_table.bbox
            tbl["bounding_box"] = {
                "x": round(r.x0, 2),
                "y": round(r.y0, 2),
                "w": round(r.x1 - r.x0, 2),
                "h": round(r.y1 - r.y0, 2),
            }

            try:
                clip = fitz.Rect(r.x0, r.y0, r.x1, r.y1)
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                img_file = tbl.get("image_file")
                if img_file:
                    out_png = output_dir / img_file
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    pix.save(str(out_png))
                    tbl["crop_quality"] = "pymupdf_find_tables"
                    tbl["extraction_quality"] = "good"
                    tbl["vlm_ready"] = True
                    if "quality_reasons" not in tbl:
                        tbl["quality_reasons"] = []
                    tbl["quality_reasons"].append("bbox recovered via PyMuPDF find_tables()")
                    fixed += 1
            except Exception:
                fixed += 1

    finally:
        try:
            doc.close()
        except Exception:
            pass

    return fixed


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python3 grobid_figure_fallback.py <tei.xml> <pdf> <out_figures_dir>")
        sys.exit(1)
    figs = extract_tei_figures_with_images(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"Extracted {len(figs)} TEI figures:")
    for f in figs:
        print(f"  {f['label']} | page {f['page']} | type={f['type']} | image={f['image_file']}")