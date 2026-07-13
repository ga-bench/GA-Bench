"""
PyMuPDF-based fallback for figures pdffigures2 couldn't extract correctly.

When pdffigures2 produces a "suspect" figure (tiny bbox, almost certainly the
page-header strip rather than the real figure), we re-render the *entire page*
where that figure's caption sits, using PyMuPDF.

This won't give a properly-cropped figure, but it guarantees the figure is
*somewhere* in the rendered image. The downstream consumer (an LLM or a human
reviewer) can find it. Far better than a 25-pixel-tall header strip.

The fallback overwrites the bad PNG produced by pdffigures2 with a full-page
render at the same DPI (150 by default, matching pdffigures2's setting).
The figure dict gets new fields:
    extraction_method: "pdffigures2" (default) or "pymupdf_full_page" (fallback)
    extraction_quality: "good" | "page_render" | "header_strip"

Usage:
    from pymupdf_fallback import apply_fallback_to_suspect_figures

    enriched_figures = apply_fallback_to_suspect_figures(
        figures=figures_from_pdffigures2,
        pdf_path="/path/to/paper.pdf",
        figures_dir="/path/to/output/figures",
        dpi=150,
    )
"""

from __future__ import annotations
from pathlib import Path

import fitz  # PyMuPDF

from typing import Iterable


# Quality thresholds — extended in v2 to catch vertical/horizontal slivers
# and extreme aspect ratios that v1 missed (e.g. paper 3's fig05: w=35px h=266px).
MIN_BBOX_HEIGHT_PX = 50
MIN_BBOX_WIDTH_PX = 60       # NEW in v2: catches vertical slivers
MIN_BBOX_AREA_PX = 5000
MIN_IMAGE_FILE_SIZE_KB = 8
MAX_ASPECT_RATIO = 10.0      # NEW in v2: w/h or h/w > 10 is almost certainly a sliver


def _is_suspect(fig: dict, figures_dir: Path) -> tuple[bool, list[str]]:
    """
    Determine whether a figure's pdffigures2 output is suspect.
    Returns (is_suspect, reasons).

    v2 changes: also flags vertical/horizontal slivers (one dimension very small)
    and extreme aspect ratios (e.g. 10:1 strips that pdffigures2 sometimes returns
    when it detects only a panel label region instead of the full figure).
    """
    reasons: list[str] = []

    # Image file checks
    img_name = fig.get("image_file")
    if not img_name:
        reasons.append("no image_file field")
    else:
        img_path = figures_dir / img_name
        if not img_path.exists():
            reasons.append(f"image file missing: {img_name}")
        else:
            size_kb = img_path.stat().st_size / 1024
            if size_kb < MIN_IMAGE_FILE_SIZE_KB:
                reasons.append(f"image too small ({size_kb:.1f} KB)")

    # Bbox checks
    bbox = fig.get("bounding_box")
    if bbox:
        h = bbox.get("h", 0)
        w = bbox.get("w", 0)
        area = h * w
        if h < MIN_BBOX_HEIGHT_PX:
            reasons.append(f"bbox height too small ({h:.0f}px)")
        if w < MIN_BBOX_WIDTH_PX:
            reasons.append(f"bbox width too small ({w:.0f}px)")
        if area < MIN_BBOX_AREA_PX:
            reasons.append(f"bbox area too small ({area:.0f}px²)")
        # Aspect-ratio sliver detection: w=35, h=266 → ratio 7.6, but if more extreme.
        if h > 0 and w > 0:
            ratio = max(h / w, w / h)
            if ratio > MAX_ASPECT_RATIO:
                reasons.append(f"extreme aspect ratio ({ratio:.1f}:1)")

    return (len(reasons) > 0), reasons


def _find_duplicate_bbox_indices(figures: list[dict]) -> set[int]:
    """Indices of figures whose bbox is shared with another figure (rounded to 1px)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, fig in enumerate(figures):
        bbox = fig.get("bounding_box")
        if not bbox:
            continue
        key = (round(bbox.get("x", 0)), round(bbox.get("y", 0)),
               round(bbox.get("w", 0)), round(bbox.get("h", 0)))
        groups[key].append(i)
    suspect = set()
    for indices in groups.values():
        if len(indices) > 1:
            suspect.update(indices)
    return suspect


def _render_pdf_page(
    pdf_path: Path,
    page_number_1indexed: int,
    out_png_path: Path,
    dpi: int = 150,
) -> bool:
    """
    Render a single PDF page to PNG at the given DPI.
    Returns True on success.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return False

    try:
        # Convert 1-indexed page to 0-indexed for PyMuPDF.
        page_idx = page_number_1indexed - 1
        if page_idx < 0 or page_idx >= doc.page_count:
            doc.close()
            return False

        page = doc[page_idx]
        # PyMuPDF's default zoom is 72 DPI; scale up to match the requested DPI.
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        out_png_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_png_path))
        doc.close()
        return True
    except Exception:
        try:
            doc.close()
        except Exception:
            pass
        return False


def apply_fallback_to_suspect_figures(
    figures: list[dict],
    pdf_path: str | Path,
    figures_dir: str | Path,
    dpi: int = 150,
) -> list[dict]:
    """
    For each suspect figure, re-render the full page with PyMuPDF and overwrite
    the bad pdffigures2 PNG. Tag every figure with extraction_method and
    extraction_quality fields.

    Args:
        figures: list of figure dicts as produced by pdffigures2_runner.run_pdffigures2()
        pdf_path: path to the source PDF
        figures_dir: directory containing the pdffigures2-extracted PNGs (and where
                     PyMuPDF will write replacements)
        dpi: render DPI for fallbacks (default 150, matching pdffigures2)

    Returns: a new list of figure dicts (originals not mutated). Each figure has
        added fields:
            extraction_method: "pdffigures2" | "pymupdf_full_page"
            extraction_quality: "good" | "page_render" | "header_strip"
            quality_reasons: list[str]  (empty for good ones)
        Suspect figures whose fallback succeeded keep their original metadata
        (label, caption, page, bounding_box) — only the image file content
        changes on disk.
    """
    pdf_path = Path(pdf_path)
    figures_dir = Path(figures_dir)

    duplicate_indices = _find_duplicate_bbox_indices(figures)

    out: list[dict] = []
    for i, fig in enumerate(figures):
        new_fig = dict(fig)  # shallow copy

        suspect, reasons = _is_suspect(fig, figures_dir)
        if i in duplicate_indices:
            reasons.append("duplicate bbox shared with another figure")
            suspect = True

        if not suspect:
            new_fig["extraction_method"] = "pdffigures2"
            new_fig["extraction_quality"] = "good"
            new_fig["quality_reasons"] = []
            out.append(new_fig)
            continue

        # Suspect — try fallback.
        page = fig.get("page")
        img_name = fig.get("image_file")

        if not page or not img_name:
            # Can't fall back without a page or a target filename.
            new_fig["extraction_method"] = "pdffigures2"
            new_fig["extraction_quality"] = "header_strip"
            new_fig["quality_reasons"] = reasons + ["no page or image_file for fallback"]
            out.append(new_fig)
            continue

        target_path = figures_dir / img_name
        ok = _render_pdf_page(pdf_path, page, target_path, dpi=dpi)

        if ok:
            new_fig["extraction_method"] = "pymupdf_full_page"
            new_fig["extraction_quality"] = "page_render"
            new_fig["quality_reasons"] = reasons
            # Refresh the on-disk file size (in case downstream cares)
            try:
                new_fig["_image_size_kb"] = target_path.stat().st_size / 1024
            except OSError:
                pass
        else:
            new_fig["extraction_method"] = "pdffigures2"
            new_fig["extraction_quality"] = "header_strip"
            new_fig["quality_reasons"] = reasons + ["pymupdf_fallback_failed"]

        out.append(new_fig)

    return out


if __name__ == "__main__":
    # Standalone smoke test: given an existing test_outputs/<paper>/ dir
    # with a source.pdf and figures/ + data/ from pdffigures2, re-evaluate
    # all figures and apply fallback. Useful for inspecting fallback output
    # without re-running pdffigures2.
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python3 pymupdf_fallback.py <paper_dir>")
        print("       where <paper_dir> contains source.pdf, figures/, data/")
        sys.exit(1)

    paper_dir = Path(sys.argv[1])
    pdf_path = paper_dir / "source.pdf"
    figures_dir = paper_dir / "figures"
    data_dir = paper_dir / "data"

    if not pdf_path.exists():
        print(f"No source.pdf at {pdf_path}")
        sys.exit(1)

    # Find pdffigures2 JSON (named after the original PDF stem).
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print(f"No pdffigures2 JSON in {data_dir}")
        sys.exit(1)
    pf_json = json_files[0]

    with open(pf_json) as f:
        raw = json.load(f)
    raw_figs = raw.get("figures", []) if isinstance(raw, dict) else raw

    # Reshape to our internal format (subset that the fallback needs).
    minimal_figs = []
    for rf in raw_figs:
        page = rf.get("page")
        page_1 = (page + 1) if isinstance(page, int) else None
        region = rf.get("regionBoundary", {}) or {}
        bbox = None
        if region:
            try:
                bbox = {
                    "x": float(region.get("x1", 0)),
                    "y": float(region.get("y1", 0)),
                    "w": max(0.0, float(region.get("x2", 0)) - float(region.get("x1", 0))),
                    "h": max(0.0, float(region.get("y2", 0)) - float(region.get("y1", 0))),
                }
            except (TypeError, ValueError):
                bbox = None
        render_url = rf.get("renderURL", "") or ""
        img_name = render_url.rsplit("/", 1)[-1] if render_url else None

        minimal_figs.append({
            "label": f"{rf.get('figType', 'Figure')} {rf.get('name', '')}",
            "type": (rf.get("figType") or "").lower() or "figure",
            "caption": rf.get("caption", ""),
            "page": page_1,
            "bounding_box": bbox,
            "image_file": img_name,
        })

    print(f"Loaded {len(minimal_figs)} figures from pdffigures2 JSON")
    enriched = apply_fallback_to_suspect_figures(minimal_figs, pdf_path, figures_dir)

    n_good = sum(1 for f in enriched if f["extraction_quality"] == "good")
    n_fallback = sum(1 for f in enriched if f["extraction_method"] == "pymupdf_full_page")
    n_header = sum(1 for f in enriched if f["extraction_quality"] == "header_strip")
    print(f"  good (pdffigures2): {n_good}")
    print(f"  fallback applied:   {n_fallback}")
    print(f"  still bad:          {n_header}")
    print()

    for f in enriched:
        marker = {
            "good": "[GOOD]    ",
            "page_render": "[FALLBACK]",
            "header_strip": "[BAD]     ",
        }.get(f["extraction_quality"], "[?]       ")
        print(f"  {marker} {f['label']:<16s} p.{f['page']!s:<4s} "
              f"method={f['extraction_method']:<18s} {f['image_file']}")
        if f["quality_reasons"]:
            print(f"             reasons: {'; '.join(f['quality_reasons'])}")