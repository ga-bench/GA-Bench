"""
ocr_engine.py  (v7.2 — FREE local OCR, no paid API)

OCR for content that survives only as images: image-based tables and equations.
Workflow:
    1. Render the region (page + bbox, PDF points) to a PNG via PyMuPDF.
    2. Recognise it with a FREE, LOCAL engine:
         - tables    -> RapidOCR (ONNX) or Tesseract  -> plain text
         - equations -> pix2tex (LaTeX-OCR)            -> LaTeX
    3. Return the transcription, or None on any failure.

No network calls, no API key, no per-use cost. Models download once (on first
use / on the login node) and are cached locally thereafter.

Fails soft: if no backend is installed, or a call errors, ocr_* returns None and
the caller flags the item `unrecovered`. The run never crashes.

Install (into your ParserEnv, on the login node so weights cache):
    pip install rapidocr_onnxruntime pillow pymupdf      # tables
    pip install "pix2tex[gui]" torch --extra-index-url ... # equations (optional)
    # Tesseract alternative for tables (needs the tesseract binary on PATH):
    pip install pytesseract pillow

Env:
    OCR_ENABLED          "1"/"0"                         (default "1")
    OCR_TABLE_ENGINE     auto|rapidocr|tesseract|none    (default "auto")
    OCR_EQUATION_ENGINE  auto|pix2tex|none               (default "auto")
    OCR_DPI              int                             (default 200)
"""

from __future__ import annotations
import os
import io

from text_cleanup import clean_equation_text, equation_block, merge_broken_equation_blocks

_CACHE: dict[tuple, str | None] = {}

# Lazily-initialised engine singletons (per process).
_TABLE_ENGINE = "UNSET"      # becomes a callable(pil)->str|None, or None
_EQ_ENGINE = "UNSET"


def _dpi() -> int:
    try:
        return int(os.environ.get("OCR_DPI", "300"))
    except ValueError:
        return 300


# ── PDF region -> PNG bytes ────────────────────────────────────────────────

def _parse_coords_regions(coords: str):
    regions = []
    if not coords:
        return regions
    for chunk in str(coords).split(";"):
        parts = chunk.split(",")
        if len(parts) >= 5:
            try:
                page = int(float(parts[0]))
                x, y, w, h = (float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4]))
                regions.append((page, x, y, w, h))
            except ValueError:
                continue
    return regions


def _union_same_page(regions):
    if not regions:
        return None
    page = regions[0][0]
    same = [r for r in regions if r[0] == page]
    x0 = min(r[1] for r in same)
    y0 = min(r[2] for r in same)
    x1 = max(r[1] + r[3] for r in same)
    y1 = max(r[2] + r[4] for r in same)
    return page, x0, y0, x1, y1


def render_crop_png(pdf_path, page_1indexed: int, bbox_xywh, pad: float = 6.0):
    try:
        import fitz
    except Exception:
        return None
    try:
        doc = fitz.open(str(pdf_path))
        try:
            if page_1indexed < 1 or page_1indexed > doc.page_count:
                return None
            page = doc[page_1indexed - 1]
            x, y, w, h = bbox_xywh
            rect = fitz.Rect(x - pad, y - pad, x + w + pad, y + h + pad) & page.rect
            if rect.is_empty or rect.width < 4 or rect.height < 4:
                return None
            zoom = _dpi() / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return None


def _png_to_pil(png_bytes):
    try:
        from PIL import Image
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        return None


# ── Free table engines ─────────────────────────────────────────────────────

def _build_rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    engine = RapidOCR()

    def run(pil):
        try:
            arr = np.array(pil)
            result, _ = engine(arr)
            if not result:
                return None
            lines = [item[1] for item in result if len(item) >= 2 and item[1]]
            text = "\n".join(lines).strip()
            return text or None
        except Exception:
            return None
    return run


def _build_tesseract():
    try:
        import pytesseract  # noqa
    except Exception:
        return None

    def run(pil):
        try:
            import pytesseract
            text = pytesseract.image_to_string(pil).strip()
            return text or None
        except Exception:
            return None
    return run


def _build_pix2tex():
    try:
        from pix2tex.cli import LatexOCR
    except Exception:
        return None
    try:
        model = LatexOCR()
    except Exception:
        return None

    def run(pil):
        try:
            latex = model(pil)
            latex = (latex or "").strip()
            return latex or None
        except Exception:
            return None
    return run


def _table_engine():
    global _TABLE_ENGINE
    if _TABLE_ENGINE != "UNSET":
        return _TABLE_ENGINE
    choice = os.environ.get("OCR_TABLE_ENGINE", "auto").lower()
    engine = None
    if choice in ("auto", "rapidocr"):
        engine = _build_rapidocr()
    if engine is None and choice in ("auto", "tesseract"):
        engine = _build_tesseract()
    if choice == "none":
        engine = None
    _TABLE_ENGINE = engine
    return engine


def _equation_engine():
    global _EQ_ENGINE
    if _EQ_ENGINE != "UNSET":
        return _EQ_ENGINE
    choice = os.environ.get("OCR_EQUATION_ENGINE", "auto").lower()
    engine = None
    if choice in ("auto", "pix2tex"):
        engine = _build_pix2tex()
    if choice == "none":
        engine = None
    _EQ_ENGINE = engine
    return engine


def ocr_enabled() -> bool:
    """True if OCR is on and at least one backend (table or equation) loaded."""
    if os.environ.get("OCR_ENABLED", "1") != "1":
        return False
    return (_table_engine() is not None) or (_equation_engine() is not None)




def _preprocess_table_pil(pil):
    """Light, safe preprocessing for small-font table OCR."""
    try:
        from PIL import ImageOps, ImageFilter, ImageEnhance
        img = pil.convert("L")
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Contrast(img).enhance(1.8)
        img = img.filter(ImageFilter.SHARPEN)
        return img.convert("RGB")
    except Exception:
        return pil


def _words_to_markdown(words: list[dict]) -> tuple[str, float, list[str]]:
    """Reconstruct rows/columns from OCR word coordinates."""
    reasons: list[str] = []
    words = [w for w in words if (w.get("text") or "").strip()]
    if not words:
        return "", 0.0, ["no_ocr_words"]

    heights = sorted([max(1.0, float(w.get("h", 1))) for w in words])
    med_h = heights[len(heights)//2]
    y_tol = max(6.0, med_h * 0.65)

    words_sorted = sorted(words, key=lambda w: (float(w.get("y", 0)), float(w.get("x", 0))))
    rows: list[list[dict]] = []
    for w in words_sorted:
        cy = float(w.get("y", 0)) + float(w.get("h", 0)) / 2.0
        placed = False
        for row in rows:
            row_cy = sum(float(x.get("y", 0)) + float(x.get("h", 0)) / 2.0 for x in row) / len(row)
            if abs(cy - row_cy) <= y_tol:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])
    rows = [sorted(r, key=lambda w: float(w.get("x", 0))) for r in rows]

    # Build column anchors from x positions. This is intentionally simple and robust.
    xs = sorted(float(w.get("x", 0)) for w in words)
    if not xs:
        return "", 0.0, ["no_x_positions"]
    gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
    median_gap = sorted(gaps)[len(gaps)//2] if gaps else 20.0
    col_tol = max(18.0, median_gap * 2.5)
    anchors: list[float] = []
    for x in xs:
        if not anchors or abs(x - anchors[-1]) > col_tol:
            anchors.append(x)
        else:
            anchors[-1] = (anchors[-1] + x) / 2.0
    if len(anchors) > 12:
        reasons.append("many_column_anchors")

    lines = []
    pipe_like_rows = 0
    for row in rows:
        cells = [""] * max(1, len(anchors))
        for w in row:
            x = float(w.get("x", 0))
            j = min(range(len(anchors)), key=lambda k: abs(x - anchors[k])) if anchors else 0
            txt = str(w.get("text") or "").strip()
            cells[j] = (cells[j] + " " + txt).strip() if cells[j] else txt
        # trim empty ends
        while cells and not cells[-1]:
            cells.pop()
        while cells and not cells[0]:
            cells.pop(0)
        if cells:
            if len(cells) >= 2:
                pipe_like_rows += 1
            lines.append(" | ".join(cells))

    markdown = "\n".join(lines).strip()
    if len(rows) < 2:
        reasons.append("few_rows")
    if pipe_like_rows < max(1, len(lines)//3):
        reasons.append("weak_column_alignment")
    structure = 0.35
    structure += min(0.25, 0.04 * len(rows))
    structure += min(0.25, 0.05 * pipe_like_rows)
    if reasons:
        structure -= 0.15
    return markdown, round(max(0.0, min(1.0, structure)), 3), reasons


def _tesseract_table_data(pil) -> tuple[list[dict], float | None]:
    try:
        import pytesseract
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config="--psm 6")
    except Exception:
        return [], None
    words = []
    confs = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data.get("conf", [None])[i])
        except Exception:
            conf = None
        if conf is not None and conf >= 0:
            confs.append(conf / 100.0)
        words.append({
            "text": txt,
            "x": float(data.get("left", [0])[i]),
            "y": float(data.get("top", [0])[i]),
            "w": float(data.get("width", [1])[i]),
            "h": float(data.get("height", [1])[i]),
            "conf": conf,
        })
    avg_conf = round(sum(confs) / len(confs), 3) if confs else None
    return words, avg_conf


def _rapidocr_table_data(pil) -> tuple[list[dict], float | None]:
    try:
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np
        engine = RapidOCR()
        result, _ = engine(np.array(pil))
    except Exception:
        return [], None
    words = []
    confs = []
    for item in result or []:
        try:
            box, txt, conf = item[0], item[1], item[2]
            if not txt:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            words.append({"text": str(txt), "x": min(xs), "y": min(ys), "w": max(xs)-min(xs), "h": max(ys)-min(ys), "conf": conf})
            if conf is not None:
                confs.append(float(conf))
        except Exception:
            continue
    avg_conf = round(sum(confs) / len(confs), 3) if confs else None
    return words, avg_conf


def ocr_table_region(pdf_path, page_1indexed: int, bbox_xywh) -> dict | None:
    """OCR a table region and return text + reconstructed markdown + confidence."""
    if os.environ.get("OCR_ENABLED", "1") != "1":
        return None
    key = (str(pdf_path), int(page_1indexed), tuple(round(v, 1) for v in bbox_xywh), "table_structured")
    if key in _CACHE:
        return _CACHE[key]
    png = render_crop_png(pdf_path, page_1indexed, bbox_xywh, pad=10.0)
    pil = _png_to_pil(png) if png else None
    if pil is None:
        _CACHE[key] = None
        return None
    pil = _preprocess_table_pil(pil)

    choice = os.environ.get("OCR_TABLE_ENGINE", "auto").lower()
    words: list[dict] = []
    avg_conf = None
    engine_name = None
    if choice in ("auto", "tesseract"):
        words, avg_conf = _tesseract_table_data(pil)
        engine_name = "tesseract"
    if not words and choice in ("auto", "rapidocr"):
        words, avg_conf = _rapidocr_table_data(pil)
        engine_name = "rapidocr"

    markdown, structure_conf, reasons = _words_to_markdown(words)
    text = markdown
    if not text:
        # Legacy fallback: whatever engine is available as plain text.
        engine = _table_engine()
        text = engine(pil) if engine else None
        markdown = text or ""
        structure_conf = 0.25 if text else 0.0
        reasons.append("plain_text_fallback")
    out = {
        "text": text or "",
        "markdown": markdown or text or "",
        "ocr_confidence": avg_conf,
        "structure_confidence": structure_conf,
        "quality_reasons": reasons,
        "engine": engine_name or choice,
    }
    _CACHE[key] = out
    return out

# ── Public OCR API (unchanged signatures) ──────────────────────────────────

def ocr_region(pdf_path, page_1indexed: int, bbox_xywh, kind: str) -> str | None:
    if os.environ.get("OCR_ENABLED", "1") != "1":
        return None
    engine = _equation_engine() if kind == "equation" else _table_engine()
    if engine is None:
        return None
    key = (str(pdf_path), int(page_1indexed),
           tuple(round(v, 1) for v in bbox_xywh), kind)
    if key in _CACHE:
        return _CACHE[key]
    png = render_crop_png(pdf_path, page_1indexed, bbox_xywh)
    pil = _png_to_pil(png) if png else None
    result = engine(pil) if pil is not None else None
    _CACHE[key] = result
    return result


def ocr_equation_coords(pdf_path, coords: str) -> str | None:
    if os.environ.get("OCR_ENABLED", "1") != "1":
        return None
    if _equation_engine() is None:
        return None
    regions = _parse_coords_regions(coords)
    u = _union_same_page(regions)
    if not u:
        return None
    page, x0, y0, x1, y1 = u
    return ocr_region(pdf_path, page, (x0, y0, x1 - x0, y1 - y0), "equation")


def resolve_equations(all_sections: list[dict], pdf_path) -> int:
    """Replace [[EQN:n]] placeholders with readable in-text equation blocks.

    Priority: OCR LaTeX when available; otherwise cleaned GROBID raw formula.
    The equation remains in the section text as [Equation N: ...], so downstream
    LLM prompts do not need a separate equation file.
    """
    n_ok = 0
    for sec in all_sections:
        eqs = sec.get("equations") or []
        if not eqs:
            continue
        for eq in eqs:
            eq_id = eq.get("id")
            placeholder = f"[[EQN:{eq_id}]]"
            latex = ocr_equation_coords(pdf_path, eq.get("coords", "")) if pdf_path else None
            if latex:
                cleaned = clean_equation_text(latex)
                replacement = equation_block(eq_id, cleaned, low_confidence=False)
                n_ok += 1
                eq["latex"] = cleaned
                eq["cleaned"] = cleaned
                eq["ocr_status"] = "ok"
                eq["quality"] = "good"
            else:
                raw_clean = clean_equation_text(eq.get("raw") or "")
                replacement = equation_block(eq_id, raw_clean, low_confidence=not bool(raw_clean))
                eq["cleaned"] = raw_clean
                eq["ocr_status"] = "unrecovered"
                eq["quality"] = "raw_cleaned" if raw_clean else "failed"
            for field in ("text", "text_no_tables"):
                if sec.get(field):
                    sec[field] = sec[field].replace(placeholder, replacement)
        for field in ("text", "text_no_tables"):
            if sec.get(field):
                sec[field] = merge_broken_equation_blocks(sec[field])
    return n_ok
