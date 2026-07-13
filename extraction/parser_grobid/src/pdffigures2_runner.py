"""
pdffigures2 subprocess wrapper.

Runs Allen AI's pdffigures2 jar against a single PDF and returns a structured
Python dict with all the figures it extracted.

CLI invoked:
    java -jar /path/to/pdffigures2.jar \
        -d <data_dir>/ \
        -m <fig_dir>/ \
        -e \
        -c \
        -t 1 \
        -i 150 \
        <pdf_path>

pdffigures2 emits:
    - PNG images: <fig_dir>/<pdf_basename>-Figure1-1.png, -Figure2-1.png,
      -Table1-1.png, etc.
    - One JSON file: <data_dir>/<pdf_basename>.json containing an array of
      figure objects: caption, captionBoundary, figType, name, page,
      regionBoundary, renderDpi, renderURL, imageText.

This wrapper:
    - Reads that JSON
    - Normalizes each entry into a consistent shape
    - Resolves image file paths
    - Detects "scheme" figures by caption text (pdffigures2 doesn't distinguish
      schemes from regular figures — it tags them all as figType="Figure")

Usage:
    from pdffigures2_runner import run_pdffigures2
    result = run_pdffigures2(
        pdf_path="/path/to/paper.pdf",
        figures_dir="/path/to/output/figures",
        data_dir="/path/to/output/data",
        jar_path="/path/to/pdffigures2.jar",
    )
    # result = {"success": bool, "figures": [...], "error": str | None,
    #           "stats": {"elapsed_seconds": float}}
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import time
from pathlib import Path

from label_utils import parse_label


# pdffigures2's CLI names the json file <pdf_basename>.json in --data-prefix dir.
# pdffigures2 names images <pdf_basename>-<FigType><Name>-<id>.png
# e.g. "10_1007_s40203-023-00143-7-Figure2-1.png"
#
# Caption text often starts with "Scheme N" for chemistry schemes, even though
# pdffigures2 still tags figType="Figure". We detect these.
SCHEME_CAPTION_RE = re.compile(r"^\s*scheme\s+\d", re.IGNORECASE)


def _classify_figure_type(fig_type_field: str, caption: str) -> str:
    """
    Determine our internal type label from pdffigures2's figType + caption.

    pdffigures2 only emits "Figure" or "Table" in figType.
    We add a third internal category, "scheme", detected via caption prefix.

    Returns: "figure", "table", or "scheme".
    """
    ft = (fig_type_field or "").lower()
    if ft == "table":
        return "table"
    if SCHEME_CAPTION_RE.match(caption or ""):
        return "scheme"
    return "figure"


def _resolve_image_path(figures_dir: Path, render_url: str) -> Path | None:
    """
    pdffigures2's renderURL is relative to the dir from which the jar was run,
    not necessarily an absolute path. The actual filename is just the basename
    of that URL, and the file lives in figures_dir.
    """
    if not render_url:
        return None
    basename = os.path.basename(render_url)
    candidate = figures_dir / basename
    if candidate.exists():
        return candidate
    return None


def run_pdffigures2(
    pdf_path: str | Path,
    figures_dir: str | Path,
    data_dir: str | Path,
    jar_path: str | Path,
    timeout_seconds: int = 300,
    dpi: int = 150,
    apply_pymupdf_fallback: bool = True,
) -> dict:
    """
    Run pdffigures2 against one PDF.

    Args:
        pdf_path: input PDF file
        figures_dir: where pdffigures2 writes PNG images
        data_dir: where pdffigures2 writes its metadata JSON
        jar_path: path to pdffigures2.jar
        timeout_seconds: hard limit for the subprocess (default 5 min)
        dpi: rendering DPI for extracted figures (default 150)

    Returns dict:
        {
            "success": bool,
            "figures": [
                {
                    "label": "Figure 2",
                    "type": "figure" | "table" | "scheme",
                    "caption": "Fig. 2 ...",
                    "page": int,            # 1-indexed
                    "bounding_box": {"x": float, "y": float, "w": float, "h": float},
                    "image_file": "<basename>.png",  # filename only
                    "image_abs_path": "<full path>", # resolved absolute path
                    "pdffigures2_name": "2",
                },
                ...
            ],
            "error": str | None,
            "stats": {
                "elapsed_seconds": float,
                "n_figures": int,
                "n_tables": int,
                "n_schemes": int,
            }
        }
    """
    pdf_path = Path(pdf_path).resolve()
    figures_dir = Path(figures_dir).resolve()
    data_dir = Path(data_dir).resolve()
    jar_path = Path(jar_path).resolve()

    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # pdffigures2 expects the output dirs to end with a separator for its
    # prefix logic, so we pass them with trailing slashes.
    fig_prefix = str(figures_dir) + os.sep
    data_prefix = str(data_dir) + os.sep

    cmd = [
        "java",
        "-jar", str(jar_path),
        "-d", data_prefix,
        "-m", fig_prefix,
        "-e",                       # ignore errors, don't abort batch
        "-c",                       # save regionless captions
        "-t", "1",                  # single-threaded internally
        "-i", str(dpi),
        str(pdf_path),
    ]

    start_time = time.time()
    error: str | None = None
    figures_out: list[dict] = []

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        # pdffigures2 exits 0 on success even with per-figure errors (because of -e).
        # Non-zero exit usually means a real crash.
        if completed.returncode != 0:
            error = (
                f"pdffigures2 exited with code {completed.returncode}. "
                f"stderr: {completed.stderr[-500:].strip()}"
            )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "figures": [],
            "error": f"pdffigures2 timed out after {timeout_seconds}s (took {elapsed:.1f}s)",
            "stats": {
                "elapsed_seconds": elapsed,
                "n_figures": 0,
                "n_tables": 0,
                "n_schemes": 0,
            },
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "figures": [],
            "error": f"Could not invoke java/jar: {e}",
            "stats": {
                "elapsed_seconds": time.time() - start_time,
                "n_figures": 0,
                "n_tables": 0,
                "n_schemes": 0,
            },
        }

    elapsed = time.time() - start_time

    # Find the JSON output. pdffigures2 names it <pdf_basename>.json.
    pdf_stem = pdf_path.stem  # filename without .pdf
    json_path = data_dir / f"{pdf_stem}.json"

    if not json_path.exists():
        # pdffigures2 ran but produced no JSON. Could mean: no figures found,
        # or a silent failure. Either way, return empty result without error
        # unless we already have one from the exit code.
        return {
            "success": error is None,
            "figures": [],
            "error": error,  # may be None (clean run, just no figures)
            "stats": {
                "elapsed_seconds": elapsed,
                "n_figures": 0,
                "n_tables": 0,
                "n_schemes": 0,
            },
        }

    # Parse the JSON.
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "success": False,
            "figures": [],
            "error": f"Could not parse pdffigures2 JSON: {e}",
            "stats": {
                "elapsed_seconds": elapsed,
                "n_figures": 0,
                "n_tables": 0,
                "n_schemes": 0,
            },
        }

    # pdffigures2's JSON structure: {"figures": [...]} OR sometimes just [...]
    if isinstance(raw, dict):
        raw_figs = raw.get("figures", [])
    elif isinstance(raw, list):
        raw_figs = raw
    else:
        raw_figs = []

    n_figures = n_tables = n_schemes = 0

    for f in raw_figs:
        caption = f.get("caption", "") or ""
        fig_type_raw = f.get("figType", "") or ""
        fig_name = str(f.get("name", "")).strip()
        page_raw = f.get("page")

        # pdffigures2 uses 0-indexed page numbers internally in some places;
        # the JSON uses 0-indexed. Normalize to 1-indexed for our output.
        page_1indexed = None
        if isinstance(page_raw, int):
            page_1indexed = page_raw + 1

        # Region bounding box.
        region = f.get("regionBoundary", {}) or {}
        bbox = None
        if region:
            try:
                x1 = float(region.get("x1", 0))
                y1 = float(region.get("y1", 0))
                x2 = float(region.get("x2", 0))
                y2 = float(region.get("y2", 0))
                bbox = {
                    "x": x1,
                    "y": y1,
                    "w": max(0.0, x2 - x1),
                    "h": max(0.0, y2 - y1),
                }
            except (TypeError, ValueError):
                bbox = None

        # Resolve the image file.
        render_url = f.get("renderURL", "") or ""
        img_path = _resolve_image_path(figures_dir, render_url)
        img_filename = img_path.name if img_path is not None else None
        img_abs = str(img_path) if img_path is not None else None

        # Build human-readable label. Prefer the real caption label so appendix
        # and supplementary assets keep labels such as Figure A.1 and Figure S8.
        our_type = _classify_figure_type(fig_type_raw, caption)
        rec = parse_label(caption)
        if rec and rec.get("kind") in {"figure", "table", "scheme"}:
            our_type = rec["kind"]
            label = rec["norm"]
            normalized_label = rec.get("normalized_label")
            figure_id_hint = rec.get("id")
        else:
            figure_id_hint = None
            normalized_label = fig_name or ""
            if our_type == "table":
                label = f"Table {fig_name}" if fig_name else "Table"
            elif our_type == "scheme":
                label = f"Scheme {fig_name}" if fig_name else "Scheme"
            else:
                label = f"Figure {fig_name}" if fig_name else "Figure"

        if our_type == "table":
            n_tables += 1
        elif our_type == "scheme":
            n_schemes += 1
        else:
            n_figures += 1

        figures_out.append({
            "label": label,
            "normalized_label": normalized_label,
            "display_label": label,
            "figure_id_hint": figure_id_hint,
            "type": our_type,
            "caption": caption.strip(),
            "page": page_1indexed,
            "bounding_box": bbox,
            "image_file": img_filename,
            "image_abs_path": img_abs,
            "pdffigures2_name": fig_name,
        })

    # Apply PyMuPDF fallback for suspect figures (header-strip outputs, etc.)
    n_fallback_applied = 0
    if apply_pymupdf_fallback and figures_out:
        try:
            from pymupdf_fallback import apply_fallback_to_suspect_figures
            figures_out = apply_fallback_to_suspect_figures(
                figures=figures_out,
                pdf_path=pdf_path,
                figures_dir=figures_dir,
                dpi=dpi,
            )
            n_fallback_applied = sum(
                1 for f in figures_out
                if f.get("extraction_method") == "pymupdf_full_page"
            )
        except Exception as e:
            # Fallback failure shouldn't kill the whole extraction.
            if error is None:
                error = f"pymupdf fallback failed: {e}"
            else:
                error = f"{error}; pymupdf fallback failed: {e}"

    return {
        "success": True,
        "figures": figures_out,
        "error": error,  # might be a warning-level message even on success
        "stats": {
            "elapsed_seconds": elapsed,
            "n_figures": n_figures,
            "n_tables": n_tables,
            "n_schemes": n_schemes,
            "n_fallback_applied": n_fallback_applied,
        },
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python3 pdffigures2_runner.py <pdf> <figures_dir> <data_dir> [jar_path]")
        sys.exit(1)

    pdf_arg = sys.argv[1]
    fig_dir_arg = sys.argv[2]
    data_dir_arg = sys.argv[3]
    jar_arg = sys.argv[4] if len(sys.argv) > 4 else \
        "./grobid_parser/bin/pdffigures2.jar"

    print(f"Running pdffigures2 on: {pdf_arg}")
    print(f"  figures_dir: {fig_dir_arg}")
    print(f"  data_dir:    {data_dir_arg}")
    print(f"  jar:         {jar_arg}")
    print()

    result = run_pdffigures2(
        pdf_path=pdf_arg,
        figures_dir=fig_dir_arg,
        data_dir=data_dir_arg,
        jar_path=jar_arg,
    )

    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Success: {result['success']}")
    print(f"Error:   {result['error']}")
    print(f"Stats:   {result['stats']}")
    print()
    print(f"FIGURES ({len(result['figures'])}):")
    for fig in result["figures"]:
        page = fig["page"] if fig["page"] is not None else "?"
        bbox = fig["bounding_box"]
        bbox_str = (f"bbox=({bbox['x']:.0f},{bbox['y']:.0f},"
                    f"{bbox['w']:.0f}x{bbox['h']:.0f})") if bbox else "bbox=?"
        img = fig["image_file"] if fig["image_file"] else "(no image)"
        cap = fig["caption"][:80].replace("\n", " ")
        method = fig.get("extraction_method", "?")
        quality = fig.get("extraction_quality", "?")
        print(f"  [{fig['type']:6s}] {fig['label']:<15s} p.{page:<4} "
              f"{quality:<13s} via {method:<18s} {img}")
        print(f"           caption: {cap}...")
        reasons = fig.get("quality_reasons", [])
        if reasons:
            print(f"           reasons: {'; '.join(reasons)}")