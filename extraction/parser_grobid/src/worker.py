"""
Per-PDF worker (v8).

v7.2 changes vs v7:
    1. REQUIRED_OUTPUTS includes both IMRaD files (skip-guard cannot leave a
       partial/stale set).
    2. pdffigures2 table-type detections are no longer discarded; they are
       passed to write_paper_outputs as `table_detections` so image-based
       tables can be OCR'd by region.
    3. DEFAULT_JAR_PATH points at the grobid_parser tree.

Entry points unchanged:
    process_one_pdf(pdf_path, grobid_url, ...) -> result dict
    worker_loop(grobid_url, work_queue_path, node_id, worker_id, ...) -> None
"""

from __future__ import annotations
import argparse
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback
import uuid
import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import requests

from tei_parser import parse_tei
from imrad_classifier import infer_methods_from_sections
from pdffigures2_runner import run_pdffigures2
from figure_section_mapper import map_figures_to_sections
from figure_validator import find_missing_references
from grobid_figure_fallback import (
    extract_tei_figures_with_images,
    merge_pdffigures2_and_tei,
    parse_tei_figures,
    recover_figures_by_caption_scan,
    enrich_stub_captions,
    crosscheck_bbox_with_pymupdf,
)
from output_writer import write_paper_outputs


DEFAULT_JAR_PATH = "./grobid_parser/bin/pdffigures2.jar"
DEFAULT_DATASET_ROOT = "./dataset_10k"

OUTPUT_FOLDER_NAME = "extracted"

REQUIRED_OUTPUTS = [
    "fulltext.json",
    "fulltext_imrad.json",
    "figures.json",
    "tables.json",
    "equations.json",
    "quality_report.json",
    "tei.xml",
    "figures",
]

PER_PAPER_TIMEOUT_S = 900

GROBID_CONNECT_TIMEOUT_S = 30
GROBID_READ_TIMEOUT_S = 300

PDFFIGURES2_TIMEOUT_S = 300

TEI_FIGURE_FALLBACK_THRESHOLD = 3


def _is_elsevier(publisher: str) -> bool:
    return "elsevier" in (publisher or "").strip().lower()


def parse_doi_from_folder(doi_folder_name: str) -> str:
    parts = doi_folder_name.split("_")
    if len(parts) >= 2:
        prefix = parts[0]
        registrant = parts[1]
        rest = "_".join(parts[2:])
        if rest:
            return f"{prefix}.{registrant}/{rest.replace('_', '.')}"
        return f"{prefix}.{registrant}"
    return doi_folder_name


def _read_sidecar_metadata(doi_folder: Path) -> dict:
    """Read <doi>_Metadata.json when available, without making it required."""
    try:
        meta_files = sorted(doi_folder.glob("*_Metadata.json"))
        if not meta_files:
            return {}
        import json
        with open(meta_files[0], "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def parse_path_metadata(pdf_path: Path) -> dict:
    """Dataset layout for this update:

        ./

    There are no publisher or journal folders in this dataset.
    """
    doi_folder = pdf_path.parent
    sidecar = _read_sidecar_metadata(doi_folder)
    return {
        "doi": sidecar.get("doi") or parse_doi_from_folder(doi_folder.name),
        "doi_folder": doi_folder,
        "publisher": sidecar.get("publisher") or sidecar.get("source_publisher") or "",
        "journal": sidecar.get("journal") or sidecar.get("journal_name") or "",
    }


def output_already_complete(doi_folder: Path) -> bool:
    output_dir = doi_folder / OUTPUT_FOLDER_NAME
    if not output_dir.is_dir():
        return False
    # Outputs are DOI-prefixed in v9, e.g. 10_1016_x_fulltext.json.
    required_suffixes = [
        "fulltext.json", "fulltext_imrad.json", "figures.json",
        "tables.json", "equations.json", "quality_report.json", "tei.xml",
    ]
    for suffix in required_suffixes:
        if suffix == "tei.xml":
            if not (output_dir / "tei.xml").exists():
                return False
            continue
        if not list(output_dir.glob(f"*_{suffix}")):
            return False
    if not (output_dir / "figures").exists():
        return False
    return True


def call_grobid(pdf_path: Path, grobid_url: str, output_tei_path: Path):
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    try:
        with open(pdf_path, "rb") as f:
            files = {"input": (pdf_path.name, f, "application/pdf")}
            data = {
                "consolidateHeader": "1",
                "consolidateCitations": "0",
                "teiCoordinates": "figure,formula,head,table",
                "segmentSentences": "0",
            }
            response = requests.post(
                endpoint, files=files, data=data,
                timeout=(GROBID_CONNECT_TIMEOUT_S, GROBID_READ_TIMEOUT_S),
            )
        if response.status_code != 200:
            return False, f"GROBID returned HTTP {response.status_code}: {response.text[:200]}"
        if len(response.content) < 500:
            return False, f"GROBID returned suspiciously short response ({len(response.content)} bytes): {response.text[:200]}"
        if b"<TEI" not in response.content[:500]:
            return False, f"GROBID response does not contain <TEI> root: {response.text[:200]}"
        output_tei_path.parent.mkdir(parents=True, exist_ok=True)
        output_tei_path.write_bytes(response.content)
        return True, None
    except requests.exceptions.Timeout:
        return False, f"GROBID timeout after {GROBID_READ_TIMEOUT_S}s"
    except requests.exceptions.ConnectionError as e:
        return False, f"GROBID connection error: {e}"
    except Exception as e:
        return False, f"GROBID call failed: {type(e).__name__}: {e}"


def _assemble_body_text(sections: list[dict]) -> str:
    parts = []
    for s in sections:
        body = (s.get("text_no_tables") or s.get("text") or "").strip()
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def process_one_pdf(pdf_path, grobid_url, jar_path=DEFAULT_JAR_PATH, scratch_root=None) -> dict:
    start_time = time.time()
    pdf_path = Path(pdf_path).resolve()

    result: dict = {
        "success": False, "status": "failed", "pdf_path": str(pdf_path),
        "doi": "", "publisher": "", "journal": "", "elapsed_seconds": 0.0,
        "imrad_complete": False, "imrad_found": [], "methods_inferred": False,
        "n_figures": 0, "n_tables": 0, "n_schemes": 0, "n_equations": 0,
        "n_references": 0, "n_fallback_applied": 0,
        "figures_recovered_caption_scan": 0, "captions_enriched": 0,
        "figures_bbox_recovered": 0, "tables_text": 0, "tables_ocr": 0,
        "tables_unrecovered": 0, "equations_ocr": 0,
        "captions_removed_from_body": 0,
        "error": None, "stage_failed": None,
    }

    if not pdf_path.exists():
        result["error"] = f"PDF not found: {pdf_path}"
        result["stage_failed"] = "input_check"
        result["elapsed_seconds"] = time.time() - start_time
        return result

    try:
        meta = parse_path_metadata(pdf_path)
    except Exception as e:
        result["error"] = f"path metadata parse failed: {e}"
        result["stage_failed"] = "path_parse"
        result["elapsed_seconds"] = time.time() - start_time
        return result

    result["doi"] = meta["doi"]
    result["publisher"] = meta["publisher"]
    result["journal"] = meta["journal"]
    doi_folder: Path = meta["doi_folder"]


    if output_already_complete(doi_folder):
        result["success"] = True
        result["status"] = "skipped"
        result["elapsed_seconds"] = time.time() - start_time
        return result

    if scratch_root is None:
        scratch_root = Path(tempfile.gettempdir()) / f"parser_scratch_{os.environ.get('USER', 'unknown')}"
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)

    paper_scratch = scratch_root / f"pid{os.getpid()}_{uuid.uuid4().hex[:8]}"
    paper_scratch.mkdir(parents=True, exist_ok=True)
    tei_path = paper_scratch / "output.tei.xml"
    pf_figures_dir = paper_scratch / "figures"
    pf_data_dir = paper_scratch / "data"

    try:
        ok, err = call_grobid(pdf_path, grobid_url, tei_path)
        if not ok:
            result["error"] = err
            result["stage_failed"] = "grobid"
            return result

        pf_result = run_pdffigures2(
            pdf_path=pdf_path, figures_dir=pf_figures_dir, data_dir=pf_data_dir,
            jar_path=jar_path, timeout_seconds=PDFFIGURES2_TIMEOUT_S,
            apply_pymupdf_fallback=True,
        )
        if not pf_result["success"]:
            result["error"] = f"pdffigures2 failed: {pf_result.get('error')}"
            result["stage_failed"] = "pdffigures2"
            return result

        try:
            paper_data = parse_tei(tei_path)
        except Exception as e:
            result["error"] = f"TEI parse failed: {type(e).__name__}: {e}"
            result["stage_failed"] = "tei_parse"
            return result

        try:
            _out_dir = doi_folder / OUTPUT_FOLDER_NAME
            _out_dir.mkdir(parents=True, exist_ok=True)
            if tei_path.exists():
                shutil.copy2(str(tei_path), str(_out_dir / "tei.xml"))
        except Exception:
            pass

        methods_report = infer_methods_from_sections(paper_data["sections"])
        result["methods_inferred"] = methods_report.get("inferred", False)
        result["n_equations"] = sum(len(s.get("equations", []) or []) for s in paper_data["sections"])

        pf_figures = pf_result["figures"]
        tei_figs_list: list[dict] = []
        if len(pf_figures) < TEI_FIGURE_FALLBACK_THRESHOLD:
            try:
                tei_figs_list = extract_tei_figures_with_images(
                    tei_xml_path=tei_path, pdf_path=pdf_path,
                    output_figures_dir=pf_figures_dir,
                )
                pf_figures = merge_pdffigures2_and_tei(pf_figures, tei_figs_list)
            except Exception:
                pass
        else:
            try:
                tei_figs_list = parse_tei_figures(tei_path)
            except Exception:
                tei_figs_list = []

        try:
            enriched_all = map_figures_to_sections(pf_figures, paper_data["sections"])
        except Exception as e:
            result["error"] = f"figure-section mapping failed: {type(e).__name__}: {e}"
            result["stage_failed"] = "figure_mapping"
            return result

        enriched_figures = [f for f in enriched_all if (f.get("type") or "") != "table"]
        enriched_tables_initial = [f for f in enriched_all if (f.get("type") or "") == "table"]

        missing_report = find_missing_references(
            sections=paper_data["sections"],
            extracted_figures=enriched_figures + enriched_tables_initial,
        )

        if missing_report["missing_figures"]:
            try:
                if not tei_figs_list:
                    tei_figs_list = extract_tei_figures_with_images(
                        tei_xml_path=tei_path, pdf_path=pdf_path,
                        output_figures_dir=pf_figures_dir,
                    )
                merged_all = merge_pdffigures2_and_tei(
                    enriched_figures + enriched_tables_initial, tei_figs_list,
                )
                enriched_all = map_figures_to_sections(merged_all, paper_data["sections"])
                enriched_figures = [f for f in enriched_all if (f.get("type") or "") != "table"]
                enriched_tables_initial = [f for f in enriched_all if (f.get("type") or "") == "table"]
            except Exception:
                pass

        try:
            recheck = find_missing_references(
                sections=paper_data["sections"],
                extracted_figures=enriched_figures + enriched_tables_initial,
            )
            still_missing_fig_or_sch = recheck["missing_figures"] + recheck["missing_schemes"]
            if still_missing_fig_or_sch or not enriched_figures:
                have_keys = set(recheck["extracted_figure_keys"]) | set(recheck["extracted_scheme_keys"])
                body_text = _assemble_body_text(paper_data["sections"])
                raw_text = "\n\n".join(s.get("text") or "" for s in paper_data.get("sections", []))
                recovered = recover_figures_by_caption_scan(
                    pdf_path=pdf_path, body_text=body_text, raw_text=raw_text,
                    output_figures_dir=pf_figures_dir, already_have_keys=have_keys,
                )
                if recovered:
                    merged_all = merge_pdffigures2_and_tei(
                        enriched_figures + enriched_tables_initial, recovered
                    )
                    enriched_all = map_figures_to_sections(merged_all, paper_data["sections"])
                    enriched_figures = [f for f in enriched_all if (f.get("type") or "") != "table"]
                    enriched_tables_initial = [f for f in enriched_all if (f.get("type") or "") == "table"]
                    result["figures_recovered_caption_scan"] = len(recovered)
        except Exception:
            pass

        try:
            if not tei_figs_list:
                tei_figs_list = parse_tei_figures(tei_path)
            result["captions_enriched"] = enrich_stub_captions(enriched_figures, tei_figs_list)
        except Exception:
            pass

        try:
            result["figures_bbox_recovered"] = crosscheck_bbox_with_pymupdf(
                enriched_figures, pdf_path, pf_figures_dir
            )
        except Exception:
            pass

        # v7.2: also bbox-recover the TABLE detections so OCR crops are accurate.
        try:
            crosscheck_bbox_with_pymupdf(enriched_tables_initial, pdf_path, pf_figures_dir)
        except Exception:
            pass

        write_result = write_paper_outputs(
            paper_data=paper_data,
            figures_enriched=enriched_figures,
            references=paper_data.get("references", []),
            doi=meta["doi"], publisher=meta["publisher"], journal=meta["journal"],
            output_dir=doi_folder / OUTPUT_FOLDER_NAME,
            figures_source_dir=pf_figures_dir,
            methods_inference_report=methods_report,
            pdf_path=pdf_path,
            table_detections=enriched_tables_initial,
        )
        if not write_result["success"]:
            result["error"] = f"output write failed: {write_result.get('error')}"
            result["stage_failed"] = "output_write"
            return result

        try:
            if tei_path.exists():
                shutil.copy2(str(tei_path), str((doi_folder / OUTPUT_FOLDER_NAME) / "tei.xml"))
        except Exception:
            pass

        result["success"] = True
        result["status"] = "done"
        result["imrad_complete"] = write_result["imrad_complete"]
        result["imrad_found"] = write_result["imrad_found"]
        result["n_figures"] = write_result["n_figures"]
        result["n_schemes"] = write_result["n_schemes"]
        result["n_references"] = write_result["n_references"]
        result["n_tables"] = write_result["tables_total"]
        result["tables_text"] = write_result.get("tables_text", 0)
        result["tables_ocr"] = write_result.get("tables_ocr", 0)
        result["tables_unrecovered"] = write_result.get("tables_unrecovered", 0)
        result["equations_ocr"] = write_result.get("equations_ocr", 0)
        result["captions_removed_from_body"] = write_result.get("captions_removed_from_body", 0)
        result["n_fallback_applied"] = pf_result["stats"].get("n_fallback_applied", 0)
        return result

    except Exception as e:
        result["error"] = f"unexpected error: {type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"
        result["stage_failed"] = "unknown"
        return result

    finally:
        try:
            shutil.rmtree(paper_scratch, ignore_errors=True)
        except Exception:
            pass
        result["elapsed_seconds"] = time.time() - start_time


def format_log_line(result: dict, node_id: str, worker_id: int) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    error_oneline = (result.get("error") or "").replace("\n", " ").replace("\t", " ")[:300]
    total_assets = result.get("n_figures", 0) + result.get("n_schemes", 0)
    return "\t".join([
        ts, f"{node_id}/w{worker_id:02d}", result.get("status", "?"),
        result.get("pdf_path", ""), f"{result.get('elapsed_seconds', 0):.2f}",
        "imrad_ok" if result.get("imrad_complete") else "imrad_partial",
        ("methods_inferred" if result.get("methods_inferred") else "methods_direct"),
        str(total_assets),
        f"figs={result.get('n_figures', 0)}",
        f"schemes={result.get('n_schemes', 0)}",
        f"tbl={result.get('n_tables', 0)}",
        f"tbl_ocr={result.get('tables_ocr', 0)}",
        f"tbl_unrec={result.get('tables_unrecovered', 0)}",
        f"eq_ocr={result.get('equations_ocr', 0)}",
        f"cap_rm={result.get('captions_removed_from_body', 0)}",
        f"refs={result.get('n_references', 0)}",
        result.get("stage_failed") or "", error_oneline,
    ])


def worker_loop(grobid_url, work_queue_path, node_id, worker_id,
                jar_path=DEFAULT_JAR_PATH, scratch_root=None, max_papers=None):
    from coordinator import claim_next_paper, mark_paper_done, mark_paper_failed
    import time as _time
    import random

    processed = 0
    consecutive_empty_polls = 0
    consecutive_queue_errors = 0
    MAX_EMPTY_POLLS_BEFORE_EXIT = 5
    EMPTY_POLL_SLEEP_S = 10
    MAX_QUEUE_ERRORS_BEFORE_EXIT = 30

    while True:
        if max_papers is not None and processed >= max_papers:
            break
        try:
            claim = claim_next_paper(work_queue_path, claimed_by=f"{node_id}/w{worker_id:02d}")
            consecutive_queue_errors = 0
        except Exception as e:
            consecutive_queue_errors += 1
            print(f"[worker {worker_id}] queue error on claim ({consecutive_queue_errors}): {e}", flush=True)
            if consecutive_queue_errors >= MAX_QUEUE_ERRORS_BEFORE_EXIT:
                print(f"[worker {worker_id}] giving up after {consecutive_queue_errors} consecutive queue errors", flush=True)
                break
            _time.sleep(5 + random.random() * 10)
            continue

        if claim is None:
            consecutive_empty_polls += 1
            if consecutive_empty_polls >= MAX_EMPTY_POLLS_BEFORE_EXIT:
                break
            time.sleep(EMPTY_POLL_SLEEP_S)
            continue
        consecutive_empty_polls = 0

        pdf_path = claim["pdf_path"]
        try:
            result = process_one_pdf(
                pdf_path=pdf_path, grobid_url=grobid_url,
                jar_path=jar_path, scratch_root=scratch_root,
            )
        except Exception as e:
            result = {
                "success": False, "status": "failed", "pdf_path": pdf_path,
                "error": f"worker loop caught: {type(e).__name__}: {e}",
                "stage_failed": "worker_loop", "elapsed_seconds": 0.0,
                "imrad_complete": False, "imrad_found": [], "n_figures": 0,
                "n_tables": 0, "n_schemes": 0, "n_equations": 0, "n_references": 0,
                "n_fallback_applied": 0, "methods_inferred": False,
                "figures_recovered_caption_scan": 0, "doi": "", "publisher": "",
                "journal": "", "tables_ocr": 0, "tables_unrecovered": 0, "equations_ocr": 0,
                "captions_removed_from_body": 0,
            }

        print(format_log_line(result, node_id, worker_id), flush=True)

        try:
            if result["success"]:
                mark_paper_done(work_queue_path, pdf_path, result)
            else:
                mark_paper_failed(work_queue_path, pdf_path, result)
            consecutive_queue_errors = 0
        except Exception as e:
            consecutive_queue_errors += 1
            print(f"[worker {worker_id}] queue error recording outcome for {pdf_path} ({consecutive_queue_errors}): {e}", flush=True)
            if consecutive_queue_errors >= MAX_QUEUE_ERRORS_BEFORE_EXIT:
                print(f"[worker {worker_id}] giving up after {consecutive_queue_errors} consecutive queue errors", flush=True)
                break

        processed += 1


def _cli_test_single(args):
    result = process_one_pdf(
        pdf_path=args.pdf, grobid_url=args.grobid_url,
        jar_path=args.jar_path, scratch_root=args.scratch_root,
    )
    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    for k, v in result.items():
        if isinstance(v, list):
            print(f"  {k}: {v}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    sys.exit(0 if result["success"] else 1)


def _cli_loop(args):
    node_id = args.node_id or socket.gethostname()
    worker_loop(
        grobid_url=args.grobid_url, work_queue_path=args.work_queue,
        node_id=node_id, worker_id=args.worker_id, jar_path=args.jar_path,
        scratch_root=args.scratch_root, max_papers=args.max_papers,
    )


def main():
    parser = argparse.ArgumentParser(description="PDF parser worker (v8)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_test = subparsers.add_parser("test", help="Process a single PDF (no queue)")
    p_test.add_argument("pdf")
    p_test.add_argument("--grobid-url", default="http://localhost:8070")
    p_test.add_argument("--jar-path", default=DEFAULT_JAR_PATH)
    p_test.add_argument("--scratch-root", default=None)
    p_test.set_defaults(func=_cli_test_single)

    p_loop = subparsers.add_parser("loop", help="Run as a queue-driven worker")
    p_loop.add_argument("--grobid-url", required=True)
    p_loop.add_argument("--work-queue", required=True)
    p_loop.add_argument("--worker-id", type=int, required=True)
    p_loop.add_argument("--node-id", default=None)
    p_loop.add_argument("--jar-path", default=DEFAULT_JAR_PATH)
    p_loop.add_argument("--scratch-root", default=None)
    p_loop.add_argument("--max-papers", type=int, default=None)
    p_loop.set_defaults(func=_cli_loop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()