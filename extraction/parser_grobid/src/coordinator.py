"""
Coordinator and work-queue management for the parallel PDF extraction job.

v3 PATCH: file-based queue (replaces SQLite, which corrupts on
Lustre/NFS under heavy concurrent writes from 64+ workers across nodes).

v7 PATCH:
    - OUTPUT_FOLDER_NAME -> "extracted".
    - This update expects DOI folders directly under the dataset root.

Queue layout on Lustre:
    <queue_root>/
        pending/        # one tiny file per PDF, named by hash of pdf_path
        in_progress/    # claimed but not finished
        done/           # completed successfully
        failed/         # crashed during processing
        index.json      # map: hash -> {pdf_path, publisher, journal, doi}
        results/        # per-paper JSON result blobs (small)
        stats.json      # rolling status counts (rewritten by monitor)
        live.log        # human-readable activity log (rewritten by monitor)

How claiming works (NFS-safe, atomic):
    Each "task" is a tiny file named <hash>.json in the pending/ directory.
    To claim a task, a worker calls os.rename(pending/<hash>.json,
    in_progress/<hash>.json). POSIX (and Linux NFS) guarantees rename()
    is atomic for files within the same filesystem: exactly one caller
    succeeds, every other caller gets ENOENT and retries another task.

    No locks. No SQLite. No coordination overhead beyond a directory listing.

    On crash/orphan: the in_progress/ file is still there; reset_orphans
    moves it back to pending/.

Two roles in one module:
  1. Helpers used BY worker processes (running on compute nodes):
        claim_next_paper(queue_root, claimed_by) -> dict | None
        mark_paper_done(queue_root, pdf_path, result)
        mark_paper_failed(queue_root, pdf_path, result)
  2. The COORDINATOR entry point (run once, on the head node):
        build_or_reopen_queue(queue_root, dataset_root)
        reset_orphans(queue_root)             — recover from crashes
        run_coordinator(...)                  — full run: spawn nodes + monitor

This module is callable as a CLI for the coordinator role:
    python3 coordinator.py build         --dataset-root <path> --db <queue_root>
    python3 coordinator.py reset-orphans --db <queue_root>
    python3 coordinator.py status        --db <queue_root>
    python3 coordinator.py write-master  --db <queue_root> --out-dir <path>
    python3 coordinator.py monitor       --db <queue_root> --log <path> --stats-json <path>

Note: the CLI flag is named --db for back-compat with the existing PBS script;
it now points to the queue *directory* root, not a SQLite file path.
"""

from __future__ import annotations
import argparse
import csv
import errno
import hashlib
import json
import os
import random
import shutil
import sys
import time
import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Queue layout helpers
# ─────────────────────────────────────────────────────────────────────────────

PENDING_DIR = "pending"
IN_PROGRESS_DIR = "in_progress"
DONE_DIR = "done"
FAILED_DIR = "failed"
RESULTS_DIR = "results"

OUTPUT_FOLDER_NAME = "extracted"
ALL_STATE_DIRS = [PENDING_DIR, IN_PROGRESS_DIR, DONE_DIR, FAILED_DIR, RESULTS_DIR]

# Throttle for the expensive done/-scan used by the live monitor.
IMRAD_SCAN_THROTTLE_S = 60
_IMRAD_SCAN_CACHE: dict = {}


def _is_elsevier(publisher: str) -> bool:
    """Legacy helper kept for compatibility; the Paper 1 dataset is already filtered."""
    return "elsevier" in (publisher or "").strip().lower()


def _ensure_layout(queue_root: Path) -> None:
    """Create all queue subdirectories if missing."""
    queue_root.mkdir(parents=True, exist_ok=True)
    for sub in ALL_STATE_DIRS:
        (queue_root / sub).mkdir(parents=True, exist_ok=True)


def _hash_pdf_path(pdf_path: str) -> str:
    """Stable short hash used as a task filename. Sha1 is plenty for uniqueness."""
    return hashlib.sha1(pdf_path.encode("utf-8")).hexdigest()


def _task_filename(pdf_path: str) -> str:
    return _hash_pdf_path(pdf_path) + ".json"


def _atomic_write_json(target: Path, payload: dict) -> None:
    """
    Write JSON atomically: write to a temp file, then rename. Avoids partial
    files if a worker is killed mid-write.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp{os.getpid()}_{random.randint(0, 1<<30)}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(target))


def _read_json(path: Path) -> dict | None:
    """Read a JSON file; return None on any IO/parse error (file may be gone)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Worker-facing helpers (imported by worker.py)
# ─────────────────────────────────────────────────────────────────────────────

def claim_next_paper(queue_root: str | Path, claimed_by: str) -> dict | None:
    """
    Atomically claim the next pending paper.
    """
    queue_root = Path(queue_root)
    pending = queue_root / PENDING_DIR
    in_progress = queue_root / IN_PROGRESS_DIR

    MAX_TRIES = 50
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        with os.scandir(pending) as it:
            entries = []
            for i, e in enumerate(it):
                entries.append(e.name)
                if i >= 5000:
                    break
    except FileNotFoundError:
        return None

    if not entries:
        try:
            sample = next(pending.iterdir(), None)
        except FileNotFoundError:
            return None
        if sample is None:
            return None
        entries = [sample.name]

    random.shuffle(entries)
    for name in entries[:MAX_TRIES]:
        src = pending / name
        dst = in_progress / name
        try:
            os.rename(str(src), str(dst))
        except FileNotFoundError:
            continue
        except OSError as e:
            if e.errno == errno.ENOENT:
                continue
            raise

        payload = _read_json(dst)
        if payload is None:
            try:
                os.unlink(str(dst))
            except OSError:
                pass
            continue

        payload["claimed_by"] = claimed_by
        payload["claimed_at"] = now
        try:
            _atomic_write_json(dst, payload)
        except OSError:
            pass

        return {
            "pdf_path": payload["pdf_path"],
            "publisher": payload.get("publisher", ""),
            "journal": payload.get("journal", ""),
            "doi": payload.get("doi", ""),
        }

    return None


def _move_to_outcome(
    queue_root: Path,
    pdf_path: str,
    outcome_dir: str,
    result_data: dict,
) -> None:
    """Move a task from in_progress/ to done/ or failed/, merging the result."""
    task_name = _task_filename(pdf_path)

    src = queue_root / IN_PROGRESS_DIR / task_name
    dst = queue_root / outcome_dir / task_name
    results = queue_root / RESULTS_DIR / task_name

    existing = _read_json(src) or {}
    payload = {**existing, **result_data}

    try:
        _atomic_write_json(results, payload)
    except OSError:
        pass

    try:
        if src.exists():
            os.replace(str(src), str(dst))
    except FileNotFoundError:
        pass
    except OSError:
        pass

    try:
        _atomic_write_json(dst, payload)
    except OSError:
        pass


def mark_paper_done(queue_root: str | Path, pdf_path: str, result: dict) -> None:
    """Record a successful processing outcome."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "pdf_path": pdf_path,
        "publisher": result.get("publisher", ""),
        "journal": result.get("journal", ""),
        "doi": result.get("doi", ""),
        "status": "done",
        "finished_at": now,
        "imrad_complete": bool(result.get("imrad_complete")),
        "imrad_found": list(result.get("imrad_found", [])),
        "methods_inferred": bool(result.get("methods_inferred")),
        "n_figures": int(result.get("n_figures", 0) or 0),
        "n_schemes": int(result.get("n_schemes", 0) or 0),
        "n_references": int(result.get("n_references", 0) or 0),
        "n_fallback": int(result.get("n_fallback_applied", 0) or 0),
        "figures_recovered_caption_scan": int(
            result.get("figures_recovered_caption_scan", 0) or 0
        ),
        "captions_enriched": int(result.get("captions_enriched", 0) or 0),
        "figures_bbox_recovered": int(result.get("figures_bbox_recovered", 0) or 0),
        "elapsed_seconds": float(result.get("elapsed_seconds", 0.0) or 0.0),
        "error": None,
        "stage_failed": None,
    }
    _move_to_outcome(Path(queue_root), pdf_path, DONE_DIR, payload)


def mark_paper_failed(queue_root: str | Path, pdf_path: str, result: dict) -> None:
    """Record a failed processing outcome."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    err = (result.get("error") or "")[:2000]
    payload = {
        "pdf_path": pdf_path,
        "publisher": result.get("publisher", ""),
        "journal": result.get("journal", ""),
        "doi": result.get("doi", ""),
        "status": "failed",
        "finished_at": now,
        "imrad_complete": False,
        "imrad_found": [],
        "n_figures": int(result.get("n_figures", 0) or 0),
        "n_schemes": int(result.get("n_schemes", 0) or 0),
        "n_references": int(result.get("n_references", 0) or 0),
        "n_fallback": int(result.get("n_fallback_applied", 0) or 0),
        "elapsed_seconds": float(result.get("elapsed_seconds", 0.0) or 0.0),
        "error": err,
        "stage_failed": result.get("stage_failed"),
    }
    _move_to_outcome(Path(queue_root), pdf_path, FAILED_DIR, payload)


# ─────────────────────────────────────────────────────────────────────────────
# Queue building (coordinator only)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_doi_from_folder(doi_folder_name: str) -> str:
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
    try:
        meta_files = sorted(doi_folder.glob("*_Metadata.json"))
        if not meta_files:
            return {}
        return json.loads(meta_files[0].read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _parse_metadata_from_path(pdf_path: Path, dataset_root: Path) -> dict:
    """Expected layout: <dataset_root>/<doi_folder>/<doi>.pdf.

    There are no publisher or journal directories in this dataset.
    Publisher/journal are read from <doi>_Metadata.json when available.
    """
    try:
        rel = pdf_path.relative_to(dataset_root)
    except ValueError:
        return {"publisher": "", "journal": "", "doi": ""}
    parts = rel.parts
    if len(parts) < 2:
        return {"publisher": "", "journal": "", "doi": ""}
    doi_folder = parts[0]
    sidecar = _read_sidecar_metadata(dataset_root / doi_folder)
    return {
        "publisher": sidecar.get("publisher") or sidecar.get("source_publisher") or "",
        "journal": sidecar.get("journal") or sidecar.get("journal_name") or "",
        "doi": sidecar.get("doi") or _parse_doi_from_folder(doi_folder),
    }


def build_or_reopen_queue(
    queue_root: str | Path,
    dataset_root: str | Path,
    verbose: bool = True,
) -> dict:
    """
    Walk the dataset root for *.pdf files (ELSEVIER ONLY) and populate the queue
    with one tiny JSON file per PDF in pending/.

    Idempotent: if a task file already exists in ANY state directory
    (pending/in_progress/done/failed), we skip it.

    Returns stats incl. {inserted, skipped_existing, skipped_non_dataset_layout, ...}
    """
    queue_root = Path(queue_root)
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    _ensure_layout(queue_root)

    if verbose:
        print(f"Discovering PDFs under {dataset_root}...", flush=True)

    pdfs = list(dataset_root.glob("*/*.pdf"))

    if verbose:
        print(f"  found {len(pdfs)} candidate PDFs", flush=True)

    # Build a set of task filenames that already exist anywhere in the queue.
    existing: set[str] = set()
    for sub in (PENDING_DIR, IN_PROGRESS_DIR, DONE_DIR, FAILED_DIR):
        try:
            with os.scandir(queue_root / sub) as it:
                for e in it:
                    if e.name.endswith(".json"):
                        existing.add(e.name)
        except FileNotFoundError:
            pass

    inserted = 0
    skipped = 0
    skipped_non_dataset_layout = 0
    for pdf in pdfs:
        pdf_str = str(pdf)
        name = _task_filename(pdf_str)
        if name in existing:
            skipped += 1
            continue

        meta = _parse_metadata_from_path(pdf, dataset_root)


        payload = {
            "pdf_path": pdf_str,
            "publisher": meta["publisher"],
            "journal": meta["journal"],
            "doi": meta["doi"],
            "enqueued_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        target = queue_root / PENDING_DIR / name
        try:
            _atomic_write_json(target, payload)
            inserted += 1
            existing.add(name)
        except OSError as e:
            if verbose:
                print(f"  WARN: failed to enqueue {pdf_str}: {e}", flush=True)

    stats = get_queue_status(queue_root)
    stats["inserted"] = inserted
    stats["skipped_existing"] = skipped
    stats["skipped_non_dataset_layout"] = skipped_non_dataset_layout
    if verbose:
        print(f"  inserted {inserted} new, skipped {skipped} already present, "
              f"skipped {skipped_non_dataset_layout} outside expected layout", flush=True)
        print(f"  queue status: {stats}", flush=True)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Resume / orphan recovery
# ─────────────────────────────────────────────────────────────────────────────

def reset_orphans(queue_root: str | Path, verbose: bool = True) -> int:
    """
    Move any task files in in_progress/ back to pending/. These are leftovers
    from a previous crashed run.

    Also best-effort cleans up partial outputs in the dataset (the .tmp
    folders output_writer leaves behind on crash).

    Returns count of rows reset.
    """
    queue_root = Path(queue_root)
    src_dir = queue_root / IN_PROGRESS_DIR
    dst_dir = queue_root / PENDING_DIR

    if not src_dir.is_dir():
        if verbose:
            print("No in_progress directory; nothing to reset.", flush=True)
        return 0

    orphan_names: list[str] = []
    try:
        with os.scandir(src_dir) as it:
            for e in it:
                if e.name.endswith(".json"):
                    orphan_names.append(e.name)
    except FileNotFoundError:
        return 0

    if not orphan_names:
        if verbose:
            print("No orphans to reset.", flush=True)
        return 0

    if verbose:
        print(f"Found {len(orphan_names)} orphans; moving back to pending/...",
              flush=True)

    # Best-effort cleanup of partial outputs (.tmp folders from crashed writes).
    for name in orphan_names:
        payload = _read_json(src_dir / name)
        if payload is None:
            continue
        pdf_str = payload.get("pdf_path", "")
        if not pdf_str:
            continue
        doi_folder = Path(pdf_str).parent
        for candidate in [
            doi_folder / (OUTPUT_FOLDER_NAME + ".tmp"),
        ]:
            if candidate.exists() and candidate.name.endswith(".tmp"):
                shutil.rmtree(candidate, ignore_errors=True)

    reset_count = 0
    for name in orphan_names:
        src = src_dir / name
        dst = dst_dir / name
        try:
            payload = _read_json(src)
            if payload is not None:
                for k in ("claimed_by", "claimed_at"):
                    payload.pop(k, None)
                try:
                    _atomic_write_json(src, payload)
                except OSError:
                    pass
            os.rename(str(src), str(dst))
            reset_count += 1
        except FileNotFoundError:
            pass
        except OSError as e:
            if verbose:
                print(f"  WARN: failed to reset {name}: {e}", flush=True)

    if verbose:
        print(f"  reset {reset_count} tasks back to pending", flush=True)
    return reset_count


# ─────────────────────────────────────────────────────────────────────────────
# Status reporting
# ─────────────────────────────────────────────────────────────────────────────

def _count_files(path: Path) -> int:
    """Count *.json files in a directory. Returns 0 if dir is missing."""
    try:
        n = 0
        with os.scandir(path) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".json"):
                    n += 1
        return n
    except FileNotFoundError:
        return 0


def get_queue_status(queue_root: str | Path) -> dict:
    """Return current counts of papers by status."""
    queue_root = Path(queue_root)
    pending = _count_files(queue_root / PENDING_DIR)
    in_progress = _count_files(queue_root / IN_PROGRESS_DIR)
    done = _count_files(queue_root / DONE_DIR)
    failed = _count_files(queue_root / FAILED_DIR)
    total = pending + in_progress + done + failed

    out = {
        "total": total,
        "pending": pending,
        "in_progress": in_progress,
        "done": done,
        "failed": failed,
        "imrad_complete_count": 0,
        "by_publisher": {},
    }

    now_ts = time.time()
    cached = _IMRAD_SCAN_CACHE.get(str(queue_root))
    if cached is not None and (now_ts - cached["ts"]) < IMRAD_SCAN_THROTTLE_S:
        out["imrad_complete_count"] = cached["imrad_complete_count"]
        out["by_publisher"] = cached["by_publisher"]
    else:
        imrad_count = _scan_done_imrad(queue_root)
        by_pub = _scan_by_publisher(queue_root)
        out["imrad_complete_count"] = imrad_count
        out["by_publisher"] = by_pub
        _IMRAD_SCAN_CACHE[str(queue_root)] = {
            "ts": now_ts,
            "imrad_complete_count": imrad_count,
            "by_publisher": by_pub,
        }

    return out


def _scan_done_imrad(queue_root: Path) -> int:
    """Count done/ tasks where imrad_complete=True. Linear-scan."""
    n = 0
    done = queue_root / DONE_DIR
    try:
        with os.scandir(done) as it:
            for e in it:
                if not e.name.endswith(".json"):
                    continue
                p = _read_json(Path(e.path))
                if p and p.get("imrad_complete"):
                    n += 1
    except FileNotFoundError:
        pass
    return n


def _scan_by_publisher(queue_root: Path) -> dict:
    """Aggregate counts by publisher across all state directories."""
    by_pub: dict[str, dict[str, int]] = {}

    for sub, key in [
        (PENDING_DIR, "pending"),
        (IN_PROGRESS_DIR, "in_progress"),
        (DONE_DIR, "done"),
        (FAILED_DIR, "failed"),
    ]:
        path = queue_root / sub
        try:
            with os.scandir(path) as it:
                for e in it:
                    if not e.name.endswith(".json"):
                        continue
                    p = _read_json(Path(e.path))
                    if p is None:
                        continue
                    pub = (p.get("publisher") or "")
                    d = by_pub.setdefault(pub or "_unknown", {
                        "total": 0, "done": 0, "failed": 0,
                    })
                    d["total"] += 1
                    if key == "done":
                        d["done"] += 1
                    elif key == "failed":
                        d["failed"] += 1
        except FileNotFoundError:
            pass

    return by_pub


def get_failure_breakdown(queue_root: str | Path, limit: int = 10) -> dict:
    """Return counts of failures grouped by stage_failed and recent error samples."""
    queue_root = Path(queue_root)
    by_stage: dict[str, int] = {}
    samples: list[dict] = []
    failed_dir = queue_root / FAILED_DIR
    try:
        with os.scandir(failed_dir) as it:
            failed_files = []
            for e in it:
                if not e.name.endswith(".json"):
                    continue
                try:
                    mt = e.stat().st_mtime
                except OSError:
                    mt = 0.0
                failed_files.append((mt, Path(e.path)))
    except FileNotFoundError:
        return {"by_stage": {}, "recent_errors": []}

    for _, path in failed_files:
        p = _read_json(path)
        if p is None:
            continue
        stage = p.get("stage_failed") or "_unspecified"
        by_stage[stage] = by_stage.get(stage, 0) + 1

    failed_files.sort(key=lambda t: t[0], reverse=True)
    for _, path in failed_files[:limit]:
        p = _read_json(path)
        if p is None:
            continue
        samples.append({
            "pdf_path": p.get("pdf_path", ""),
            "stage_failed": p.get("stage_failed"),
            "error": (p.get("error") or "")[:200],
            "processed_at": p.get("finished_at", ""),
        })

    return {"by_stage": by_stage, "recent_errors": samples}


# ─────────────────────────────────────────────────────────────────────────────
# Live stats writer (used by the monitor thread)
# ─────────────────────────────────────────────────────────────────────────────

def render_stats_block(queue_root: str | Path) -> str:
    """Build the human-readable stats block written to extraction_live.log."""
    s = get_queue_status(queue_root)
    f = get_failure_breakdown(queue_root, limit=0)
    done = s["done"]
    failed = s["failed"]
    total = s["total"]
    remaining = s["pending"] + s["in_progress"]
    processed = done + failed
    pct = (100.0 * processed / total) if total else 0.0

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []
    lines.append("=" * 75)
    lines.append(f" EXTRACTION STATISTICS — {now}")
    lines.append("=" * 75)
    lines.append(f" Total in queue:    {total:>7d}")
    lines.append(f" Processed:         {processed:>7d}  ({pct:5.2f}%)")
    lines.append(f" Remaining:         {remaining:>7d}")
    lines.append("")
    lines.append(" Outcomes:")
    lines.append(f"   \u2713 Success:       {done:>7d}")
    lines.append(f"   \u2717 Failed:        {failed:>7d}")
    lines.append(f"   \u27f3 In progress:   {s['in_progress']:>7d}")
    lines.append("")
    if done > 0:
        imrad_pct = (100.0 * s["imrad_complete_count"] / done) if done else 0.0
        lines.append(" Of successful:")
        lines.append(f"   Full IMRaD:      {s['imrad_complete_count']:>7d}  ({imrad_pct:5.2f}%)")
        lines.append(f"   Partial IMRaD:   {done - s['imrad_complete_count']:>7d}")
    # Keep live PBS logs short. Full publisher details remain available in
    # extraction_stats.json under payload["queue"]["by_publisher"].
    publisher_group_count = len(s.get("by_publisher", {}))
    if publisher_group_count:
        lines.append("")
        lines.append(f" Publisher groups: {publisher_group_count}  (details saved in extraction_stats.json)")
    if f["by_stage"]:
        lines.append("")
        lines.append(" Failure reasons:")
        for stage, n in f["by_stage"].items():
            lines.append(f"   {stage:<20s} {n:>6d}")
    lines.append("=" * 75)
    return "\n".join(lines)


def write_stats_files(queue_root: str | Path, log_path: Path, stats_json_path: Path) -> None:
    """Append a stats block to extraction_live.log and rewrite extraction_stats.json."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stats_json_path.parent.mkdir(parents=True, exist_ok=True)

    block = render_stats_block(queue_root)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(block)
        f.write("\n")

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "queue": get_queue_status(queue_root),
        "failures": get_failure_breakdown(queue_root, limit=20),
    }
    _atomic_write_json(stats_json_path, payload)


# ─────────────────────────────────────────────────────────────────────────────
# Master file writer
# ─────────────────────────────────────────────────────────────────────────────

def write_master_files(queue_root: str | Path, out_dir: str | Path) -> dict:
    """
    Read every result blob (done/failed) and write
    extraction_master.json + extraction_master.csv.
    """
    queue_root = Path(queue_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "extraction_master.json"
    csv_path = out_dir / "extraction_master.csv"

    rows: list[dict] = []
    for sub in (DONE_DIR, FAILED_DIR):
        path = queue_root / sub
        try:
            with os.scandir(path) as it:
                for e in it:
                    if not e.name.endswith(".json"):
                        continue
                    p = _read_json(Path(e.path))
                    if p is None:
                        continue
                    rows.append(p)
        except FileNotFoundError:
            pass

    for sub, status in ((PENDING_DIR, "pending"), (IN_PROGRESS_DIR, "in_progress")):
        path = queue_root / sub
        try:
            with os.scandir(path) as it:
                for e in it:
                    if not e.name.endswith(".json"):
                        continue
                    p = _read_json(Path(e.path))
                    if p is None:
                        continue
                    p["status"] = status
                    rows.append(p)
        except FileNotFoundError:
            pass

    rows.sort(key=lambda r: (r.get("publisher", ""), r.get("journal", ""), r.get("pdf_path", "")))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_columns = [
        "pdf_path", "publisher", "journal", "doi", "status",
        "imrad_complete", "imrad_found",
        "n_figures", "n_schemes", "n_references", "n_fallback",
        "elapsed_seconds", "stage_failed", "error",
        "claimed_by", "claimed_at", "finished_at",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        for r in rows:
            writer.writerow([
                r.get("pdf_path", ""),
                r.get("publisher", ""),
                r.get("journal", ""),
                r.get("doi", ""),
                r.get("status", ""),
                "1" if r.get("imrad_complete") else "0",
                "|".join(r.get("imrad_found") or []),
                r.get("n_figures", 0) or 0,
                r.get("n_schemes", 0) or 0,
                r.get("n_references", 0) or 0,
                r.get("n_fallback", 0) or 0,
                r.get("elapsed_seconds", 0) or 0,
                r.get("stage_failed", "") or "",
                (r.get("error") or "").replace("\n", " ")[:500],
                r.get("claimed_by", "") or "",
                r.get("claimed_at", "") or "",
                r.get("finished_at", "") or "",
            ])

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "row_count": len(rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Monitor loop
# ─────────────────────────────────────────────────────────────────────────────

def monitor_loop(
    queue_root: str | Path,
    log_path: str | Path,
    stats_json_path: str | Path,
    poll_interval_s: int = 5,
    stats_every_n_papers: int = 10,
    stall_warn_minutes: int = 15,
) -> None:
    """Run the live-monitoring loop until the queue is empty."""
    queue_root = Path(queue_root)
    log_path = Path(log_path)
    stats_json_path = Path(stats_json_path)
    last_stats_threshold = 0
    stall_seen_at: dict = {}

    while True:
        try:
            s = get_queue_status(queue_root)
        except OSError as e:
            print(f"[monitor] queue read error: {e}", flush=True)
            time.sleep(poll_interval_s)
            continue

        processed = s["done"] + s["failed"]
        remaining = s["pending"] + s["in_progress"]

        if s["in_progress"] > 0 and stall_warn_minutes > 0:
            now = time.time()
            try:
                with os.scandir(queue_root / IN_PROGRESS_DIR) as it:
                    current_names = set()
                    for e in it:
                        if not e.name.endswith(".json"):
                            continue
                        current_names.add(e.name)
                        if e.name not in stall_seen_at:
                            try:
                                stall_seen_at[e.name] = e.stat().st_mtime
                            except OSError:
                                stall_seen_at[e.name] = now
                        elif now - stall_seen_at[e.name] > stall_warn_minutes * 60:
                            with open(log_path, "a", encoding="utf-8") as f:
                                age = (now - stall_seen_at[e.name]) / 60.0
                                f.write(f"[monitor] STALL WARNING: {e.name} "
                                        f"in_progress for {age:.1f} min\n")
                            stall_seen_at[e.name] = now + 9999
                    for k in list(stall_seen_at.keys()):
                        if k not in current_names:
                            del stall_seen_at[k]
            except OSError:
                pass

        threshold = (processed // stats_every_n_papers) * stats_every_n_papers
        if threshold > last_stats_threshold and processed > 0:
            try:
                write_stats_files(queue_root, log_path, stats_json_path)
            except Exception as e:
                print(f"[monitor] failed to write stats: {e}", flush=True)
            last_stats_threshold = threshold

        if remaining == 0:
            try:
                write_stats_files(queue_root, log_path, stats_json_path)
            except Exception:
                pass
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[monitor] queue drained ({s['done']} done, "
                        f"{s['failed']} failed)\n")
            return

        time.sleep(poll_interval_s)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli_build(args):
    stats = build_or_reopen_queue(args.db, args.dataset_root, verbose=True)
    print(json.dumps(stats, indent=2))


def _cli_reset_orphans(args):
    n = reset_orphans(args.db, verbose=True)
    print(f"reset {n} orphan tasks")


def _cli_status(args):
    s = get_queue_status(args.db)
    print(json.dumps(s, indent=2))


def _cli_write_master(args):
    out = write_master_files(args.db, args.out_dir)
    print(json.dumps(out, indent=2))


def _cli_monitor(args):
    monitor_loop(
        queue_root=args.db,
        log_path=args.log,
        stats_json_path=args.stats_json,
        poll_interval_s=args.poll_interval,
        stats_every_n_papers=args.stats_every,
    )


def main():
    parser = argparse.ArgumentParser(description="Coordinator / file-based work-queue manager (v7)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="Build/populate the work queue")
    p.add_argument("--db", required=True, help="Queue directory root")
    p.add_argument("--dataset-root", required=True)
    p.set_defaults(func=_cli_build)

    p = sub.add_parser("reset-orphans", help="Reset in_progress tasks to pending")
    p.add_argument("--db", required=True, help="Queue directory root")
    p.set_defaults(func=_cli_reset_orphans)

    p = sub.add_parser("status", help="Show queue status")
    p.add_argument("--db", required=True, help="Queue directory root")
    p.set_defaults(func=_cli_status)

    p = sub.add_parser("write-master", help="Write extraction_master.{json,csv}")
    p.add_argument("--db", required=True, help="Queue directory root")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=_cli_write_master)

    p = sub.add_parser("monitor", help="Run the live monitor loop")
    p.add_argument("--db", required=True, help="Queue directory root")
    p.add_argument("--log", required=True, help="Path to extraction_live.log")
    p.add_argument("--stats-json", required=True, help="Path to extraction_stats.json")
    p.add_argument("--poll-interval", type=int, default=5)
    p.add_argument("--stats-every", type=int, default=10)
    p.set_defaults(func=_cli_monitor)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()