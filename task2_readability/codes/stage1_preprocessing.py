#!/usr/bin/env python3

import csv
import json
import hashlib
import sys
import re
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is not installed in this Python environment.")
    sys.exit(1)


CONFIG_PATH = Path(
    "./task2_readability/config/stage1_config.json"
)

SCIMAGO_PATH = Path(
    "./task2_readability/dependencies/scimagojr_2024.csv"
)

OPTIONAL_MASTER_PATH = Path(
    "./task2_readability/dependencies/master.csv"
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_string(value):
    return str(value or "").strip()


def normalize_doi_key(value):
    s = clean_string(value)
    if not s:
        return ""
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def normalize_year(value):
    if value in [None, ""]:
        return ""

    text = str(value)
    tokens = text.replace("-", " ").replace("/", " ").replace(".", " ").split()

    for token in tokens:
        if token.isdigit() and len(token) == 4:
            year = int(token)
            if 1900 <= year <= 2100:
                return year

    if text.isdigit() and len(text) == 4:
        year = int(text)
        if 1900 <= year <= 2100:
            return year

    return ""


def normalize_journal_for_match(name):
    s = clean_string(name).lower()

    if not s:
        return ""

    s = s.replace("&", " and ")
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\bthe\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def journal_match_keys(name):
    base = normalize_journal_for_match(name)

    if not base:
        return []

    keys = {base}

    if base.startswith("the "):
        keys.add(base[4:])

    if base.startswith("journal of "):
        keys.add(base.replace("journal of ", "", 1))

    keys.add(base.replace(" and ", " "))
    keys.add(base.replace(" ", ""))

    return [k for k in keys if k]


def split_issns(value):
    text = clean_string(value)

    if not text:
        return []

    parts = re.split(r"[,;\s]+", text)
    out = []

    for p in parts:
        p = p.strip().upper()

        if not p:
            continue

        p = re.sub(r"[^0-9X]", "", p)

        if len(p) == 8:
            out.append(f"{p[:4]}-{p[4:]}")
            out.append(p)
        elif p:
            out.append(p)

    return sorted(set(out))


def read_csv_rows(path):
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8-sig", newline="", errors="replace") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except Exception:
            first_line = sample.splitlines()[0] if sample else ""
            if first_line.count(";") > first_line.count(","):
                dialect = csv.excel()
                dialect.delimiter = ";"
            elif first_line.count("\t") > first_line.count(","):
                dialect = csv.excel()
                dialect.delimiter = "\t"
            else:
                dialect = csv.excel()

        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)


def pick(row, names, default=""):
    lower_map = {
        str(k).strip().lower(): k
        for k in row.keys()
        if k is not None and str(k).strip()
    }

    for name in names:
        key = lower_map.get(str(name).strip().lower())
        if key is not None and clean_string(row.get(key)):
            return clean_string(row.get(key))

    return default


def read_json(path):
    if not path or not Path(path).exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_nested(data, keys, default=""):
    for key in keys:
        cur = data
        ok = True

        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break

        if ok and cur not in [None, "", [], {}]:
            return cur

    return default


def first_nonempty(*values):
    for value in values:
        s = clean_string(value)
        if s:
            return s
    return ""


def write_csv(path, rows, fieldnames):
    path = Path(path)
    make_dir(path.parent)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# OUTPUT RESET
# =============================================================================

def reset_outputs(output_root):
    output_root = Path(output_root)

    for sub in ["index", "image_metadata", "duplicates", "summaries", "reports"]:
        make_dir(output_root / sub)

    files = [
        output_root / "index" / "stage1_ga_index.csv",
        output_root / "index" / "stage1_ga_index.jsonl",
        output_root / "index" / "stage1_missing_files_report.csv",
        output_root / "index" / "stage1_unmatched_domain_report.csv",

        output_root / "image_metadata" / "stage1_image_metadata.csv",
        output_root / "image_metadata" / "stage1_corrupted_images.csv",
        output_root / "image_metadata" / "stage1_low_resolution_images.csv",
        output_root / "image_metadata" / "stage1_image_format_summary.csv",

        output_root / "duplicates" / "stage1_exact_duplicate_images.csv",
        output_root / "duplicates" / "stage1_near_duplicate_images.csv",
        output_root / "duplicates" / "stage1_duplicate_cluster_summary.csv",

        output_root / "summaries" / "stage1_year_summary.csv",
        output_root / "summaries" / "stage1_domain_summary.csv",
        output_root / "summaries" / "stage1_journal_summary.csv",
        output_root / "summaries" / "stage1_publisher_summary.csv",
        output_root / "summaries" / "stage1_year_domain_summary.csv",
        output_root / "summaries" / "stage1_dataset_health_summary.csv",

        output_root / "reports" / "stage1_preprocessing_report.txt",
    ]

    for fp in files:
        if fp.exists():
            fp.unlink()


# =============================================================================
# SCIMAGO + OPTIONAL MASTER LOADING
# =============================================================================

def load_scimago():
    by_issn = {}
    by_title = {}

    rows = read_csv_rows(SCIMAGO_PATH)

    for row in rows:
        title = pick(row, [
            "Title", "title", "Source Title", "Source title", "Source", "Journal"
        ])

        areas = pick(row, [
            "Areas", "Subject Area", "Subject area", "subject_area", "Subject Areas"
        ])

        categories = pick(row, [
            "Categories", "Subject Categories", "Subject categories", "subject_categories"
        ])

        issn = pick(row, [
            "Issn", "ISSN", "issn", "Issns", "ISSNs"
        ])

        publisher = pick(row, ["Publisher", "publisher"])
        sourceid = pick(row, ["Sourceid", "Source ID", "sourceid"])
        source_type = pick(row, ["Type", "type"])

        if not title and not issn:
            continue

        entry = {
            "title": title,
            "areas": areas,
            "categories": categories,
            "issns": split_issns(issn),
            "publisher": publisher,
            "sourceid": sourceid,
            "type": source_type,
        }

        for one_issn in entry["issns"]:
            by_issn[one_issn] = entry
            by_issn[one_issn.replace("-", "")] = entry

        for key in journal_match_keys(title):
            by_title[key] = entry

    return by_issn, by_title


def load_optional_master():
    by_folder = {}
    by_doi = {}

    if not OPTIONAL_MASTER_PATH.exists():
        return by_folder, by_doi

    for row in read_csv_rows(OPTIONAL_MASTER_PATH):
        folder = pick(row, ["doi_folder", "folder", "paper_id", "doi_key"])
        doi = pick(row, ["doi", "metadata_doi", "DOI", "article_doi"])

        if folder:
            by_folder[normalize_doi_key(folder)] = row

        if doi:
            by_doi[normalize_doi_key(doi)] = row

    return by_folder, by_doi


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_metadata_file(folder, metadata_suffix):
    expected = folder / f"{folder.name}{metadata_suffix}"

    if expected.exists():
        return expected, ""

    matches = sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.name.endswith(metadata_suffix)
    ])

    if not matches:
        return None, ""

    if len(matches) == 1:
        return matches[0], ""

    return matches[0], f"Multiple metadata files found; selected {matches[0].name}"


def find_pdf_file(folder, pdf_extension):
    expected = folder / f"{folder.name}{pdf_extension}"

    if expected.exists():
        return expected, ""

    matches = sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == pdf_extension.lower()
    ])

    if not matches:
        return None, ""

    if len(matches) == 1:
        return matches[0], ""

    return matches[0], f"Multiple PDF files found; selected {matches[0].name}"


def find_ga_file(folder, accepted_extensions, keywords):
    patterns = [
        f"{folder.name}_Graphical_Abstract.jpg",
        f"{folder.name}_Graphical_Abstract.jpeg",
        f"{folder.name}_Graphical_Abstract.png",
        "*_Graphical_Abstract.jpg",
        "*_Graphical_Abstract.jpeg",
        "*_Graphical_Abstract.png",
        "*graphical*abstract*.jpg",
        "*graphical*abstract*.jpeg",
        "*graphical*abstract*.png",
    ]

    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0], ""

    accepted_extensions = [x.lower() for x in accepted_extensions]
    keywords = [x.lower() for x in keywords]

    candidates = []

    for p in folder.iterdir():
        if not p.is_file():
            continue

        if p.suffix.lower() not in accepted_extensions:
            continue

        name_lower = p.name.lower()

        if any(k in name_lower for k in keywords):
            candidates.append(p)

    if not candidates:
        return None, ""

    selected = candidates[0]

    note = ""
    if len(candidates) > 1:
        note = f"Multiple GA candidates found; selected {selected.name}"

    return selected, note


def find_extracted_outputs(folder):
    extracted_dir = folder / "extracted"

    if not extracted_dir.exists() or not extracted_dir.is_dir():
        return {
            "extracted_dir": None,
            "fulltext_json": None,
            "fulltext_imrad_json": None,
        }

    fulltext_json = extracted_dir / "fulltext.json"
    fulltext_imrad_json = extracted_dir / "fulltext_imrad.json"

    if not fulltext_json.exists():
        matches = sorted(extracted_dir.glob("*fulltext.json"))
        fulltext_json = matches[0] if matches else None

    if not fulltext_imrad_json.exists():
        matches = sorted(extracted_dir.glob("*fulltext_imrad.json"))
        fulltext_imrad_json = matches[0] if matches else None

    return {
        "extracted_dir": extracted_dir,
        "fulltext_json": fulltext_json if fulltext_json and fulltext_json.exists() else None,
        "fulltext_imrad_json": fulltext_imrad_json if fulltext_imrad_json and fulltext_imrad_json.exists() else None,
    }


# =============================================================================
# METADATA + DOMAIN LOGIC
# =============================================================================

def metadata_fields(meta, folder):
    doi = get_nested(meta, [
        "doi", "DOI", "prism:doi", "coredata.prism:doi",
        "dc:identifier", "dc.identifier.doi"
    ])

    if isinstance(doi, str):
        doi = doi.replace("doi:", "").strip()

    title = get_nested(meta, [
        "title", "dc:title", "coredata.dc:title",
        "coredata.title", "article_title"
    ])

    journal = get_nested(meta, [
        "journal", "journal_name", "publicationName",
        "prism:publicationName", "prism.publicationName",
        "coredata.prism:publicationName"
    ])

    publisher = get_nested(meta, [
        "publisher", "dc:publisher", "coredata.dc:publisher"
    ])

    year_raw = get_nested(meta, [
        "year", "publication_year", "coverDate", "prism:coverDate",
        "prism.coverDate", "coredata.prism:coverDate",
        "published", "published_date", "publication_date"
    ])

    issn = get_nested(meta, [
        "issn", "ISSN", "prism:issn", "prism.issn",
        "coredata.prism:issn"
    ])

    issn_l = get_nested(meta, [
        "issn_l", "issn-l", "ISSN-L"
    ])

    eissn = get_nested(meta, [
        "eIssn", "eissn", "electronic_issn",
        "prism:eIssn", "prism.eIssn"
    ])

    pissn = get_nested(meta, [
        "pIssn", "pissn", "print_issn",
        "prism:pIssn", "prism.pIssn"
    ])

    pdf_source = get_nested(meta, [
        "pdf_source", "pdfSource", "source"
    ])

    return {
        "metadata_doi": clean_string(doi) or folder.name.replace("_", "/", 1),
        "title": clean_string(title),
        "journal": clean_string(journal),
        "publisher": clean_string(publisher),
        "year_raw": clean_string(year_raw),
        "publication_year": normalize_year(year_raw),
        "issn": clean_string(issn),
        "issn_l": clean_string(issn_l),
        "eIssn": clean_string(eissn),
        "pIssn": clean_string(pissn),
        "source_issn": "",
        "pdf_source": clean_string(pdf_source),
    }


def parser_metadata_fallback(fulltext_doc, imrad_doc, key):
    return first_nonempty(
        fulltext_doc.get(key),
        imrad_doc.get(key),

        (fulltext_doc.get("metadata") or {}).get(key)
        if isinstance(fulltext_doc.get("metadata"), dict) else "",

        (imrad_doc.get("metadata") or {}).get(key)
        if isinstance(imrad_doc.get("metadata"), dict) else "",

        (fulltext_doc.get("article_metadata") or {}).get(key)
        if isinstance(fulltext_doc.get("article_metadata"), dict) else "",

        (imrad_doc.get("article_metadata") or {}).get(key)
        if isinstance(imrad_doc.get("article_metadata"), dict) else "",
    )


def enrich_metadata(meta_f, fulltext_doc, imrad_doc, master_row):
    master_row = master_row or {}
    out = dict(meta_f)

    parser_journal = first_nonempty(
        parser_metadata_fallback(fulltext_doc, imrad_doc, "journal"),
        parser_metadata_fallback(fulltext_doc, imrad_doc, "publicationName"),
        parser_metadata_fallback(fulltext_doc, imrad_doc, "container-title"),
        parser_metadata_fallback(fulltext_doc, imrad_doc, "sourceTitle"),
    )

    master_journal = pick(master_row, [
        "journal", "publicationName", "publication_name", "source_title",
        "source title", "container-title", "container_title", "scimago_title"
    ])

    out["journal"] = first_nonempty(out.get("journal"), parser_journal, master_journal)

    parser_title = parser_metadata_fallback(fulltext_doc, imrad_doc, "title")
    master_title = pick(master_row, ["title", "article_title", "dc:title"])
    out["title"] = first_nonempty(out.get("title"), parser_title, master_title)

    master_year = pick(master_row, [
        "year_clean", "year", "publication_year",
        "published_year", "coverDate", "cover_date"
    ])

    if not out.get("publication_year") and master_year:
        out["year_raw"] = master_year
        out["publication_year"] = normalize_year(master_year)

    out["issn"] = first_nonempty(
        out.get("issn"),
        pick(master_row, ["issn", "ISSN", "source_issn"])
    )

    out["issn_l"] = first_nonempty(
        out.get("issn_l"),
        pick(master_row, ["issn_l", "issn-l", "ISSN-L"])
    )

    out["eIssn"] = first_nonempty(
        out.get("eIssn"),
        pick(master_row, ["eIssn", "eissn", "electronic_issn", "prism:eIssn"])
    )

    out["pIssn"] = first_nonempty(
        out.get("pIssn"),
        pick(master_row, ["pIssn", "pissn", "print_issn", "prism:pIssn"])
    )

    return out


def domain_return(entry, source, source_issn="", score=""):
    return {
        "domain": entry.get("areas", "") or "unknown/unmatched",
        "subject_area": entry.get("areas", "") or "unknown/unmatched",
        "subject_categories": entry.get("categories", "") or "unknown/unmatched",
        "scimago_title": entry.get("title", ""),
        "scimago_sourceid": entry.get("sourceid", ""),
        "scimago_type": entry.get("type", ""),
        "scimago_issn": "; ".join(entry.get("issns", [])),
        "scimago_publisher": entry.get("publisher", ""),
        "source_issn": source_issn,
        "domain_match_source": source,
        "domain_match_score": score,
    }


def domain_fields(meta_f, scimago_by_issn, scimago_by_title):
    candidate_issns = []

    for key in ["issn", "issn_l", "eIssn", "pIssn"]:
        candidate_issns.extend(split_issns(meta_f.get(key, "")))

    for issn in candidate_issns:
        entry = scimago_by_issn.get(issn) or scimago_by_issn.get(issn.replace("-", ""))
        if entry:
            return domain_return(entry, "scimago_issn", issn, "1.000")

    journal = meta_f.get("journal", "")

    for key in journal_match_keys(journal):
        entry = scimago_by_title.get(key)
        if entry:
            return domain_return(entry, "scimago_name_exact", "", "1.000")

    journal_key = normalize_journal_for_match(journal)

    if journal_key and len(journal_key) >= 8:
        best_entry = None
        best_score = 0.0

        for key, entry in scimago_by_title.items():
            if not key or len(key) < 8 or " " not in key:
                continue

            score = SequenceMatcher(None, journal_key, key).ratio()

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= 0.94:
            return domain_return(
                best_entry,
                "scimago_name_fuzzy",
                "",
                f"{best_score:.3f}"
            )

    return {
        "domain": "unknown/unmatched",
        "subject_area": "unknown/unmatched",
        "subject_categories": "unknown/unmatched",
        "scimago_title": "",
        "scimago_sourceid": "",
        "scimago_type": "",
        "scimago_issn": "",
        "scimago_publisher": "",
        "source_issn": "",
        "domain_match_source": "unmatched",
        "domain_match_score": "",
    }


# =============================================================================
# IMAGE METADATA
# =============================================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def simple_dhash(img, hash_size=8):
    img = img.convert("L")
    img = img.resize((hash_size + 1, hash_size))

    pixels = list(img.getdata())
    bits = []

    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits.append(1 if left > right else 0)

    value = 0

    for bit in bits:
        value = (value << 1) | bit

    return f"{value:016x}"


def hamming_distance_hex(a, b):
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:
        return 999


# =============================================================================
# INDEX BUILDING
# =============================================================================

def build_index(config, scimago_by_issn, scimago_by_title, master_by_folder, master_by_doi):
    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_root"])

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    index_rows = []
    missing_rows = []
    unmatched_rows = []

    doi_folders = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    for folder in doi_folders:
        paper_id = folder.name
        notes = []

        metadata_path, metadata_note = find_metadata_file(folder, config["metadata_suffix"])
        pdf_path, pdf_note = find_pdf_file(folder, config["pdf_extension"])
        ga_path, ga_note = find_ga_file(
            folder,
            config["accepted_ga_extensions"],
            config["expected_ga_name_keywords"]
        )

        if metadata_note:
            notes.append(metadata_note)
        if pdf_note:
            notes.append(pdf_note)
        if ga_note:
            notes.append(ga_note)

        extracted = find_extracted_outputs(folder)
        extracted_folder = extracted.get("extracted_dir")

        metadata = {}
        metadata_parse_status = "missing"

        if metadata_path:
            metadata = read_json(metadata_path)
            metadata_parse_status = "parsed" if metadata else "parse_failed"

        meta_f = metadata_fields(metadata, folder)

        doi_key = normalize_doi_key(meta_f.get("metadata_doi")) or normalize_doi_key(folder.name)
        folder_key = normalize_doi_key(folder.name)

        master_row = master_by_doi.get(doi_key) or master_by_folder.get(folder_key)

        fulltext_doc = read_json(extracted.get("fulltext_json"))
        imrad_doc = read_json(extracted.get("fulltext_imrad_json"))

        meta_f = enrich_metadata(meta_f, fulltext_doc, imrad_doc, master_row)

        dom = domain_fields(meta_f, scimago_by_issn, scimago_by_title)
        meta_f["source_issn"] = dom.get("source_issn", "")

        article_type = get_nested(metadata, [
            "article_type", "type", "subtype", "prism.aggregationType"
        ])

        source_platform = get_nested(metadata, [
            "source_platform", "platform"
        ])

        oa_status = get_nested(metadata, [
            "oa_status", "open_access_status", "oa"
        ])

        license_value = get_nested(metadata, [
            "license", "license_url", "open_access_license"
        ])

        has_pdf = int(pdf_path is not None)
        has_ga = int(ga_path is not None)
        has_metadata = int(metadata_path is not None)
        has_extracted_folder = int(extracted_folder is not None)

        missing_fields = []

        if not meta_f.get("publication_year"):
            missing_fields.append("publication_year")
        if not meta_f.get("journal"):
            missing_fields.append("journal")
        if not meta_f.get("publisher"):
            missing_fields.append("publisher")
        if dom["domain_match_source"] == "unmatched":
            missing_fields.append("domain")

        is_complete_for_stage1 = int(
            has_ga == 1
            and has_metadata == 1
            and meta_f.get("publication_year") != ""
        )

        row = {
            "paper_id": paper_id,
            "doi_folder_name": folder.name,
            "doi": meta_f.get("metadata_doi", ""),
            "doi_normalized": normalize_doi_key(meta_f.get("metadata_doi", "")),
            "dataset_folder_path": str(folder),
            "pdf_path": str(pdf_path) if pdf_path else "",
            "ga_path": str(ga_path) if ga_path else "",
            "metadata_path": str(metadata_path) if metadata_path else "",
            "extracted_folder_path": str(extracted_folder) if extracted_folder else "",

            "has_pdf": has_pdf,
            "has_ga": has_ga,
            "has_metadata": has_metadata,
            "has_extracted_folder": has_extracted_folder,
            "is_complete_for_stage1": is_complete_for_stage1,

            "publication_year": meta_f.get("publication_year", ""),
            "journal": meta_f.get("journal", ""),
            "publisher": meta_f.get("publisher", ""),
            "domain": dom.get("domain", ""),
            "subject_area": dom.get("subject_area", ""),
            "subject_categories": dom.get("subject_categories", ""),
            "article_type": clean_string(article_type),
            "source_platform": clean_string(source_platform),
            "pdf_source": meta_f.get("pdf_source", ""),
            "oa_status": clean_string(oa_status),
            "license": clean_string(license_value),
            "title": meta_f.get("title", ""),

            "issn": meta_f.get("issn", ""),
            "issn_l": meta_f.get("issn_l", ""),
            "eIssn": meta_f.get("eIssn", ""),
            "pIssn": meta_f.get("pIssn", ""),
            "source_issn": meta_f.get("source_issn", ""),

            "scimago_match_status": dom.get("domain_match_source", ""),
            "domain_match_source": dom.get("domain_match_source", ""),
            "domain_match_score": dom.get("domain_match_score", ""),
            "scimago_title": dom.get("scimago_title", ""),
            "scimago_sourceid": dom.get("scimago_sourceid", ""),
            "scimago_type": dom.get("scimago_type", ""),
            "scimago_issn": dom.get("scimago_issn", ""),
            "scimago_publisher": dom.get("scimago_publisher", ""),
            "scimago_areas": dom.get("subject_area", ""),
            "scimago_categories": dom.get("subject_categories", ""),

            "metadata_parse_status": metadata_parse_status,
            "metadata_missing_fields": ";".join(missing_fields),
            "stage1_index_status": "complete" if is_complete_for_stage1 else "incomplete",
            "stage1_index_notes": " | ".join(notes),
        }

        index_rows.append(row)

        if dom["domain_match_source"] == "unmatched":
            unmatched_rows.append({
                "paper_id": paper_id,
                "journal": meta_f.get("journal", ""),
                "issn": meta_f.get("issn", ""),
                "issn_l": meta_f.get("issn_l", ""),
                "eIssn": meta_f.get("eIssn", ""),
                "pIssn": meta_f.get("pIssn", ""),
                "metadata_path": str(metadata_path) if metadata_path else "",
                "notes": "domain unmatched after ISSN, exact-title, and fuzzy matching",
            })

        if not has_pdf or not has_ga or not has_metadata or not has_extracted_folder:
            missing_rows.append({
                "paper_id": paper_id,
                "doi_folder_name": folder.name,
                "missing_pdf": int(not has_pdf),
                "missing_ga": int(not has_ga),
                "missing_metadata": int(not has_metadata),
                "missing_extracted_folder": int(not has_extracted_folder),
                "folder_path": str(folder),
                "notes": " | ".join(notes),
            })

    index_fields = [
        "paper_id", "doi_folder_name", "doi", "doi_normalized",
        "dataset_folder_path", "pdf_path", "ga_path", "metadata_path",
        "extracted_folder_path", "has_pdf", "has_ga", "has_metadata",
        "has_extracted_folder", "is_complete_for_stage1",

        "publication_year", "journal", "publisher", "domain",
        "subject_area", "subject_categories", "article_type", "source_platform",
        "pdf_source", "oa_status", "license", "title",

        "issn", "issn_l", "eIssn", "pIssn", "source_issn",

        "scimago_match_status", "domain_match_source", "domain_match_score",
        "scimago_title", "scimago_sourceid", "scimago_type",
        "scimago_issn", "scimago_publisher", "scimago_areas",
        "scimago_categories",

        "metadata_parse_status", "metadata_missing_fields",
        "stage1_index_status", "stage1_index_notes",
    ]

    write_csv(
        output_root / "index" / "stage1_ga_index.csv",
        index_rows,
        index_fields
    )

    with open(output_root / "index" / "stage1_ga_index.jsonl", "w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(
        output_root / "index" / "stage1_missing_files_report.csv",
        missing_rows,
        [
            "paper_id", "doi_folder_name", "missing_pdf", "missing_ga",
            "missing_metadata", "missing_extracted_folder", "folder_path", "notes"
        ]
    )

    write_csv(
        output_root / "index" / "stage1_unmatched_domain_report.csv",
        unmatched_rows,
        [
            "paper_id", "journal", "issn", "issn_l", "eIssn",
            "pIssn", "metadata_path", "notes"
        ]
    )

    return index_rows


# =============================================================================
# IMAGE EXTRACTION + DUPLICATES
# =============================================================================

def extract_image_metadata(config, index_rows):
    output_root = Path(config["output_root"])
    image_rows = []
    corrupted_rows = []
    low_resolution_rows = []

    for row in index_rows:
        paper_id = row["paper_id"]
        ga_path = row["ga_path"]

        image_row = {
            "paper_id": paper_id,
            "ga_path": ga_path,
            "image_open_status": "not_attempted",
            "image_width": "",
            "image_height": "",
            "pixel_count": "",
            "aspect_ratio": "",
            "file_size_bytes": "",
            "file_extension": "",
            "detected_image_format": "",
            "color_mode": "",
            "num_channels": "",
            "has_alpha_channel": "",
            "is_corrupted": 0,
            "corruption_error_message": "",
            "low_resolution_flag": "",
            "extremely_low_resolution_flag": "",
            "very_wide_flag": "",
            "very_tall_flag": "",
            "abnormal_aspect_ratio_flag": "",
            "tiny_file_flag": "",
            "huge_file_flag": "",
            "exact_image_hash": "",
            "perceptual_hash": "",
            "duplicate_cluster_id": "",
            "is_exact_duplicate": 0,
            "is_near_duplicate": 0,
            "stage1_image_status": "",
            "stage1_image_notes": "",
        }

        if not ga_path:
            image_row["image_open_status"] = "missing_ga"
            image_row["is_corrupted"] = 1
            image_row["corruption_error_message"] = "GA path missing"
            image_row["stage1_image_status"] = "invalid"
            image_rows.append(image_row)
            corrupted_rows.append(image_row.copy())
            continue

        path = Path(ga_path)

        try:
            image_row["file_size_bytes"] = path.stat().st_size
            image_row["file_extension"] = path.suffix.lower()
            image_row["exact_image_hash"] = sha256_file(path)

            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                width, height = img.size
                pixel_count = width * height
                aspect_ratio = width / height if height else 0

                image_row["image_open_status"] = "opened"
                image_row["image_width"] = width
                image_row["image_height"] = height
                image_row["pixel_count"] = pixel_count
                image_row["aspect_ratio"] = round(aspect_ratio, 6)
                image_row["detected_image_format"] = img.format
                image_row["color_mode"] = img.mode
                image_row["num_channels"] = len(img.getbands())
                image_row["has_alpha_channel"] = int("A" in img.getbands())
                image_row["perceptual_hash"] = simple_dhash(img)

                low_resolution = int(
                    width < config["low_resolution_width_threshold"]
                    or height < config["low_resolution_height_threshold"]
                    or pixel_count < config["low_resolution_pixel_threshold"]
                )

                extremely_low_resolution = int(
                    width < config["extreme_low_resolution_width_threshold"]
                    or height < config["extreme_low_resolution_height_threshold"]
                    or pixel_count < config["extreme_low_resolution_pixel_threshold"]
                )

                very_wide = int(aspect_ratio > config["very_wide_aspect_ratio_threshold"])
                very_tall = int(aspect_ratio < config["very_tall_aspect_ratio_threshold"])

                abnormal_aspect_ratio = int(
                    aspect_ratio > config["abnormal_wide_aspect_ratio_threshold"]
                    or aspect_ratio < config["abnormal_tall_aspect_ratio_threshold"]
                )

                tiny_file = int(image_row["file_size_bytes"] < config["tiny_file_threshold_bytes"])
                huge_file = int(image_row["file_size_bytes"] > config["huge_file_threshold_bytes"])

                image_row["low_resolution_flag"] = low_resolution
                image_row["extremely_low_resolution_flag"] = extremely_low_resolution
                image_row["very_wide_flag"] = very_wide
                image_row["very_tall_flag"] = very_tall
                image_row["abnormal_aspect_ratio_flag"] = abnormal_aspect_ratio
                image_row["tiny_file_flag"] = tiny_file
                image_row["huge_file_flag"] = huge_file

                notes = []

                if low_resolution:
                    notes.append("low_resolution")
                if extremely_low_resolution:
                    notes.append("extremely_low_resolution")
                if very_wide:
                    notes.append("very_wide")
                if very_tall:
                    notes.append("very_tall")
                if abnormal_aspect_ratio:
                    notes.append("abnormal_aspect_ratio")
                if tiny_file:
                    notes.append("tiny_file")
                if huge_file:
                    notes.append("huge_file")

                image_row["stage1_image_status"] = "valid"
                image_row["stage1_image_notes"] = ";".join(notes)

        except Exception as e:
            image_row["image_open_status"] = "failed"
            image_row["is_corrupted"] = 1
            image_row["corruption_error_message"] = str(e)
            image_row["stage1_image_status"] = "invalid"

        image_rows.append(image_row)

        if image_row["is_corrupted"] == 1:
            corrupted_rows.append(image_row.copy())

        if image_row["low_resolution_flag"] == 1:
            low_resolution_rows.append(image_row.copy())

    assign_duplicate_flags(config, image_rows)

    image_fields = [
        "paper_id", "ga_path", "image_open_status", "image_width",
        "image_height", "pixel_count", "aspect_ratio", "file_size_bytes",
        "file_extension", "detected_image_format", "color_mode",
        "num_channels", "has_alpha_channel", "is_corrupted",
        "corruption_error_message", "low_resolution_flag",
        "extremely_low_resolution_flag", "very_wide_flag", "very_tall_flag",
        "abnormal_aspect_ratio_flag", "tiny_file_flag", "huge_file_flag",
        "exact_image_hash", "perceptual_hash", "duplicate_cluster_id",
        "is_exact_duplicate", "is_near_duplicate", "stage1_image_status",
        "stage1_image_notes"
    ]

    write_csv(output_root / "image_metadata" / "stage1_image_metadata.csv", image_rows, image_fields)
    write_csv(output_root / "image_metadata" / "stage1_corrupted_images.csv", corrupted_rows, image_fields)
    write_csv(output_root / "image_metadata" / "stage1_low_resolution_images.csv", low_resolution_rows, image_fields)

    return image_rows


def assign_duplicate_flags(config, image_rows):
    valid_rows = [r for r in image_rows if r["stage1_image_status"] == "valid"]

    exact_groups = defaultdict(list)

    for row in valid_rows:
        exact_groups[row["exact_image_hash"]].append(row)

    exact_cluster_number = 1

    for exact_hash, rows in exact_groups.items():
        if len(rows) > 1:
            cluster_id = f"exact_dup_{exact_cluster_number}"
            exact_cluster_number += 1

            for row in rows:
                row["duplicate_cluster_id"] = cluster_id
                row["is_exact_duplicate"] = 1

    threshold = int(config["near_duplicate_hash_distance_threshold"])
    phash_rows = [r for r in valid_rows if r["perceptual_hash"]]

    visited = set()
    near_cluster_number = 1

    for i, row1 in enumerate(phash_rows):
        if row1["paper_id"] in visited:
            continue

        cluster = [row1]
        visited.add(row1["paper_id"])

        for row2 in phash_rows[i + 1:]:
            if row2["paper_id"] in visited:
                continue

            distance = hamming_distance_hex(row1["perceptual_hash"], row2["perceptual_hash"])

            if distance <= threshold:
                cluster.append(row2)
                visited.add(row2["paper_id"])

        if len(cluster) > 1:
            cluster_id = f"near_dup_{near_cluster_number}"
            near_cluster_number += 1

            for row in cluster:
                if not row["duplicate_cluster_id"]:
                    row["duplicate_cluster_id"] = cluster_id
                row["is_near_duplicate"] = 1


def write_duplicate_reports(config, image_rows):
    output_root = Path(config["output_root"])

    exact_groups = defaultdict(list)
    near_groups = defaultdict(list)

    for row in image_rows:
        if row["is_exact_duplicate"] == 1:
            exact_groups[row["exact_image_hash"]].append(row)

        if row["is_near_duplicate"] == 1:
            near_groups[row["duplicate_cluster_id"]].append(row)

    exact_rows = []

    for exact_hash, rows in exact_groups.items():
        exact_rows.append({
            "exact_image_hash": exact_hash,
            "duplicate_count": len(rows),
            "paper_ids": ";".join(r["paper_id"] for r in rows),
            "ga_paths": ";".join(r["ga_path"] for r in rows),
        })

    near_rows = []

    for cluster_id, rows in near_groups.items():
        near_rows.append({
            "duplicate_cluster_id": cluster_id,
            "cluster_size": len(rows),
            "paper_ids": ";".join(r["paper_id"] for r in rows),
            "ga_paths": ";".join(r["ga_path"] for r in rows),
            "hash_distance_summary": "simple_dhash_cluster",
        })

    summary_rows = [
        {
            "duplicate_type": "exact",
            "num_clusters": len(exact_rows),
            "num_images_in_clusters": sum(r["duplicate_count"] for r in exact_rows),
        },
        {
            "duplicate_type": "near",
            "num_clusters": len(near_rows),
            "num_images_in_clusters": sum(r["cluster_size"] for r in near_rows),
        },
    ]

    write_csv(
        output_root / "duplicates" / "stage1_exact_duplicate_images.csv",
        exact_rows,
        ["exact_image_hash", "duplicate_count", "paper_ids", "ga_paths"]
    )

    write_csv(
        output_root / "duplicates" / "stage1_near_duplicate_images.csv",
        near_rows,
        ["duplicate_cluster_id", "cluster_size", "paper_ids", "ga_paths", "hash_distance_summary"]
    )

    write_csv(
        output_root / "duplicates" / "stage1_duplicate_cluster_summary.csv",
        summary_rows,
        ["duplicate_type", "num_clusters", "num_images_in_clusters"]
    )


# =============================================================================
# SUMMARIES + REPORT
# =============================================================================

def merge_index_image_rows(index_rows, image_rows):
    image_by_id = {row["paper_id"]: row for row in image_rows}
    merged_rows = []

    for row in index_rows:
        merged = dict(row)
        image_row = image_by_id.get(row["paper_id"], {})

        for key, value in image_row.items():
            merged[f"image_{key}"] = value

        merged_rows.append(merged)

    return merged_rows


def make_group_summary(merged_rows, group_column):
    groups = defaultdict(list)

    for row in merged_rows:
        key = row.get(group_column) or "UNKNOWN"
        groups[key].append(row)

    output_rows = []

    for group_name, rows in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        years = [
            int(r["publication_year"])
            for r in rows
            if str(r.get("publication_year", "")).isdigit()
        ]

        widths = [
            int(r["image_image_width"])
            for r in rows
            if str(r.get("image_image_width", "")).isdigit()
        ]

        heights = [
            int(r["image_image_height"])
            for r in rows
            if str(r.get("image_image_height", "")).isdigit()
        ]

        pixels = [
            int(r["image_pixel_count"])
            for r in rows
            if str(r.get("image_pixel_count", "")).isdigit()
        ]

        output_rows.append({
            group_column: group_name,
            "num_papers": len(rows),
            "publication_year_min": min(years) if years else "",
            "publication_year_max": max(years) if years else "",
            "num_valid_ga_images": sum(1 for r in rows if r.get("image_stage1_image_status") == "valid"),
            "mean_width": round(sum(widths) / len(widths), 2) if widths else "",
            "mean_height": round(sum(heights) / len(heights), 2) if heights else "",
            "mean_pixel_count": round(sum(pixels) / len(pixels), 2) if pixels else "",
            "low_resolution_rate": round(
                sum(1 for r in rows if r.get("image_low_resolution_flag") == 1) / len(rows), 4
            ) if rows else "",
            "corrupted_rate": round(
                sum(1 for r in rows if r.get("image_is_corrupted") == 1) / len(rows), 4
            ) if rows else "",
            "duplicate_rate": round(
                sum(
                    1 for r in rows
                    if r.get("image_is_exact_duplicate") == 1
                    or r.get("image_is_near_duplicate") == 1
                ) / len(rows), 4
            ) if rows else "",
        })

    return output_rows


def build_health_summary(index_rows, image_rows):
    years = [
        int(r["publication_year"])
        for r in index_rows
        if str(r.get("publication_year", "")).isdigit()
    ]

    journals = set(r["journal"] for r in index_rows if r.get("journal"))
    publishers = set(r["publisher"] for r in index_rows if r.get("publisher"))
    domains = set(r["domain"] for r in index_rows if r.get("domain") and r.get("domain") != "unknown/unmatched")

    metrics = {
        "total_doi_folders": len(index_rows),
        "total_with_ga": sum(int(r["has_ga"]) for r in index_rows),
        "total_with_metadata": sum(int(r["has_metadata"]) for r in index_rows),
        "total_with_pdf": sum(int(r["has_pdf"]) for r in index_rows),
        "total_complete_for_stage1": sum(int(r["is_complete_for_stage1"]) for r in index_rows),

        "total_domain_matched": sum(
            1 for r in index_rows
            if r["domain_match_source"] != "unmatched"
        ),
        "total_domain_unmatched": sum(
            1 for r in index_rows
            if r["domain_match_source"] == "unmatched"
        ),
        "total_scimago_issn_matches": sum(
            1 for r in index_rows
            if r["domain_match_source"] == "scimago_issn"
        ),
        "total_scimago_exact_name_matches": sum(
            1 for r in index_rows
            if r["domain_match_source"] == "scimago_name_exact"
        ),
        "total_scimago_fuzzy_name_matches": sum(
            1 for r in index_rows
            if r["domain_match_source"] == "scimago_name_fuzzy"
        ),

        "total_valid_images": sum(1 for r in image_rows if r["stage1_image_status"] == "valid"),
        "total_corrupted_images": sum(1 for r in image_rows if r["is_corrupted"] == 1),
        "total_low_resolution_images": sum(1 for r in image_rows if r["low_resolution_flag"] == 1),
        "total_extremely_low_resolution_images": sum(
            1 for r in image_rows if r["extremely_low_resolution_flag"] == 1
        ),
        "total_exact_duplicate_images": sum(1 for r in image_rows if r["is_exact_duplicate"] == 1),
        "total_near_duplicate_images": sum(1 for r in image_rows if r["is_near_duplicate"] == 1),

        "year_min": min(years) if years else "",
        "year_max": max(years) if years else "",
        "num_journals": len(journals),
        "num_publishers": len(publishers),
        "num_domains": len(domains),
    }

    return [{"metric": key, "value": value} for key, value in metrics.items()]


def write_summaries(config, index_rows, image_rows):
    output_root = Path(config["output_root"])
    merged_rows = merge_index_image_rows(index_rows, image_rows)

    year_groups = defaultdict(list)

    for row in merged_rows:
        year = row.get("publication_year")
        if year != "":
            year_groups[year].append(row)

    year_summary_rows = []

    for year, rows in sorted(year_groups.items()):
        year_summary_rows.append({
            "publication_year": year,
            "num_papers": len(rows),
            "num_valid_ga_images": sum(1 for r in rows if r.get("image_stage1_image_status") == "valid"),
            "num_low_resolution": sum(1 for r in rows if r.get("image_low_resolution_flag") == 1),
            "num_corrupted": sum(1 for r in rows if r.get("image_is_corrupted") == 1),
            "num_duplicates": sum(
                1 for r in rows
                if r.get("image_is_exact_duplicate") == 1
                or r.get("image_is_near_duplicate") == 1
            ),
        })

    write_csv(
        output_root / "summaries" / "stage1_year_summary.csv",
        year_summary_rows,
        ["publication_year", "num_papers", "num_valid_ga_images", "num_low_resolution", "num_corrupted", "num_duplicates"]
    )

    domain_summary = make_group_summary(merged_rows, "domain")
    journal_summary = make_group_summary(merged_rows, "journal")
    publisher_summary = make_group_summary(merged_rows, "publisher")

    write_csv(
        output_root / "summaries" / "stage1_domain_summary.csv",
        domain_summary,
        [
            "domain", "num_papers", "publication_year_min", "publication_year_max",
            "num_valid_ga_images", "mean_width", "mean_height", "mean_pixel_count",
            "low_resolution_rate", "corrupted_rate", "duplicate_rate"
        ]
    )

    write_csv(
        output_root / "summaries" / "stage1_journal_summary.csv",
        journal_summary,
        [
            "journal", "num_papers", "publication_year_min", "publication_year_max",
            "num_valid_ga_images", "mean_width", "mean_height", "mean_pixel_count",
            "low_resolution_rate", "corrupted_rate", "duplicate_rate"
        ]
    )

    write_csv(
        output_root / "summaries" / "stage1_publisher_summary.csv",
        publisher_summary,
        [
            "publisher", "num_papers", "publication_year_min", "publication_year_max",
            "num_valid_ga_images", "mean_width", "mean_height", "mean_pixel_count",
            "low_resolution_rate", "corrupted_rate", "duplicate_rate"
        ]
    )

    year_domain_groups = defaultdict(list)

    for row in merged_rows:
        year = row.get("publication_year")
        domain = row.get("domain")

        if year != "" and domain:
            year_domain_groups[(year, domain)].append(row)

    year_domain_rows = []

    for (year, domain), rows in sorted(year_domain_groups.items()):
        pixels = [
            int(r["image_pixel_count"])
            for r in rows
            if str(r.get("image_pixel_count", "")).isdigit()
        ]

        year_domain_rows.append({
            "publication_year": year,
            "domain": domain,
            "num_papers": len(rows),
            "num_valid_ga_images": sum(1 for r in rows if r.get("image_stage1_image_status") == "valid"),
            "mean_pixel_count": round(sum(pixels) / len(pixels), 2) if pixels else "",
            "low_resolution_rate": round(
                sum(1 for r in rows if r.get("image_low_resolution_flag") == 1) / len(rows), 4
            ) if rows else "",
        })

    write_csv(
        output_root / "summaries" / "stage1_year_domain_summary.csv",
        year_domain_rows,
        ["publication_year", "domain", "num_papers", "num_valid_ga_images", "mean_pixel_count", "low_resolution_rate"]
    )

    health_summary = build_health_summary(index_rows, image_rows)

    write_csv(
        output_root / "summaries" / "stage1_dataset_health_summary.csv",
        health_summary,
        ["metric", "value"]
    )

    format_counts = Counter(
        row["detected_image_format"]
        for row in image_rows
        if row["detected_image_format"]
    )

    format_rows = [
        {"detected_image_format": key, "count": value}
        for key, value in format_counts.most_common()
    ]

    write_csv(
        output_root / "image_metadata" / "stage1_image_format_summary.csv",
        format_rows,
        ["detected_image_format", "count"]
    )

    return health_summary


def write_report(config, health_summary, index_rows):
    output_root = Path(config["output_root"])
    report_path = output_root / "reports" / "stage1_preprocessing_report.txt"

    metrics = {row["metric"]: row["value"] for row in health_summary}

    total = int(metrics.get("total_doi_folders", 0) or 0)
    complete = int(metrics.get("total_complete_for_stage1", 0) or 0)
    valid_images = int(metrics.get("total_valid_images", 0) or 0)
    corrupted = int(metrics.get("total_corrupted_images", 0) or 0)
    unmatched = int(metrics.get("total_domain_unmatched", 0) or 0)

    ready = "YES"
    reason = "Dataset has valid Stage 1 outputs."

    if total == 0:
        ready = "NO"
        reason = "No DOI folders found."
    elif complete / max(total, 1) < 0.90:
        ready = "NO"
        reason = "Less than 90% of records are complete for Stage 1."
    elif valid_images / max(total, 1) < 0.90:
        ready = "NO"
        reason = "Less than 90% of records have valid GA images."
    elif corrupted / max(total, 1) > 0.05:
        ready = "NO"
        reason = "More than 5% of GA images are corrupted."
    elif unmatched > 0:
        ready = "NO"
        reason = "Some papers still have unmatched domains."

    top_journals = Counter(row["journal"] or "UNKNOWN" for row in index_rows).most_common(10)
    top_publishers = Counter(row["publisher"] or "UNKNOWN" for row in index_rows).most_common(10)
    domain_distribution = Counter(row["domain"] or "UNKNOWN" for row in index_rows).most_common()
    domain_match_status = Counter(row["domain_match_source"] for row in index_rows).most_common()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Task 2 Stage 1 Preprocessing Report\n")
        f.write("==================================\n\n")

        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Dataset path: {config['dataset_root']}\n")
        f.write(f"Output path: {config['output_root']}\n")
        f.write(f"SCImago path: {SCIMAGO_PATH}\n")
        f.write(f"Optional master path: {OPTIONAL_MASTER_PATH}\n\n")

        f.write("Dataset health summary\n")
        f.write("----------------------\n")
        for row in health_summary:
            f.write(f"{row['metric']}: {row['value']}\n")

        f.write("\nDomain match status\n")
        f.write("-------------------\n")
        for name, count in domain_match_status:
            f.write(f"{name}: {count}\n")

        f.write("\nTop journals\n")
        f.write("------------\n")
        for name, count in top_journals:
            f.write(f"{name}: {count}\n")

        f.write("\nTop publishers\n")
        f.write("--------------\n")
        for name, count in top_publishers:
            f.write(f"{name}: {count}\n")

        f.write("\nDomain distribution\n")
        f.write("-------------------\n")
        for name, count in domain_distribution:
            f.write(f"{name}: {count}\n")

        f.write("\nSTAGE 1 VERDICT\n")
        f.write("---------------\n")
        f.write(f"READY_FOR_STAGE_2 = {ready}\n")
        f.write(f"Reason: {reason}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("============================================================")
    print("Task 2 Stage 1 preprocessing started")
    print(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    print("============================================================")

    config = load_config()
    output_root = Path(config["output_root"])

    reset_outputs(output_root)

    print("Loading SCImago...")
    scimago_by_issn, scimago_by_title = load_scimago()
    print(f"SCImago ISSN keys: {len(scimago_by_issn)}")
    print(f"SCImago title keys: {len(scimago_by_title)}")

    print("Loading optional master.csv if available...")
    master_by_folder, master_by_doi = load_optional_master()
    print(f"Master folder keys: {len(master_by_folder)}")
    print(f"Master DOI keys: {len(master_by_doi)}")

    print("Building Stage 1 GA index...")
    index_rows = build_index(
        config,
        scimago_by_issn,
        scimago_by_title,
        master_by_folder,
        master_by_doi
    )
    print(f"Total DOI folders indexed: {len(index_rows)}")

    print("Extracting image metadata...")
    image_rows = extract_image_metadata(config, index_rows)
    print(f"Total image rows processed: {len(image_rows)}")

    print("Writing duplicate reports...")
    write_duplicate_reports(config, image_rows)

    print("Writing summary files...")
    health_summary = write_summaries(config, index_rows, image_rows)

    print("Writing final Stage 1 report...")
    write_report(config, health_summary, index_rows)

    print("============================================================")
    print("Task 2 Stage 1 preprocessing completed")
    print(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Report: {output_root / 'reports' / 'stage1_preprocessing_report.txt'}")
    print("============================================================")


if __name__ == "__main__":
    main()