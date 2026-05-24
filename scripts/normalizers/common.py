"""Shared safe helpers for silver normalizers."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import unicodedata
from collections.abc import Generator, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import yaml

from scripts.normalizers.schema import COLUMN_ORDER, LABELS, NULLABLE_COLUMNS, SCHEMA_VERSION, SilverRecord


class RowCapExceeded(RuntimeError):
    """Raised when a module exceeds the default write cap without override."""


class DuplicateRecordIdError(ValueError):
    """Raised when silver rows contain duplicate record IDs."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_ROOT = PROJECT_ROOT / "data" / "bronze_raw"
SILVER_ROOT = PROJECT_ROOT / "data" / "silver_normalized"
DEFAULT_ROW_CAP = 50_000


def make_record_id(source_dataset: str, unique_value: str) -> str:
    """Create a deterministic record id."""

    digest = hashlib.sha1(str(unique_value).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{source_dataset}::{digest}"


def clean_text(value: Any) -> str | None:
    """Normalize text to UTF-8-ish NFC and remove most control characters."""

    if value is None or pd.isna(value):
        return None
    text = unicodedata.normalize("NFC", str(value))
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = text.strip()
    return text or None


def safe_json_dumps(obj: Any, max_bytes: int = 8192) -> str | None:
    """Serialize JSON if it fits the per-row feature cap."""

    if obj is None:
        return None
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    if len(payload.encode("utf-8")) > max_bytes:
        return None
    return payload


def normalize_binary_label(value: Any) -> int:
    """Normalize common binary labels to 0 or 1."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "malicious", "phishing", "jailbreak", "attack", "vulnerable", "yes"}:
        return 1
    if text in {"0", "false", "benign", "benign_url", "non_vulnerable", "safe", "no"}:
        return 0
    raise ValueError(f"ambiguous binary label: {value!r}")


def normalize_severity(
    cvss_v2: float | None = None,
    cvss_v3: float | None = None,
    cvss_v4: float | None = None,
    vendor_severity: str | None = None,
) -> tuple[str, float | None]:
    """Normalize severity to a canonical label and score."""

    score = next((s for s in (cvss_v4, cvss_v3, cvss_v2) if s is not None), None)
    if score is not None:
        score = float(score)
        if score >= 9.0:
            return "critical", score
        if score >= 7.0:
            return "high", score
        if score >= 4.0:
            return "medium", score
        if score > 0:
            return "low", score
        return "info", score
    sev = (vendor_severity or "unknown").strip().lower()
    aliases = {"moderate": "medium", "important": "high", "none": "info"}
    sev = aliases.get(sev, sev)
    if sev not in {"critical", "high", "medium", "low", "info"}:
        sev = "unknown"
    return sev, None


def normalize_timestamp(value: Any) -> pd.Timestamp | None:
    """Convert a value to UTC pandas timestamp."""

    if value is None or pd.isna(value):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def extract_domain_from_url(url: str | None) -> str | None:
    """Extract lowercased host without performing any network request."""

    if not url:
        return None
    candidate = str(url).strip()
    try:
        parsed = urlparse(candidate if "://" in candidate else f"http://{candidate}")
    except ValueError:
        return None
    host = parsed.hostname
    return host.lower() if host else None


def defang_url_for_log(url: str) -> str:
    """Defang a URL for logs."""

    return str(url).replace("http://", "hxxp://").replace("https://", "hxxps://").replace(".", "[.]")


def detect_file_encoding(path: Path, sample_bytes: int = 65536) -> str:
    """Return a conservative encoding guess."""

    sample = path.read_bytes()[:sample_bytes]
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def safe_read_csv_chunks(path: Path, chunksize: int = 50000, encoding: str | None = None) -> Iterable[pd.DataFrame]:
    """Read CSV-like files in chunks, including gzip-compressed CSV."""

    enc = encoding or detect_file_encoding(path)
    compression = "infer"
    if path.suffix.lower() == ".gz":
        with path.open("rb") as fh:
            if fh.read(2) != b"\x1f\x8b":
                compression = None
    return pd.read_csv(path, chunksize=chunksize, encoding=enc, compression=compression, on_bad_lines="skip")


def safe_read_json_stream(path: Path) -> Generator[Any, None, None]:
    """Stream JSON/JSONL records without network access."""

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return
    try:
        import ijson

        with path.open("rb") as fh:
            for item in ijson.items(fh, "vulnerabilities.item"):
                yield item
        return
    except Exception:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            for key in ("vulnerabilities", "CVE_Items", "objects"):
                if isinstance(data.get(key), list):
                    yield from data[key]
                    return
            yield data


def safe_read_yaml_dir(directory: Path) -> Generator[tuple[Path, dict[str, Any]], None, None]:
    """Yield YAML documents from a directory tree while ignoring git/cache paths."""

    for path in directory.rglob("*"):
        if any(part in {".git", "__pycache__", ".cache"} for part in path.parts):
            continue
        if path.suffix.lower() not in {".yml", ".yaml"}:
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, dict):
            yield path, data


def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    h.update(str(path.relative_to(BRONZE_ROOT) if path.is_relative_to(BRONZE_ROOT) else path).encode())
    h.update(str(path.stat().st_size).encode())
    h.update(str(path.stat().st_mtime_ns).encode())
    with path.open("rb") as fh:
        h.update(fh.read(64 * 1024))
    return h.hexdigest()


def compute_input_hash(paths: list[Path]) -> str:
    """Compute a stable input hash from file fingerprints."""

    h = hashlib.sha256()
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            count = 0
            for child in path.rglob("*"):
                if not child.is_file() or ".git" in child.parts:
                    continue
                try:
                    stat = child.stat()
                    h.update(str(child.relative_to(BRONZE_ROOT) if child.is_relative_to(BRONZE_ROOT) else child).encode())
                    h.update(str(stat.st_size).encode())
                    h.update(str(stat.st_mtime_ns).encode())
                except OSError:
                    continue
                count += 1
                if count >= 20_000:
                    h.update(f"truncated-dir-file-count-at:{count}".encode())
                    break
        elif path.exists():
            files.append(path)
    for path in sorted(files):
        try:
            h.update(_file_fingerprint(path).encode())
        except OSError:
            continue
    return h.hexdigest()


def ensure_unified_schema(
    df: pd.DataFrame,
    source_dataset: str,
    source_type: str,
    main_category: str,
    license: str,
) -> pd.DataFrame:
    """Fill schema defaults, normalize nulls, and order columns."""

    out = df.copy()
    now = pd.Timestamp.now(tz=UTC)
    defaults: dict[str, Any] = {
        "source_dataset": source_dataset,
        "source_type": source_type,
        "main_category": main_category,
        "license": license,
        "schema_version": SCHEMA_VERSION,
        "ingested_at": now,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)
    for col in COLUMN_ORDER:
        if col not in out.columns:
            out[col] = pd.NA if col in NULLABLE_COLUMNS else defaults.get(col)
    out = out.replace({"": pd.NA})
    for col in ("raw_text", "notes", "attack_name", "attack_family"):
        if col in out.columns:
            out[col] = out[col].map(clean_text)
    out["binary_label"] = out["binary_label"].map(normalize_binary_label).astype("int8")
    out["timestamp"] = out["timestamp"].map(normalize_timestamp)
    out["ingested_at"] = pd.to_datetime(out["ingested_at"], utc=True)
    out = out[COLUMN_ORDER].sort_values("record_id").reset_index(drop=True)
    return out


def validate_against_schema(df: pd.DataFrame) -> None:
    """Validate column order, required null policy, controlled vocabs, and rows."""

    if list(df.columns) != COLUMN_ORDER:
        raise ValueError("silver dataframe columns do not match COLUMN_ORDER")
    required = [c for c in COLUMN_ORDER if c not in NULLABLE_COLUMNS]
    missing = [c for c in required if df[c].isna().any()]
    if missing:
        raise ValueError(f"required columns contain nulls: {missing}")
    if df["record_id"].duplicated().any():
        dupes = df.loc[df["record_id"].duplicated(), "record_id"].head(10).tolist()
        raise DuplicateRecordIdError(f"duplicate record_id values: {dupes}")
    invalid_labels = sorted(set(df["label"].dropna()) - LABELS)
    if invalid_labels:
        raise ValueError(f"invalid labels: {invalid_labels}")
    for column in ("timestamp", "ingested_at"):
        for value in df[column].dropna():
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                raise ValueError(f"timestamp values in {column} must be timezone-aware UTC")
            if ts.tz_convert("UTC") != ts:
                raise ValueError(f"timestamp values in {column} must be UTC")
    for record in df.to_dict("records"):
        SilverRecord(**{k: (None if pd.isna(v) else v) for k, v in record.items()})


def sample_balanced(df: pd.DataFrame, label_col: str, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """Return a balanced sample up to max_rows."""

    if len(df) <= max_rows:
        return df
    groups = list(df.groupby(label_col, dropna=False))
    per = max(max_rows // max(len(groups), 1), 1)
    sampled = [g.sample(n=min(len(g), per), random_state=random_state) for _, g in groups]
    out = pd.concat(sampled, ignore_index=True)
    if len(out) < max_rows:
        rest = df.drop(out.index, errors="ignore")
        if not rest.empty:
            out = pd.concat([out, rest.sample(n=min(max_rows - len(out), len(rest)), random_state=random_state)])
    return out.head(max_rows).reset_index(drop=True)


def _atomic_write_bytes(data: bytes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, output)


def write_metadata_json(meta: dict[str, Any], output_path: Path) -> None:
    """Atomically write metadata JSON."""

    _atomic_write_bytes(json.dumps(meta, indent=2, sort_keys=True, default=str).encode("utf-8"), output_path)


def write_silver(df: pd.DataFrame, output_stem: Path, max_rows: int | None = DEFAULT_ROW_CAP) -> dict[str, Any]:
    """Atomically write parquet, CSV.GZ, and return output metadata."""

    if max_rows is not None and len(df) > max_rows:
        raise RowCapExceeded(f"{len(df)} rows exceeds cap {max_rows}")
    validate_against_schema(df)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    parquet = output_stem.with_suffix(".parquet")
    csv_gz = output_stem.with_suffix(".csv.gz")
    tmp_parquet = parquet.with_suffix(parquet.suffix + ".tmp")
    tmp_csv = csv_gz.with_suffix(csv_gz.suffix + ".tmp")
    df.to_parquet(tmp_parquet, index=False)
    os.replace(tmp_parquet, parquet)
    with gzip.open(tmp_csv, "wt", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp_csv, csv_gz)
    try:
        parquet_path = str(parquet.relative_to(PROJECT_ROOT))
        csv_path = str(csv_gz.relative_to(PROJECT_ROOT))
    except ValueError:
        parquet_path = str(parquet)
        csv_path = str(csv_gz)
    return {
        "parquet": parquet_path,
        "csv_gz": csv_path,
        "parquet_bytes": parquet.stat().st_size,
        "csv_gz_bytes": csv_gz.stat().st_size,
    }


def license_compatibility(license_name: str) -> str:
    """Classify license compatibility for reporting."""

    text = license_name.lower()
    if "academic" in text or "unsw" in text or "cic" in text:
        return "restricted_academic"
    if text.startswith("restricted:") or "tos" in text or "gpl" in text:
        return "restricted_other"
    if text in {"unknown", ""}:
        return "unknown"
    return "permissive"
