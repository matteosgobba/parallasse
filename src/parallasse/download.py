"""Download only the declared, checksum-verified public dataset files."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_record(record_id: str) -> dict:
    request = urllib.request.Request(
        f"https://zenodo.org/api/records/{record_id}",
        headers={"User-Agent": "Parallasse/0.1 (independent scientific study)"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def download_declared_file(spec: dict, raw_dir: Path, metadata_dir: Path) -> dict:
    """Fail closed on changed metadata, size, version, or checksum."""
    record = fetch_record(spec["record_id"])
    metadata = record["metadata"]
    version = metadata.get("version")
    if version != spec["record_version"]:
        raise ValueError(f"Unexpected record version: {version!r}")
    entry = next(item for item in record["files"] if item["key"] == spec["filename"])
    expected_checksum = f"md5:{spec['published_md5']}"
    if entry["checksum"] != expected_checksum:
        raise ValueError(f"Published checksum changed for {spec['filename']}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / spec["filename"]
    url = entry["links"]["self"]
    started = time.monotonic()
    if not path.exists():
        temporary = path.with_suffix(".npz.part")
        request = urllib.request.Request(url, headers={"User-Agent": "Parallasse/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if temporary.stat().st_size != entry["size"]:
            raise ValueError(f"File size mismatch for {path.name}")
        if md5_file(temporary) != spec["published_md5"]:
            raise ValueError(f"MD5 mismatch for {path.name}; .part preserved")
        temporary.replace(path)
        os.chmod(path, 0o444)
    if path.stat().st_size != entry["size"] or md5_file(path) != spec["published_md5"]:
        raise ValueError(f"Cached file failed integrity verification: {path}")
    summary = {
        **spec,
        "record_url": f"https://zenodo.org/records/{spec['record_id']}",
        "api_url": f"https://zenodo.org/api/records/{spec['record_id']}",
        "download_url": url,
        "doi": record.get("doi"),
        "title": metadata.get("title"),
        "creators": [creator["name"] for creator in metadata.get("creators", [])],
        "license": metadata.get("license"),
        "access_right": metadata.get("access_right"),
        "size_bytes": path.stat().st_size,
        "md5_verified": True,
        "sha256": sha256_file(path),
        "verified_utc": datetime.now(UTC).isoformat(),
        "download_and_hash_seconds": round(time.monotonic() - started, 3),
    }
    (metadata_dir / f"{spec['generator']}_provenance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
