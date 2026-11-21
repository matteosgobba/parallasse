import hashlib

import pytest

from parallasse.download import download_declared_file


def test_cached_file_checksum_and_metadata_are_checked(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = b"stand-in for public data; no network needed"
    (raw / "sample.npz").write_bytes(payload)
    checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    spec = dict(
        record_id="123",
        record_version="v1",
        filename="sample.npz",
        published_md5=checksum,
        generator="pythia",
    )
    record = {
        "metadata": {"version": "v1", "license": {"id": "cc-by-4.0"}},
        "files": [
            {
                "key": "sample.npz",
                "checksum": f"md5:{checksum}",
                "size": len(payload),
                "links": {"self": "https://zenodo.org/unused"},
            }
        ],
    }
    monkeypatch.setattr("parallasse.download.fetch_record", lambda _: record)
    result = download_declared_file(spec, raw, tmp_path / "metadata")
    assert result["md5_verified"]
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    (raw / "sample.npz").write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="Cached file failed"):
        download_declared_file(spec, raw, tmp_path / "metadata")


def test_changed_record_version_does_not_download(tmp_path, monkeypatch):
    spec = dict(record_id="123", record_version="v1")
    monkeypatch.setattr(
        "parallasse.download.fetch_record", lambda _: {"metadata": {"version": "v2"}}
    )
    with pytest.raises(ValueError, match="Unexpected record version"):
        download_declared_file(spec, tmp_path, tmp_path)
