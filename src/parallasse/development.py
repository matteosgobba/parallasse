"""Checked access to source train/tune rows; held-out cohorts are not allowed."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from parallasse.data import validate_jets
from parallasse.download import sha256_file


@dataclass
class DevelopmentData:
    train_particles: NDArray
    train_labels: NDArray
    tune_particles: NDArray
    tune_labels: NDArray
    train_rows: NDArray
    tune_rows: NDArray
    provenance: dict


def development_rows(root: Path, name: str, limit: int) -> tuple[NDArray, dict]:
    """Reject held-out cohort names before opening any files."""
    if name not in ("train", "tune"):
        raise ValueError("Development access permits only source train and tune")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("A positive integer row limit is required")
    manifest = json.loads((root / "results/task0001/cohorts.json").read_text())
    item = manifest["cohorts"][name]
    if item["generator"] != "pythia" or item["original_file"] != "QG_jets.npz":
        raise ValueError("Development cohort must belong to the declared Pythia source")
    path = root / item["row_ids_path"]
    if sha256_file(path) != item["row_ids_sha256"]:
        raise ValueError("Persisted development row IDs changed")
    rows = np.load(path, allow_pickle=False)
    if limit > len(rows):
        raise ValueError("Requested development subset exceeds the persisted cohort")
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Row IDs must be a one-dimensional integer array")
    selected = rows[:limit].copy()
    return selected, {
        "cohort": name,
        "full_row_ids_sha256": item["row_ids_sha256"],
        "selected_rows": len(selected),
        "selection": "persisted shuffled cohort prefix",
        "subset_sha256_int64_le_bytes": hashlib.sha256(
            selected.astype("<i8").tobytes()
        ).hexdigest(),
    }


def load_source_development(root: Path, train_limit: int, tune_limit: int) -> DevelopmentData:
    root = root.resolve()
    validation = json.loads((root / "results/task0001/validation.json").read_text())
    for relative in ("results/task0001/cohorts.json", "results/task0001/preprocessing.json"):
        if sha256_file(root / relative) != validation["evidence_sha256"][relative]:
            raise ValueError(f"Phase 1 contract changed: {relative}")
    train_rows, train_provenance = development_rows(root, "train", train_limit)
    tune_rows, tune_provenance = development_rows(root, "tune", tune_limit)
    if np.intersect1d(train_rows, tune_rows).size:
        raise ValueError("Training and tuning subsets overlap")
    manifest = json.loads((root / "results/task0001/cohorts.json").read_text())
    path = root / "data/raw/QG_jets.npz"
    raw_hash = sha256_file(path)
    if raw_hash != manifest["raw_sha256"]["source"]:
        raise ValueError("Raw source bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"X", "y"}:
            raise ValueError("Unexpected source archive schema")
        x, y = archive["X"], archive["y"]
        all_rows = np.concatenate((train_rows, tune_rows))
        if np.any(all_rows < 0) or np.any(all_rows >= len(y)):
            raise ValueError("Development row ID out of range")
        train_x, tune_x = x[train_rows], x[tune_rows]
        train_y, tune_y = y[train_rows], y[tune_rows]
    validate_jets(train_x, train_y)
    validate_jets(tune_x, tune_y)
    return DevelopmentData(
        train_x,
        train_y,
        tune_x,
        tune_y,
        train_rows,
        tune_rows,
        {
            "source_file": str(path.relative_to(root)),
            "source_sha256": raw_hash,
            "train": train_provenance,
            "tune": tune_provenance,
            "opened_cohorts": ["train", "tune"],
            "target_file_opened": False,
            "heldout_row_arrays_opened": False,
            "note": (
                "Compressed source container decoded; only train/tune rows used for features/fits."
            ),
        },
    )
