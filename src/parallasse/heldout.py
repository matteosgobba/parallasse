"""Verify frozen evaluation inputs and prepare only the requested cohort."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from parallasse.data import class_counts
from parallasse.download import sha256_file
from parallasse.features import aggregate_features
from parallasse.heldout_metrics import KEYS, calibration_cut
from parallasse.preprocessing import encode_pid, kinematics

INPUT_FREEZE = Path("results/task0003/20261110-source-fits-01/evaluation_freeze.json")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_hashes(hashes: dict, root: Path = Path(".")) -> None:
    for name, expected in hashes.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Frozen bytes changed: {name}")


def verify_input_freeze() -> dict:
    validation = read_json(Path("results/task0003/validation.json"))
    verify_hashes({str(INPUT_FREEZE): validation["evidence_sha256"][str(INPUT_FREEZE)]})
    frozen = read_json(INPUT_FREEZE)
    verify_hashes(frozen["input_files_sha256"])
    verify_hashes({frozen["source_run_record"]: frozen["source_run_record_sha256"]})
    verify_hashes(frozen["source_files_sha256"])
    verify_hashes(frozen["source_files_sha256"], Path(frozen["source_snapshot"]))
    verify_hashes({m["checkpoint"]: m["checkpoint_sha256"] for m in frozen["models"]})
    actual_keys = tuple(f"{m['model']}-{m['seed']}" for m in frozen["models"])
    if actual_keys != KEYS or frozen["state"] != "models_and_rules_frozen":
        raise ValueError("Unexpected model set or freeze state")
    run = read_json(Path(frozen["source_run_record"]))
    if sys.version != run["python"] or any(
        importlib.metadata.version(name) != value for name, value in run["versions"].items()
    ):
        raise ValueError("Inference environment differs from the frozen source fits")
    return frozen


def derive_cuts(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if (
            tuple(data["keys"]) != KEYS
            or data["labels"].shape != (5000,)
            or np.any(data["labels"] != 0)
            or data["scores"].shape != (5000, 12)
        ):
            raise ValueError("Numerical cuts require exactly the frozen calibration gluons")
        return {key: calibration_cut(data["scores"][:, i]) for i, key in enumerate(KEYS)}


def verified_cuts(record: dict, path: Path, run_id: str) -> dict:
    if record.get("status") != "cuts_frozen":
        raise ValueError("Final scoring requires the prior numerical cut freeze")
    verify_hashes({str(path): record["cut_freeze_sha256"]})
    lock = read_json(path)
    if (
        lock["run_id"] != run_id
        or lock["operator"] != ">"
        or lock["model_freeze_sha256"] != record["model_freeze_sha256"]
        or lock["calibration_cohort"] != "source calibration gluons only"
        or lock["background_acceptance_target"] != 0.1
    ):
        raise ValueError("Numerical cuts do not belong to this frozen evaluation")
    calibration = lock["calibration_scores"]
    verify_hashes({calibration["path"]: calibration["sha256"]})
    if derive_cuts(Path(calibration["path"])) != lock["cuts"]:
        raise ValueError("Numerical cuts disagree with the calibration scores")
    return {key: value["threshold"] for key, value in lock["cuts"].items()}


def prepare_cohort(frozen: dict, cohort: str, destination: Path) -> tuple[np.ndarray, np.ndarray]:
    if cohort not in ("calibration", "test", "target"):
        raise ValueError("Unknown evaluation cohort")
    dataset = frozen["dataset_contract"]
    item = dataset["cohorts"][cohort]
    role = "target" if cohort == "target" else "source"
    raw = Path("data/raw") / item["original_file"]
    rows_file = Path(item["row_ids_path"])
    verify_hashes({str(raw): dataset["raw_sha256"][role], str(rows_file): item["row_ids_sha256"]})
    rows = np.load(rows_file, allow_pickle=False)
    if (
        rows.shape != (item["rows"],)
        or not np.issubdtype(rows.dtype, np.integer)
        or len(np.unique(rows)) != len(rows)
    ):
        raise ValueError("Invalid frozen cohort row IDs")
    with np.load(raw, allow_pickle=False) as archive:
        labels_all = archive["y"]
        if np.any(rows < 0) or np.any(rows >= len(labels_all)):
            raise ValueError("Frozen row ID outside raw file")
        labels = labels_all[rows]
        if not np.all(np.isin(labels, [0, 1])) or class_counts(labels) != item["class_counts"]:
            raise ValueError("Frozen cohort label counts do not match")
        if cohort == "calibration":
            rows, labels = rows[labels == 0], labels[labels == 0]
        particles = archive["X"]
    n, width = len(rows), particles.shape[1]
    aggregate = np.empty((n, 6), dtype=np.float32)
    coordinates = np.empty((n, width, 3), dtype=np.float32)
    mask = np.empty((n, width), dtype=np.bool_)
    pid = np.empty((n, width), dtype=np.int16)
    vocabulary = frozen["pid_contract"]["pid_vocabulary"]
    for start in range(0, n, 4096):
        section = slice(start, start + 4096)
        chunk = particles[rows[section]]
        coordinates[section], mask[section] = kinematics(chunk)
        pid[section] = encode_pid(chunk, vocabulary)
        aggregate[section] = aggregate_features(chunk)
    np.savez(destination, aggregate=aggregate, coordinates=coordinates, mask=mask, pid=pid)
    return rows, labels
