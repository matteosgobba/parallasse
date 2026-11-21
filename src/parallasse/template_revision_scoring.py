"""Isolated PFN scoring with v0.10 version and full-evidence access checks."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.replication import prepare_replication_inputs
from parallasse.score_template_design import CHECKPOINTS, verify_reservation
from parallasse.template_revision import _record, authorize_revision_role
from parallasse.template_workflow import atomic_json, check_score_job, selected_model_contract


def check_revision_score_job(job):
    # Both modes authorize before opening even the encoded-input/row sidecars.
    if job["mode"] == "old_source_smoke":
        return check_score_job(job)
    if job["mode"] != "revision_reserved_role":
        raise ValueError("Unknown revised PFN scoring mode")
    record, engine, config = _record(Path(job["study_record"]))
    authorize_revision_role(record, engine, config, job["generator"], job["role"])
    if (
        job["engine_sha256"] != record["engine_sha256"]
        or job["protocol_version"] != record["protocol_version"]
    ):
        raise ValueError("Worker version differs from its revision study")
    expected = next(s["filename"] for s in config["files"] if s["generator"] == job["generator"])
    if job["raw_file"] != str(Path("data/raw") / expected):
        raise ValueError("Revision worker raw-file role mismatch")
    freeze = verify_reservation(Path(engine["reservation"]))
    verify_hashes(
        {
            job["row_proof"]: job["row_proof_sha256"],
            job["inputs"]: job["inputs_sha256"],
            job["contract"]: job["contract_sha256"],
        }
    )
    contract = read_json(Path(job["contract"]))
    if contract != selected_model_contract(freeze):
        raise ValueError("Revised worker must use the original model/cut/preprocessing contract")
    verify_hashes({m["checkpoint"]: m["checkpoint_sha256"] for m in contract["models"]})
    with (
        np.load(job["row_proof"], allow_pickle=False) as rows,
        np.load(freeze["archives"]["cohorts"]["path"], allow_pickle=False) as cohorts,
    ):
        prefix = f"{job['generator']}_{job['role']}"
        if not np.array_equal(rows["rows"], cohorts[f"{prefix}_rows"]) or not np.array_equal(
            rows["labels"], cohorts[f"{prefix}_labels"]
        ):
            raise ValueError("Revised worker rows/labels differ from their exact reserved role")
    return contract


def score_rows(raw, rows, labels, contract, directory, proof):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "inputs.npz"
    destination = directory / "predictions.npz"
    actual = prepare_replication_inputs(Path(raw), np.asarray(rows), contract, inputs)
    if not np.array_equal(actual, labels):
        raise ValueError("Raw labels differ from the frozen authorized rows")
    contract_path = directory / "models.json"
    atomic_json(contract_path, contract)
    row_proof = directory / "rows.npz"
    with row_proof.open("xb") as handle:
        np.savez(handle, rows=rows, labels=labels)
    job = {
        **proof,
        "raw_file": str(raw),
        "inputs": str(inputs),
        "inputs_sha256": sha256_file(inputs),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "row_proof": str(row_proof),
        "row_proof_sha256": sha256_file(row_proof),
        "rows": np.asarray(rows).tolist() if proof["mode"] == "old_source_smoke" else None,
        "destination": str(destination),
    }
    job_path = directory / "job.json"
    atomic_json(job_path, job)
    with (directory / "worker.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "scripts/template_revision_score_worker.py",
                str(job_path),
                sha256_file(job_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with np.load(destination, allow_pickle=False) as a:
        if tuple(a["keys"]) != CHECKPOINTS or a["scores"].shape != (len(rows), 6):
            raise ValueError("Revised worker returned malformed PFN probabilities")
        scores = a["scores"]
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Revised worker probabilities outside [0,1]")
    return scores


def score_revision_role(record_path, generator, role, destination):
    record, engine, config = _record(record_path)
    authorize_revision_role(record, engine, config, generator, role)
    freeze = verify_reservation(Path(engine["reservation"]))
    with np.load(freeze["archives"]["cohorts"]["path"], allow_pickle=False) as a:
        rows, labels = a[f"{generator}_{role}_rows"], a[f"{generator}_{role}_labels"]
    raw = Path("data/raw") / next(
        s["filename"] for s in config["files"] if s["generator"] == generator
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    proof = {
        "mode": "revision_reserved_role",
        "study_record": str(record_path),
        "generator": generator,
        "role": role,
        "engine_sha256": record["engine_sha256"],
        "protocol_version": record["protocol_version"],
    }
    with tempfile.TemporaryDirectory(
        prefix=f"{generator}-{role}-", dir=destination.parent
    ) as scratch:
        scores = score_rows(
            raw, rows, labels, selected_model_contract(freeze), Path(scratch), proof
        )
        with destination.open("xb") as handle:
            np.savez_compressed(
                handle,
                rows=rows,
                labels=labels,
                keys=np.array(CHECKPOINTS),
                scores=scores,
                generator=np.asarray(generator),
                role=np.asarray(role),
            )
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "rows": len(rows),
        "generator": generator,
        "role": role,
    }
