"""Hash-checked executable freezes, stage access and isolated PFN scoring."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.replication import prepare_replication_inputs
from parallasse.score_template_design import CHECKPOINTS, verify_reservation

RESERVATION = Path("results/task0011/20261113-template-reservation-01/design_freeze.json")


def now():
    return datetime.now(UTC).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, suffix=".part"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_engine(path):
    path = Path(path)
    engine = read_json(path)
    if engine.get("state") != "executable_frozen" or engine.get("smoke_only") is not False:
        raise ValueError("A completed executable freeze is required")
    record = read_json(path.parent / "run.json")
    if record.get("status") != "completed" or record.get("engine_freeze_sha256") != sha256_file(
        path
    ):
        raise ValueError("Executable freeze does not match its completed run")
    verify_hashes(engine["implementation_sha256"])
    verify_hashes(engine["implementation_sha256"], Path(engine["source_snapshot"]))
    verify_hashes(engine["input_sha256"])
    verify_reservation(Path(engine["reservation"]))
    if not engine["resource_gate_passed"]:
        raise ValueError("Full-campaign resource gate failed")
    return engine


def checked_gate(record, name, config):
    ref = record.get(name)
    if not ref:
        raise ValueError(f"Required {name} is absent")
    verify_hashes({ref["path"]: ref["sha256"]})
    gate = read_json(Path(ref["path"]))
    expected = config["calibration"][
        "source_cells" if name == "source_gate" else "simulation_cells"
    ]
    if (
        gate.get("passed") is not True
        or gate.get("scope") != "frozen_full_grid"
        or gate.get("cells") != expected
        or gate.get("repetitions") != 1000
        or gate.get("smoke_only") is not False
    ):
        raise ValueError(f"Required {name} is not a complete validated gate")
    return gate


def authorize_role(record, generator, role, config):
    if (generator, role) not in [
        (g, r) for g in ("pythia", "herwig") for r in ("template", "evaluation")
    ]:
        raise ValueError("Unknown reserved data role")
    status = record["status"]
    if role == "template" and generator == "pythia" and status == "source_running":
        return
    if role == "template" and status == "model_running":
        checked_gate(record, "source_gate", config)
        return
    if status == "evaluation_running":
        checked_gate(record, "source_gate", config)
        checked_gate(record, "model_gate", config)
        return
    raise ValueError("Reserved role access is not authorized by the current stage")


def selected_model_contract(freeze):
    protocol = read_json(Path(freeze["protocol_path"]))
    root = Path(freeze["protocol_path"]).parent
    contract = read_json(root / "model_cut_contract.json")
    original = read_json(Path(protocol["model_freeze"]))
    return {
        "models": contract["models"],
        "pid_contract": original["pid_contract"],
        "inference": original["inference"],
    }


def check_score_job(job):
    """Called in the fresh Torch worker before opening any encoded inputs."""
    if job["mode"] == "old_source_smoke":
        if job["raw_file"] != "data/raw/QG_jets.npz":
            raise ValueError("Smoke jobs must not access reserved files")
        verify_hashes(job["source_proof_sha256"])
        source = np.load("data/processed/test_rows.npy", allow_pickle=False)
        requested = np.asarray(job["rows"], dtype=np.int64)
        if not 0 < len(requested) <= 256 or not np.array_equal(requested, source[: len(requested)]):
            raise ValueError("Smoke rows differ from the bounded original source-test prefix")
    elif job["mode"] == "reserved_role":
        engine = load_engine(job["engine_freeze"])
        record = read_json(Path(job["study_record"]))
        config = read_json(Path(engine["protocol_path"]))
        authorize_role(record, job["generator"], job["role"], config)
        expected = next(
            s["filename"] for s in config["files"] if s["generator"] == job["generator"]
        )
        if job["raw_file"] != str(Path("data/raw") / expected):
            raise ValueError("Scoring job raw file differs from its reserved role")
    else:
        raise ValueError("Unknown scoring job mode")
    verify_hashes({job["inputs"]: job["inputs_sha256"], job["contract"]: job["contract_sha256"]})
    contract = read_json(Path(job["contract"]))
    expected = selected_model_contract(verify_reservation(Path(job["reservation"])))
    if contract != expected:
        raise ValueError("Scoring models/settings differ from the reserved frozen contract")
    keys = tuple(f"{m['model']}-{m['seed']}" for m in contract["models"])
    if keys != CHECKPOINTS:
        raise ValueError("Only the six frozen PFNs may be scored")
    verify_hashes({m["checkpoint"]: m["checkpoint_sha256"] for m in contract["models"]})
    return contract


def score_rows(raw, rows, expected_labels, contract, directory, proof):
    """Encode only explicitly authorized rows; launch Torch in a separate executable."""
    raw, directory = Path(raw), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inputs, destination = directory / "inputs.npz", directory / "predictions.npz"
    labels = prepare_replication_inputs(raw, np.asarray(rows), contract, inputs)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("Scoring labels differ from the frozen original row IDs")
    contract_path = directory / "models.json"
    atomic_json(contract_path, contract)
    job = {
        **proof,
        "raw_file": str(raw),
        "inputs": str(inputs),
        "rows": np.asarray(rows).tolist() if proof["mode"] == "old_source_smoke" else None,
        "inputs_sha256": sha256_file(inputs),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "destination": str(destination),
    }
    atomic_json(directory / "job.json", job)
    with (directory / "worker.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "scripts/template_score_worker.py",
                str(directory / "job.json"),
                sha256_file(directory / "job.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with np.load(destination, allow_pickle=False) as result:
        if tuple(result["keys"]) != CHECKPOINTS or result["scores"].shape != (len(rows), 6):
            raise ValueError("PFN worker returned malformed predictions")
        scores = result["scores"]
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("PFN probabilities outside their contract")
    return scores


def score_reserved_role(record_path, generator, role, engine_path, destination):
    """The access check happens before raw data preparation, including cached scores."""
    record_path, destination = Path(record_path), Path(destination)
    engine = load_engine(engine_path)
    config = read_json(Path(engine["protocol_path"]))
    record = read_json(record_path)
    authorize_role(record, generator, role, config)
    freeze = verify_reservation(Path(engine["reservation"]))
    with np.load(freeze["archives"]["cohorts"]["path"], allow_pickle=False) as a:
        rows, labels = a[f"{generator}_{role}_rows"], a[f"{generator}_{role}_labels"]
    raw = Path("data/raw") / next(
        s["filename"] for s in config["files"] if s["generator"] == generator
    )
    proof = {
        "mode": "reserved_role",
        "engine_freeze": str(engine_path),
        "study_record": str(record_path),
        "generator": generator,
        "role": role,
        "reservation": engine["reservation"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
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
