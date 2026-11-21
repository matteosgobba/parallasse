"""Freeze the score-template protocol and audit/reserve unused records, without scoring."""

import argparse
import fcntl
import importlib.metadata
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from evaluate_heldout import save_arrays, write_json

from parallasse.data import CANONICAL_VERSION, fingerprints, integrity_summary, load_jets
from parallasse.download import download_declared_file, fetch_record, sha256_file
from parallasse.heldout import read_json, verify_hashes, verify_input_freeze
from parallasse.score_template_design import (
    CHECKPOINTS,
    reserve_template_roles,
    validate_design,
    verify_reservation,
)

CONFIG = Path("configs/score_templates.json")


def now():
    return datetime.now(UTC).isoformat()


def historical_inputs(config):
    """Use only verified metadata, identity archives, labels and frozen inputs."""
    v9 = read_json(Path("results/task0009/validation.json"))
    v10 = read_json(Path(config["feasibility_validation"]))
    previous_path = Path(config["previous_efn"])
    replication_path = Path(config["previous_replication"])
    verify_hashes({str(previous_path): v9["evidence_sha256"][str(previous_path)]})
    verify_hashes(v10["evidence_sha256"])
    previous, replication = read_json(previous_path), read_json(replication_path)
    if previous["status"] != "completed" or replication["status"] != "completed":
        raise ValueError("Prior completed evidence required")
    frozen = verify_input_freeze()
    hashes = {
        str(previous_path): sha256_file(previous_path),
        str(replication_path): previous["input_sha256"][str(replication_path)],
        config["model_freeze"]: sha256_file(Path(config["model_freeze"])),
        config["core_evaluation"]: previous["input_sha256"][config["core_evaluation"]],
        config["feasibility_run"]: v10["evidence_sha256"][config["feasibility_run"]],
        config["feasibility_validation"]: sha256_file(Path(config["feasibility_validation"])),
    }
    core = read_json(Path(config["core_evaluation"]))
    hashes[core["cut_freeze"]] = core["cut_freeze_sha256"]
    selected = [m for m in frozen["models"] if f"{m['model']}-{m['seed']}" in CHECKPOINTS]
    hashes.update({m["checkpoint"]: m["checkpoint_sha256"] for m in selected})
    refs = [replication["identities"], previous["archives"]["identities"]]
    prior_hashes = []
    for ref, keys in zip(
        refs, [("old_source", "old_target", "source", "target"), ("source", "target")], strict=True
    ):
        hashes[ref["path"]] = ref["sha256"]
        verify_hashes({ref["path"]: ref["sha256"]})
        with np.load(ref["path"], allow_pickle=False) as archive:
            prior_hashes.extend(archive[key] for key in keys)
    old_specs = (
        read_json(Path("configs/data.json"))["files"]
        + read_json(Path("configs/replication.json"))["files"]
        + read_json(Path("configs/efn.json"))["files"]
    )
    prior_labels = []
    for spec in old_specs:
        path = Path("data/raw") / spec["filename"]
        hashes[str(path)] = previous["input_sha256"][str(path)]
        verify_hashes({str(path): hashes[str(path)]})
        with np.load(path, allow_pickle=False) as archive:
            prior_labels.append(archive["y"])
    verify_hashes(hashes)
    all_labels, all_hashes = np.concatenate(prior_labels), np.concatenate(prior_hashes)
    if len(all_labels) != 600000 or len(all_hashes) != len(all_labels):
        raise ValueError("Expected all six previous raw files in exact identity order")
    return hashes, all_labels, all_hashes, selected, core


def reserve(run_id):
    output, artifacts = Path("results/task0011") / run_id, Path("artifacts/task0011") / run_id
    if output.exists() or artifacts.exists():
        raise FileExistsError("Reservation run already exists; evidence is never overwritten")
    lock_path = Path("work/score-template-locks") / f"{run_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or artifacts.exists():
            raise FileExistsError("Reservation run already exists; evidence is never overwritten")
        config = read_json(CONFIG)
        rank = validate_design(config)
        output.mkdir(parents=True)
        artifacts.mkdir(parents=True)
        record = {
            "task": "TASK-0011",
            "run_id": run_id,
            "status": "preparing",
            "started_utc": now(),
            "protocol_version": config["protocol_version"],
            "new_classifier_fits": 0,
            "fresh_scores_computed": False,
            "feature_performance_inspected": False,
            "python": sys.version,
            "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy")},
            "calibration_order_statistic_one_based": rank,
        }
        write_json(output / "run.json", record)
        paths = [
            CONFIG,
            Path("reports/score_template_protocol.md"),
            Path("scripts/reserve_score_template_data.py"),
            Path("src/parallasse/score_template_design.py"),
            Path("tests/test_score_template_design.py"),
            Path("src/parallasse/data.py"),
            Path("src/parallasse/download.py"),
            Path("src/parallasse/mixtures.py"),
            Path("src/parallasse/heldout.py"),
            Path("src/parallasse/heldout_metrics.py"),
            Path("scripts/evaluate_heldout.py"),
            Path("requirements-lock.txt"),
            Path("pyproject.toml"),
        ]
        snapshot = artifacts / "source_snapshot"
        for path in paths:
            (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot / path)
        record.update(
            source_snapshot=str(snapshot),
            source_files_sha256={str(p): sha256_file(p) for p in paths},
        )
        write_json(output / "protocol.json", config)
        record.update(
            protocol_sha256=sha256_file(output / "protocol.json"),
            design_frozen_before_data_utc=now(),
        )
        write_json(output / "run.json", record, "w")
        started = time.perf_counter()
        try:
            hashes, prior_labels, prior_hashes, selected, core = historical_inputs(config)
            old_bins = Path(config["binning"]["reference"])
            feasibility = read_json(Path(config["feasibility_run"]))
            verify_hashes({str(old_bins): feasibility["artifact_sha256"][str(old_bins)]})
            hashes[str(old_bins)] = feasibility["artifact_sha256"][str(old_bins)]
            with np.load(old_bins, allow_pickle=False) as archive:
                if tuple(archive["keys"]) != CHECKPOINTS or archive["edges"].shape != (6, 9):
                    raise ValueError("Unexpected frozen source-only binning")
                bins = save_arrays(
                    artifacts / "bin_edges.npz", keys=archive["keys"], edges=archive["edges"]
                )
            identities, labels, summaries = {}, {}, {}
            download_seconds = 0.0
            for spec in config["files"]:
                metadata = fetch_record(spec["record_id"])
                entry = next(item for item in metadata["files"] if item["key"] == spec["filename"])
                if (
                    entry["size"] != spec["size_bytes"]
                    or metadata["metadata"].get("license", {}).get("id") != "cc-by-4.0"
                ):
                    raise ValueError("Published size/licence differs from the declared record")
                write_json(artifacts / f"zenodo_{spec['record_id']}.json", metadata)
                print(f"Verifying/downloading {spec['filename']}", flush=True)
                before = time.perf_counter()
                provenance = download_declared_file(spec, Path("data/raw"), output)
                download_seconds += time.perf_counter() - before
                raw = Path("data/raw") / spec["filename"]
                if raw.stat().st_size != spec["size_bytes"]:
                    raise ValueError("Declared file size mismatch")
                hashes[str(raw)] = provenance["sha256"]
                particles, y = load_jets(raw)
                if len(y) != spec["expected_rows"]:
                    raise ValueError("Unexpected new-file row count")
                h = fingerprints(particles)
                generator = spec["generator"]
                labels[generator], identities[generator] = y, h
                summaries[generator] = integrity_summary(particles, y, h)
                del particles
                print(f"Audited {generator}: {len(y)} exact constituent records", flush=True)
            arrays, counts = reserve_template_roles(
                labels,
                identities,
                prior_labels,
                prior_hashes,
                config["cohorts"]["partition_seeds"],
                config["cohorts"]["minimum_rows_per_class_per_role"],
            )
            archives = {
                "cohorts": save_arrays(artifacts / "cohorts.npz", **arrays),
                "identities": save_arrays(
                    artifacts / "identities.npz", prior=prior_hashes, **identities
                ),
                "bin_edges": bins,
            }
            write_json(
                output / "audit.json",
                {
                    "canonical_definition": CANONICAL_VERSION,
                    "previous_identity_records": len(prior_hashes),
                    "total_compared_identity_records": len(prior_hashes)
                    + sum(len(h) for h in identities.values()),
                    "new_datasets": summaries,
                    "retained_counts": counts,
                    "archives": archives,
                    "scope": (
                        "Exact constituent records; cached verified prior identities, "
                        "newly computed identities; original collision IDs unavailable."
                    ),
                    "fresh_scores_computed": False,
                },
            )
            cut_record = read_json(Path(core["cut_freeze"]))
            write_json(
                output / "model_cut_contract.json",
                {
                    "keys": CHECKPOINTS,
                    "models": selected,
                    "preprocessing": read_json(Path(config["model_freeze"]))["pid_contract"],
                    "cuts": {key: cut_record["cuts"][key] for key in CHECKPOINTS},
                    "operator": ">",
                    "new_training": False,
                },
            )
            outputs = {
                str(output / name): sha256_file(output / name)
                for name in (
                    "protocol.json",
                    "audit.json",
                    "model_cut_contract.json",
                    "pythia_provenance.json",
                    "herwig_provenance.json",
                )
            }
            freeze = {
                "state": "design_and_roles_frozen",
                "run_id": run_id,
                "frozen_utc": now(),
                "fresh_scores_computed": False,
                "protocol_path": str(output / "protocol.json"),
                "input_sha256": hashes,
                "output_sha256": outputs,
                "archives": archives,
                "audit_source_sha256": record["source_files_sha256"],
                "audit_source_snapshot": str(snapshot),
                "executable_inference_freeze": "required_in_TASK-0012",
                "calibrated_numerical_lr_cutoffs": "pending_TASK-0013_modelling_only",
                "evaluation_access": "not_granted_by_this_reservation",
            }
            write_json(output / "design_freeze.json", freeze)
            verify_reservation(output / "design_freeze.json")
            record.update(
                status="completed",
                completed_utc=now(),
                elapsed_seconds=time.perf_counter() - started,
                download_and_hash_seconds=download_seconds,
                input_sha256=hashes,
                archives=archives,
                output_sha256={
                    **outputs,
                    str(output / "design_freeze.json"): sha256_file(output / "design_freeze.json"),
                },
                next_task="TASK-0012 executable implementation/freeze; no scoring in TASK-0011",
            )
            write_json(output / "run.json", record, "w")
            print(
                "Design and four disjoint data roles frozen; no new classifier scores.", flush=True
            )
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(output / "run.json", record, "w")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("Use a simple unique run ID")
    if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        parser.error("Run from the canonical project root")
    reserve(args.run_id)
