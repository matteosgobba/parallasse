"""Two-stage evaluation of frozen models: calibrate first, score final cohorts second."""

import argparse
import csv
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from parallasse.download import sha256_file
from parallasse.heldout import (
    INPUT_FREEZE,
    derive_cuts,
    prepare_cohort,
    read_json,
    verified_cuts,
    verify_hashes,
    verify_input_freeze,
)
from parallasse.heldout_metrics import KEYS, METRICS, checked_scores, cohort_metrics, summarize


def write_json(path: Path, value: dict, mode: str = "x") -> None:
    with path.open(mode) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_arrays(path: Path, **arrays) -> dict:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {"path": str(path), "sha256": sha256_file(path)}


def score_cohort(frozen: dict, freeze_hash: str, cohort: str, artifacts: Path) -> dict:
    print(f"Scoring {cohort} with 12 frozen checkpoints", flush=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="inference-", dir=artifacts) as scratch_dir:
        scratch = Path(scratch_dir)
        inputs = scratch / "cohort.npz"
        rows, labels = prepare_cohort(frozen, cohort, inputs)
        scores = {}
        for family in ("xgb", "neural"):
            destination = scratch / f"{family}.npz"
            with (artifacts / f"{cohort}-{family}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/heldout_worker.py",
                        family,
                        str(INPUT_FREEZE),
                        freeze_hash,
                        str(inputs),
                        str(destination),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            with np.load(destination, allow_pickle=False) as family_scores:
                if set(scores) & set(family_scores.files):
                    raise ValueError("Duplicate inference model keys")
                scores.update(
                    {key: checked_scores(family_scores[key]) for key in family_scores.files}
                )
        if set(scores) != set(KEYS) or any(
            score.shape != labels.shape for score in scores.values()
        ):
            raise ValueError("Inference did not produce one score per row and frozen checkpoint")
        result = save_arrays(
            artifacts / f"{cohort}_scores.npz",
            rows=rows,
            labels=labels,
            keys=np.asarray(KEYS),
            scores=np.column_stack([scores[key] for key in KEYS]),
        )
    return {**result, "rows": len(rows), "wall_seconds": time.perf_counter() - started}


def snapshot_source(artifacts: Path) -> dict:
    paths = sorted(Path("src/parallasse").glob("*.py")) + [
        Path("scripts/evaluate_heldout.py"),
        Path("scripts/heldout_worker.py"),
        Path("tests/test_heldout.py"),
        Path("configs/evaluation_rules.json"),
        Path("requirements-lock.txt"),
    ]
    snapshot = artifacts / "source_snapshot"
    for path in paths:
        (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot / path)
    hashes = {str(path): sha256_file(path) for path in paths}
    verify_hashes(hashes, snapshot)
    return {"source_snapshot": str(snapshot), "source_files_sha256": hashes}


def calibrate(frozen: dict, run_id: str, output: Path, artifacts: Path) -> None:
    if output.exists() or artifacts.exists():
        raise FileExistsError("Evaluation run already exists; evidence is never overwritten")
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    run_path = output / "run.json"
    record = {
        "task": "TASK-0004",
        "run_id": run_id,
        "protocol_version": "0.4-heldout-rules",
        "status": "calibrating",
        "started_utc": datetime.now(UTC).isoformat(),
        "model_freeze": str(INPUT_FREEZE),
        "model_freeze_sha256": sha256_file(INPUT_FREEZE),
        "platform": platform.platform(),
        "inference": frozen["inference"],
        "final_scoring_started_utc": None,
        "heldout_quality_opened": False,
        **snapshot_source(artifacts),
    }
    write_json(run_path, record)
    try:
        calibration = score_cohort(frozen, record["model_freeze_sha256"], "calibration", artifacts)
        cuts = derive_cuts(Path(calibration["path"]))
        lock = output / "cut_freeze.json"
        write_json(
            lock,
            {
                "run_id": run_id,
                "frozen_utc": datetime.now(UTC).isoformat(),
                "model_freeze_sha256": record["model_freeze_sha256"],
                "operator": ">",
                "calibration_cohort": frozen["rules"]["calibration_cohort"],
                "background_acceptance_target": frozen["rules"]["background_acceptance_target"],
                "threshold_rule": frozen["rules"]["threshold_rule"],
                "calibration_scores": calibration,
                "cuts": cuts,
            },
        )
        record.update(
            status="cuts_frozen", cut_freeze=str(lock), cut_freeze_sha256=sha256_file(lock)
        )
        verified_cuts(record, lock, run_id)
        write_json(run_path, record, "w")
        print(f"Frozen {len(cuts)} cuts from 5,000 calibration gluons; no final scores", flush=True)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(run_path, record, "w")
        raise


def evaluate(frozen: dict, run_id: str, output: Path, artifacts: Path) -> None:
    run_path = output / "run.json"
    record = read_json(run_path)
    if record["run_id"] != run_id or record["model_freeze_sha256"] != sha256_file(INPUT_FREEZE):
        raise ValueError("Calibration belongs to a different model freeze or run")
    cuts = verified_cuts(record, output / "cut_freeze.json", run_id)
    verify_hashes(record["source_files_sha256"])
    verify_hashes(record["source_files_sha256"], Path(record["source_snapshot"]))
    record.update(
        status="scoring",
        final_scoring_started_utc=datetime.now(UTC).isoformat(),
        heldout_quality_opened=True,
    )
    write_json(run_path, record, "w")
    started = time.perf_counter()
    try:
        per_seed, draws, cohort_artifacts = {}, {}, {}
        plan = frozen["rules"]["bootstrap"]
        streams = np.random.SeedSequence(plan["seed"]).spawn(2)
        for cohort, stream in zip(("test", "target"), streams, strict=True):
            result = score_cohort(frozen, record["model_freeze_sha256"], cohort, artifacts)
            cohort_artifacts[cohort] = result
            record["score_artifacts"] = cohort_artifacts
            write_json(run_path, record, "w")
            print(f"Computing {plan['replicates']} paired bootstrap draws for {cohort}", flush=True)
            with np.load(result["path"], allow_pickle=False) as data:
                per_seed[cohort], draws[cohort] = cohort_metrics(
                    data["labels"],
                    data["scores"],
                    cuts,
                    np.random.Generator(np.random.PCG64(stream)),
                    plan["replicates"],
                )
        report = summarize(per_seed, draws)
        report.update(
            {
                "run_id": run_id,
                "bootstrap": plan,
                "cut_freeze_sha256": record["cut_freeze_sha256"],
                "class_counts": {
                    cohort: {
                        "quarks": rows[0]["total_quarks"],
                        "gluons": rows[0]["total_gluons"],
                    }
                    for cohort, rows in per_seed.items()
                },
                "score_artifacts": cohort_artifacts,
            }
        )
        draws_record = save_arrays(
            artifacts / "bootstrap.npz",
            **draws,
            keys=np.asarray(KEYS),
            metrics=np.asarray(METRICS),
        )
        write_json(output / "summary.json", report)
        with (output / "per_seed.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["cohort", *per_seed["test"][0]])
            writer.writeheader()
            for cohort, rows in per_seed.items():
                writer.writerows({"cohort": cohort, **row} for row in rows)
        verify_hashes({record["cut_freeze"]: record["cut_freeze_sha256"]})
        record.update(
            status="completed",
            completed_utc=datetime.now(UTC).isoformat(),
            evaluation_wall_seconds=time.perf_counter() - started,
            bootstrap_artifact=draws_record,
            output_sha256={
                str(output / name): sha256_file(output / name)
                for name in ("summary.json", "per_seed.csv", "cut_freeze.json")
            },
        )
        write_json(run_path, record, "w")
        print(f"Held-out evaluation complete in {record['evaluation_wall_seconds']:.1f} seconds")
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(run_path, record, "w")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        raise ValueError("Use a simple, unique run ID")
    frozen = verify_input_freeze()
    output = Path("results/task0004") / args.run_id
    artifacts = Path("artifacts/task0004") / args.run_id
    operation = calibrate if args.stage == "calibrate" else evaluate
    operation(frozen, args.run_id, output, artifacts)


if __name__ == "__main__":
    main()
