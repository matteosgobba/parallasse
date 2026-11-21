"""Run bounded PFN feasibility or frozen-config source fits without held-out data."""

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from parallasse.development import load_source_development
from parallasse.download import sha256_file
from parallasse.features import FEATURE_NAMES, FEATURE_VERSION, aggregate_features
from parallasse.preprocessing import encode_pid, kinematics


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def matching_pairs(models: list, seeds: list[int]) -> None:
    by_key = {(item["model"], item["seed"]): item for item in models}
    for seed in seeds:
        k, pid = (by_key[(name, seed)] for name in ("PFN-K", "PFN-ID"))
        for key in ("parameter_count", "initial_weights_sha256"):
            if k[key] != pid[key]:
                raise ValueError(f"PFN pair mismatch at seed {seed}: {key}")
        for left, right in zip(k["history"], pid["history"], strict=False):
            if left["train_order_sha256"] != right["train_order_sha256"]:
                raise ValueError(f"PFN pair minibatch order mismatch at seed {seed}")


def prepare_inputs(config: dict, output: Path) -> dict:
    schedule = config["schedule"]
    data = load_source_development(Path.cwd(), schedule["train_rows"], schedule["tune_rows"])
    arrays = {}
    for split in ("train", "tune"):
        particles = getattr(data, f"{split}_particles")
        parts = {"aggregate": [], "coordinates": [], "mask": [], "pid": []}
        for start in range(0, len(particles), 4096):
            chunk = particles[start : start + 4096]
            coords, mask = kinematics(chunk)
            parts["coordinates"].append(coords)
            parts["mask"].append(mask)
            parts["pid"].append(encode_pid(chunk, config["pid_vocabulary"]).astype(np.int16))
            parts["aggregate"].append(aggregate_features(chunk))
        for key, values in parts.items():
            arrays[f"{split}_{key}"] = np.concatenate(values)
        arrays[f"{split}_rows"] = getattr(data, f"{split}_rows")
        arrays[f"{split}_labels"] = getattr(data, f"{split}_labels")
    np.savez(output, **arrays)
    return {
        **data.provenance,
        "development_artifact": str(output),
        "development_sha256": sha256_file(output),
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
    }


def evaluation_freeze(path: Path, record: dict, record_path: Path) -> None:
    files = [
        Path(name)
        for name in (
            "results/task0001/cohorts.json",
            "results/task0001/preprocessing.json",
            "configs/evaluation_rules.json",
            "requirements-lock.txt",
        )
    ]
    write_json(
        path,
        {
            "protocol_version": "0.4-evaluation-input-freeze",
            "state": "models_and_rules_frozen",
            "source_run_record": str(record_path),
            "source_run_record_sha256": sha256_file(record_path),
            "input_files_sha256": {str(p): sha256_file(p) for p in files},
            "dataset_contract": read_json(files[0]),
            "pid_contract": read_json(files[1]),
            "rules": read_json(files[2]),
            "source_snapshot": record["source_snapshot"],
            "source_files_sha256": record["source_files_sha256"],
            "inference": {
                key: record["config"][key] for key in ("device", "cpu_threads", "batch_size")
            },
            "models": [
                {
                    key: item[key]
                    for key in (
                        "model",
                        "seed",
                        "checkpoint",
                        "checkpoint_sha256",
                        "trees_used",
                        "model_arguments",
                        "scaler",
                    )
                    if key in item
                }
                for item in record["models"]
            ],
            "numerical_cuts": None,
            "next": "Freeze source-calibration cuts before scoring either final cohort.",
            "calibration_scored": False,
            "heldout_quality_opened": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot-id")
    args = parser.parse_args()
    for value in (args.run_id, args.pilot_id):
        if value is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise ValueError("Use a simple, unique run ID")
    config_path = Path("configs/source_fits.json")
    declared = read_json(config_path)
    schedule = declared[args.mode]
    config_hash = sha256_file(config_path)
    historical_path = Path("results/task0002/20261105-baseline-pilot-01/run.json")
    historical_validation = read_json(Path("results/task0002/validation.json"))
    if (
        sha256_file(historical_path)
        != historical_validation["evidence_sha256"][str(historical_path)]
    ):
        raise ValueError("Baseline selection evidence changed")
    historical = read_json(historical_path)
    if historical["selected_xgb_pilot_variant"] != f"xgb_depth{declared['xgb_depth']}":
        raise ValueError("Full XGB depth differs from the previous selection")
    if args.mode == "full":
        if not args.pilot_id:
            raise ValueError("A completed PFN pilot is required before full fitting")
        pilot_path = Path("results/task0003") / args.pilot_id / "run.json"
        pilot = read_json(pilot_path)
        if (
            pilot["status"] != "completed"
            or pilot["mode"] != "pilot"
            or pilot["source_config_sha256"] != config_hash
            or not pilot["paired_pfn_checks_passed"]
        ):
            raise ValueError("PFN pilot not complete under this configuration")
        if any(sha256_file(Path(p)) != h for p, h in pilot["source_files_sha256"].items()):
            raise ValueError("Source changed since the PFN pilot")
    config = {
        **declared,
        "schedule": schedule,
        "xgb": historical["config"]["xgb"],
        "pid_vocabulary": read_json(Path("results/task0001/preprocessing.json"))["pid_vocabulary"],
    }
    output = Path("results/task0003") / args.run_id
    artifacts = Path("artifacts/task0003") / args.run_id
    if output.exists() or artifacts.exists():
        raise FileExistsError("Run already exists; evidence is never overwritten")
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    snapshot = artifacts / "source_snapshot"
    paths = sorted(Path("src/parallasse").glob("*.py")) + [
        Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),
        Path("scripts/source_fit_worker.py"),
        config_path,
        Path("configs/evaluation_rules.json"),
        Path("requirements-lock.txt"),
    ]
    source_hashes = {str(path): sha256_file(path) for path in paths}
    for path in paths:
        (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot / path)
    started = time.perf_counter()
    record = {
        "run_id": args.run_id,
        "mode": args.mode,
        "status": "running",
        "task": "TASK-0003",
        "started_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "source_config_sha256": config_hash,
        "source_files_sha256": source_hashes,
        "source_snapshot": str(snapshot),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "xgboost", "scikit-learn")
        },
        "model_runtime_isolation": "fresh executable process per model and seed",
        "opened_cohorts": ["source train", "source tune"],
        "heldout_quality_opened": False,
        "models": [],
    }
    if args.mode == "full":
        record["required_pfn_pilot"] = {"path": str(pilot_path), "sha256": sha256_file(pilot_path)}
    record_path = output / "run.json"
    write_json(record_path, record)
    try:
        record["inputs"] = prepare_inputs(config, artifacts / "development.npz")
        write_json(record_path, record)
        for model_id in schedule["models"]:
            for seed in schedule["seeds"]:
                key = f"{model_id}-{seed}"
                print(f"Fitting {key} ({args.mode})", flush=True)
                with (artifacts / f"{key}.log").open("x") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/source_fit_worker.py",
                            args.run_id,
                            model_id,
                            str(seed),
                        ],
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                item = read_json(artifacts / key / "fit.json")
                record["models"].append(item)
                write_json(record_path, record)
                print(f"{key}: tuning AUC {item['tune']['roc_auc']:.6f}", flush=True)
        matching_pairs(record["models"], schedule["seeds"])
        with (output / "tuning.csv").open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["model", "seed", "tuning_auc", "tuning_log_loss", "selected_epoch_or_trees"]
            )
            for item in record["models"]:
                writer.writerow(
                    [
                        item["model"],
                        item["seed"],
                        item["tune"]["roc_auc"],
                        item["tune"]["log_loss"],
                        item.get("best_epoch", item.get("trees_used")),
                    ]
                )
        record.update(
            status="completed",
            paired_pfn_checks_passed=True,
            wall_seconds=time.perf_counter() - started,
            completed_utc=datetime.now(UTC).isoformat(),
        )
        write_json(record_path, record)
        if args.mode == "full":
            evaluation_freeze(output / "evaluation_freeze.json", record, record_path)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(record_path, record)
        raise


if __name__ == "__main__":
    main()
