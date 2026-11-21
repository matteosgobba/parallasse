"""Bounded source-only pilots with isolated native model-family processes."""

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from parallasse.data import class_counts
from parallasse.development import load_source_development
from parallasse.download import sha256_file
from parallasse.features import FEATURE_NAMES, FEATURE_VERSION, aggregate_features


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_budget(config: dict) -> None:
    if config["scope"] != "source_only_pilot_not_final_fits":
        raise ValueError("This entry point is only for development pilots")
    if not 0 < config["train_limit"] <= 5000 or not 0 < config["tune_limit"] <= 1000:
        raise ValueError("Pilot row budget exceeded")
    depths = config["xgb"]["max_depth_candidates"]
    if not 1 <= len(depths) <= 3 or len(set(depths)) != len(depths):
        raise ValueError("At most three distinct XGB depth candidates are allowed")
    if not 1 <= config["mlp"]["max_epochs"] <= 3:
        raise ValueError("The neural runtime pilot is limited to three epochs")
    if config["forbidden_cohorts"] != ["calibration", "test", "target"]:
        raise ValueError("Held-out cohort boundaries must remain explicit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.json"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        raise ValueError("Run IDs must be a simple alphanumeric identifier")
    root = Path.cwd().resolve()
    config = json.loads(args.config.read_text())
    validate_budget(config)
    result_dir = root / "results/task0002" / args.run_id
    artifact_dir = root / "artifacts/task0002" / args.run_id
    if result_dir.exists() or artifact_dir.exists():
        raise FileExistsError("Run already exists; evidence is never overwritten")
    result_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    source_files = sorted((root / "src/parallasse").glob("*.py")) + [
        root / "scripts/run_baseline_pilot.py",
        root / "scripts/baseline_worker.py",
        root / args.config,
        root / "configs/pilot.json",
        root / "requirements-lock.txt",
    ]
    record = {
        "schema_version": 1,
        "run_id": args.run_id,
        "task": "TASK-0002",
        "protocol_version": config["protocol_version"],
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "purpose": "Source-only implementation/runtime pilot, not final performance evidence",
        "config": config,
        "source_files_sha256": {str(p.relative_to(root)): sha256_file(p) for p in source_files},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "xgboost", "torch")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "device": "cpu",
        "opened_cohorts": ["train", "tune"],
        "heldout_quality_opened": False,
        "isolated_model_family_workers": True,
        "models": [],
    }
    write_json(result_dir / "config.json", config)
    write_json(result_dir / "run.json", record)
    try:
        data = load_source_development(root, config["train_limit"], config["tune_limit"])
        feature_started = time.perf_counter()
        train_x = aggregate_features(data.train_particles)
        tune_x = aggregate_features(data.tune_particles)
        feature_seconds = time.perf_counter() - feature_started
        inputs_path = artifact_dir / "inputs.npz"
        np.savez_compressed(
            inputs_path,
            train_features=train_x,
            tune_features=tune_x,
            train_labels=data.train_labels,
            tune_labels=data.tune_labels,
            train_rows=data.train_rows,
            tune_rows=data.tune_rows,
        )
        write_json(
            result_dir / "features.json",
            {
                "feature_names": list(FEATURE_NAMES),
                "feature_version": FEATURE_VERSION,
                "dtype": str(train_x.dtype),
                "train_shape": list(train_x.shape),
                "tune_shape": list(tune_x.shape),
                "train_class_counts": class_counts(data.train_labels),
                "tune_class_counts": class_counts(data.tune_labels),
                "source_provenance": data.provenance,
                "input_artifact": str(inputs_path.relative_to(root)),
                "input_artifact_sha256": sha256_file(inputs_path),
                "feature_extraction_seconds": feature_seconds,
                "pid_used_as_input": False,
            },
        )
        record["features_manifest_sha256"] = sha256_file(result_dir / "features.json")
        all_scores = {}
        for family, metadata in (("xgb", "xgb_models.json"), ("mlp", "mlp_model.json")):
            worker_started = time.perf_counter()
            worker = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/baseline_worker.py"),
                    "--family",
                    family,
                    "--run-id",
                    args.run_id,
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=180,
            )
            (artifact_dir / f"{family}_stdout.txt").write_text(worker.stdout)
            (artifact_dir / f"{family}_stderr.txt").write_text(worker.stderr)
            if worker.returncode != 0:
                raise RuntimeError(
                    f"{family} worker failed with status {worker.returncode}; see logs"
                )
            family_record = json.loads((result_dir / metadata).read_text())
            record["models"].extend(family_record["models"])
            record[f"{family}_worker_wall_seconds"] = time.perf_counter() - worker_started
            with np.load(artifact_dir / f"{family}_tune_scores.npz", allow_pickle=False) as scores:
                all_scores.update({name: scores[name] for name in scores.files})
            write_json(result_dir / "run.json", record)
            print(worker.stdout, end="", flush=True)
        scores_path = artifact_dir / "tune_scores.npz"
        np.savez_compressed(scores_path, **all_scores)
        best_xgb = max(
            (item for item in record["models"] if item["model"] == "XGB"),
            key=lambda item: (item["tune"]["roc_auc"], -item["max_depth"]),
        )
        record.update(
            status="completed",
            completed_utc=datetime.now(UTC).isoformat(),
            wall_seconds=time.perf_counter() - started,
            selected_xgb_pilot_variant=best_xgb["variant"],
            tune_score_artifact=str(scores_path.relative_to(root)),
            tune_score_artifact_sha256=sha256_file(scores_path),
            scientific_final_fits=0,
            note=(
                "One seed and identical 5k/1k source prefixes; three MLP epochs. "
                "No final-model or transfer conclusion follows."
            ),
        )
        write_json(result_dir / "run.json", record)
        with (result_dir / "pilot_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "variant",
                    "train_rows",
                    "tune_rows",
                    "tune_roc_auc",
                    "tune_log_loss",
                    "selected_iteration_or_epoch",
                    "training_loop_seconds",
                ],
            )
            writer.writeheader()
            for item in record["models"]:
                writer.writerow(
                    {
                        "variant": item["variant"],
                        "train_rows": len(train_x),
                        "tune_rows": len(tune_x),
                        "tune_roc_auc": item["tune"]["roc_auc"],
                        "tune_log_loss": item["tune"]["log_loss"],
                        "selected_iteration_or_epoch": item.get(
                            "trees_used", item.get("best_epoch")
                        ),
                        "training_loop_seconds": item.get(
                            "fit_seconds", item.get("fit_seconds_including_tune_checks")
                        ),
                    }
                )
        print(f"Completed {args.run_id}; held-out quality remains unopened.", flush=True)
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(result_dir / "run.json", record)
        raise


if __name__ == "__main__":
    main()
