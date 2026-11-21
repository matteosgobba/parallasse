"""Isolated Torch worker for EFN source fitting and frozen-model inference."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from evaluate_heldout import write_json

from parallasse.download import sha256_file
from parallasse.efn import EnergyFlowNetwork, efn_prediction_checks
from parallasse.heldout import read_json, verify_hashes
from parallasse.source_neural import fit_neural, neural_scores


def runtime(threads):
    if "xgboost" in sys.modules:
        raise RuntimeError("The EFN worker must not load XGBoost")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def fit(record_path, schedule_name, seed):
    record = read_json(record_path)
    if record["status"] != "fitting":
        raise ValueError("An active EFN source fit is required")
    config = record["config"]
    schedule = config["training"][schedule_name]
    if seed not in schedule["seeds"]:
        raise ValueError("Seed outside the declared EFN schedule")
    runtime(config["training"]["cpu_threads"])
    torch.manual_seed(seed)
    dataref = record["development"][schedule_name]
    verify_hashes({dataref["path"]: dataref["sha256"]})
    output = Path(record["artifacts"]) / f"{schedule_name}-EFN-{seed}"
    output.mkdir(exist_ok=False)
    with np.load(dataref["path"], allow_pickle=False) as data:
        train = tuple(torch.from_numpy(data[f"train_{k}"]) for k in ("coordinates", "mask"))
        tune = tuple(torch.from_numpy(data[f"tune_{k}"]) for k in ("coordinates", "mask"))
        model = EnergyFlowNetwork(**config["architecture"])
        scores, metrics = fit_neural(
            model,
            train,
            data["train_labels"],
            tune,
            data["tune_labels"],
            config["training"],
            seed,
            schedule["neural_max_epochs"],
        )
        weights = output / "weights.pt"
        torch.save(model.state_dict(), weights)
        restored = EnergyFlowNetwork(**config["architecture"])
        restored.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        np.testing.assert_array_equal(
            neural_scores(restored, tune, config["training"]["batch_size"]), scores
        )
        checks = efn_prediction_checks(
            restored, data["probe_particles"], config["invariance_tolerance"]
        )
    predictions = output / "tune_scores.npy"
    np.save(predictions, scores, allow_pickle=False)
    write_json(
        output / "fit.json",
        {
            **metrics,
            "model": "EFN",
            "seed": seed,
            "mode": schedule_name,
            "model_arguments": config["architecture"],
            "checkpoint": str(weights),
            "checkpoint_sha256": sha256_file(weights),
            "tune_scores": str(predictions),
            "tune_scores_sha256": sha256_file(predictions),
            "checkpoint_replay_exact": True,
            "invariance": checks,
            "train_rows": schedule["train_rows"],
            "tune_rows": schedule["tune_rows"],
        },
    )


def predict(manifest_path, expected_hash, inputs, destination):
    verify_hashes({str(manifest_path): expected_hash})
    manifest = read_json(manifest_path)
    runtime(manifest["cpu_threads"])
    values = {}
    with np.load(inputs, allow_pickle=False) as data:
        tensors = tuple(torch.from_numpy(data[k]) for k in ("coordinates", "mask"))
        for record in manifest["models"]:
            verify_hashes({record["checkpoint"]: record["checkpoint_sha256"]})
            model = EnergyFlowNetwork(**record["model_arguments"])
            model.load_state_dict(
                torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
            )
            values[f"EFN-{record['seed']}"] = neural_scores(model, tensors, manifest["batch_size"])
    with destination.open("xb") as handle:
        np.savez(handle, **values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    training = commands.add_parser("fit")
    training.add_argument("record", type=Path)
    training.add_argument("schedule", choices=("pilot", "full"))
    training.add_argument("seed", type=int)
    inference = commands.add_parser("predict")
    inference.add_argument("manifest", type=Path)
    inference.add_argument("manifest_hash")
    inference.add_argument("inputs", type=Path)
    inference.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.operation == "fit":
        fit(args.record, args.schedule, args.seed)
    else:
        predict(args.manifest, args.manifest_hash, args.inputs, args.destination)


if __name__ == "__main__":
    main()
