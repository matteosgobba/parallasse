"""Internal single model/seed fit, with one native model library per process."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from parallasse.download import sha256_file


def xgb_fit(data, config, seed, output):
    import xgboost as xgb

    from parallasse.xgb_baseline import fit_xgb_pilot, xgb_quark_scores

    if "torch" in sys.modules:
        raise RuntimeError("XGB worker must not load Torch")
    train, tune = data["train_aggregate"], data["tune_aggregate"]
    model, scores, record = fit_xgb_pilot(
        train,
        data["train_labels"],
        tune,
        data["tune_labels"],
        config["xgb"],
        config["xgb_depth"],
        seed,
        config["cpu_threads"],
    )
    path = output / "weights.ubj"
    model.save_model(path)
    restored = xgb.XGBClassifier(n_jobs=config["cpu_threads"], device="cpu")
    restored.load_model(path)
    np.testing.assert_array_equal(xgb_quark_scores(restored, tune), scores)
    return scores, record, path


def neural_fit(data, config, model_id, seed, output):
    import torch

    from parallasse.baselines import fit_training_scaler, scaler_from_record, scaler_record
    from parallasse.mlp_baseline import AggregateMLP
    from parallasse.pfn import ParticleFlowNetwork
    from parallasse.source_neural import fit_neural, neural_scores, pfn_prediction_checks

    if "xgboost" in sys.modules:
        raise RuntimeError("Neural worker must not load XGBoost")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    extra = {}
    if model_id == "MLP":
        model_args = {"hidden_sizes": config["mlp_hidden_sizes"]}
        model_type = AggregateMLP
        raw_train, raw_tune = data["train_aggregate"], data["tune_aggregate"]
        scaler = fit_training_scaler(raw_train)
        scaling = scaler_record(scaler)
        scaling["fit_cohort"] = "full_source_training_only"
        extra["scaler"] = scaling
        train = (torch.from_numpy(scaler.transform(raw_train)),)
        tune = (torch.from_numpy(scaler.transform(raw_tune)),)
        replay_inputs = (torch.from_numpy(scaler_from_record(scaling).transform(raw_tune)),)
    else:
        model_args = {
            "vocabulary_size": len(config["pid_vocabulary"]),
            "embedding_width": config["pid_embedding_width"],
            "particle_widths": config["pfn_particle_hidden_sizes"],
            "head_widths": config["pfn_head_hidden_sizes"],
            "use_pid": model_id == "PFN-ID",
        }
        model_type = ParticleFlowNetwork

        def particles(split):
            return (
                torch.from_numpy(data[f"{split}_coordinates"]),
                torch.from_numpy(data[f"{split}_mask"]),
                torch.from_numpy(data[f"{split}_pid"].astype(np.int64)),
            )

        train, tune = particles("train"), particles("tune")
        replay_inputs = tune
    model = model_type(**model_args)
    scores, record = fit_neural(
        model,
        train,
        data["train_labels"],
        tune,
        data["tune_labels"],
        config,
        seed,
        config["schedule"]["neural_max_epochs"],
    )
    path = output / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = model_type(**model_args)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    np.testing.assert_array_equal(
        neural_scores(restored, replay_inputs, config["batch_size"]), scores
    )
    if model_id.startswith("PFN-"):
        extra["invariance"] = pfn_prediction_checks(restored, tune)
    return scores, {**record, **extra, "model_arguments": model_args}, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("model", choices=("XGB", "MLP", "PFN-K", "PFN-ID"))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    base = Path("results/task0003") / args.run_id
    progress = json.loads((base / "run.json").read_text())
    config = progress["config"]
    schedule = config["schedule"]
    if (
        progress["status"] != "running"
        or args.model not in schedule["models"]
        or args.seed not in schedule["seeds"]
    ):
        raise ValueError("Worker must belong to a declared active source run")
    artifacts = Path("artifacts/task0003") / args.run_id
    output = artifacts / f"{args.model}-{args.seed}"
    output.mkdir(exist_ok=False)
    with np.load(artifacts / "development.npz", allow_pickle=False) as data:
        if args.model == "XGB":
            scores, record, path = xgb_fit(data, config, args.seed, output)
        else:
            scores, record, path = neural_fit(data, config, args.model, args.seed, output)
    predictions = output / "tune_scores.npy"
    np.save(predictions, scores, allow_pickle=False)
    record.update(
        {
            "model": args.model,
            "seed": args.seed,
            "train_rows": schedule["train_rows"],
            "tune_rows": schedule["tune_rows"],
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "tune_scores": str(predictions),
            "tune_scores_sha256": sha256_file(predictions),
            "checkpoint_replay_exact": True,
        }
    )
    (output / "fit.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
