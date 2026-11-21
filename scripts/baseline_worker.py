"""One model-family process: native XGB and Torch runtimes never coexist."""

import argparse
import json
from pathlib import Path

import numpy as np

from parallasse.download import sha256_file


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["xgb", "mlp"], required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / "results/task0002" / args.run_id
    artifacts = root / "artifacts/task0002" / args.run_id
    config = json.loads((output / "config.json").read_text())
    with np.load(artifacts / "inputs.npz", allow_pickle=False) as inputs:
        x, y = inputs["train_features"], inputs["train_labels"]
        t, v = inputs["tune_features"], inputs["tune_labels"]
    if args.family == "xgb":
        import xgboost as xgb

        from parallasse.xgb_baseline import fit_xgb_pilot, xgb_quark_scores

        records, predictions = [], {}
        for depth in config["xgb"]["max_depth_candidates"]:
            model, scores, record = fit_xgb_pilot(
                x, y, t, v, config["xgb"], depth, config["seed"], config["cpu_threads"]
            )
            name = f"xgb_depth{depth}"
            checkpoint = artifacts / f"{name}.ubj"
            model.save_model(checkpoint)
            restored = xgb.XGBClassifier(n_jobs=config["cpu_threads"], device="cpu")
            restored.load_model(checkpoint)
            np.testing.assert_array_equal(xgb_quark_scores(restored, t), scores)
            record.update(
                variant=name,
                checkpoint=str(checkpoint.relative_to(root)),
                checkpoint_sha256=sha256_file(checkpoint),
                checkpoint_replay_exact=True,
            )
            records.append(record)
            predictions[name] = scores
            write_json(output / "xgb_models.json", {"models": records})
            print(
                json.dumps(
                    {"variant": name, "tune": record["tune"], "fit_seconds": record["fit_seconds"]}
                ),
                flush=True,
            )
        np.savez_compressed(artifacts / "xgb_tune_scores.npz", **predictions)
    else:
        import torch

        from parallasse.baselines import scaler_from_record
        from parallasse.mlp_baseline import AggregateMLP, fit_mlp_pilot, mlp_quark_scores

        model, scaler, scores, record = fit_mlp_pilot(
            x, y, t, v, config["mlp"], config["seed"], config["cpu_threads"]
        )
        checkpoint = artifacts / "mlp_weights.pt"
        torch.save(model.state_dict(), checkpoint)
        scaler_path = artifacts / "mlp_scaler.json"
        write_json(scaler_path, record["scaler"])
        restored = AggregateMLP(config["mlp"]["hidden_sizes"])
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        restored_scaler = scaler_from_record(json.loads(scaler_path.read_text()))
        np.testing.assert_array_equal(mlp_quark_scores(restored, restored_scaler, t), scores)
        record.update(
            variant="mlp",
            checkpoint=str(checkpoint.relative_to(root)),
            checkpoint_sha256=sha256_file(checkpoint),
            checkpoint_replay_exact=True,
            scaler_path=str(scaler_path.relative_to(root)),
            scaler_sha256=sha256_file(scaler_path),
        )
        write_json(output / "mlp_model.json", {"models": [record]})
        np.savez_compressed(artifacts / "mlp_tune_scores.npz", mlp=scores)
        print(
            json.dumps(
                {
                    "variant": "mlp",
                    "tune": record["tune"],
                    "parameter_count": record["parameter_count"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
