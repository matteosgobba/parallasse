"""Internal inference-only worker; Torch and XGBoost never share a process."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from parallasse.download import sha256_file


def xgb_scores(frozen: dict, inputs) -> dict:
    import xgboost as xgb

    from parallasse.xgb_baseline import xgb_quark_scores

    if "torch" in sys.modules:
        raise RuntimeError("XGB worker must not load Torch")
    aggregate = inputs["aggregate"]
    scores = {}
    for item in frozen["models"]:
        if item["model"] != "XGB":
            continue
        model = xgb.XGBClassifier(n_jobs=frozen["inference"]["cpu_threads"], device="cpu")
        model.load_model(item["checkpoint"])
        if model.best_iteration + 1 != item["trees_used"]:
            raise ValueError("Selected tree range changed")
        scores[f"XGB-{item['seed']}"] = xgb_quark_scores(model, aggregate)
    return scores


def neural_scores_for_cohort(frozen: dict, inputs) -> dict:
    import torch

    from parallasse.baselines import scaler_from_record
    from parallasse.mlp_baseline import AggregateMLP
    from parallasse.pfn import ParticleFlowNetwork
    from parallasse.source_neural import neural_scores

    if "xgboost" in sys.modules:
        raise RuntimeError("Neural worker must not load XGBoost")
    torch.set_num_threads(frozen["inference"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    particles = (
        torch.from_numpy(inputs["coordinates"]),
        torch.from_numpy(inputs["mask"]),
        torch.from_numpy(inputs["pid"].astype(np.int64)),
    )
    aggregate = inputs["aggregate"]
    scores = {}
    for item in frozen["models"]:
        name = item["model"]
        if name == "XGB":
            continue
        if name == "MLP":
            model = AggregateMLP(**item["model_arguments"])
            scaler = scaler_from_record(item["scaler"])
            values = (torch.from_numpy(scaler.transform(aggregate)),)
        else:
            model = ParticleFlowNetwork(**item["model_arguments"])
            values = particles
        model.load_state_dict(torch.load(item["checkpoint"], map_location="cpu", weights_only=True))
        scores[f"{name}-{item['seed']}"] = neural_scores(
            model, values, frozen["inference"]["batch_size"]
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("xgb", "neural"))
    parser.add_argument("freeze", type=Path)
    parser.add_argument("freeze_sha256")
    parser.add_argument("inputs", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if sha256_file(args.freeze) != args.freeze_sha256:
        raise ValueError("Model freeze changed during inference")
    frozen = json.loads(args.freeze.read_text())
    with np.load(args.inputs, allow_pickle=False) as inputs:
        scorer = xgb_scores if args.family == "xgb" else neural_scores_for_cohort
        scores = scorer(frozen, inputs)
    with args.destination.open("xb") as handle:
        np.savez(handle, **scores)


if __name__ == "__main__":
    main()
