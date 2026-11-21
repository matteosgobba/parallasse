"""Compact PyTorch baseline; execute separately from XGBoost on macOS."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import nn

from parallasse.baselines import (
    binary_labels,
    classification_metrics,
    feature_matrix,
    fit_training_scaler,
    scaler_record,
)
from parallasse.features import FEATURE_NAMES


class AggregateMLP(nn.Module):
    def __init__(self, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        size = len(FEATURE_NAMES)
        for width in hidden_sizes:
            layers.extend((nn.Linear(size, width), nn.ReLU()))
            size = width
        layers.append(nn.Linear(size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def mlp_quark_scores(model: AggregateMLP, scaler: StandardScaler, features: NDArray) -> NDArray:
    scaled = scaler.transform(feature_matrix(features)).astype(np.float32)
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(scaled))).numpy().copy()


def fit_mlp_pilot(
    train_x: NDArray,
    train_y: NDArray,
    tune_x: NDArray,
    tune_y: NDArray,
    config: dict,
    seed: int,
    threads: int,
) -> tuple:
    if (
        config.get("activation", "relu") != "relu"
        or config.get("optimizer", "Adam") != "Adam"
        or config.get("loss", "BCEWithLogitsLoss") != "BCEWithLogitsLoss"
    ):
        raise ValueError("This declared baseline uses ReLU, Adam, and BCEWithLogitsLoss")
    x, t = feature_matrix(train_x), feature_matrix(tune_x)
    y, v = binary_labels(train_y, len(x)), binary_labels(tune_y, len(t))
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    scaler = fit_training_scaler(x)
    tensor_x = torch.from_numpy(scaler.transform(x).astype(np.float32))
    tensor_y = torch.from_numpy(y.astype(np.float32))
    model = AggregateMLP(config["hidden_sizes"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()
    shuffle = torch.Generator(device="cpu").manual_seed(seed)
    history, best_state, best_auc, best_epoch = [], None, -np.inf, 0
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for batch in torch.randperm(len(x), generator=shuffle).split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tensor_x[batch]), tensor_y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite MLP training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        scores = mlp_quark_scores(model, scaler, t)
        metrics = classification_metrics(v, scores)
        history.append({"epoch": epoch, "train_bce": loss_sum / len(x), "tune": metrics})
        if metrics["roc_auc"] > best_auc:
            best_state = copy.deepcopy(model.state_dict())
            best_auc, best_epoch = metrics["roc_auc"], epoch
    if best_state is None:
        raise ValueError("No MLP checkpoint was selected")
    model.load_state_dict(best_state)
    scores = mlp_quark_scores(model, scaler, t)
    return (
        model,
        scaler,
        scores,
        {
            "model": "MLP",
            "parameters": config,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "seed": seed,
            "cpu_threads": threads,
            "best_epoch": best_epoch,
            "epochs_trained": len(history),
            "history": history,
            "fit_seconds_including_tune_checks": time.perf_counter() - started,
            "scaler": scaler_record(scaler),
            "tune": classification_metrics(v, scores),
        },
    )
