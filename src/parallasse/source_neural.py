"""Reproducible CPU fitting and checkpoint checks for MLP and PFN."""

from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn

from parallasse.baselines import binary_labels, classification_metrics


class BestCheckpoint:
    """Highest tuning AUC; exact ties retain the first epoch and use patience."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.auc = -np.inf
        self.epoch = 0
        self.weights: dict | None = None

    def update(self, epoch: int, auc: float, model: nn.Module) -> bool:
        if auc > self.auc:
            self.auc, self.epoch = auc, epoch
            self.weights = copy.deepcopy(model.state_dict())
        return epoch - self.epoch >= self.patience


def weight_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def neural_scores(model: nn.Module, inputs: tuple[torch.Tensor, ...], batch: int) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return np.concatenate(
            [
                torch.sigmoid(model(*(item[start : start + batch] for item in inputs)))
                .cpu()
                .numpy()
                for start in range(0, len(inputs[0]), batch)
            ]
        )


def fit_neural(
    model: nn.Module,
    train: tuple[torch.Tensor, ...],
    labels: np.ndarray,
    tune: tuple[torch.Tensor, ...],
    tune_labels: np.ndarray,
    config: dict,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, dict]:
    train_y = torch.from_numpy(binary_labels(labels, len(train[0])).astype(np.float32))
    tune_y = binary_labels(tune_labels, len(tune[0]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["neural_learning_rate"],
        weight_decay=config["neural_weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order_hash = hashlib.sha256()
    initial_weights = weight_digest(model)
    selector = BestCheckpoint(config["neural_patience"])
    history = []
    batch = config["batch_size"]
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_y), generator=generator)
        order_bytes = order.numpy().astype("<i8", copy=False).tobytes()
        order_hash.update(order_bytes)
        loss_sum = torch.zeros((), dtype=torch.float64)
        for rows in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*(item[rows] for item in train)), train_y[rows])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite neural training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().double() * len(rows)
        metrics = classification_metrics(tune_y, neural_scores(model, tune, batch))
        history.append(
            {
                "epoch": epoch,
                "train_bce": float(loss_sum) / len(train_y),
                "tune": metrics,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_order_sha256": hashlib.sha256(order_bytes).hexdigest(),
            }
        )
        print(f"epoch {epoch:02d}: tuning AUC {metrics['roc_auc']:.6f}", flush=True)
        if selector.update(epoch, metrics["roc_auc"], model):
            break
    if selector.weights is None:
        raise ValueError("No neural checkpoint selected")
    model.load_state_dict(selector.weights)
    scores = neural_scores(model, tune, batch)
    return scores, {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_weights_sha256": initial_weights,
        "batch_order_sha256": order_hash.hexdigest(),
        "best_epoch": selector.epoch,
        "epochs_trained": len(history),
        "fit_seconds_including_tune_checks": time.perf_counter() - started,
        "history": history,
        "tune": classification_metrics(tune_y, scores),
    }


def pfn_prediction_checks(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict:
    coordinates, mask, tokens = (item[:64] for item in inputs)
    original = neural_scores(model, (coordinates, mask, tokens), 64)
    permutation = torch.randperm(mask.shape[1], generator=torch.Generator().manual_seed(901))
    permuted = neural_scores(
        model, (coordinates[:, permutation], mask[:, permutation], tokens[:, permutation]), 64
    )
    n, width = mask.shape
    # Include nonzero/invalid sentinels at padded positions to catch encoder leakage.
    extended_x = torch.cat((coordinates, torch.full((n, 7, 3), float("nan"))), dim=1)
    extended_mask = torch.cat((mask, torch.zeros((n, 7), dtype=torch.bool)), dim=1)
    extended_tokens = torch.cat((tokens, torch.full((n, 7), -991, dtype=torch.long)), dim=1)
    padded = neural_scores(model, (extended_x, extended_mask, extended_tokens), 64)
    np.testing.assert_allclose(original, permuted, rtol=1e-5, atol=2e-6)
    np.testing.assert_array_equal(original, padded)
    return {
        "jets_checked": n,
        "original_width": width,
        "permutation_max_abs_probability_delta": float(np.max(np.abs(original - permuted))),
        "extra_padding_max_abs_probability_delta": float(np.max(np.abs(original - padded))),
        "permutation_tolerances": {"rtol": 1e-5, "atol": 2e-6},
    }
