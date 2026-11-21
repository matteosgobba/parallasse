"""Shared baseline validation, metrics, and training-only scaling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from parallasse.features import FEATURE_NAMES


def feature_matrix(values: NDArray) -> NDArray:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not len(x):
        raise ValueError("Expected a nonempty matrix with the six declared features")
    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite baseline input")
    return x


def binary_labels(labels: NDArray, rows: int) -> NDArray:
    y = np.asarray(labels)
    if y.shape != (rows,) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Expected one binary quark=1/gluon=0 label per row")
    if np.unique(y).size != 2:
        raise ValueError("Both classes are required for this pilot")
    return y.astype(np.int64)


def classification_metrics(labels: NDArray, quark_scores: NDArray) -> dict:
    scores = np.asarray(quark_scores)
    y = binary_labels(labels, len(scores))
    if scores.shape != y.shape or not np.all(np.isfinite(scores)):
        raise ValueError("Expected one finite quark score per row")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Quark scores must be probabilities in [0,1]")
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "rows": len(y),
        "quarks": int(y.sum()),
        "gluons": int((1 - y).sum()),
    }


def fit_training_scaler(training_features: NDArray) -> StandardScaler:
    """No tuning input is accepted; fit only the caller's training rows."""
    return StandardScaler().fit(feature_matrix(training_features))


def scaler_record(scaler: StandardScaler) -> dict:
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "variance": scaler.var_.tolist(),
        "n_samples_seen": int(scaler.n_samples_seen_),
        "fit_cohort": "source_training_pilot_subset_only",
    }


def scaler_from_record(record: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(record["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(record["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(record["variance"], dtype=np.float64)
    scaler.n_features_in_ = len(FEATURE_NAMES)
    scaler.n_samples_seen_ = int(record["n_samples_seen"])
    return scaler
