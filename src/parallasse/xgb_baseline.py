"""XGBoost baseline; execute separately from PyTorch on macOS."""

from __future__ import annotations

import time

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from parallasse.baselines import binary_labels, classification_metrics, feature_matrix


def xgb_quark_scores(model: xgb.XGBClassifier, features: NDArray) -> NDArray:
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Unexpected XGBoost class order")
    return model.predict_proba(
        feature_matrix(features), iteration_range=(0, model.best_iteration + 1)
    )[:, 1]


def fit_xgb_pilot(
    train_x: NDArray,
    train_y: NDArray,
    tune_x: NDArray,
    tune_y: NDArray,
    config: dict,
    max_depth: int,
    seed: int,
    threads: int,
) -> tuple:
    x, t = feature_matrix(train_x), feature_matrix(tune_x)
    y, v = binary_labels(train_y, len(x)), binary_labels(tune_y, len(t))
    params = {key: value for key, value in config.items() if key != "max_depth_candidates"}
    params.update(max_depth=max_depth, random_state=seed, n_jobs=threads, device="cpu")
    model = xgb.XGBClassifier(**params)
    started = time.perf_counter()
    model.fit(x, y, eval_set=[(t, v)], verbose=False)
    fit_seconds = time.perf_counter() - started
    scores = xgb_quark_scores(model, t)
    history = model.evals_result()["validation_0"]["auc"]
    return (
        model,
        scores,
        {
            "model": "XGB",
            "max_depth": max_depth,
            "parameters": params,
            "best_iteration_zero_based": int(model.best_iteration),
            "trees_used": int(model.best_iteration + 1),
            "rounds_trained": len(history),
            "tune_auc_history": [float(value) for value in history],
            "fit_seconds": fit_seconds,
            "tune": classification_metrics(v, scores),
        },
    )
