"""Paired five-family summaries for the EFN extension."""

from __future__ import annotations

import numpy as np

from parallasse.heldout_metrics import METRICS, BinaryMetrics
from parallasse.heldout_metrics import MODELS as BASE_MODELS
from parallasse.mixture_uncertainty import inference_summary

MODELS = (*BASE_MODELS, "EFN")
KEYS = tuple(f"{m}-{s}" for m in MODELS for s in (17, 29, 43))


def mixture_rows(mode, fraction, size, accepted, eq, eg, nq, ng):
    rows = []
    for key, metrics in zip(
        KEYS, inference_summary(accepted, size, fraction, eq, eg, nq, ng), strict=True
    ):
        model, seed = key.rsplit("-", 1)
        rows.append(
            {
                "mode": mode,
                "fraction": fraction,
                "sample_size": size,
                "model": model,
                "seed": int(seed),
                **metrics,
            }
        )
    return rows


def classification(labels, scores, cuts, rng, replicates):
    """Reuse the validated rank engine, pairing every model and seed on each draw."""
    if (
        labels.ndim != 1
        or scores.shape != (len(labels), len(KEYS))
        or set(cuts) != set(KEYS)
        or not np.all(np.isin(labels, [0, 1]))
    ):
        raise ValueError("Classification rows, checkpoint keys and binary labels must align")
    q, g = labels == 1, labels == 0
    nq, ng = int(q.sum()), int(g.sum())
    if not nq or not ng:
        raise ValueError("Both classes are required")
    engines = [BinaryMetrics(scores[q, i], scores[g, i], cuts[k]) for i, k in enumerate(KEYS)]
    onesq, onesg = np.ones(nq, dtype=np.int64), np.ones(ng, dtype=np.int64)
    rows = []
    for key, engine in zip(KEYS, engines, strict=True):
        auc, tq, fg = engine.weighted(onesq, onesg)
        model, seed = key.rsplit("-", 1)
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "threshold": cuts[key],
                "auc": float(auc),
                "accepted_quarks": tq,
                "total_quarks": nq,
                "tpr": tq / nq,
                "accepted_gluons": fg,
                "total_gluons": ng,
                "fpr": fg / ng,
            }
        )
    draws = np.empty((replicates, len(KEYS), len(METRICS)), dtype=float)
    for b in range(replicates):
        wq = np.bincount(rng.integers(nq, size=nq), minlength=nq)
        wg = np.bincount(rng.integers(ng, size=ng), minlength=ng)
        for i, engine in enumerate(engines):
            auc, tq, fg = engine.weighted(wq, wg)
            draws[b, i] = auc, tq / nq, fg / ng
    return rows, draws


def distribution_summary(point, draws):
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(np.mean(point)),
        "seed_min": float(np.min(point)),
        "seed_max": float(np.max(point)),
        "bootstrap_95_lo": float(lo),
        "bootstrap_95_hi": float(hi),
    }


def classification_summary(rows, draws):
    points = np.array([[r[k] for k in METRICS] for r in rows]).reshape(5, 3, 3)
    samples = draws.reshape(-1, 5, 3, 3).mean(axis=2)
    models = {
        m: {
            name: distribution_summary(points[i, :, j], samples[:, i, j])
            for j, name in enumerate(METRICS)
        }
        for i, m in enumerate(MODELS)
    }
    contrast = distribution_summary(
        points[4, :, 0] - points[2, :, 0], samples[:, 4, 0] - samples[:, 2, 0]
    )
    return {"models": models, "EFN_minus_PFN-K_auc": contrast}


def bias_summary(point, draws):
    p = np.asarray(point).reshape(5, 3)
    d = np.asarray(draws).reshape(-1, 5, 3)
    models = {
        m: {
            "signed_bias": distribution_summary(p[i], d[:, i].mean(axis=1)),
            "absolute_bias": distribution_summary(np.abs(p[i]), np.abs(d[:, i]).mean(axis=1)),
        }
        for i, m in enumerate(MODELS)
    }
    contrasts = {}
    for name, index in (("PFN-K", 2), ("PFN-ID", 3), ("XGB", 0)):
        contrasts[f"EFN_minus_{name}"] = distribution_summary(
            np.abs(p[4]) - np.abs(p[index]), (np.abs(d[:, 4]) - np.abs(d[:, index])).mean(axis=1)
        )
    return {"models": models, "contrasts": contrasts}
