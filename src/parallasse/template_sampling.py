"""Checkpoint-paired row sampling for normalized template experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class ClassPool:
    rows: np.ndarray
    bins: np.ndarray
    passed: np.ndarray

    def __post_init__(self):
        if (
            self.rows.ndim != 1
            or not np.issubdtype(self.rows.dtype, np.integer)
            or np.any(self.rows < 0)
            or not len(self.rows)
            or self.bins.shape != self.passed.shape
            or self.bins.shape != (len(self.rows), 6)
            or self.passed.dtype != np.bool_
            or not np.issubdtype(self.bins.dtype, np.integer)
            or np.any((self.bins < 0) | (self.bins > 7))
            or len(np.unique(self.rows)) != len(self.rows)
        ):
            raise ValueError("Unique original rows and six aligned histogram/pass columns required")


def class_pools(scores, labels, rows, edges, cuts):
    scores, labels, rows = np.asarray(scores), np.asarray(labels), np.asarray(rows)
    if (
        scores.shape != (len(rows), 6)
        or labels.shape != rows.shape
        or edges.shape != (6, 9)
        or np.any(~np.isfinite(scores))
        or np.any((scores < 0) | (scores > 1))
        or not np.all(np.isin(labels, [0, 1]))
        or np.any(np.diff(edges, axis=1) <= 0)
    ):
        raise ValueError("Scores, binary labels, frozen source bins and rows must align")
    bins = np.column_stack(
        [
            np.minimum(7, np.searchsorted(edges[i, 1:-1], scores[:, i], side="right"))
            for i in range(6)
        ]
    ).astype(np.int8)
    passed = scores > np.asarray(cuts)
    return [
        ClassPool(rows[labels == label], bins[labels == label], passed[labels == label])
        for label in (1, 0)
    ]


def histograms(pools):
    return np.array(
        [[np.bincount(pool.bins[:, i], minlength=8) for pool in pools] for i in range(6)],
        dtype=np.int64,
    )


def draw_experiment(pools, fraction, size, rng, alpha=None, repeated=True, population=None):
    """Draw all class measurements independently, then one IID shared mixture.

    Only histogram sufficient statistics and a row-draw fingerprint need be saved;
    RNG cell state reconstructs exact rows without tens of gigabytes of indices.
    """
    laws = pools if population is None else population
    if (
        len(pools) not in (2, 4)
        or len(laws) not in (2, 4)
        or (len(laws) == 4) != (alpha is not None)
    ):
        raise ValueError("Class-pool count and simulation variation disagree")
    if not 0 <= fraction <= 1 or (alpha is not None and not 0 <= alpha <= 1) or size <= 0:
        raise ValueError("Valid mixture probabilities and positive size required")
    digest = hashlib.sha256(b"parallasse-template-draw-v1\0")
    auxiliary = np.zeros((6, len(pools), 8), dtype=np.int64)
    response_pass = np.zeros((6, len(pools)), dtype=np.int64)
    for j, pool in enumerate(pools):
        indices = (
            rng.integers(len(pool.rows), size=len(pool.rows))
            if repeated
            else np.arange(len(pool.rows))
        )
        digest.update(np.array([j, len(indices)], dtype="<i8").tobytes())
        digest.update(pool.rows[indices].astype("<i8").tobytes())
        for i in range(6):
            auxiliary[i, j] = np.bincount(pool.bins[indices, i], minlength=8)
        response_pass[:, j] = pool.passed[indices].sum(axis=0)
    if len(laws) not in (2, 4):
        raise ValueError("Population must contain both classes, optionally both generators")
    quark = rng.random(size) < fraction
    alternative = rng.random(size) < alpha if alpha is not None else np.zeros(size, dtype=bool)
    memberships = (1 - quark.astype(np.int8)) + 2 * alternative
    if np.max(memberships) >= len(laws):
        raise ValueError("Mixture asks for an unavailable generator law")
    counts = np.bincount(memberships, minlength=len(laws))
    observed = np.zeros((6, 8), dtype=np.int64)
    accepted = np.zeros(6, dtype=np.int64)
    for j, pool in enumerate(laws):
        indices = rng.integers(len(pool.rows), size=int(counts[j]))
        digest.update(np.array([100 + j, len(indices)], dtype="<i8").tobytes())
        digest.update(pool.rows[indices].astype("<i8").tobytes())
        for i in range(6):
            observed[i] += np.bincount(pool.bins[indices, i], minlength=8)
        accepted += pool.passed[indices].sum(axis=0)
    return {
        "auxiliary": auxiliary,
        "response_pass": response_pass,
        "observed": observed,
        "accepted": accepted,
        "class_sizes": np.array([len(p.rows) for p in pools]),
        "membership_counts": counts,
        "row_draw_sha256": digest.hexdigest(),
    }
