"""Single-cut mixture inference and paired sampling from finite class pools."""

from __future__ import annotations

import numpy as np


def infer_fraction(acceptance, quark_efficiency, gluon_efficiency) -> np.ndarray:
    """Invert r = f * e_q + (1-f) * e_g, retaining estimates outside [0, 1]."""
    r, eq, eg = np.broadcast_arrays(
        np.asarray(acceptance, dtype=np.float64),
        np.asarray(quark_efficiency, dtype=np.float64),
        np.asarray(gluon_efficiency, dtype=np.float64),
    )
    for value in (r, eq, eg):
        if not np.all(np.isfinite(value)) or np.any((value < 0) | (value > 1)):
            raise ValueError("Acceptance and efficiencies must be finite probabilities")
    difference = eq - eg
    if np.any(np.abs(difference) <= 1e-12):
        raise ValueError("Class efficiencies do not identify a mixture fraction")
    return (r - eg) / difference


def source_partition(rows: np.ndarray, labels: np.ndarray, seed: int) -> tuple:
    """Split each source class equally, returning disjoint archive indices."""
    rows, labels = np.asarray(rows), np.asarray(labels)
    if (
        rows.ndim != 1
        or not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
        or len(np.unique(rows)) != len(rows)
        or labels.shape != rows.shape
        or not np.all(np.isin(labels, [0, 1]))
    ):
        raise ValueError("Unique nonnegative row IDs must align with binary labels")
    rng = np.random.Generator(np.random.PCG64(seed))
    response, population = [], []
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        half = len(indices) // 2
        if half < 1:
            raise ValueError("Each class needs at least two rows")
        response.append(indices[:half])
        population.append(indices[half:])
    return np.sort(np.concatenate(response)), np.sort(np.concatenate(population))


def mixture_counts(
    quark_pass: np.ndarray,
    gluon_pass: np.ndarray,
    fraction: float,
    sample_size: int,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """IID mixtures with replacement, sharing sampled rows across all model columns."""
    q, g = np.asarray(quark_pass), np.asarray(gluon_pass)
    if (
        q.ndim != 2
        or g.ndim != 2
        or not len(q)
        or not len(g)
        or q.shape[1] != g.shape[1]
        or not q.shape[1]
        or q.dtype != np.bool_
        or g.dtype != np.bool_
    ):
        raise ValueError("Both classes need nonempty, aligned boolean pass matrices")
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Mixture fraction must be in [0, 1]")
    for value in (sample_size, replicates):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Sample size and replicate count must be positive integers")
    quark_counts = rng.binomial(sample_size, fraction, size=replicates)
    accepted = np.empty((replicates, q.shape[1]), dtype=np.int64)
    for i, nq in enumerate(quark_counts):
        # Sampling a row selects the same jet for every model and training seed.
        q_indices = rng.integers(len(q), size=nq)
        g_indices = rng.integers(len(g), size=sample_size - nq)
        accepted[i] = q[q_indices].sum(axis=0) + g[g_indices].sum(axis=0)
    return accepted, quark_counts


def closure_summary(
    accepted: np.ndarray,
    sample_size: int,
    fraction: float,
    response_q: np.ndarray,
    response_g: np.ndarray,
    population_q: np.ndarray,
    population_g: np.ndarray,
) -> list[dict]:
    """Compare pseudoexperiment means with the exact fixed-pool expectation."""
    estimates = infer_fraction(accepted / sample_size, response_q, response_g)
    r = fraction * population_q + (1 - fraction) * population_g
    expected = infer_fraction(r, response_q, response_g)
    variance = r * (1 - r) / (sample_size * (response_q - response_g) ** 2)
    mcse = np.sqrt(variance / len(accepted))
    result = []
    for i in range(estimates.shape[1]):
        values = estimates[:, i]
        result.append(
            {
                "mean_fraction": float(values.mean()),
                "expected_fraction_fixed_pools": float(expected[i]),
                "bias": float(values.mean() - fraction),
                "expected_bias_fixed_pools": float(expected[i] - fraction),
                "standard_deviation": float(values.std(ddof=1)),
                "rmse": float(np.sqrt(np.mean((values - fraction) ** 2))),
                "mean_mcse_fixed_pools": float(mcse[i]),
                "mean_minus_expectation": float(values.mean() - expected[i]),
                "outside_unit_interval_rate": float(np.mean((values < 0) | (values > 1))),
            }
        )
    return result
