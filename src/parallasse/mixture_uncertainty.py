"""Delta intervals and paired bootstrap for frozen-cut mixture inference."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from parallasse.heldout_metrics import MODELS
from parallasse.mixtures import infer_fraction

Z95 = NormalDist().inv_cdf(0.975)


def fraction_interval(accepted, sample_size: int, eq, eg, nq: int, ng: int) -> tuple:
    """First-order propagation of three independent binomial measurements."""
    for count in (sample_size, nq, ng):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Sample and response sizes must be positive integers")
    counts = np.asarray(accepted)
    if not np.issubdtype(counts.dtype, np.integer) or np.any((counts < 0) | (counts > sample_size)):
        raise ValueError("Accepted counts must be integers between zero and sample size")
    r = counts / sample_size
    eq, eg = np.asarray(eq), np.asarray(eg)
    estimate = infer_fraction(r, eq, eg)
    variance = (
        r * (1 - r) / sample_size
        + estimate**2 * eq * (1 - eq) / nq
        + (1 - estimate) ** 2 * eg * (1 - eg) / ng
    ) / (eq - eg) ** 2
    se = np.sqrt(variance)
    return estimate, estimate - Z95 * se, estimate + Z95 * se, se


def inference_summary(accepted, sample_size: int, fraction: float, eq, eg, nq: int, ng: int):
    estimates, lower, upper, se = fraction_interval(accepted, sample_size, eq, eg, nq, ng)
    if estimates.ndim != 2 or len(estimates) < 2 or not 0 <= fraction <= 1:
        raise ValueError("Summary requires repeated model columns and a fraction in [0, 1]")
    hits = (lower <= fraction) & (fraction <= upper)
    coverage = hits.mean(axis=0)
    trials = len(estimates)
    # Wilson intervals express Monte Carlo precision even for zero coverage hits.
    scale = 1 + Z95**2 / trials
    center = (coverage + Z95**2 / (2 * trials)) / scale
    half = Z95 / scale * np.sqrt(coverage * (1 - coverage) / trials + Z95**2 / (4 * trials**2))
    rows = []
    for i in range(estimates.shape[1]):
        errors = estimates[:, i] - fraction
        rows.append(
            {
                "mean_fraction": float(estimates[:, i].mean()),
                "bias": float(errors.mean()),
                "standard_deviation": float(estimates[:, i].std(ddof=1)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_mcse": float(estimates[:, i].std(ddof=1) / np.sqrt(trials)),
                "coverage_95": float(coverage[i]),
                "coverage_hits": int(hits[:, i].sum()),
                "coverage_trials": trials,
                "coverage_mc_95_lo": float(max(0, center[i] - half[i])),
                "coverage_mc_95_hi": float(min(1, center[i] + half[i])),
                "mean_interval_width": float((upper[:, i] - lower[:, i]).mean()),
                "mean_standard_error": float(se[:, i].mean()),
                "outside_unit_interval_rate": float(
                    np.mean((estimates[:, i] < 0) | (estimates[:, i] > 1))
                ),
            }
        )
    return rows


def bootstrap_acceptance(passed, sample_size: int, replicates: int, rng) -> np.ndarray:
    """Resample class rows once per draw and share indices across all columns."""
    values = np.asarray(passed)
    if values.ndim != 2 or not values.size or values.dtype != np.bool_:
        raise ValueError("A nonempty boolean pass matrix is required")
    for count in (sample_size, replicates):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Bootstrap sizes must be positive integers")
    counts = np.empty((replicates, values.shape[1]), dtype=np.int64)
    for i in range(replicates):
        counts[i] = values[rng.integers(len(values), size=sample_size)].sum(axis=0)
    return counts


def expected_bias(fraction, eq, eg, population_q, population_g) -> np.ndarray:
    """Exact conditional bias of the linear accepted-count inverse."""
    return (
        infer_fraction(fraction * population_q + (1 - fraction) * population_g, eq, eg) - fraction
    )


def bias_contrasts(point: np.ndarray, bootstrap: np.ndarray) -> dict:
    """Absolute bias is formed per seed before averaging the three seeds."""
    point = np.asarray(point).reshape(len(MODELS), 3)
    samples = np.asarray(bootstrap).reshape(-1, len(MODELS), 3)
    models = {}
    for i, model in enumerate(MODELS):
        absolute = np.abs(samples[:, i]).mean(axis=1)
        lo, hi = np.quantile(absolute, [0.025, 0.975])
        models[model] = {
            "mean_signed_bias": float(point[i].mean()),
            "mean_absolute_bias": float(np.abs(point[i]).mean()),
            "seed_bias_min": float(point[i].min()),
            "seed_bias_max": float(point[i].max()),
            "absolute_bias_bootstrap_95_lo": float(lo),
            "absolute_bias_bootstrap_95_hi": float(hi),
        }
    contrasts = {}
    for name, a, b in (("PFN-ID_minus_PFN-K", 3, 2), ("PFN-K_minus_XGB", 2, 0)):
        pair = np.abs(point[a]) - np.abs(point[b])
        draws = (np.abs(samples[:, a]) - np.abs(samples[:, b])).mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        contrasts[name] = {
            "mean": float(pair.mean()),
            "seed_min": float(pair.min()),
            "seed_max": float(pair.max()),
            "bootstrap_95_lo": float(lo),
            "bootstrap_95_hi": float(hi),
        }
    return {"models": models, "contrasts": contrasts}
