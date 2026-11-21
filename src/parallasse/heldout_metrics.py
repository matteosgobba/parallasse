"""Fixed-cut metrics and paired, class-stratified conditional bootstrap."""

from __future__ import annotations

import numpy as np

METRICS = ("auc", "tpr", "fpr")
MODELS = ("XGB", "MLP", "PFN-K", "PFN-ID")
KEYS = tuple(f"{name}-{seed}" for name in MODELS for seed in (17, 29, 43))


def checked_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("Expected a nonempty vector of finite scores")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected quark probabilities in [0, 1]")
    return scores


def calibration_cut(gluon_scores: np.ndarray) -> dict:
    """Freeze an order statistic for a strict, tie-conservative 10% cut."""
    scores = checked_scores(gluon_scores)
    n = len(scores)
    budget = n // 10
    position = n - budget - 1
    threshold = float(np.partition(scores, position)[position])
    accepted = int(np.count_nonzero(scores > threshold))
    if accepted > budget:
        raise ValueError("Calibration cut exceeded the fixed acceptance budget")
    return {
        "threshold": threshold,
        "gluon_count": n,
        "acceptance_budget": budget,
        "order_statistic_index": position,
        "accepted_gluons": accepted,
        "tied_at_threshold": int(np.count_nonzero(scores == threshold)),
        "fpr": accepted / n,
    }


class BinaryMetrics:
    """Cache score ranks so event resampling does not sort each replicate."""

    def __init__(self, quarks: np.ndarray, gluons: np.ndarray, threshold: float) -> None:
        quarks, gluons = checked_scores(quarks), checked_scores(gluons)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid frozen threshold")
        self.order = np.argsort(gluons)
        sorted_gluons = gluons[self.order]
        self.left = np.searchsorted(sorted_gluons, quarks, side="left")
        self.right = np.searchsorted(sorted_gluons, quarks, side="right")
        self.quark_accept = quarks > threshold
        self.gluon_accept = gluons > threshold
        self.nq, self.ng = len(quarks), len(gluons)
        self.prefix = np.empty(self.ng + 1, dtype=np.int64)
        self.prefix[0] = 0

    def weighted(self, quark_weight: np.ndarray, gluon_weight: np.ndarray) -> tuple:
        """Weights are integer counts from class-size-preserving draws."""
        np.cumsum(gluon_weight[self.order], out=self.prefix[1:])
        # Left and right weighted ranks differ only on ties: credit half a tie.
        twice_wins = quark_weight @ (self.prefix[self.left] + self.prefix[self.right])
        nq_accepted = int(quark_weight @ self.quark_accept)
        ng_accepted = int(gluon_weight @ self.gluon_accept)
        return twice_wins / (2 * self.nq * self.ng), nq_accepted, ng_accepted


def cohort_metrics(
    labels: np.ndarray, scores: np.ndarray, cuts: dict, rng: np.random.Generator, replicates: int
) -> tuple[list[dict], np.ndarray]:
    if (
        labels.ndim != 1
        or not np.all(np.isin(labels, [0, 1]))
        or scores.shape != (len(labels), len(KEYS))
        or set(cuts) != set(KEYS)
    ):
        raise ValueError("Scores, binary labels, and frozen model keys must align")
    quarks, gluons = labels == 1, labels == 0
    nq, ng = int(quarks.sum()), int(gluons.sum())
    if not nq or not ng:
        raise ValueError("Both classes must be present")
    engines = [
        BinaryMetrics(scores[quarks, i], scores[gluons, i], cuts[key]) for i, key in enumerate(KEYS)
    ]
    ones_q, ones_g = np.ones(nq, dtype=np.int64), np.ones(ng, dtype=np.int64)
    rows = []
    for key, engine in zip(KEYS, engines, strict=True):
        auc, q, g = engine.weighted(ones_q, ones_g)
        name, seed = key.rsplit("-", 1)
        rows.append(
            {
                "model": name,
                "seed": int(seed),
                "threshold": cuts[key],
                "auc": float(auc),
                "accepted_quarks": q,
                "total_quarks": nq,
                "tpr": q / nq,
                "accepted_gluons": g,
                "total_gluons": ng,
                "fpr": g / ng,
            }
        )
    draws = np.empty((replicates, len(KEYS), len(METRICS)), dtype=np.float64)
    for replicate in range(replicates):
        # A single pair of draws is reused for all checkpoints in this cohort.
        wq = np.bincount(rng.integers(nq, size=nq), minlength=nq)
        wg = np.bincount(rng.integers(ng, size=ng), minlength=ng)
        for index, engine in enumerate(engines):
            auc, q, g = engine.weighted(wq, wg)
            draws[replicate, index] = auc, q / nq, g / ng
    return rows, draws


def summary(seeds: np.ndarray, draws: np.ndarray) -> dict:
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(seeds.mean()),
        "seed_min": float(seeds.min()),
        "seed_max": float(seeds.max()),
        "bootstrap_95_lo": float(lo),
        "bootstrap_95_hi": float(hi),
    }


def summarize(per_seed: dict, draws: dict) -> dict:
    seed_metrics, mean_draws, cohorts = {}, {}, {}
    for cohort in ("test", "target"):
        # Each point is a metric per model/seed, never an averaged score array.
        seed_metrics[cohort] = np.array([[row[m] for m in METRICS] for row in per_seed[cohort]])
        seed_metrics[cohort] = seed_metrics[cohort].reshape(len(MODELS), 3, len(METRICS))
        mean_draws[cohort] = draws[cohort].reshape(-1, len(MODELS), 3, len(METRICS)).mean(axis=2)
        cohorts[cohort] = {
            name: {
                metric: summary(seed_metrics[cohort][i, :, j], mean_draws[cohort][:, i, j])
                for j, metric in enumerate(METRICS)
            }
            for i, name in enumerate(MODELS)
        }
    auc_gap = {
        name: summary(
            seed_metrics["test"][i, :, 0] - seed_metrics["target"][i, :, 0],
            mean_draws["test"][:, i, 0] - mean_draws["target"][:, i, 0],
        )
        for i, name in enumerate(MODELS)
    }
    contrasts = {}
    for cohort in ("test", "target"):
        points, samples = seed_metrics[cohort], mean_draws[cohort]
        contrasts[cohort] = {
            "PFN-K_minus_XGB": summary(
                points[2, :, 0] - points[0, :, 0], samples[:, 2, 0] - samples[:, 0, 0]
            ),
            "PFN-ID_minus_PFN-K": summary(
                points[3, :, 0] - points[2, :, 0], samples[:, 3, 0] - samples[:, 2, 0]
            ),
        }
    p, h = seed_metrics["test"], seed_metrics["target"]
    dp, dh = mean_draws["test"], mean_draws["target"]
    interaction = summary(
        (p[3, :, 0] - p[2, :, 0]) - (h[3, :, 0] - h[2, :, 0]),
        (dp[:, 3, 0] - dp[:, 2, 0]) - (dh[:, 3, 0] - dh[:, 2, 0]),
    )
    return {
        "cohort_metrics": cohorts,
        "auc_gap_pythia_minus_herwig": auc_gap,
        "auc_contrasts": contrasts,
        "pid_auc_gain_pythia_minus_herwig": interaction,
    }
