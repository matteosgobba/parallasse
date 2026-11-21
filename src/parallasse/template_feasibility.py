"""Bounded feasibility tools for normalized multinomial template inference.

This prototype is not the frozen TASK-0012 evaluation engine. Profile cutoffs
are asymptotic candidates; their frequentist coverage requires later validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import softmax, xlogy

LR95 = 3.841458820694124


def mixture_weights(fraction: float, alpha: float | None = None) -> tuple:
    """Component order is Pythia quark/gluon, then alternative quark/gluon."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Fraction must lie in [0, 1]")
    if alpha is None:
        return np.array([fraction, 1 - fraction]), np.array([1.0, -1.0]), None
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Variation must lie in [0, 1]")
    return (
        np.array(
            [
                fraction * (1 - alpha),
                (1 - fraction) * (1 - alpha),
                fraction * alpha,
                (1 - fraction) * alpha,
            ]
        ),
        np.array([1 - alpha, -(1 - alpha), alpha, -alpha]),
        np.array([-fraction, -(1 - fraction), fraction, 1 - fraction]),
    )


@dataclass
class TemplateFit:
    fraction: float
    alpha: float | None
    nll: float
    templates: np.ndarray
    success: bool
    iterations: int
    message: str


class MultinomialTemplates:
    """Joint pseudo-data and independent fixed-total class-template likelihood.

    With finite_templates=False, auxiliary histograms only define plug-in
    probabilities. Otherwise every class histogram contributes its multinomial
    likelihood and B-1 softmax coordinates preserve its unit normalization.
    """

    def __init__(self, data, auxiliary, finite_templates=True):
        self.data = np.asarray(data, dtype=float)
        self.auxiliary = np.asarray(auxiliary, dtype=float)
        if (
            self.data.ndim != 1
            or self.auxiliary.ndim != 2
            or self.auxiliary.shape[1] != len(self.data)
            or self.auxiliary.shape[0] not in (2, 4)
            or len(self.data) < 2
            or not np.all(np.isfinite(self.data))
            or not np.all(np.isfinite(self.auxiliary))
            or np.any(self.data < 0)
            or np.any(self.auxiliary < 0)
            or self.data.sum() <= 0
            or np.any(self.auxiliary.sum(axis=1) <= 0)
        ):
            raise ValueError("Nonnegative finite counts and nonempty class samples required")
        self.finite = bool(finite_templates)
        self.classes, self.bins = self.auxiliary.shape
        self.nominal = self.auxiliary / self.auxiliary.sum(axis=1, keepdims=True)
        if not self.finite and np.any((self.nominal.sum(axis=0) == 0) & (self.data > 0)):
            raise ValueError("Observed data outside fixed-template support")
        self.scale = self.data.sum() + (self.auxiliary.sum() if self.finite else 0)
        # Half-counts initialize logits only. They never enter the likelihood.
        initial = self.auxiliary + 0.5
        self.initial_logits = np.log(initial[:, :-1] / initial[:, -1:]).ravel()

    def objective(self, x, fixed_fraction=None, fixed_alpha=None):
        """Return scaled negative log likelihood and its analytic gradient."""
        x = np.asarray(x, dtype=float)
        offset = 0
        if fixed_fraction is None:
            fraction, offset = x[0], 1
        else:
            fraction = fixed_fraction
        alpha = None
        if self.classes == 4:
            if fixed_alpha is None:
                alpha, offset = x[offset], offset + 1
            else:
                alpha = fixed_alpha
        w, df, da = mixture_weights(fraction, alpha)
        templates = self.nominal
        if self.finite:
            logits = x[offset:].reshape(self.classes, self.bins - 1)
            templates = softmax(np.column_stack([logits, np.zeros(self.classes)]), axis=1)
        p = w @ templates
        if np.any((p <= 0) & (self.data > 0)):
            return np.inf, np.zeros_like(x)
        ratio = np.divide(self.data, p, out=np.zeros_like(p), where=p > 0)
        nll = np.sum(xlogy(self.data, self.data / self.data.sum()))
        nll -= np.sum(xlogy(self.data, p))
        gradient = []
        if fixed_fraction is None:
            gradient.append(-float(ratio @ (df @ templates)))
        if self.classes == 4 and fixed_alpha is None:
            gradient.append(-float(ratio @ (da @ templates)))
        if self.finite:
            nll += np.sum(xlogy(self.auxiliary, self.nominal))
            nll -= np.sum(xlogy(self.auxiliary, templates))
            total = self.auxiliary.sum(axis=1, keepdims=True)
            g = total * templates - self.auxiliary
            g -= w[:, None] * templates * (ratio - (templates @ ratio)[:, None])
            gradient.extend(g[:, :-1].ravel())
        return float(nll / self.scale), np.asarray(gradient) / self.scale

    def fit(self, fixed_fraction=None, fixed_alpha=None, method="L-BFGS-B") -> TemplateFit:
        """Use three variation starts, with explicit endpoint constraints."""
        for value in (fixed_fraction, fixed_alpha):
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Fixed parameters must lie in [0, 1]")
        free_alpha = self.classes == 4 and fixed_alpha is None
        starts = (0.0, 0.5, 1.0) if free_alpha else (fixed_alpha,)
        candidates = []
        for alpha in starts:
            initial, bounds = [], []
            if fixed_fraction is None:
                initial.append(0.5)
                bounds.append((0, 1))
            if free_alpha:
                initial.append(alpha)
                bounds.append((0, 1))
            if self.finite:
                initial.extend(self.initial_logits)
                bounds.extend([(-35, 35)] * len(self.initial_logits))
            if not initial:
                value, _ = self.objective([], fixed_fraction, fixed_alpha)
                return TemplateFit(
                    fixed_fraction,
                    fixed_alpha,
                    value * self.scale,
                    self.nominal.copy(),
                    True,
                    0,
                    "No free parameters",
                )
            options = {"maxiter": 1000, "ftol": 1e-13}
            if method == "L-BFGS-B":
                options.update(gtol=1e-9, maxls=40)
            result = minimize(
                self.objective,
                initial,
                args=(fixed_fraction, fixed_alpha),
                jac=True,
                bounds=bounds,
                method=method,
                options=options,
            )
            offset = int(fixed_fraction is None)
            fraction = float(result.x[0]) if fixed_fraction is None else fixed_fraction
            fitted_alpha = None
            if self.classes == 4:
                fitted_alpha = float(result.x[offset]) if free_alpha else fixed_alpha
                offset += int(free_alpha)
            templates = self.nominal.copy()
            if self.finite:
                logits = result.x[offset:].reshape(self.classes, self.bins - 1)
                templates = softmax(np.column_stack([logits, np.zeros(self.classes)]), axis=1)
            candidates.append(
                TemplateFit(
                    fraction,
                    fitted_alpha,
                    float(result.fun * self.scale),
                    templates,
                    bool(result.success and np.isfinite(result.fun)),
                    int(result.nit),
                    str(result.message),
                )
            )
        valid = [fit for fit in candidates if fit.success]
        return min(valid or candidates, key=lambda fit: fit.nll)

    def confidence_set(self, cutoff=LR95, grid_size=9) -> dict:
        """Scan then bracket LR crossings; retain disconnected components.

        This bounded prototype can miss islands narrower than its scan grid.
        Adaptive/global checks remain required for the final evaluation engine.
        """
        best = self.fit()
        if not best.success:
            raise RuntimeError(f"Unconditional fit failed: {best.message}")
        cache = {}

        def profile(fraction):
            key = float(fraction)
            if key not in cache:
                fit = self.fit(fixed_fraction=key)
                if not fit.success:
                    raise RuntimeError(f"Profile fit failed: {fit.message}")
                lr = 2 * (fit.nll - best.nll)
                if lr < -1e-5:
                    raise RuntimeError("Profile scan found a better unconditional optimum")
                cache[key] = max(0.0, lr)
            return cache[key] - cutoff

        grid = np.unique(np.r_[np.linspace(0, 1, grid_size), best.fraction])
        values = [profile(f) for f in grid]
        crossings = []
        for a, b, va, vb in zip(grid[:-1], grid[1:], values[:-1], values[1:], strict=True):
            if va * vb < 0:
                crossings.append(float(brentq(profile, a, b, xtol=2e-6)))
        boundaries = np.unique([0.0, *crossings, 1.0])
        intervals = []
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
            if profile((a + b) / 2) <= 0:
                intervals.append([float(a), float(b)])
        return {
            "estimate": best.fraction,
            "alpha": best.alpha,
            "intervals": intervals,
            "length": sum(b - a for a, b in intervals),
            "span": intervals[-1][1] - intervals[0][0] if intervals else 0,
            "profile_evaluations": len(cache),
            "cutoff": cutoff,
            "boundary_solution": best.fraction < 1e-6 or best.fraction > 1 - 1e-6,
        }


def identification(
    templates, fraction=0.5, alpha=0.5, sample_size=10000, auxiliary_size=25000
) -> dict:
    """Local Fisher geometry with and without finite multinomial templates."""
    t = np.asarray(templates, dtype=float)
    if t.ndim != 2 or t.shape[0] != 4 or np.any(t <= 0):
        raise ValueError("Four strictly positive normalized templates required")
    if not np.allclose(t.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("Templates must be normalized")
    w, df, da = mixture_weights(fraction, alpha)
    p = w @ t
    derivative = np.column_stack([df @ t, da @ t])
    singular = np.linalg.svd(derivative / np.sqrt(p[:, None]), compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] else 0.0
    bins = t.shape[1]
    simplex = np.vstack([np.eye(bins - 1), -np.ones(bins - 1)])
    nuisance = np.hstack([weight * simplex for weight in w])
    full = np.column_stack([derivative, nuisance])
    fisher = sample_size * full.T @ (full / p[:, None])
    for j in range(4):
        section = slice(2 + j * (bins - 1), 2 + (j + 1) * (bins - 1))
        fisher[section, section] += auxiliary_size * (
            np.diag(1 / t[j, :-1]) + np.ones((bins - 1, bins - 1)) / t[j, -1]
        )
    effective = fisher[:2, :2] - fisher[:2, 2:] @ np.linalg.solve(fisher[2:, 2:], fisher[2:, :2])
    identified = bool(ratio > 1e-8 and np.linalg.eigvalsh(effective)[0] > 1e-8)
    finite_width, fixed_width = None, None
    if identified:
        finite_width = float(2 * np.sqrt(LR95 * np.linalg.inv(effective)[0, 0]))
        fixed_width = float(2 * np.sqrt(LR95 * np.linalg.inv(fisher[:2, :2])[0, 0]))
    return {
        "locally_identified": identified,
        "singular_value_ratio": ratio,
        "condition_number": float(1 / ratio) if ratio > 0 else None,
        "local_finite_template_width": finite_width,
        "local_fixed_template_width": fixed_width,
        "effective_information": effective.tolist(),
    }


def source_bins(scores, bins=8) -> np.ndarray:
    """Equal-frequency edges from pooled source response only; cover [0, 1]."""
    values = np.asarray(scores)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("Finite nonempty source scores required")
    if np.any((values < 0) | (values > 1)) or not isinstance(bins, int) or bins < 2:
        raise ValueError("Probability scores and at least two bins required")
    edges = np.r_[0.0, np.quantile(values, np.arange(1, bins) / bins), 1.0]
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Tied source quantiles do not define distinct bins")
    return edges


def synthetic_templates() -> np.ndarray:
    """An identifiable factorized six-bin law with independent class/variation axes."""
    return np.array(
        [
            np.outer(x, y).ravel()
            for y in ([0.6, 0.3, 0.1], [0.1, 0.3, 0.6])
            for x in ([0.2, 0.8], [0.8, 0.2])
        ]
    )


def trial_summary(rows, truth=0.5) -> dict:
    """All attempted fits are the coverage denominator; failures are noncoverage."""
    trials = len(rows)
    if not trials:
        raise ValueError("At least one attempted fit required")
    successful = [row for row in rows if row["success"]]
    hits = sum(any(a <= truth <= b for a, b in row["intervals"]) for row in successful)
    coverage = hits / trials
    z = np.sqrt(LR95)
    scale = 1 + z * z / trials
    center = (coverage + z * z / (2 * trials)) / scale
    half = z / scale * np.sqrt(coverage * (1 - coverage) / trials + z * z / (4 * trials * trials))
    errors = np.array([row["estimate"] - truth for row in successful])
    lengths = [row["length"] for row in successful]
    return {
        "trials": trials,
        "successful": len(successful),
        "failures": trials - len(successful),
        "coverage_hits": hits,
        "coverage": coverage,
        "coverage_mc_wilson95": [max(0, center - half), min(1, center + half)],
        "mean_bias_successful": float(errors.mean()) if len(errors) else None,
        "rmse_successful": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "median_length_successful": float(np.median(lengths)) if lengths else None,
        "disconnected_sets": sum(len(row["intervals"]) > 1 for row in successful),
        "boundary_solutions": sum(row.get("boundary_solution", False) for row in successful),
    }
