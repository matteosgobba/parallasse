"""Production likelihood/profile and interval metrics for the frozen v0.9 design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import softmax, xlogy

from parallasse.mixture_uncertainty import fraction_interval
from parallasse.score_template_design import calibration_rank
from parallasse.template_feasibility import LR95, MultinomialTemplates


class NumericalFailure(RuntimeError):
    """An attempted draw remains in the denominator when its inference fails."""


@dataclass
class ProfileFit:
    fraction: float
    alpha: float | None
    templates: np.ndarray
    nll: float
    saturated: bool = False
    branches: tuple = ()


class ProfileLikelihood(MultinomialTemplates):
    """Exact prototype objective with warm starts and successive scan validation."""

    def __init__(self, data, auxiliary, numerics, finite_templates=True):
        super().__init__(data, auxiliary, finite_templates)
        self.numerics = numerics
        self.cache = {}
        self.best = None
        self.fit_calls = 0
        self.saturation_checks = 0
        self.optimizer_fallback_checks = 0
        self.failed_optimizer_starts = 0
        self.aux_totals = self.auxiliary.sum(axis=1, keepdims=True)
        self.empirical = self.data / self.data.sum()

    def objective(self, x, fixed_fraction=None, fixed_alpha=None):
        """Same normalized likelihood, with fixed terms cached outside optimizer calls."""
        offset = int(fixed_fraction is None)
        f = x[0] if fixed_fraction is None else fixed_fraction
        if self.classes == 4:
            a = x[offset] if fixed_alpha is None else fixed_alpha
            offset += int(fixed_alpha is None)
            w = np.array([f * (1 - a), (1 - f) * (1 - a), f * a, (1 - f) * a])
            df = np.array([1 - a, -(1 - a), a, -a])
            da = np.array([-f, -(1 - f), f, 1 - f])
        else:
            w = np.array([f, 1 - f])
            df = np.array([1.0, -1.0])
        t = self.nominal
        if self.finite:
            logits = np.asarray(x[offset:]).reshape(self.classes, self.bins - 1)
            exponent = np.exp(logits)
            denominator = 1 + exponent.sum(axis=1, keepdims=True)
            t = np.column_stack([exponent / denominator, 1 / denominator])
        p = w @ t
        if np.any((p <= 0) & (self.data > 0)):
            return np.inf, np.zeros_like(x)
        ratio = np.divide(self.data, p, out=np.zeros_like(p), where=p > 0)
        ratio_to_model = np.divide(self.empirical, p, out=np.ones_like(p), where=p > 0)
        value = float(np.sum(xlogy(self.data, ratio_to_model)))
        gradient = []
        if fixed_fraction is None:
            gradient.append(-float(ratio @ (df @ t)))
        if self.classes == 4 and fixed_alpha is None:
            gradient.append(-float(ratio @ (da @ t)))
        if self.finite:
            value += float(np.sum(xlogy(self.auxiliary, self.nominal / t)))
            g = self.aux_totals * t - self.auxiliary
            g -= w[:, None] * t * (ratio - (t @ ratio)[:, None])
            gradient.extend(g[:, :-1].ravel())
        return value / self.scale, np.asarray(gradient) / self.scale

    def _solve(self, fraction=None, warm=None, independent=False):
        free_alpha = self.classes == 4
        starts = [0.0, 0.5, 1.0] if free_alpha else [None]
        if warm is not None and free_alpha and fraction is not None:
            starts[1] = warm.alpha
        candidates = []
        failed_starts = 0
        for alpha in dict.fromkeys(starts):
            initial, bounds = [], []
            if fraction is None:
                initial.append(0.5 if warm is None else warm.fraction)
                bounds.append((0, 1))
            if free_alpha:
                initial.append(alpha)
                bounds.append((0, 1))
            if self.finite:
                if warm is None:
                    logits = self.initial_logits
                else:
                    logits = np.log(warm.templates[:, :-1] / warm.templates[:, -1:]).ravel()
                initial.extend(np.clip(logits, -35, 35))
                bounds.extend([(-35, 35)] * len(logits))
            if not initial:
                value, _ = self.objective([], fraction)
                return ProfileFit(fraction, None, self.nominal, value * self.scale)
            options = {"maxiter": 1000, "ftol": 1e-13}
            method = "SLSQP" if independent else "L-BFGS-B"
            if not independent:
                options.update(gtol=1e-9, maxls=40)
            result = minimize(
                self.objective,
                initial,
                args=(fraction, None),
                jac=True,
                bounds=bounds,
                method=method,
                options=options,
            )
            self.fit_calls += 1
            if not result.success or not np.isfinite(result.fun):
                failed_starts += 1
                self.failed_optimizer_starts += 1
                continue
            offset = int(fraction is None)
            fitted_fraction = float(result.x[0]) if fraction is None else float(fraction)
            fitted_alpha = float(result.x[offset]) if free_alpha else None
            offset += int(free_alpha)
            templates, saturated = self.nominal, False
            if self.finite:
                logits = result.x[offset:].reshape(self.classes, self.bins - 1)
                templates = softmax(np.column_stack([logits, np.zeros(self.classes)]), axis=1)
                saturated = bool(np.any(np.abs(logits) > 34.99))
            candidates.append(
                ProfileFit(
                    fitted_fraction,
                    fitted_alpha,
                    templates,
                    float(result.fun * self.scale),
                    saturated,
                )
            )
        if not candidates:
            if not independent:
                self.optimizer_fallback_checks += 1
                return self._solve(fraction, warm, independent=True)
            raise NumericalFailure("All optimizer starts failed")
        best = min(candidates, key=lambda fit: fit.nll)
        best.branches = tuple(sorted({fit.fraction for fit in candidates}))
        if failed_starts and not independent:
            check = self._solve(fraction, best, independent=True)
            self.optimizer_fallback_checks += 1
            if abs(check.nll - best.nll) > self.numerics["optimizer_nll_agreement"]:
                raise NumericalFailure("Flagged optimizers disagree beyond tolerance")
            if check.nll < best.nll:
                best = check
        if best.saturated and not independent:
            check = self._solve(fraction, best, independent=True)
            self.saturation_checks += 1
            if abs(check.nll - best.nll) > self.numerics["optimizer_nll_agreement"]:
                raise NumericalFailure("Saturated-logit optimizers disagree")
            if check.nll < best.nll:
                best = check
        return best

    def global_fit(self):
        if self.best is None:
            self.best = self._solve()
        return self.best

    def profile(self, fraction):
        fraction = float(fraction)
        if not 0 <= fraction <= 1:
            raise ValueError("Profile fraction outside [0,1]")
        if fraction not in self.cache:
            self.global_fit()
            warm = (
                self.cache[min(self.cache, key=lambda f: abs(f - fraction))]
                if self.cache
                else self.best
            )
            self.cache[fraction] = self._solve(fraction, warm)
        candidate = self.cache[fraction]
        tolerance = self.numerics["negative_profile_lr_tolerance"]
        if 2 * (candidate.nll - self.best.nll) < -tolerance:
            repaired = self._solve(warm=candidate)
            if repaired.nll > candidate.nll + tolerance / 2:
                # The constrained point itself is a feasible unconditional point.
                repaired = candidate
            self.best = repaired
        return candidate.nll

    def lr_at(self, fraction):
        self.profile(fraction)
        value = 2 * (self.cache[float(fraction)].nll - self.best.nll)
        if value < -self.numerics["negative_profile_lr_tolerance"]:
            raise NumericalFailure("Unresolved negative profile LR")
        return max(0.0, float(value))

    def _scan(self, nodes, cutoff):
        best = self.global_fit()
        grid = np.unique(np.r_[np.linspace(0, 1, nodes), best.fraction, best.branches])
        for f in grid:
            self.profile(f)
        # Include branch minima found on the scan; reevaluate LR relative to the repaired best.
        local = [
            grid[i]
            for i in range(1, len(grid) - 1)
            if self.cache[float(grid[i])].nll
            <= min(self.cache[float(grid[i - 1])].nll, self.cache[float(grid[i + 1])].nll)
        ]
        grid = np.unique(np.r_[grid, local, self.best.fraction])

        def value(f):
            return self.lr_at(f) - cutoff

        values = [value(f) for f in grid]
        roots = [float(f) for f, v in zip(grid, values, strict=True) if v == 0]
        for a, b, va, vb in zip(grid[:-1], grid[1:], values[:-1], values[1:], strict=True):
            if va * vb < 0:
                root = float(brentq(value, a, b, xtol=self.numerics["fraction_root_atol"]))
                if abs(value(root)) > self.numerics["endpoint_lr_atol"]:
                    raise NumericalFailure("Profile endpoint LR residual exceeds tolerance")
                roots.append(root)
        points = np.unique(np.r_[0.0, roots, 1.0])
        components = []
        for a, b in zip(points[:-1], points[1:], strict=True):
            if value((a + b) / 2) <= 0:
                components.append([float(a), float(b)])
        if not components:
            raise NumericalFailure("No confidence component around the fitted minimum")
        return components

    def confidence_set(self, cutoff=LR95):
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("Positive finite LR cutoff required")
        previous = None
        refinements = self.numerics["scan_nodes"]
        for nodes in refinements:
            old_minimum = self.global_fit().nll
            components = self._scan(nodes, cutoff)
            # A scan which repaired a global minimum must be repeated at the same resolution.
            if self.best.nll < old_minimum - self.numerics["negative_profile_lr_tolerance"] / 2:
                components = self._scan(nodes, cutoff)
                previous = None
            if previous is not None and len(previous) == len(components):
                if (
                    np.max(np.abs(np.asarray(previous) - components))
                    <= self.numerics["scan_stability_fraction_atol"]
                ):
                    return {
                        "success": True,
                        "estimate": self.best.fraction,
                        "alpha": self.best.alpha,
                        "intervals": components,
                        "length": sum(b - a for a, b in components),
                        "span": components[-1][1] - components[0][0],
                        "scan_nodes": nodes,
                        "profile_evaluations": len(self.cache),
                        "optimizer_calls": self.fit_calls,
                        "boundary_solution": self.best.fraction < 1e-6
                        or self.best.fraction > 1 - 1e-6,
                        "saturated_logit_checks": self.saturation_checks,
                        "optimizer_fallback_checks": self.optimizer_fallback_checks,
                        "failed_optimizer_starts": self.failed_optimizer_starts,
                        "cutoff": cutoff,
                    }
            previous = components
        raise NumericalFailure("Confidence components unresolved at the final scan resolution")


def infer_histogram(data, auxiliary, numerics, cutoff=LR95, truth=None, calibration=False):
    """One attempted fit; failures never disappear or become clipped estimates."""
    try:
        model = ProfileLikelihood(data, auxiliary, numerics)
        if calibration:
            if truth is None:
                raise ValueError("Calibration needs its declared truth fraction")
            lr = model.lr_at(truth)
            return {"success": True, "lr": lr, "estimate": model.best.fraction}
        return model.confidence_set(cutoff)
    except (NumericalFailure, ValueError, FloatingPointError) as error:
        return {
            "success": False,
            "error": f"{type(error).__name__}: {error}",
            **({"lr": None} if calibration else {}),
        }


def calibrated_cutoff(cells, config):
    """Conservative maximum of the frozen order statistic; failures are +infinity."""
    if not cells:
        raise ValueError("No calibration cells supplied")
    cal = config["calibration"]
    rank = calibration_rank(
        cal["repetitions_per_cell"],
        cal["nominal_coverage"],
        cal["family_quantile_error"],
        cal["family_cells"],
    )
    values = []
    for rows in cells:
        if len(rows) != cal["repetitions_per_cell"]:
            raise ValueError("Incomplete calibration cell")
        failures = sum(not row["success"] for row in rows)
        if failures / len(rows) > config["primary_decision"]["maximum_failure_rate"]:
            raise NumericalFailure("Calibration failure ceiling exceeded")
        array = [row["lr"] if row["success"] else np.inf for row in rows]
        if any(value is None or value < 0 or np.isnan(value) for value in array):
            raise ValueError("Invalid LR calibration statistic")
        values.append(float(np.partition(array, rank - 1)[rank - 1]))
    cutoff = max([cal["minimum_lr_cutoff"], *values])
    if not np.isfinite(cutoff):
        raise NumericalFailure("Calibration yielded an infinite cutoff")
    return cutoff


def cut_result(accepted, size, efficiencies, class_sizes):
    estimate, lo, hi, _ = fraction_interval(
        np.array([accepted], dtype=np.int64), size, *efficiencies, *map(int, class_sizes)
    )
    return {
        "success": True,
        "estimate": float(estimate[0]),
        "intervals": [[float(lo[0]), float(hi[0])]],
        "length": float(hi[0] - lo[0]),
        "span": float(hi[0] - lo[0]),
        "outside_unit": bool(estimate[0] < 0 or estimate[0] > 1),
    }


def summarize_trials(rows, truth):
    if not rows:
        raise ValueError("At least one attempted trial required")
    good = [row for row in rows if row["success"]]
    hits = sum(any(a <= truth <= b for a, b in row["intervals"]) for row in good)
    coverage, n = hits / len(rows), len(rows)
    z = np.sqrt(LR95)
    denom = 1 + z * z / n
    center = (coverage + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(coverage * (1 - coverage) / n + z * z / (4 * n * n))
    errors = np.array([row["estimate"] - truth for row in good])
    length = [row["length"] for row in good]
    span = [row["span"] for row in good]
    return {
        "attempts": n,
        "successful": len(good),
        "failures": n - len(good),
        "coverage_hits": hits,
        "coverage": coverage,
        "coverage_mc95": [max(0.0, center - half), min(1.0, center + half)],
        "bias": float(errors.mean()) if len(good) else None,
        "absolute_bias": abs(float(errors.mean())) if len(good) else None,
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(good) else None,
        "mean_length": float(np.mean(length)) if good else None,
        "median_length": float(np.median(length)) if good else None,
        "median_span": float(np.median(span)) if good else None,
        "span_95_quantile": float(np.quantile(span, 0.95)) if good else None,
        "disconnected_sets": sum(len(row["intervals"]) > 1 for row in good),
        "boundary_solutions": sum(row.get("boundary_solution", False) for row in good),
        "outside_unit": sum(row.get("outside_unit", False) for row in good),
    }


def validation_gate(summaries, config):
    n = config["validation"]["repetitions"]
    limit = 5 * np.sqrt(0.95 * 0.05 / n)
    return bool(
        summaries
        and all(
            r["attempts"] == n
            and abs(r["coverage"] - 0.95) <= limit
            and r["failures"] / n <= config["primary_decision"]["maximum_failure_rate"]
            and r["median_span"] is not None
            for r in summaries
        )
    )


def paired_contrasts(trials_by_method, fraction, bootstrap_indices):
    """Intervals quantify paired pseudoexperiment Monte Carlo precision only."""
    methods = list(trials_by_method)
    n = len(trials_by_method[methods[0]])
    indices = np.asarray(bootstrap_indices)
    if any(len(rows) != n for rows in trials_by_method.values()) or indices.ndim != 2:
        raise ValueError("Methods and bootstrap draws must share attempted trial indices")
    if not np.issubdtype(indices.dtype, np.integer) or np.any((indices < 0) | (indices >= n)):
        raise ValueError("Invalid paired bootstrap indices")
    hit = {
        m: np.array(
            [
                row["success"] and any(a <= fraction <= b for a, b in row["intervals"])
                for row in rows
            ],
            dtype=float,
        )
        for m, rows in trials_by_method.items()
    }
    baseline = hit["original_cut_delta"]
    result = {}
    for method in methods:
        if method == "original_cut_delta":
            continue
        difference = hit[method] - baseline
        lo, hi = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
        result[method] = {
            "coverage_difference": float(difference.mean()),
            "paired_mc95": [float(lo), float(hi)],
            "attempts": n,
        }
        for metric in ("bias", "absolute_bias", "rmse", "mean_span"):
            quantities = []
            for name in (method, "original_cut_delta"):
                values = np.array(
                    [
                        row.get("span", np.nan)
                        if metric == "mean_span"
                        else row.get("estimate", np.nan) - fraction
                        for row in trials_by_method[name]
                    ],
                    dtype=float,
                )
                if metric == "rmse":
                    values = values**2
                valid = np.isfinite(values)
                sums = np.where(valid, values, 0)[indices].sum(axis=1)
                counts = valid[indices].sum(axis=1)
                boot = np.divide(sums, counts, out=np.full(len(indices), np.nan), where=counts > 0)
                point = float(values[valid].mean()) if valid.any() else np.nan
                if metric == "absolute_bias":
                    point, boot = abs(point), np.abs(boot)
                if metric == "rmse":
                    point, boot = np.sqrt(point), np.sqrt(boot)
                quantities.append((point, boot))
            point = quantities[0][0] - quantities[1][0]
            boot = quantities[0][1] - quantities[1][1]
            valid = np.isfinite(boot)
            interval = np.quantile(boot[valid], [0.025, 0.975]).tolist() if valid.any() else None
            result[method][metric] = {
                "difference": float(point) if np.isfinite(point) else None,
                "paired_mc95": interval,
                "successful_bootstrap_draws": int(valid.sum()),
            }
    return result
