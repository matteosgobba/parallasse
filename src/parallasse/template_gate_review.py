"""Deterministic operating-characteristic review; no scoring or access authority."""

from numbers import Integral

import numpy as np
from scipy.integrate import quad
from scipy.stats import beta, binom


def _integer(value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("An integer in the declared domain is required")


def exact_lower_bound(hits, attempts, error):
    """One-sided exact lower bound with failed attempts counted as misses."""
    _integer(attempts, 1)
    _integer(hits)
    if hits > attempts or not 0 < error < 1:
        raise ValueError("Invalid binomial counts or error probability")
    return 0.0 if hits == 0 else float(beta.ppf(error, hits, attempts - hits + 1))


def required_hits(attempts, coverage_floor, error):
    """Return None if even all successes cannot establish the requested floor."""
    _integer(attempts, 1)
    if not 0 < coverage_floor < 1 or not 0 < error < 1:
        raise ValueError("Interior coverage/error probabilities required")
    hits = np.arange(1, attempts + 1)
    accepted = np.flatnonzero(beta.ppf(error, hits, attempts - hits + 1) >= coverage_floor)
    return int(accepted[0] + 1) if len(accepted) else None


def assess_proposed_cell(hits, attempts, failures, median_span, qualification):
    """Numerical design predicate only; cannot create a source/modelling access gate."""
    _integer(failures)
    if attempts != qualification["repetitions"] or failures > attempts - hits:
        raise ValueError("Declared attempts and consistent hit/failure counts required")
    error = qualification["family_error_probability"] / qualification["family_cells"]
    lower = exact_lower_bound(hits, attempts, error)
    valid_span = median_span is not None and np.isfinite(median_span) and 0 <= median_span <= 1
    checks = {
        "coverage": lower >= qualification["minimum_coverage"],
        "failures": failures <= qualification["maximum_failure_rate"] * attempts,
        "usefulness": bool(valid_span and median_span <= qualification["maximum_median_set_span"]),
    }
    return {"lower_coverage_bound": lower, "checks": checks, "qualified": all(checks.values())}


def ideal_operating_characteristics(
    calibration_size,
    rank,
    truth_cells,
    validation_size,
    lower_hits,
    upper_hits,
    minimum_content=0.95,
):
    """Exact illustrative continuous-pivot law, not a forecast for the actual PFNs.

    Each truth has the same continuous LR CDF. For independent calibration samples,
    its transformed rank statistic is Beta(rank, M+1-rank). The selected content is
    the maximum of the fixed base content and all truth-cell rank contents.
    Validation cells are conditionally independent binomials at that content.
    """
    for value in (calibration_size, rank, truth_cells, validation_size):
        _integer(value, 1)
    if rank > calibration_size or not 0 <= lower_hits <= upper_hits <= validation_size:
        raise ValueError("Invalid order statistic or validation band")
    if not 0 <= minimum_content < 1:
        raise ValueError("Invalid base content")
    law = beta(rank, calibration_size + 1 - rank)
    mass = float(law.cdf(minimum_content) ** truth_cells)

    def density(p):
        return truth_cells * law.pdf(p) * law.cdf(p) ** (truth_cells - 1)

    def average(function):
        value, error = quad(
            lambda p: function(p) * density(p), minimum_content, 1, epsabs=1e-11, epsrel=1e-10
        )
        if error > 1e-8:
            raise ArithmeticError("Operating-characteristic integral unresolved")
        return float(mass * function(minimum_content) + value)

    def pass_probability(p):
        return float(
            binom.cdf(upper_hits, validation_size, p)
            - binom.cdf(lower_hits - 1, validation_size, p)
        )

    upper = average(lambda p: binom.sf(upper_hits, validation_size, p))
    lower = average(lambda p: binom.cdf(lower_hits - 1, validation_size, p))
    return {
        "calibration_truth_cells": truth_cells,
        "mean_selected_content": average(lambda p: p),
        "one_cell_upper_veto_probability": upper,
        "one_cell_lower_veto_probability": lower,
        "one_cell_veto_probability": upper + lower,
        "any_validation_cell_veto_probability": average(
            lambda p: 1 - pass_probability(p) ** truth_cells
        ),
        "scope": (
            "Illustrative identical continuous pivots, independent truth-cell samples; "
            "no actual generator/paired-checkpoint forecast"
        ),
    }
