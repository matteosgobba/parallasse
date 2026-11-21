import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from parallasse.mixtures import infer_fraction
from parallasse.template_feasibility import (
    MultinomialTemplates,
    identification,
    mixture_weights,
    source_bins,
    synthetic_templates,
    trial_summary,
)


def test_joint_multinomial_gradient_and_normalization():
    templates = synthetic_templates()
    data = 1000 * (mixture_weights(0.4, 0.3)[0] @ templates)
    model = MultinomialTemplates(data, 2000 * templates)
    x = np.r_[0.43, 0.26, model.initial_logits]
    actual = model.objective(x)[1]
    numerical = approx_derivative(lambda v: model.objective(v)[0], x).ravel()
    np.testing.assert_allclose(actual, numerical, atol=1e-8, rtol=1e-5)
    fit = model.fit()
    assert fit.success
    assert fit.fraction == pytest.approx(0.4, abs=1e-4)
    assert fit.alpha == pytest.approx(0.3, abs=1e-4)
    np.testing.assert_allclose(fit.templates.sum(axis=1), 1, atol=1e-12)


def test_two_bin_joint_mle_matches_unbounded_inverse_when_interior():
    model = MultinomialTemplates([600, 400], [[250, 750], [800, 200]])
    fit = model.fit()
    assert fit.success and fit.nll < 1e-6
    assert fit.fraction == pytest.approx(float(infer_fraction(0.4, 0.75, 0.2)), abs=1e-5)
    bounded = MultinomialTemplates([1000, 0], [[250, 750], [800, 200]], False).fit()
    assert bounded.fraction == 0
    assert infer_fraction(0, 0.75, 0.2) < 0


def test_rank_detects_identical_classes_and_confounded_variation():
    t = synthetic_templates()
    assert identification(t)["locally_identified"]
    assert not identification(t[:, :2] / t[:, :2].sum(axis=1)[:, None])["locally_identified"]
    confounded = np.array([t[0], t[1], 0.8 * t[0] + 0.2 * t[1], t[1]])
    assert not identification(confounded)["locally_identified"]
    identical = np.tile(t[0], (4, 1))
    assert not identification(identical)["locally_identified"]
    interval = MultinomialTemplates(1000 * t[0], 5000 * identical, False).confidence_set()
    np.testing.assert_allclose(interval["intervals"], [[0, 1]])


def test_finite_template_precision_increases_local_profile_width():
    t = synthetic_templates()
    small = identification(t, auxiliary_size=500)
    large = identification(t, auxiliary_size=25000)
    assert small["local_finite_template_width"] > large["local_finite_template_width"]
    assert large["local_finite_template_width"] > large["local_fixed_template_width"]


def test_independent_optimizer_and_profile_boundary():
    t = synthetic_templates()
    data = 1000 * (mixture_weights(0.35, 0)[0] @ t)
    model = MultinomialTemplates(data, 5000 * t)
    a, b = model.fit(), model.fit(method="SLSQP")
    assert a.success and b.success
    assert abs(a.nll - b.nll) < 1e-4
    assert a.fraction == pytest.approx(0.35, abs=1e-4)
    assert a.alpha == pytest.approx(0, abs=1e-4)
    result = model.confidence_set()
    assert any(lo <= 0.35 <= hi for lo, hi in result["intervals"])
    assert result["length"] < 0.3


def test_zero_counts_are_not_unsupported_fixed_probabilities_or_pseudocounts():
    with pytest.raises(ValueError, match="support"):
        MultinomialTemplates([4, 5, 1], [[10, 0, 0], [0, 10, 0]], False)
    model = MultinomialTemplates([4, 5, 1], [[10, 0, 0], [0, 10, 0]])
    fit = model.fit()
    assert fit.success and np.all(fit.templates[:, 2] > 0)
    # The original zero auxiliary counts remain exactly zero in the likelihood.
    np.testing.assert_array_equal(model.auxiliary, [[10, 0, 0], [0, 10, 0]])
    for data, aux in (([-1, 1], [[1, 2], [2, 1]]), ([1, 2], [[0, 0], [1, 2]])):
        with pytest.raises(ValueError, match="counts"):
            MultinomialTemplates(data, aux)


def test_source_binning_and_failure_denominator():
    edges = source_bins(np.linspace(0.01, 0.99, 80))
    assert edges[0] == 0 and edges[-1] == 1 and len(edges) == 9
    with pytest.raises(ValueError, match="Tied"):
        source_bins(np.full(20, 0.5))
    summary = trial_summary(
        [
            {"success": True, "estimate": 0.5, "intervals": [[0.4, 0.6]], "length": 0.2},
            {"success": False},
        ]
    )
    assert summary["coverage_hits"] == 1 and summary["trials"] == 2
    assert summary["coverage"] == 0.5 and summary["failures"] == 1


def test_pilot_run_id_reentry_is_rejected_before_data_access(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("pilot_template_inference")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/task0010/completed").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="never overwritten"):
        runner.run_pilot("completed")
