import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from parallasse.template_feasibility import (
    MultinomialTemplates,
    mixture_weights,
    synthetic_templates,
)
from parallasse.template_inference import (
    NumericalFailure,
    ProfileLikelihood,
    calibrated_cutoff,
    cut_result,
    paired_contrasts,
    summarize_trials,
)
from parallasse.template_sampling import class_pools, draw_experiment, histograms


def config():
    return json.loads(Path("configs/score_templates.json").read_text())


def test_adaptive_profile_agrees_with_known_noiseless_truth_and_separate_optimizer():
    t = synthetic_templates()
    model = ProfileLikelihood(
        1000 * (mixture_weights(0.35, 0.4)[0] @ t), 5000 * t, config()["numerics"]
    )
    x = np.r_[0.43, 0.26, model.initial_logits]
    old_value, old_gradient = MultinomialTemplates.objective(model, x)
    value, gradient = model.objective(x)
    assert value == pytest.approx(old_value, abs=1e-12)
    np.testing.assert_allclose(gradient, old_gradient, atol=1e-12, rtol=0)
    with threadpool_limits(1):
        result = model.confidence_set()
        independent = model._solve(independent=True)
    assert result["success"] and result["scan_nodes"] >= 33
    assert result["estimate"] == pytest.approx(0.35, abs=1e-4)
    assert any(a <= 0.35 <= b for a, b in result["intervals"])
    assert abs(independent.nll - model.best.nll) < 1e-4
    np.testing.assert_allclose(model.best.templates.sum(axis=1), 1, atol=1e-12)


def test_real_disconnected_profile_keeps_components_and_full_span():
    t = np.array([[0.2, 0.8], [0.8, 0.2], [0.8, 0.2], [0.2, 0.8]])
    model = ProfileLikelihood(
        10000 * (mixture_weights(0.9, 0)[0] @ t),
        5000 * t,
        config()["numerics"],
        finite_templates=False,
    )
    with threadpool_limits(1):
        result = model.confidence_set()
    assert len(result["intervals"]) == 2 and result["span"] == 1
    assert result["length"] < 0.3
    assert result["intervals"][0][1] == pytest.approx(0.11443, abs=0.0001)
    assert result["intervals"][1][0] == pytest.approx(0.88557, abs=0.0001)
    flat = ProfileLikelihood(1000 * t[0], np.tile(5000 * t[0], (4, 1)), config()["numerics"], False)
    np.testing.assert_allclose(flat.confidence_set()["intervals"], [[0, 1]])


def test_scan_ceiling_is_a_failure_and_zero_count_nuisance_is_not_a_pseudocount():
    numerics = {**config()["numerics"], "scan_nodes": [17]}
    model = ProfileLikelihood([600, 400], [[250, 750], [800, 200]], numerics)
    with pytest.raises(NumericalFailure, match="unresolved"):
        model.confidence_set()
    zero = ProfileLikelihood([4, 5, 1], [[10, 0, 0], [0, 10, 0]], config()["numerics"])
    fit = zero.global_fit()
    assert np.all(fit.templates[:, 2] > 0)
    np.testing.assert_array_equal(zero.auxiliary, [[10, 0, 0], [0, 10, 0]])
    cut = cut_result(0, 100, [0.7, 0.2], [5000, 5000])
    assert cut["estimate"] < 0 and cut["intervals"][0][0] < 0 and cut["outside_unit"]


def test_calibration_rank_failures_and_no_optional_validation_selection():
    cfg = config()
    rows = [{"success": True, "lr": float(i) / 100} for i in range(1000)]
    assert calibrated_cutoff([rows], cfg) == pytest.approx(9.72)
    altered = [dict(r) for r in rows]
    for i in range(11):
        altered[i] = {"success": False, "lr": None}
    with pytest.raises(NumericalFailure, match="failure ceiling"):
        calibrated_cutoff([altered], cfg)
    with pytest.raises(ValueError, match="Incomplete"):
        calibrated_cutoff([rows[:-1]], cfg)


def test_failure_denominators_span_and_paired_bootstrap_semantics():
    good = {"success": True, "estimate": 0.5, "intervals": [[0.4, 0.6]], "length": 0.2, "span": 0.2}
    miss = {"success": True, "estimate": 0.2, "intervals": [[0.1, 0.3]], "length": 0.2, "span": 0.2}
    rows = [good, {"success": False}, miss, good]
    stats = summarize_trials(rows, 0.5)
    assert stats["coverage_hits"] == 2 and stats["attempts"] == 4 and stats["successful"] == 3
    assert stats["failures"] == 1 and stats["coverage"] == 0.5
    draws = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    contrast = paired_contrasts(
        {"original_cut_delta": rows, "source_multinomial_profile": rows}, 0.5, draws
    )
    assert contrast["source_multinomial_profile"]["paired_mc95"] == [0, 0]


def test_checkpoint_paired_sampling_replays_independent_rows_and_histograms():
    scores = np.repeat(np.linspace(0, 1, 40)[:, None], 6, axis=1)
    labels = np.tile([1, 0], 20)
    edges = np.repeat(np.linspace(0, 1, 9)[None, :], 6, axis=0)
    pools = class_pools(scores, labels, np.arange(40), edges, np.full(6, 0.5))
    # A score at a bin boundary goes to its right; strict cuts keep ties rejected.
    boundary = class_pools(
        np.tile([0.5], (4, 6)), np.array([1, 0, 1, 0]), np.arange(4), edges, np.full(6, 0.5)
    )
    assert np.all(boundary[0].bins == 4) and not boundary[0].passed.any()
    a = draw_experiment(pools, 0.5, 100, np.random.default_rng(17))
    b = draw_experiment(pools, 0.5, 100, np.random.default_rng(17))
    for key in ["auxiliary", "observed", "accepted", "response_pass"]:
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a[key], np.repeat(a[key][:1], 6, axis=0))
    assert a["row_draw_sha256"] == b["row_draw_sha256"]
    assert np.all(a["observed"].sum(axis=1) == 100)
    assert np.all(histograms(pools).sum(axis=2) == 20)
    fixed = draw_experiment(pools, 0.5, 100, np.random.default_rng(18), repeated=False)
    np.testing.assert_array_equal(fixed["auxiliary"], histograms(pools))


def test_failed_old_source_profiles_recover_without_changing_the_likelihood():
    fixture = json.loads(Path("tests/fixtures/template_optimizer_source.json").read_text())
    with threadpool_limits(1):
        for case in fixture["cases"]:
            model = ProfileLikelihood(case["data"], case["auxiliary"], config()["numerics"])
            result = model.confidence_set(case["cutoff"])
            independent = model._solve(independent=True)
            assert result["success"] and result["scan_nodes"] >= 33
            assert abs(model.best.nll - independent.nll) < 1e-4
            for a, b in result["intervals"]:
                for endpoint in (a, b):
                    if 0 < endpoint < 1:
                        assert abs(model.lr_at(endpoint) - case["cutoff"]) <= 0.002


def test_unsuccessful_native_status_requires_successful_independent_fallback(monkeypatch):
    from types import SimpleNamespace

    import parallasse.template_inference as engine

    actual = engine.minimize

    def forced(fun, x0, *args, **kwargs):
        if kwargs.get("method") == "L-BFGS-B":
            return SimpleNamespace(success=False, fun=-123456.0)
        return actual(fun, x0, *args, **kwargs)

    monkeypatch.setattr(engine, "minimize", forced)
    model = ProfileLikelihood([600, 400], [[250, 750], [800, 200]], config()["numerics"])
    fit = model.global_fit()
    assert fit.fraction == pytest.approx((0.4 - 0.2) / (0.75 - 0.2), abs=1e-4)
    assert fit.nll > -1e-4 and model.optimizer_fallback_checks > 0
