import json

import numpy as np
import pytest

from parallasse.download import sha256_file
from parallasse.mixture_uncertainty import (
    Z95,
    bias_contrasts,
    bootstrap_acceptance,
    expected_bias,
    fraction_interval,
    inference_summary,
)
from parallasse.mixtures import mixture_counts


def test_delta_variance_matches_independent_derivatives_and_response_precision():
    estimate, lo, hi, se = fraction_interval(np.array([400]), 1000, 0.7, 0.2, 5000, 6000)
    f = (0.4 - 0.2) / (0.7 - 0.2)
    variance = (0.4 * 0.6 / 1000 + f * f * 0.7 * 0.3 / 5000 + (1 - f) ** 2 * 0.2 * 0.8 / 6000) / (
        0.7 - 0.2
    ) ** 2
    assert estimate[0] == pytest.approx(f)
    assert se[0] ** 2 == pytest.approx(variance)
    assert lo[0] == pytest.approx(f - Z95 * np.sqrt(variance))
    assert hi[0] == pytest.approx(f + Z95 * np.sqrt(variance))
    larger = fraction_interval(np.array([400]), 1000, 0.7, 0.2, 10000, 12000)
    assert larger[3][0] < se[0]
    assert larger[0][0] == estimate[0]


def test_intervals_keep_boundaries_unbounded_and_reject_invalid_measurements():
    estimate, lo, hi, _ = fraction_interval(np.array([0, 100]), 100, 0.7, 0.2, 5000, 5000)
    assert estimate[0] < 0 and estimate[1] > 1
    assert lo[0] < 0 and hi[1] > 1
    for count in (np.array([0.5]), np.array([-1]), np.array([101])):
        with pytest.raises(ValueError, match="Accepted counts"):
            fraction_interval(count, 100, 0.7, 0.2, 5000, 5000)
    with pytest.raises(ValueError, match="positive integers"):
        fraction_interval(np.array([50]), 100, 0.7, 0.2, 0, 5000)
    with pytest.raises(ValueError, match="identify"):
        fraction_interval(np.array([50]), 100, 0.2, 0.2, 5000, 5000)


def test_response_bootstrap_pairs_columns_and_reproduces_row_draws():
    values = np.array([[True, False], [False, True], [True, False]])
    actual = bootstrap_acceptance(values, 20, 8, np.random.default_rng(71))
    assert np.all(actual.sum(axis=1) == 20)
    rng = np.random.default_rng(71)
    expected = np.stack([values[rng.integers(3, size=20)].sum(axis=0) for _ in range(8)])
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="boolean"):
        bootstrap_acceptance(values.astype(int), 20, 8, rng)


def test_summary_records_zero_coverage_monte_carlo_precision_and_outside_estimates():
    stats = inference_summary(
        np.zeros((20, 1), dtype=int), 100, 0.5, np.array([0.7]), np.array([0.2]), 5000, 5000
    )[0]
    assert stats["coverage_hits"] == 0 and stats["coverage_trials"] == 20
    assert stats["coverage_95"] == 0
    assert stats["coverage_mc_95_hi"] > 0
    assert stats["outside_unit_interval_rate"] == 1
    assert stats["bias"] == pytest.approx(-0.9)


def test_repeated_response_coverage_control_uses_common_law_and_paired_events():
    rng = np.random.default_rng(317)
    q = np.repeat((rng.random(400) < 0.7)[:, None], 2, axis=1)
    g = np.repeat((rng.random(400) < 0.2)[:, None], 2, axis=1)
    cq = bootstrap_acceptance(q, 1000, 1000, np.random.default_rng(12))
    cg = bootstrap_acceptance(g, 1000, 1000, np.random.default_rng(13))
    counts, _ = mixture_counts(q, g, 0.5, 1000, 1000, np.random.default_rng(14))
    stats = inference_summary(counts, 1000, 0.5, cq / 1000, cg / 1000, 1000, 1000)
    assert stats[0] == stats[1]
    assert 0.90 < stats[0]["coverage_95"] < 0.99
    assert abs(stats[0]["bias"]) < 5 * stats[0]["mean_mcse"]


def test_absolute_bias_contrast_averages_seedwise_absolute_values():
    point = np.zeros((4, 3))
    point[2] = [0.1, -0.1, 0.1]
    point[3] = [0.15, -0.05, 0.15]
    report = bias_contrasts(point.ravel(), np.tile(point.ravel(), (50, 1)))
    contrast = report["contrasts"]["PFN-ID_minus_PFN-K"]
    expected = np.mean(np.abs(point[3]) - np.abs(point[2]))
    assert contrast["mean"] == pytest.approx(expected)
    assert contrast["bootstrap_95_lo"] == pytest.approx(expected)
    assert contrast["bootstrap_95_hi"] == pytest.approx(expected)
    np.testing.assert_allclose(expected_bias(0.5, 0.7, 0.2, 0.7, 0.2), 0, atol=1e-15)


def test_target_gate_requires_validated_source_and_rejects_tampered_inputs(tmp_path, monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("evaluate_mixture_transfer")
    record = {"status": "source_sampling"}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="coverage gate"):
        runner.verified_source_stage(tmp_path)
    summary = tmp_path / "source_summary.json"
    summary.write_text(json.dumps({"source_gate_passed": False}))
    code = tmp_path / "code.py"
    code.write_text("original")
    archive = tmp_path / "source.npz"
    archive.write_bytes(b"fixture")
    record.update(
        status="source_validated",
        source_files_sha256={str(code): sha256_file(code)},
        source_snapshot=str(tmp_path),
        input_sha256={},
        output_sha256={str(summary): sha256_file(summary)},
        source_archive={"path": str(archive), "sha256": sha256_file(archive)},
    )
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="did not pass"):
        runner.verified_source_stage(tmp_path)
    summary.write_text(json.dumps({"source_gate_passed": True}))
    with pytest.raises(ValueError, match="Frozen bytes"):
        runner.verified_source_stage(tmp_path)
    record["output_sha256"][str(summary)] = sha256_file(summary)
    path.write_text(json.dumps(record))
    assert runner.verified_source_stage(tmp_path)["status"] == "source_validated"
    code.write_text("changed")
    with pytest.raises(ValueError, match="Frozen bytes"):
        runner.verified_source_stage(tmp_path)
