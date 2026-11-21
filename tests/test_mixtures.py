import numpy as np
import pytest

from parallasse.mixtures import (
    closure_summary,
    infer_fraction,
    mixture_counts,
    source_partition,
)


def test_fraction_inverse_closes_at_boundaries_and_with_reversed_response():
    fractions = np.array([0, 0.25, 0.5, 0.75, 1])[:, None]
    eq, eg = np.array([0.7, 0.1]), np.array([0.2, 0.8])
    np.testing.assert_allclose(
        infer_fraction(fractions * eq + (1 - fractions) * eg, eq, eg),
        np.broadcast_to(fractions, (5, 2)),
        atol=1e-15,
    )
    assert infer_fraction(0.05, 0.7, 0.2) < 0
    assert infer_fraction(0.9, 0.7, 0.2) > 1


def test_fraction_rejects_nonidentifiable_and_invalid_probabilities():
    with pytest.raises(ValueError, match="identify"):
        infer_fraction(0.5, 0.3, 0.3)
    for invalid in (np.nan, np.inf, -0.1, 1.1):
        for inputs in ((invalid, 0.7, 0.2), (0.5, invalid, 0.2), (0.5, 0.7, invalid)):
            with pytest.raises(ValueError, match="probabilities"):
                infer_fraction(*inputs)


def test_source_partition_is_disjoint_exhaustive_stratified_and_reproducible():
    rows, labels = np.arange(24) + 100, np.tile([0, 1], 12)
    response, population = source_partition(rows, labels, 71)
    assert len(response) == len(population) == 12
    assert np.intersect1d(rows[response], rows[population]).size == 0
    np.testing.assert_array_equal(np.sort(np.r_[response, population]), np.arange(24))
    np.testing.assert_array_equal(np.bincount(labels[response]), [6, 6])
    again = source_partition(rows, labels, 71)
    np.testing.assert_array_equal(response, again[0])
    np.testing.assert_array_equal(population, again[1])
    with pytest.raises(ValueError, match="Unique"):
        source_partition(np.zeros(24, dtype=int), labels, 71)
    with pytest.raises(ValueError, match="two rows"):
        source_partition(np.arange(3), np.array([0, 0, 1]), 71)


def test_mixture_sampling_pairs_models_and_fluctuates_class_counts():
    q = np.ones((8, 2), dtype=bool)
    g = np.zeros((8, 2), dtype=bool)
    counts, truth = mixture_counts(q, g, 0.5, 30, 100, np.random.default_rng(12))
    np.testing.assert_array_equal(counts[:, 0], truth)
    np.testing.assert_array_equal(counts[:, 1], truth)
    assert len(np.unique(truth)) > 1
    again = mixture_counts(q, g, 0.5, 30, 100, np.random.default_rng(12))
    np.testing.assert_array_equal(counts, again[0])
    for f in (0, 1):
        accepted, labels = mixture_counts(q, g, f, 30, 5, np.random.default_rng(12))
        assert np.all(accepted == 30 * f) and np.all(labels == 30 * f)
    with pytest.raises(ValueError, match="positive integers"):
        mixture_counts(q, g, 0.5, True, 10, np.random.default_rng(12))


def test_closure_mean_targets_finite_pool_expectation_not_response_reuse():
    response_q, response_g = np.array([0.8]), np.array([0.2])
    population_q, population_g = np.array([0.7]), np.array([0.1])
    result = closure_summary(
        np.array([[40], [40]]),
        100,
        0.5,
        response_q,
        response_g,
        population_q,
        population_g,
    )[0]
    assert result["expected_fraction_fixed_pools"] == pytest.approx(1 / 3)
    assert result["expected_bias_fixed_pools"] == pytest.approx(-1 / 6)
    assert result["mean_minus_expectation"] == pytest.approx(0)
    assert result["bias"] == pytest.approx(-1 / 6)
    assert result["mean_mcse_fixed_pools"] == pytest.approx(np.sqrt(0.4 * 0.6 / (100 * 0.6**2 * 2)))


def test_source_entry_reads_completed_run_and_keeps_target_scores_closed(tmp_path, monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_source_closure")
    monkeypatch.chdir(tmp_path)
    rows = np.arange(20000)
    labels = np.tile([0, 1], 10000)
    scores = np.repeat(labels[:, None], 12, axis=1).astype(float)
    rows_dir = tmp_path / "data/processed"
    rows_dir.mkdir(parents=True)
    np.save(rows_dir / "test_rows.npy", rows)
    np.savez("source.npz", rows=rows, labels=labels, scores=scores, keys=runner.KEYS)
    run = {
        "status": "completed",
        "run_id": "fixture",
        "cut_freeze": "cut.json",
        "cut_freeze_sha256": "hash",
        "score_artifacts": {
            "test": {"path": "source.npz", "sha256": "hash"},
            "target": {"path": "target_must_remain_closed.npz", "sha256": "hash"},
        },
    }
    lock = {
        "operator": ">",
        "run_id": "fixture",
        "cuts": {k: {"threshold": 0.5} for k in runner.KEYS},
    }
    documents = {
        "results/run.json": run,
        "cut.json": lock,
        "results/task0004/validation.json": {"evidence_sha256": {"results/run.json": "hash"}},
        "results/task0001/validation.json": {
            "evidence_sha256": {"results/task0001/cohorts.json": "hash"}
        },
        "results/task0001/cohorts.json": {
            "cohorts": {"test": {"rows": 20000, "row_ids_sha256": "hash"}}
        },
    }
    monkeypatch.setattr(runner, "read_json", lambda p: documents[str(p)])
    monkeypatch.setattr(runner, "verify_hashes", lambda values: None)
    config = {"source_evaluation_run": "results/run.json"}
    loaded_rows, loaded_labels, passed, inputs = runner.source_inputs(config)
    np.testing.assert_array_equal(loaded_rows, rows)
    np.testing.assert_array_equal(loaded_labels, labels)
    np.testing.assert_array_equal(passed, scores.astype(bool))
    assert all("target" not in path for path in inputs)
    run["status"] = "cuts_frozen"
    with pytest.raises(ValueError, match="completed"):
        runner.source_inputs(config)
    run["status"] = "completed"
    lock["operator"] = ">="
    with pytest.raises(ValueError, match="strict frozen cuts"):
        runner.source_inputs(config)
