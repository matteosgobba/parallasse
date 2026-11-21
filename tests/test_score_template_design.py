import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from parallasse.download import sha256_file
from parallasse.score_template_design import (
    calibration_rank,
    reserve_template_roles,
    validate_design,
    verify_reservation,
)


def fixtures():
    labels = {g: np.tile([0, 1], 12) for g in ("pythia", "herwig")}
    hashes = {g: np.array([f"{g}{i}" for i in range(24)], dtype="S64") for g in labels}
    return labels, hashes


def test_both_generators_have_disjoint_deterministic_template_and_evaluation_roles():
    labels, hashes = fixtures()
    seeds = {"pythia": 17, "herwig": 29}
    arrays, counts = reserve_template_roles(
        labels, hashes, np.array([]), np.array([], dtype="S64"), seeds, minimum_per_class=2
    )
    for generator in labels:
        t, e = [arrays[f"{generator}_{role}_rows"] for role in ("template", "evaluation")]
        assert not np.intersect1d(t, e).size
        np.testing.assert_array_equal(np.sort(np.r_[t, e]), np.arange(24))
        rng = np.random.default_rng(seeds[generator])
        expected = [rng.permutation(np.flatnonzero(labels[generator] == k))[:6] for k in (0, 1)]
        np.testing.assert_array_equal(t, np.sort(np.concatenate(expected)))
        assert counts[generator]["roles"]["template"]["class_counts"] == {
            "gluon_0": 6,
            "quark_1": 6,
        }


def test_prior_duplicates_cross_generator_overlap_and_label_conflicts():
    labels, hashes = fixtures()
    hashes["pythia"][2] = hashes["pythia"][0]
    hashes["herwig"][4] = hashes["pythia"][4]
    prior = np.array([hashes["pythia"][0], hashes["herwig"][1]], dtype="S64")
    arrays, counts = reserve_template_roles(
        labels, hashes, np.array([0, 1]), prior, {"pythia": 17, "herwig": 29}, minimum_per_class=2
    )
    assert counts["pythia"]["removed_rows"] == 2
    assert counts["herwig"]["removed_rows"] == 2
    for generator in labels:
        retained = hashes[generator][arrays[f"{generator}_retained_rows"]]
        assert not np.intersect1d(retained, prior).size
    assert not np.intersect1d(
        hashes["pythia"][arrays["pythia_retained_rows"]],
        hashes["herwig"][arrays["herwig_retained_rows"]],
    ).size
    with pytest.raises(ValueError, match="conflicting labels"):
        reserve_template_roles(
            labels,
            hashes,
            np.array([1, 1]),
            prior,
            {"pythia": 17, "herwig": 29},
            minimum_per_class=2,
        )
    with pytest.raises(ValueError, match="Too few"):
        reserve_template_roles(
            labels,
            hashes,
            np.array([0, 1]),
            prior,
            {"pythia": 17, "herwig": 29},
            minimum_per_class=20,
        )


def test_calibration_order_statistic_is_minimal_conservative_family_bound():
    rank = calibration_rank()
    assert binom.cdf(rank - 1, 1000, 0.95) >= 1 - 0.05 / 144
    assert binom.cdf(rank - 2, 1000, 0.95) < 1 - 0.05 / 144
    with pytest.raises(ValueError, match="Too few"):
        calibration_rank(repetitions=10)
    with pytest.raises(ValueError, match="Positive"):
        calibration_rank(repetitions=True)
    config = json.loads(Path("configs/score_templates.json").read_text())
    assert validate_design(config) == rank
    config["rng"]["source_validation"] = config["rng"]["source_calibration"]
    with pytest.raises(ValueError, match="must differ"):
        validate_design(config)


def test_reservation_rejects_tampered_role_bytes_and_never_grants_scoring(tmp_path):
    role = tmp_path / "cohorts.npz"
    role.write_bytes(b"original role bytes")
    config = tmp_path / "protocol.json"
    config.write_text(Path("configs/score_templates.json").read_text())
    freeze = {
        "state": "design_and_roles_frozen",
        "fresh_scores_computed": False,
        "executable_inference_freeze": "required_in_TASK-0012",
        "input_sha256": {},
        "output_sha256": {str(config): sha256_file(config)},
        "audit_source_sha256": {},
        "audit_source_snapshot": str(tmp_path),
        "protocol_path": str(config),
        "archives": {"cohorts": {"path": str(role), "sha256": sha256_file(role)}},
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze))
    assert verify_reservation(path)["fresh_scores_computed"] is False
    role.write_bytes(b"different role bytes")
    with pytest.raises(ValueError, match="Frozen bytes"):
        verify_reservation(path)
    freeze["fresh_scores_computed"] = True
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="not a scoring grant"):
        verify_reservation(path)


def test_completed_reservation_refuses_reentry_before_network_access(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("reserve_score_template_data")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/task0011/completed").mkdir(parents=True)
    monkeypatch.setattr(
        runner, "fetch_record", lambda *args: pytest.fail("Network access forbidden")
    )
    with pytest.raises(FileExistsError, match="never overwritten"):
        runner.reserve("completed")
