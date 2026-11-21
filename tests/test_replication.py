import json
from pathlib import Path

import numpy as np
import pytest

from parallasse.download import sha256_file
from parallasse.features import aggregate_features
from parallasse.preprocessing import encode_pid, kinematics
from parallasse.replication import prepare_replication_inputs, replication_cohorts


def test_new_cohorts_exclude_previous_and_cross_generator_identities():
    source_labels = np.tile([0, 1], 10)
    source_hashes = np.array([f"s{i}" for i in range(20)], dtype="S64")
    source_hashes[2] = source_hashes[0]
    target_labels = np.tile([0, 1], 6)
    target_hashes = np.array([f"t{i}" for i in range(12)], dtype="S64")
    target_hashes[0] = source_hashes[4]
    previous_hashes = np.array([source_hashes[0], target_hashes[3]], dtype="S64")
    previous_labels = np.array([0, 1])
    cohorts, counts = replication_cohorts(
        source_labels,
        source_hashes,
        target_labels,
        target_hashes,
        previous_labels,
        previous_hashes,
        17,
    )
    assert counts["source_removed"] == 2 and counts["target_removed"] == 2
    source = source_hashes[cohorts["source_rows"]]
    target = target_hashes[cohorts["target_rows"]]
    assert not np.intersect1d(source, previous_hashes).size
    assert not np.intersect1d(target, previous_hashes).size
    assert not np.intersect1d(source, target).size
    np.testing.assert_array_equal(
        np.sort(np.r_[cohorts["response_rows"], cohorts["population_rows"]]), cohorts["source_rows"]
    )
    assert not np.intersect1d(cohorts["response_rows"], cohorts["population_rows"]).size
    assert counts["response_quarks"] + counts["population_quarks"] == 10


def test_audit_rejects_conflicting_labels_even_for_previously_seen_identities():
    labels = np.tile([0, 1], 4)
    hashes = np.array([f"s{i}" for i in range(8)], dtype="S64")
    with pytest.raises(ValueError, match="conflicting labels"):
        replication_cohorts(
            labels,
            hashes,
            labels,
            np.array([f"t{i}" for i in range(8)], dtype="S64"),
            np.array([1]),
            np.array([hashes[0]], dtype="S64"),
            17,
        )


def test_new_input_preparation_reuses_frozen_features_and_pid_with_original_row_order(tmp_path):
    particles = np.array(
        [
            [[3, 0.1, 3.12, 211], [1, 0.2, -3.12, 22], [0, 0, 0, 0]],
            [[2, -0.2, 0.3, -211], [2, 0.1, 0.4, 130], [0, 0, 0, 0]],
        ],
        dtype=float,
    )
    labels = np.array([1, 0])
    raw = tmp_path / "new.npz"
    np.savez(raw, X=particles, y=labels)
    rows = np.array([1, 0])
    frozen = {"pid_contract": {"pid_vocabulary": [22, 211]}}
    output = tmp_path / "inputs.npz"
    returned = prepare_replication_inputs(raw, rows, frozen, output)
    np.testing.assert_array_equal(returned, labels[rows])
    with np.load(output) as a:
        expected, mask = kinematics(particles[rows])
        np.testing.assert_array_equal(a["coordinates"], expected)
        np.testing.assert_array_equal(a["mask"], mask)
        np.testing.assert_array_equal(a["aggregate"], aggregate_features(particles[rows]))
        np.testing.assert_array_equal(a["pid"], encode_pid(particles[rows], [22, 211]))
        assert np.all(a["pid"][0, :2] == 1)
    with pytest.raises(FileExistsError):
        prepare_replication_inputs(raw, rows, frozen, output)
    with pytest.raises(ValueError, match="outside"):
        prepare_replication_inputs(raw, np.array([2]), frozen, tmp_path / "invalid.npz")


def test_replication_stage_rejects_unvalidated_state_and_changed_cohort_bytes(
    tmp_path, monkeypatch
):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("replicate_mixture_inference")
    record = {"status": "audited"}
    run = tmp_path / "run.json"
    run.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="source_validated"):
        runner.verified_stage(tmp_path, "source_validated")
    archive = tmp_path / "cohorts.npz"
    archive.write_bytes(b"original")
    reference = {"path": str(archive), "sha256": sha256_file(archive)}
    record.update(
        input_sha256={},
        output_sha256={},
        source_files_sha256={},
        source_snapshot=str(tmp_path),
        cohorts=reference,
        identities=reference,
    )
    run.write_text(json.dumps(record))
    assert runner.verified_stage(tmp_path, "audited")["status"] == "audited"
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen bytes"):
        runner.verified_stage(tmp_path, "audited")
