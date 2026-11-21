import numpy as np
import pytest

from parallasse.data import (
    canonical_jet_hash,
    load_jets,
    particle_mask,
    save_array_once,
    source_splits,
    unique_rows,
    validate_jets,
    verify_cohorts,
)


@pytest.fixture
def particles():
    return np.array(
        [
            [[3, 0.1, 3.12, 211], [1, 0.2, -3.12, 22], [0, 0, 0, 0]],
            [[2, -0.2, 0.3, -211], [2, 0.1, 0.4, 130], [0, 0, 0, 0]],
        ],
        dtype=np.float64,
    )


def test_hash_is_order_padding_dtype_and_signed_zero_invariant():
    jet = np.array([[1, 0, -0.0, 22], [2, 0.5, 0.25, 211]], dtype=np.float32)
    reordered = np.pad(jet[::-1].astype(np.float64), ((0, 3), (0, 0)))
    reordered[reordered == 0] = -0.0
    assert canonical_jet_hash(jet) == canonical_jet_hash(reordered)
    changed = jet.astype(np.float64)
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    assert canonical_jet_hash(jet) != canonical_jet_hash(changed)


def test_duplicate_policy_rejects_label_conflicts():
    hashes = np.array([b"a", b"b", b"a"])
    np.testing.assert_array_equal(unique_rows(np.array([1, 0, 1]), hashes), [0, 1])
    with pytest.raises(ValueError, match="conflicting labels"):
        unique_rows(np.array([1, 0, 0]), hashes)


def test_source_splits_are_stratified_stable_and_group_isolated():
    labels = np.tile([0, 1], 100)
    hashes = np.asarray([f"jet-{i}" for i in range(200)], dtype="S64")
    labels = np.append(labels, labels[0])
    hashes = np.append(hashes, hashes[0])
    fractions = dict(train=0.6, tune=0.1, calibration=0.1, test=0.2)
    splits = source_splits(labels, hashes, fractions, 20261104)
    repeat = source_splits(labels, hashes, fractions, 20261104)
    assert [len(splits[name]) for name in fractions] == [120, 20, 20, 40]
    for name, rows in splits.items():
        np.testing.assert_array_equal(rows, repeat[name])
        assert int(labels[rows].sum()) == len(rows) // 2
        assert 200 not in rows
    verify_cohorts(splits, hashes, np.array([0]), np.array([b"different"]))
    with pytest.raises(ValueError, match="Source/target constituent overlap"):
        verify_cohorts(splits, hashes, np.array([0]), hashes[:1])
    with pytest.raises(ValueError, match="more than one cohort"):
        verify_cohorts(
            {"a": np.array([0]), "b": np.array([0])},
            hashes,
            np.array([0]),
            np.array([b"different"]),
        )


@pytest.mark.parametrize("kind", ["nan", "negative_pt", "dirty_padding", "empty", "pid_fraction"])
def test_invalid_particles_are_rejected(particles, kind):
    if kind == "nan":
        particles[0, 0, 1] = np.nan
    elif kind == "negative_pt":
        particles[0, 0, 0] = -1
    elif kind == "dirty_padding":
        particles[0, 2, 1] = 0.1
    elif kind == "empty":
        particles[0] = 0
    else:
        particles[0, 0, 3] = 0.5
    with pytest.raises(ValueError):
        particle_mask(particles)


def test_invalid_label_convention_is_rejected(particles):
    with pytest.raises(ValueError, match="quark=1 or gluon=0"):
        validate_jets(particles, np.array([1, -1]))


def test_loader_preserves_original_dtypes_and_validates_keys(particles, tmp_path):
    path = tmp_path / "valid.npz"
    np.savez(path, X=particles, y=np.array([1, 0], dtype=np.float32))
    x, y = load_jets(path)
    assert x.dtype == np.float64 and y.dtype == np.float32
    np.savez(path, X=particles, y=np.array([1, 0]), unexpected=np.array([1]))
    with pytest.raises(ValueError, match="Unexpected arrays"):
        load_jets(path)


def test_persisted_rows_cannot_be_silently_changed(tmp_path):
    path = tmp_path / "rows.npy"
    save_array_once(path, np.array([7, 2, 3]))
    save_array_once(path, np.array([7, 2, 3]))
    with pytest.raises(ValueError, match="would change"):
        save_array_once(path, np.array([2, 7, 3]))


def test_fraction_sum_is_validated():
    with pytest.raises(ValueError, match="sum to one"):
        source_splits(
            np.array([0, 1]),
            np.array([b"a", b"b"]),
            dict(train=0.6, tune=0.1, calibration=0.1, test=0.3),
            1,
        )
