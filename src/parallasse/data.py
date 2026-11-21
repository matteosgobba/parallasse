"""Integrity, exact jet identity, and persisted cohort construction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

COHORTS = ("train", "tune", "calibration", "test")
CANONICAL_VERSION = "jet-sha256-v1:real-positive-pt;float64-le;zero-normalized;sorted-rows"


def load_jets(path: Path) -> tuple[NDArray, NDArray]:
    """Read numeric arrays only; pickle loading is deliberately disabled."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"X", "y"}:
            raise ValueError(f"Unexpected arrays in {path.name}: {archive.files}")
        particles, labels = archive["X"], archive["y"]
    validate_jets(particles, labels)
    return particles, labels


def particle_mask(particles: NDArray) -> NDArray:
    if particles.ndim != 3 or particles.shape[-1] != 4:
        raise ValueError("Particles must have shape (jets, padded_particles, 4)")
    if not np.issubdtype(particles.dtype, np.number):
        raise ValueError("Particle arrays must be numeric")
    if not np.all(np.isfinite(particles)):
        raise ValueError("Non-finite particle or padding entry")
    mask = particles[..., 0] > 0
    if np.any(particles[~mask] != 0):
        raise ValueError("Nonpositive-pt entries must be entirely zero padding")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("Empty jet")
    pids = particles[..., 3][mask]
    if np.any(pids != np.rint(pids)) or np.any(np.abs(pids) > np.iinfo(np.int32).max):
        raise ValueError("Constituent PID must be a finite int32-valued category")
    return mask


def validate_jets(particles: NDArray, labels: NDArray) -> NDArray:
    mask = particle_mask(particles)
    if labels.shape != (particles.shape[0],):
        raise ValueError("One scalar label is required per jet")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be quark=1 or gluon=0")
    return mask


def canonical_jet_hash(particles: NDArray) -> str:
    """Ignore ordering/padding; preserve exact represented numerical values.

    There is no rounding, tolerance, rotation, or attempt to identify physically
    equivalent but numerically different events. The dtype conversion unifies
    float32/float64 only when their represented values really are identical.
    """
    real = np.asarray(particles[particles[:, 0] > 0], dtype="<f8").copy()
    real[real == 0] = 0.0  # Normalize IEEE signed zeros.
    order = np.lexsort(tuple(real[:, column] for column in (3, 2, 1, 0)))
    real = np.ascontiguousarray(real[order], dtype="<f8")
    payload = b"parallasse-jet-v1\0" + len(real).to_bytes(8, "little") + real.tobytes()
    return hashlib.sha256(payload).hexdigest()


def fingerprints(particles: NDArray) -> NDArray:
    particle_mask(particles)
    return np.asarray([canonical_jet_hash(jet) for jet in particles], dtype="S64")


def unique_rows(labels: NDArray, hashes: NDArray) -> NDArray:
    """Retain the lowest original row per exact group; reject label conflicts."""
    if len(labels) != len(hashes):
        raise ValueError("Fingerprint/label count mismatch")
    _, first, inverse, counts = np.unique(
        hashes, return_index=True, return_inverse=True, return_counts=True
    )
    for group in np.flatnonzero(counts > 1):
        if np.unique(labels[inverse == group]).size != 1:
            raise ValueError("Identical constituent records have conflicting labels")
    return np.sort(first).astype(np.int64)


def integrity_summary(particles: NDArray, labels: NDArray, hashes: NDArray) -> dict:
    mask = validate_jets(particles, labels)
    unique = unique_rows(labels, hashes)
    _, group_sizes = np.unique(hashes, return_counts=True)
    pids = np.unique(particles[..., 3][mask]).astype(np.int64)
    counts = mask.sum(axis=1)
    return {
        "rows": len(labels),
        "shape": list(particles.shape),
        "particle_dtype": str(particles.dtype),
        "label_dtype": str(labels.dtype),
        "class_counts": class_counts(labels),
        "real_constituents": int(mask.sum()),
        "padding_slots": int((~mask).sum()),
        "min_constituents_per_jet": int(counts.min()),
        "max_constituents_per_jet": int(counts.max()),
        "pid_categories_integrity_only": pids.tolist(),
        "all_entries_finite": True,
        "padding_all_zero": True,
        "all_real_pt_positive": True,
        "empty_jets": 0,
        "exact_duplicate_groups": int(np.sum(group_sizes > 1)),
        "exact_duplicate_extra_rows": int(len(labels) - len(unique)),
        "unique_rows": int(len(unique)),
        "conflicting_label_groups": 0,
        "canonical_definition": CANONICAL_VERSION,
    }


def class_counts(labels: NDArray) -> dict[str, int]:
    return {"gluon_0": int(np.sum(labels == 0)), "quark_1": int(np.sum(labels == 1))}


def source_splits(labels: NDArray, hashes: NDArray, fractions: dict, seed: int) -> dict:
    """Stratify unique jets; distribute integer remainders deterministically."""
    if set(fractions) != set(COHORTS):
        raise ValueError("Expected train/tune/calibration/test fractions")
    weights = np.asarray([fractions[name] for name in COHORTS], dtype=np.float64)
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Fractions must be positive and sum to one")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Invalid binary labels")
    rows = unique_rows(labels, hashes)
    rng = np.random.Generator(np.random.PCG64(seed))
    split_lists = {name: [] for name in COHORTS}
    for label in (0, 1):
        candidates = rng.permutation(rows[labels[rows] == label])
        exact = len(candidates) * weights
        counts = np.floor(exact).astype(int)
        remainder_order = np.argsort(-(exact - counts), kind="stable")
        counts[remainder_order[: len(candidates) - int(counts.sum())]] += 1
        for name, indices in zip(
            COHORTS, np.split(candidates, np.cumsum(counts)[:-1]), strict=True
        ):
            split_lists[name].extend(indices.tolist())
    return {
        name: rng.permutation(np.asarray(split_lists[name], dtype=np.int64)) for name in COHORTS
    }


def verify_cohorts(
    splits: dict, source_hashes: NDArray, target_rows: NDArray, target_hashes: NDArray
) -> None:
    source_ids = np.concatenate(list(splits.values()))
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("Source rows occur in more than one cohort")
    selected_hashes = source_hashes[source_ids]
    if len(np.unique(selected_hashes)) != len(selected_hashes):
        raise ValueError("Duplicate source groups cross cohorts or remain undeduplicated")
    if len(np.unique(target_rows)) != len(target_rows):
        raise ValueError("Duplicate target row IDs")
    if len(np.unique(target_hashes[target_rows])) != len(target_rows):
        raise ValueError("Duplicate target constituent records")
    if np.intersect1d(selected_hashes, target_hashes[target_rows]).size:
        raise ValueError("Source/target constituent overlap")


def save_array_once(path: Path, array: NDArray) -> None:
    """Re-runs may verify cohort arrays, never silently change their identities."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), array):
            raise ValueError(f"Existing cohort artifact would change: {path}")
        return
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
