"""New-event cohort isolation and frozen preprocessing for replication."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from parallasse.data import unique_rows
from parallasse.features import aggregate_features
from parallasse.mixtures import source_partition
from parallasse.preprocessing import encode_pid, kinematics


def replication_cohorts(
    source_labels,
    source_hashes,
    target_labels,
    target_hashes,
    previous_labels,
    previous_hashes,
    seed: int,
) -> tuple[dict, dict]:
    """Exclude prior identities before assigning source roles and target membership."""
    # Check conflicting labels even for identities that will later be excluded.
    unique_rows(
        np.concatenate([previous_labels, source_labels, target_labels]),
        np.concatenate([previous_hashes, source_hashes, target_hashes]),
    )
    source = unique_rows(source_labels, source_hashes)
    source = source[~np.isin(source_hashes[source], previous_hashes)]
    target = unique_rows(target_labels, target_hashes)
    target = target[~np.isin(target_hashes[target], previous_hashes)]
    target = target[~np.isin(target_hashes[target], source_hashes[source])]
    response, population = source_partition(source, source_labels[source], seed)
    if not len(target) or not all(np.any(target_labels[target] == k) for k in (0, 1)):
        raise ValueError("Both target classes must remain after the identity audit")
    cohorts = {
        "source_rows": source,
        "response_indices": response,
        "population_indices": population,
        "response_rows": source[response],
        "population_rows": source[population],
        "source_labels": source_labels[source],
        "target_rows": target,
        "target_labels": target_labels[target],
    }
    counts = {
        "source_retained": len(source),
        "target_retained": len(target),
        "source_removed": len(source_labels) - len(source),
        "target_removed": len(target_labels) - len(target),
        "response_quarks": int(np.sum(source_labels[source[response]] == 1)),
        "response_gluons": int(np.sum(source_labels[source[response]] == 0)),
        "population_quarks": int(np.sum(source_labels[source[population]] == 1)),
        "population_gluons": int(np.sum(source_labels[source[population]] == 0)),
        "target_quarks": int(np.sum(target_labels[target] == 1)),
        "target_gluons": int(np.sum(target_labels[target] == 0)),
    }
    return cohorts, counts


def prepare_replication_inputs(raw: Path, rows: np.ndarray, frozen: dict, destination: Path):
    """Reuse the original feature functions and training-only PID vocabulary."""
    rows = np.asarray(rows)
    if (
        rows.ndim != 1
        or not np.issubdtype(rows.dtype, np.integer)
        or len(np.unique(rows)) != len(rows)
    ):
        raise ValueError("Expected unique integer raw row IDs")
    with np.load(raw, allow_pickle=False) as data:
        particles, labels = data["X"], data["y"]
    if not len(rows) or np.any(rows < 0) or np.any(rows >= len(labels)):
        raise ValueError("Replication row IDs outside the raw file")
    n, width = len(rows), particles.shape[1]
    aggregate = np.empty((n, 6), dtype=np.float32)
    coordinates = np.empty((n, width, 3), dtype=np.float32)
    mask = np.empty((n, width), dtype=bool)
    pid = np.empty((n, width), dtype=np.int16)
    vocabulary = frozen["pid_contract"]["pid_vocabulary"]
    for start in range(0, n, 4096):
        section = slice(start, start + 4096)
        chunk = particles[rows[section]]
        coordinates[section], mask[section] = kinematics(chunk)
        pid[section] = encode_pid(chunk, vocabulary)
        aggregate[section] = aggregate_features(chunk)
    with destination.open("xb") as handle:
        np.savez(handle, aggregate=aggregate, coordinates=coordinates, mask=mask, pid=pid)
    return labels[rows]
