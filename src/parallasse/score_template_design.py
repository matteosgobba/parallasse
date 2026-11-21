"""Prospective score-template design, symmetric data roles and reservation checks.

This module audits identities and freezes inputs. It does not compute classifier
scores or implement the final statistical inference engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import binom

from parallasse.data import class_counts, unique_rows
from parallasse.heldout import read_json, verify_hashes
from parallasse.mixtures import source_partition

GENERATORS = ("pythia", "herwig")
CHECKPOINTS = tuple(f"{m}-{s}" for m in ("PFN-K", "PFN-ID") for s in (17, 29, 43))


def calibration_rank(repetitions=1000, probability=0.95, family_error=0.05, cells=144):
    """One-based upper tolerance order statistic, with family Monte Carlo error."""
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or isinstance(cells, bool)
        or not isinstance(cells, int)
        or cells < 1
        or not 0 < probability < 1
        or not 0 < family_error < 1
    ):
        raise ValueError("Positive counts and probabilities strictly inside (0,1) required")
    rank = int(binom.ppf(1 - family_error / cells, repetitions, probability)) + 1
    if rank > repetitions:
        raise ValueError("Too few calibration draws for the declared tolerance bound")
    return rank


def validate_design(config):
    """Fail closed on unsupported cohort/model/grid or inconsistent work counts."""
    if (
        config["protocol_version"] != "0.9-score-template-design"
        or config["with_bc"] is not False
        or config["models"] != ["PFN-K", "PFN-ID"]
        or config["seeds"] != [17, 29, 43]
        or [spec["generator"] for spec in config["files"]] != list(GENERATORS)
        or any("withbc" in spec["filename"] for spec in config["files"])
        or config["grid"]["fractions"] != [0.25, 0.5, 0.75]
        or config["grid"]["sample_sizes"] != [1000, 10000]
        or config["grid"]["modelling_variations"] != [0.0, 0.5, 1.0]
        or config["grid"]["repetitions"] != 1000
        or config["grid"]["primary_fraction"] != 0.5
        or config["grid"]["primary_sample_size"] != 10000
        or config["grid"]["primary_generator"] != "herwig"
        or config["binning"]["bins"] != 8
    ):
        raise ValueError("Unsupported prospective score-template design")
    cal, resources = config["calibration"], config["resources"]
    if (
        (cal["source_cells"], cal["simulation_cells"], cal["family_cells"]) != (36, 108, 144)
        or resources["source_confidence_sets"] != 108000
        or resources["simulation_confidence_sets"] != 180000
        or resources["source_calibration_lr_pairs"] != 36000
        or resources["simulation_calibration_lr_pairs"] != 108000
    ):
        raise ValueError("Calibration and execution counts disagree with the declared grid")
    seeds = list(config["cohorts"]["partition_seeds"].values()) + [
        value for value in config["rng"].values() if isinstance(value, int)
    ]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Partition, calibration, validation and evaluation roots must differ")
    return calibration_rank(
        cal["repetitions_per_cell"],
        cal["nominal_coverage"],
        cal["family_quantile_error"],
        cal["family_cells"],
    )


def reserve_template_roles(
    labels, identities, prior_labels, prior_identities, seeds, minimum_per_class=20000
):
    """Deduplicate/exclude prior identities, then split both generators identically."""
    if set(labels) != set(GENERATORS) or set(identities) != set(GENERATORS):
        raise ValueError("Both declared generators are required")
    for name in GENERATORS:
        if not np.all(np.isin(labels[name], [0, 1])):
            raise ValueError("Binary source/target labels required")
    unique_rows(
        np.concatenate([prior_labels, *[labels[g] for g in GENERATORS]]),
        np.concatenate([prior_identities, *[identities[g] for g in GENERATORS]]),
    )
    arrays, counts = {}, {}
    earlier = np.asarray(prior_identities)
    for generator in GENERATORS:
        y, h = labels[generator], identities[generator]
        unique = unique_rows(y, h)
        retained = unique[~np.isin(h[unique], earlier)]
        template, evaluation = source_partition(retained, y[retained], seeds[generator])
        arrays[f"{generator}_retained_rows"] = retained
        arrays[f"{generator}_retained_labels"] = y[retained]
        role_counts = {}
        for role, indices in (("template", template), ("evaluation", evaluation)):
            rows = retained[indices]
            role_y = y[rows]
            if any(np.sum(role_y == label) < minimum_per_class for label in (0, 1)):
                raise ValueError("Too few retained rows per class/role; revise before scoring")
            arrays[f"{generator}_{role}_rows"] = rows
            arrays[f"{generator}_{role}_labels"] = role_y
            role_counts[role] = {"rows": len(rows), "class_counts": class_counts(role_y)}
        counts[generator] = {
            "input_rows": len(y),
            "retained_rows": len(retained),
            "within_duplicate_extra_rows": len(y) - len(unique),
            "prior_or_preceding_generator_overlap_rows": len(unique) - len(retained),
            "removed_rows": len(y) - len(retained),
            "roles": role_counts,
        }
        earlier = np.concatenate([earlier, h[retained]])
    return arrays, counts


def verify_reservation(freeze_path: Path):
    """Verify prospective design/data bytes; this grants no scoring access."""
    freeze = read_json(freeze_path)
    if (
        freeze.get("state") != "design_and_roles_frozen"
        or freeze.get("fresh_scores_computed") is not False
        or freeze.get("executable_inference_freeze") != "required_in_TASK-0012"
    ):
        raise ValueError("A completed design/role freeze is required, not a scoring grant")
    verify_hashes(freeze["input_sha256"])
    verify_hashes(freeze["output_sha256"])
    verify_hashes(freeze["audit_source_sha256"], Path(freeze["audit_source_snapshot"]))
    for item in freeze["archives"].values():
        verify_hashes({item["path"]: item["sha256"]})
    config = read_json(Path(freeze["protocol_path"]))
    validate_design(config)
    return freeze
