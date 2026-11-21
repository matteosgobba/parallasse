"""Verify integrity, freeze row IDs, and define training-only preprocessing."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from parallasse.data import (
    CANONICAL_VERSION,
    class_counts,
    fingerprints,
    integrity_summary,
    load_jets,
    save_array_once,
    source_splits,
    unique_rows,
    verify_cohorts,
)
from parallasse.download import md5_file, sha256_file
from parallasse.preprocessing import fit_pid_vocabulary


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/task0001"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["with_bc"] is not False:
        raise ValueError("This protocol uses light-quark files only")
    started = time.monotonic()
    all_hashes, all_labels, summaries, raw_sha256 = {}, {}, {}, {}
    specs = {spec["role"]: spec for spec in config["files"]}
    if set(specs) != {"source", "target"} or len(config["files"]) != 2:
        raise ValueError("Expected exactly one declared source and one target file")
    for role in ("source", "target"):
        spec = specs[role]
        path = args.raw_dir / spec["filename"]
        if md5_file(path) != spec["published_md5"]:
            raise ValueError(f"Checksum changed: {path}")
        particles, labels = load_jets(path)
        if len(labels) != spec["expected_rows"]:
            raise ValueError("Unexpected sample count")
        hashes = fingerprints(particles)
        summaries[role] = integrity_summary(particles, labels, hashes)
        all_hashes[role], all_labels[role] = hashes, labels
        raw_sha256[role] = sha256_file(path)
        save_array_once(args.processed_dir / f"{spec['generator']}_fingerprints.npy", hashes)
        print(f"Audited {role}: {len(labels)} rows", flush=True)
        del particles
    source_hashes, target_hashes = all_hashes["source"], all_hashes["target"]
    overlap = np.intersect1d(source_hashes, target_hashes)
    splits = source_splits(
        all_labels["source"], source_hashes, config["source_fractions"], config["split_seed"]
    )
    target_unique = unique_rows(all_labels["target"], target_hashes)
    target_rows = target_unique[~np.isin(target_hashes[target_unique], source_hashes)]
    verify_cohorts(splits, source_hashes, target_rows, target_hashes)
    cohorts = {}
    for name, rows in {**splits, "target": target_rows}.items():
        role = "target" if name == "target" else "source"
        path = args.processed_dir / f"{name}_rows.npy"
        save_array_once(path, rows)
        cohorts[name] = {
            "generator": specs[role]["generator"],
            "original_file": specs[role]["filename"],
            "record_id": specs[role]["record_id"],
            "row_ids_path": path.as_posix(),
            "row_ids_sha256": sha256_file(path),
            "rows": len(rows),
            "class_counts": class_counts(all_labels[role][rows]),
        }
    source, _ = load_jets(args.raw_dir / specs["source"]["filename"])
    vocabulary = fit_pid_vocabulary(source[splits["train"]])
    del source
    write_json(
        args.output_dir / "preprocessing.json",
        {
            "schema_version": 1,
            "protocol_version": config["protocol_version"],
            "vocabulary_fit_cohort": "train",
            "vocabulary_fit_row_ids_sha256": cohorts["train"]["row_ids_sha256"],
            "pid_vocabulary": vocabulary,
            "padding_token": 0,
            "unknown_or_identity_off_token": 1,
            "known_tokens": "sorted vocabulary position + 2",
            "particle_inputs": [
                "pt/sum_pt",
                "y-pt_weighted_mean_y",
                "wrapped_phi-circular_mean_phi",
            ],
            "azimuth_center": "atan2(sum(z*sin(phi)), sum(z*cos(phi)))",
            "azimuth_wrapping": "atan2(sin(delta_phi), cos(delta_phi))",
            "ambiguous_center_policy": "reject circular resultant magnitude below 1e-12",
            "accumulation_dtype": "float64",
            "output_dtype": "float32",
            "truncation": "none; retain all real constituents",
            "padding_policy": "all transformed values and tokens zero; mask encoder outputs",
            "learned_scaling": (
                "none in Phase 1; aggregate scaling may be fit on train only in Phase 2"
            ),
        },
    )
    write_json(
        args.output_dir / "cohorts.json",
        {
            "schema_version": 1,
            "protocol_version": config["protocol_version"],
            "split_seed": config["split_seed"],
            "rng": "numpy.random.Generator(PCG64)",
            "numpy_version": np.__version__,
            "fractions": config["source_fractions"],
            "rounding": "per-class largest remainder; ties in train/tune/calibration/test order",
            "config_sha256": sha256_file(args.config),
            "raw_sha256": raw_sha256,
            "duplicate_policy": config["cohort_policy"],
            "cohorts": cohorts,
        },
    )
    write_json(
        args.output_dir / "audit.json",
        {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_version": config["protocol_version"],
            "canonical_definition": CANONICAL_VERSION,
            "datasets": summaries,
            "cross_generator_exact_overlap_groups": int(len(overlap)),
            "target_rows_excluded_for_source_overlap": int(len(target_unique) - len(target_rows)),
            "source_cohorts_disjoint": True,
            "source_target_disjoint_after_policy": True,
            "target_use": (
                "integrity and provenance only; no model scores or label-conditioned features"
            ),
            "audit_wall_seconds": round(time.monotonic() - started, 3),
            "raw_sha256": raw_sha256,
        },
    )
    print(json.dumps({name: entry["rows"] for name, entry in cohorts.items()}), flush=True)


if __name__ == "__main__":
    main()
