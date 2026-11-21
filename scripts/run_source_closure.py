"""Validate mixture inference using the frozen Pythia scores and disjoint pools."""

import argparse
import csv
import platform
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from evaluate_heldout import save_arrays, write_json

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.heldout_metrics import KEYS, checked_scores
from parallasse.mixtures import (
    closure_summary,
    infer_fraction,
    mixture_counts,
    source_partition,
)

CONFIG = Path("configs/mixtures.json")


def source_inputs(config: dict) -> tuple:
    run_path = Path(config["source_evaluation_run"])
    previous = read_json(Path("results/task0004/validation.json"))["evidence_sha256"]
    verify_hashes({str(run_path): previous[str(run_path)]})
    run = read_json(run_path)
    if run["status"] != "completed":
        raise ValueError("Source closure requires the completed evaluation")
    cut_path = Path(run["cut_freeze"])
    verify_hashes({str(cut_path): run["cut_freeze_sha256"]})
    lock = read_json(cut_path)
    if lock["operator"] != ">" or lock["run_id"] != run["run_id"] or set(lock["cuts"]) != set(KEYS):
        raise ValueError("Expected the original strict frozen cuts for all checkpoints")
    cuts = {key: item["threshold"] for key, item in lock["cuts"].items()}
    source = run["score_artifacts"]["test"]
    rows_path = Path("data/processed/test_rows.npy")
    cohorts_path = Path("results/task0001/cohorts.json")
    validation = read_json(Path("results/task0001/validation.json"))
    verify_hashes({str(cohorts_path): validation["evidence_sha256"][str(cohorts_path)]})
    cohort = read_json(cohorts_path)["cohorts"]["test"]
    hashes = {
        str(run_path): previous[str(run_path)],
        str(cut_path): run["cut_freeze_sha256"],
        source["path"]: source["sha256"],
        str(cohorts_path): validation["evidence_sha256"][str(cohorts_path)],
        str(rows_path): cohort["row_ids_sha256"],
    }
    verify_hashes(hashes)
    with np.load(source["path"], allow_pickle=False) as archive:
        rows, labels, scores = (archive[k] for k in ("rows", "labels", "scores"))
        if tuple(archive["keys"]) != KEYS or scores.shape != (
            cohort["rows"],
            len(KEYS),
        ):
            raise ValueError("Unexpected model keys or source score shape")
    if not np.array_equal(rows, np.load(rows_path, allow_pickle=False)):
        raise ValueError("Source score IDs differ from the persisted test IDs")
    if labels.shape != rows.shape or any(np.sum(labels == k) != 10000 for k in (0, 1)):
        raise ValueError("Expected the frozen balanced Pythia final-test cohort")
    for column in scores.T:
        checked_scores(column)
    return rows, labels, scores > np.array([cuts[k] for k in KEYS]), hashes


def run_closure(run_id: str) -> None:
    output = Path("results/task0006") / run_id
    artifacts = Path("artifacts/task0006") / run_id
    if output.exists() or artifacts.exists():
        raise FileExistsError("Closure run already exists; evidence is never overwritten")
    config = read_json(CONFIG)
    if (
        config["protocol_version"] != "0.5-mixture-inference"
        or config["numerical_singularity_tolerance"] != 1e-12
        or tuple(config["seeds"]) != (17, 29, 43)
    ):
        raise ValueError("Unsupported mixture protocol")
    rows, labels, passed, inputs = source_inputs(config)
    output.mkdir(parents=True)
    record = {
        "task": "TASK-0006",
        "run_id": run_id,
        "protocol_version": config["protocol_version"],
        "status": "preparing",
        "started_utc": datetime.now(UTC).isoformat(),
        "interpretation": "Exploratory Pythia closure conditional on fixed observed pools and cuts",
        "python": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "input_sha256": inputs,
        "target_performance_opened": False,
        "new_fits": 0,
        "new_inference_calls": 0,
    }
    write_json(output / "run.json", record)
    started = time.perf_counter()
    try:
        artifacts.mkdir(parents=True)
        paths = [
            CONFIG,
            Path("scripts/run_source_closure.py"),
            Path("src/parallasse/mixtures.py"),
            Path("tests/test_mixtures.py"),
            Path("scripts/evaluate_heldout.py"),
            Path("src/parallasse/heldout.py"),
            Path("src/parallasse/heldout_metrics.py"),
            Path("requirements-lock.txt"),
        ]
        snapshot = artifacts / "source_snapshot"
        for path in paths:
            (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot / path)
        record["source_files_sha256"] = {str(p): sha256_file(p) for p in paths}
        record["source_snapshot"] = str(snapshot)
        verify_hashes(record["source_files_sha256"], snapshot)
        write_json(output / "protocol.json", config)
        response, population = source_partition(rows, labels, config["partition_seed"])
        pools = {}
        for name, indices in (("response", response), ("population", population)):
            pools[name] = [passed[indices][labels[indices] == k] for k in (1, 0)]
        eq, eg = [p.mean(axis=0) for p in pools["response"]]
        pq, pg = [p.mean(axis=0) for p in pools["population"]]
        infer_fraction(0.5, eq, eg)
        splits = save_arrays(
            artifacts / "source_partition.npz",
            response_rows=rows[response],
            population_rows=rows[population],
            response_indices=response,
            population_indices=population,
            source_rows=rows,
            source_labels=labels,
        )
        write_json(
            output / "response.json",
            {
                "keys": KEYS,
                "response_quarks": len(pools["response"][0]),
                "response_gluons": len(pools["response"][1]),
                "population_quarks": len(pools["population"][0]),
                "population_gluons": len(pools["population"][1]),
                "response_quark_acceptance": eq.tolist(),
                "response_gluon_acceptance": eg.tolist(),
                "population_quark_acceptance": pq.tolist(),
                "population_gluon_acceptance": pg.tolist(),
                "partition": splits,
            },
        )
        record.update(
            status="design_frozen",
            partition=splits,
            design_frozen_utc=datetime.now(UTC).isoformat(),
            protocol_sha256=sha256_file(output / "protocol.json"),
        )
        write_json(output / "run.json", record, "w")
        cells = [(f, n) for f in config["fractions"] for n in config["sample_sizes"]]
        streams = np.random.SeedSequence(config["sampling_seed"]).spawn(len(cells))
        metrics, arrays = [], {"keys": np.asarray(KEYS)}
        for cell, ((fraction, size), stream) in enumerate(zip(cells, streams, strict=True)):
            accepted, truth = mixture_counts(
                *pools["population"],
                fraction,
                size,
                config["replicates"],
                np.random.Generator(np.random.PCG64(stream)),
            )
            arrays[f"accepted_{cell}"] = accepted
            arrays[f"quark_counts_{cell}"] = truth
            stats = closure_summary(accepted, size, fraction, eq, eg, pq, pg)
            for key, values in zip(KEYS, stats, strict=True):
                model, seed = key.rsplit("-", 1)
                metrics.append(
                    {
                        "fraction": fraction,
                        "sample_size": size,
                        "model": model,
                        "seed": int(seed),
                        **values,
                    }
                )
        draws = save_arrays(artifacts / "pseudoexperiments.npz", **arrays)
        with (output / "closure.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=metrics[0])
            writer.writeheader()
            writer.writerows(metrics)
        standardized = [
            abs(r["mean_minus_expectation"]) / r["mean_mcse_fixed_pools"] for r in metrics
        ]
        summary = {
            "cells": len(cells),
            "replicates_per_cell": config["replicates"],
            "per_seed_metric_rows": len(metrics),
            "keys": KEYS,
            "max_absolute_bias": max(abs(r["bias"]) for r in metrics),
            "max_absolute_expected_bias_fixed_pools": max(
                abs(r["expected_bias_fixed_pools"]) for r in metrics
            ),
            "max_mcse_deviation_from_fixed_pool_expectation": max(standardized),
            "all_means_within_five_mcse_of_fixed_pool_expectation": max(standardized) <= 5,
            "uncertainty_scope": (
                "MC standard errors condition on the two observed source pools. "
                "Response uncertainty and coverage require TASK-0007."
            ),
            "draws": draws,
        }
        write_json(output / "summary.json", summary)
        verify_hashes(inputs)
        record.update(
            status="completed",
            completed_utc=datetime.now(UTC).isoformat(),
            wall_seconds=time.perf_counter() - started,
            draws=draws,
            output_sha256={
                str(output / name): sha256_file(output / name)
                for name in (
                    "protocol.json",
                    "response.json",
                    "closure.csv",
                    "summary.json",
                )
            },
        )
        write_json(output / "run.json", record, "w")
        print(f"Saved {len(metrics)} Pythia closure rows from {len(cells)} mixture cells.")
        print(
            f"Largest mean deviation from its fixed-pool expectation: {max(standardized):.2f} MCSE."
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", record, "w")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        raise ValueError("Use a simple, unique run ID")
    run_closure(args.run_id)


if __name__ == "__main__":
    main()
