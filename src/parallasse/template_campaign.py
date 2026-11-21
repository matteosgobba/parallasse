"""Resumable fixed-design source/modelling validation and reserved-pool comparison."""

from __future__ import annotations

import gzip
import json
import resource
from pathlib import Path

import numpy as np

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.score_template_design import CHECKPOINTS, verify_reservation
from parallasse.template_inference import (
    NumericalFailure,
    calibrated_cutoff,
    cut_result,
    infer_histogram,
    paired_contrasts,
    summarize_trials,
    validation_gate,
)
from parallasse.template_sampling import class_pools, draw_experiment, histograms
from parallasse.template_workflow import (
    atomic_json,
    authorize_role,
    load_engine,
    now,
    score_reserved_role,
)

SOURCE = "source_multinomial_profile"
SIMULATION = "simulation_multinomial_profile"
CUT = "original_cut_delta"


def cpu_seconds():
    return sum(
        getattr(resource.getrusage(who), field)
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
        for field in ("ru_utime", "ru_stime")
    )


class ResourceCeiling(RuntimeError):
    pass


class CampaignBudget:
    def __init__(self, record, path, limit):
        self.record, self.path, self.limit = record, Path(path), limit
        self.initial = float(record.get("cpu_seconds", 0))
        self.started = cpu_seconds()

    def save(self):
        self.record["cpu_seconds"] = self.initial + cpu_seconds() - self.started
        atomic_json(self.path, self.record)

    def check(self):
        self.save()
        if self.record["cpu_seconds"] >= self.limit:
            self.record["resource_ceiling_reached"] = True
            self.save()
            raise ResourceCeiling(
                "CPU-hour ceiling reached; partial evidence saved, scope review required"
            )


def cell_grid(config, stage):
    variations = (
        config["grid"]["modelling_variations"] if stage.startswith("simulation") else [None]
    )
    return [
        (n, f, a)
        for n in config["grid"]["sample_sizes"]
        for f in config["grid"]["fractions"]
        for a in variations
    ]


def load_score_pools(reference, engine):
    verify_hashes({reference["path"]: reference["sha256"]})
    freeze = verify_reservation(Path(engine["reservation"]))
    contract = read_json(Path(engine["protocol_path"]).parent / "model_cut_contract.json")
    with np.load(freeze["archives"]["bin_edges"]["path"], allow_pickle=False) as a:
        edges = a["edges"]
    with np.load(reference["path"], allow_pickle=False) as a:
        if tuple(a["keys"]) != CHECKPOINTS:
            raise ValueError("Score keys differ from six frozen checkpoints")
        generator, role = reference["generator"], reference["role"]
        if str(a["generator"]) != generator or str(a["role"]) != role:
            raise ValueError("Score archive role metadata mismatch")
        with np.load(freeze["archives"]["cohorts"]["path"], allow_pickle=False) as cohorts:
            if not np.array_equal(
                a["rows"], cohorts[f"{generator}_{role}_rows"]
            ) or not np.array_equal(a["labels"], cohorts[f"{generator}_{role}_labels"]):
                raise ValueError("Score row IDs/labels differ from the reserved role")
        return class_pools(
            a["scores"],
            a["labels"],
            a["rows"],
            edges,
            np.array([contract["cuts"][k]["threshold"] for k in CHECKPOINTS]),
        )


def evaluate_draw(draw, config, methods, cutoffs, size, fraction, calibration=False):
    rows = []
    for i, key in enumerate(CHECKPOINTS):
        for method in methods:
            if method == CUT:
                eq = draw["response_pass"][i, :2] / draw["class_sizes"][:2]
                try:
                    outcome = cut_result(
                        int(draw["accepted"][i]), size, eq, draw["class_sizes"][:2]
                    )
                except ValueError as error:
                    outcome = {"success": False, "error": str(error)}
            else:
                auxiliary = draw["auxiliary"][i, :2] if method == SOURCE else draw["auxiliary"][i]
                cutoff = (
                    config["calibration"]["minimum_lr_cutoff"]
                    if calibration
                    else cutoffs[method][key][str(size)]
                )
                outcome = infer_histogram(
                    draw["observed"][i],
                    auxiliary,
                    config["numerics"],
                    cutoff,
                    truth=fraction,
                    calibration=calibration,
                )
            rows.append({"key": key, "method": method, **outcome})
    return rows


def run_cell(
    directory,
    pools,
    config,
    cell,
    stream,
    methods,
    kind,
    cutoffs,
    budget,
    population=None,
    repetitions=None,
    chunk_size=25,
):
    """Immutable chunks plus atomic RNG checkpoints; a restart repeats no committed draw."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.json"
    size, fraction, alpha = cell
    count = config["grid"]["repetitions"] if repetitions is None else repetitions
    identity = {
        "cell": list(cell),
        "methods": methods,
        "kind": kind,
        "repetitions": count,
        "seed_entropy": stream.entropy,
        "seed_spawn_key": list(stream.spawn_key),
    }
    rng = np.random.default_rng(stream)
    if index_path.exists():
        index = read_json(index_path)
        if any(index.get(k) != v for k, v in identity.items()):
            raise ValueError("A resumed cell differs from its frozen design/seed")
        for chunk in index["chunks"]:
            verify_hashes(chunk["sha256"])
        rng.bit_generator.state = index["rng_state"]
    else:
        index = {**identity, "completed": 0, "chunks": [], "rng_state": rng.bit_generator.state}
        atomic_json(index_path, index)
    while index["completed"] < count:
        start = index["completed"]
        stop = min(count, start + chunk_size)
        trial_rows = []
        arrays = {
            key: []
            for key in (
                "auxiliary",
                "observed",
                "response_pass",
                "accepted",
                "membership_counts",
                "row_draw_sha256",
            )
        }
        for replicate in range(start, stop):
            budget.check()
            draw = draw_experiment(
                pools,
                fraction,
                size,
                rng,
                alpha=alpha,
                repeated=kind != "evaluation",
                population=population,
            )
            outcomes = evaluate_draw(
                draw, config, methods, cutoffs, size, fraction, kind == "calibration"
            )
            trial_rows.extend({"replicate": replicate, **r} for r in outcomes)
            for name in arrays:
                arrays[name].append(draw[name])
        rows_path = directory / f"trials_{start:04d}_{stop:04d}.json.gz"
        counts_path = directory / f"counts_{start:04d}_{stop:04d}.npz"
        # A process killed between writing a chunk and its index leaves an uncommitted
        # chunk. Verify exact contents before reusing; never overwrite different bytes.
        row_bytes = gzip.compress(
            json.dumps(trial_rows, sort_keys=True, allow_nan=False).encode(), mtime=0
        )
        if rows_path.exists():
            if rows_path.read_bytes() != row_bytes:
                raise ValueError("Uncommitted trial chunk changed; preserve and audit it")
        else:
            with rows_path.open("xb") as handle:
                handle.write(row_bytes)
        converted = {k: np.asarray(v) for k, v in arrays.items()}
        if counts_path.exists():
            with np.load(counts_path, allow_pickle=False) as a:
                if set(a.files) != set(converted) or any(
                    not np.array_equal(a[k], v) for k, v in converted.items()
                ):
                    raise ValueError("Uncommitted sampling chunk changed")
        else:
            with counts_path.open("xb") as handle:
                np.savez_compressed(handle, **converted)
        index["chunks"].append(
            {
                "start": start,
                "stop": stop,
                "rows": str(rows_path),
                "sha256": {
                    str(rows_path): sha256_file(rows_path),
                    str(counts_path): sha256_file(counts_path),
                },
            }
        )
        index.update(completed=stop, rng_state=rng.bit_generator.state)
        atomic_json(index_path, index)
        budget.save()
    return index


def read_trials(index):
    rows = []
    for chunk in index["chunks"]:
        verify_hashes(chunk["sha256"])
        rows.extend(json.loads(gzip.decompress(Path(chunk["rows"]).read_bytes())))
    return rows


def run_grid(
    root, pools, config, stage, methods, kind, cutoffs, budget, population=None, root_seed=None
):
    cells = cell_grid(config, stage)
    root_seed = config["rng"][stage] if root_seed is None else root_seed
    streams = np.random.SeedSequence(root_seed).spawn(len(cells))
    results = []
    for number, (cell, stream) in enumerate(zip(cells, streams, strict=True)):
        index = run_cell(
            Path(root) / f"cell_{number:02d}",
            pools,
            config,
            cell,
            stream,
            methods,
            kind,
            cutoffs,
            budget,
            population=population,
        )
        results.append((cell, index))
    return results


def calibrate_stage(root, pools, config, stage, method, budget):
    indices = run_grid(root, pools, config, stage, [method], "calibration", {}, budget)
    grouped = {key: {str(n): [] for n in config["grid"]["sample_sizes"]} for key in CHECKPOINTS}
    for (size, _, _), index in indices:
        rows = read_trials(index)
        for key in CHECKPOINTS:
            grouped[key][str(size)].append([r for r in rows if r["key"] == key])
    return {
        method: {
            key: {n: calibrated_cutoff(cells, config) for n, cells in per_size.items()}
            for key, per_size in grouped.items()
        }
    }


def validate_stage(root, pools, config, stage, methods, cutoffs, budget):
    results = run_grid(root, pools, config, stage, methods, "validation", cutoffs, budget)
    summaries = []
    method = SOURCE if stage.startswith("source") else SIMULATION
    for (size, fraction, alpha), index in results:
        rows = read_trials(index)
        for key in CHECKPOINTS:
            summaries.append(
                {
                    "key": key,
                    "sample_size": size,
                    "fraction": fraction,
                    "alpha": alpha,
                    **summarize_trials(
                        [r for r in rows if r["key"] == key and r["method"] == method], fraction
                    ),
                }
            )
    return {
        "passed": validation_gate(summaries, config),
        "scope": "frozen_full_grid",
        "smoke_only": False,
        "repetitions": config["validation"]["repetitions"],
        "cells": len(summaries),
        "method": method,
        "validated_utc": now(),
        "summaries": summaries,
    }


def evaluate_population(root, pools, population, config, generator, cutoffs, budget):
    # Both auxiliary generator measurements stay fixed; the actual endpoint population
    # supplies its two classes directly, without constructing a generator mixture.
    results = run_grid(
        root,
        pools,
        config,
        "endpoint_evaluation",
        config["methods"],
        "evaluation",
        cutoffs,
        budget,
        population=population,
        root_seed=config["rng"][f"{generator}_evaluation"],
    )
    bootstrap_streams = np.random.SeedSequence(config["rng"]["paired_metric_bootstrap"]).spawn(12)
    summaries = []
    contrasts = []
    for number, ((size, fraction, _), index) in enumerate(results):
        rows = read_trials(index)
        rng = np.random.default_rng(bootstrap_streams[number + (6 if generator == "herwig" else 0)])
        indices = rng.integers(
            config["grid"]["repetitions"],
            size=(
                config["pairing_and_uncertainty"]["bootstrap_repetitions"],
                config["grid"]["repetitions"],
            ),
        )
        for key in CHECKPOINTS:
            groups = {
                method: [r for r in rows if r["key"] == key and r["method"] == method]
                for method in config["methods"]
            }
            for method, group in groups.items():
                summaries.append(
                    {
                        "generator": generator,
                        "sample_size": size,
                        "fraction": fraction,
                        "key": key,
                        "method": method,
                        **summarize_trials(group, fraction),
                    }
                )
            contrasts.append(
                {
                    "generator": generator,
                    "sample_size": size,
                    "fraction": fraction,
                    "key": key,
                    "contrasts": paired_contrasts(groups, fraction, indices),
                }
            )
    return summaries, contrasts


def create_study(run_id, engine_path):
    output, artifacts = Path("results/task0013") / run_id, Path("artifacts/task0013") / run_id
    if output.exists() or artifacts.exists():
        raise FileExistsError("Study ID already exists; evidence is never overwritten")
    engine = load_engine(engine_path)
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    record = {
        "run_id": run_id,
        "status": "ready",
        "started_utc": now(),
        "engine_freeze": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "protocol_path": engine["protocol_path"],
        "cpu_seconds": 0.0,
        "score_archives": {},
        "artifacts": str(artifacts),
    }
    atomic_json(output / "run.json", record)
    return record


def execute_stage(run_id, stage):
    output = Path("results/task0013") / run_id
    record_path = output / "run.json"
    record = read_json(record_path)
    expected = {
        "source": ("ready", "source_running"),
        "model": ("source_validated", "model_running"),
        "evaluate": ("model_validated", "evaluation_running"),
    }
    if stage not in expected or record["status"] not in expected[stage]:
        raise ValueError(
            "Stage requires its preceding successful gate; completed/failed runs reject reentry"
        )
    verify_hashes({record["engine_freeze"]: record["engine_sha256"]})
    engine = load_engine(record["engine_freeze"])
    config = read_json(Path(engine["protocol_path"]))
    artifacts = Path(record["artifacts"])
    record["status"] = {
        "source": "source_running",
        "model": "model_running",
        "evaluate": "evaluation_running",
    }[stage]
    atomic_json(record_path, record)
    budget = CampaignBudget(
        record, record_path, config["resources"]["full_machine_cpu_hour_ceiling"] * 3600
    )

    def pools(generator, role):
        authorize_role(record, generator, role, config)
        name = f"{generator}_{role}"
        if name not in record["score_archives"]:
            ref = score_reserved_role(
                record_path,
                generator,
                role,
                record["engine_freeze"],
                artifacts / f"{name}_scores.npz",
            )
            record["score_archives"][name] = ref
            budget.save()
        return load_score_pools(record["score_archives"][name], engine)

    try:
        budget.check()
        pythia = pools("pythia", "template")
        model_pools = pythia if stage == "source" else pythia + pools("herwig", "template")
        if np.any(histograms(model_pools) < config["binning"]["minimum_template_bin_count"]):
            raise NumericalFailure("Measured template bin count below the frozen minimum")
        if stage in ("source", "model"):
            prefix = "source" if stage == "source" else "simulation"
            method = SOURCE if stage == "source" else SIMULATION
            cutoff_path = output / f"{prefix}_cutoffs.json"
            if cutoff_path.exists():
                reference = record.get(f"{prefix}_cutoffs")
                if not reference:
                    raise ValueError("Uncommitted cutoff file requires an audit")
                verify_hashes({reference["path"]: reference["sha256"]})
                cutoffs = read_json(cutoff_path)["cutoffs"]
            else:
                cutoffs = calibrate_stage(
                    artifacts / f"{prefix}_calibration",
                    model_pools,
                    config,
                    f"{prefix}_calibration",
                    method,
                    budget,
                )
                atomic_json(
                    cutoff_path,
                    {"cutoffs": cutoffs, "frozen_utc": now(), "scope": "frozen_full_grid"},
                )
                record[f"{prefix}_cutoffs"] = {
                    "path": str(cutoff_path),
                    "sha256": sha256_file(cutoff_path),
                }
                budget.save()
            methods = [CUT, SOURCE] if stage == "source" else [SIMULATION]
            gate = validate_stage(
                artifacts / f"{prefix}_validation",
                model_pools,
                config,
                f"{prefix}_validation",
                methods,
                cutoffs,
                budget,
            )
            gate_path = output / f"{stage}_gate.json"
            atomic_json(gate_path, gate)
            record[f"{stage}_gate"] = {"path": str(gate_path), "sha256": sha256_file(gate_path)}
            record["status"] = (
                ("source_validated" if stage == "source" else "model_validated")
                if gate["passed"]
                else f"{stage}_failed"
            )
        else:
            cutoffs = {}
            for prefix in ("source", "simulation"):
                ref = record[f"{prefix}_cutoffs"]
                verify_hashes({ref["path"]: ref["sha256"]})
                cutoffs.update(read_json(Path(ref["path"]))["cutoffs"])
            summaries = []
            contrasts = []
            for generator in ("pythia", "herwig"):
                a, b = evaluate_population(
                    artifacts / f"{generator}_evaluation",
                    model_pools,
                    pools(generator, "evaluation"),
                    config,
                    generator,
                    cutoffs,
                    budget,
                )
                summaries.extend(a)
                contrasts.extend(b)
            primary = [
                r
                for r in summaries
                if r["generator"] == "herwig"
                and r["sample_size"] == 10000
                and r["fraction"] == 0.5
                and r["method"] == SIMULATION
            ]
            delta = {
                r["key"]: r["contrasts"][SIMULATION]
                for r in contrasts
                if r["generator"] == "herwig" and r["sample_size"] == 10000 and r["fraction"] == 0.5
            }
            criteria = config["primary_decision"]
            passed = all(
                r["coverage"] >= criteria["coverage_floor"]
                and r["failures"] / r["attempts"] <= criteria["maximum_failure_rate"]
                and r["median_span"] is not None
                and r["median_span"] <= criteria["maximum_median_set_span"]
                and delta[r["key"]]["paired_mc95"][0] > 0
                for r in primary
            )
            atomic_json(
                output / "evaluation_summary.json",
                {
                    "summaries": summaries,
                    "contrasts": contrasts,
                    "primary_criterion_met": bool(len(primary) == 6 and passed),
                    "scope": "conditional_on_observed_pools",
                },
            )
            record.update(status="completed", completed_utc=now())
        budget.save()
    except ResourceCeiling:
        budget.save()
        raise
    except Exception as error:
        record.update(status=f"{stage}_failed", error=f"{type(error).__name__}: {error}")
        budget.save()
        raise
    return record
