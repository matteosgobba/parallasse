"""Version-bound v0.10 execution, fresh control evidence and inherited CPU budget.

The historical v0.9 engine and its failed run are never edited or resumed here.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.score_template_design import CHECKPOINTS
from parallasse.template_campaign import (
    CUT,
    SIMULATION,
    SOURCE,
    CampaignBudget,
    ResourceCeiling,
    calibrate_stage,
    cell_grid,
    evaluate_population,
    load_score_pools,
    run_grid,
)
from parallasse.template_gate_review import assess_proposed_cell
from parallasse.template_inference import NumericalFailure, summarize_trials
from parallasse.template_sampling import histograms
from parallasse.template_workflow import atomic_json, load_engine, now

REVIEW = Path("results/task0013_review/20261115-validation-review-02/revision_design_freeze.json")
REVISION = Path("configs/template_validation_revision.json")
ROOT = Path("results/task0013_revision")
ARTIFACTS = Path("artifacts/task0013_revision")


def resolve_protocol(revision):
    config = copy.deepcopy(read_json(Path(revision["parent_protocol"])))
    config["protocol_version"] = (
        revision["protocol_version"].removesuffix("-design") + "-executable"
    )
    config["qualification"] = copy.deepcopy(revision["qualification"])
    config["validation"]["gate"] = revision["qualification"]["coverage_requirement"]
    for name in ("source_validation", "simulation_validation"):
        config["rng"][name] = revision["fresh_validation_streams_reserved_not_drawn"][name]
    config["resources"]["source_calibration_lr_pairs"] = 0
    config["revision_context"] = {
        "design": str(REVISION),
        "design_sha256": sha256_file(REVISION),
        "failed_parent": revision["preserved_failed_run"],
        "source_cutoffs": revision["source_cutoffs_to_reuse"],
        "cumulative_profile_sets": revision["work_accounting"][
            "cumulative_profile_sets_including_failed_v09"
        ],
        "interpretation": (
            "Post-source-control prospective revision; original failed gate remains false"
        ),
    }
    return config


def load_revision_engine(path):
    path = Path(path)
    engine = read_json(path)
    if engine.get("state") != "revision_executable_frozen" or engine.get("smoke_only") is not False:
        raise ValueError("A complete revision executable freeze is required")
    run = read_json(path.parent / "run.json")
    if run.get("status") != "completed" or run.get("engine_freeze_sha256") != sha256_file(path):
        raise ValueError("Revision executable freeze differs from completed run")
    verify_hashes(engine["implementation_sha256"])
    verify_hashes(engine["implementation_sha256"], Path(engine["source_snapshot"]))
    verify_hashes(engine["input_sha256"])
    load_engine(engine["base_engine"])
    revision = read_json(REVISION)
    review = read_json(REVIEW)
    if (
        review.get("state") != "design_only_not_executable"
        or engine["review_design_sha256"] != sha256_file(REVIEW)
        or review["configuration_sha256"] != sha256_file(REVISION)
    ):
        raise ValueError("Revision design binding changed")
    protocol = read_json(Path(engine["protocol_path"]))
    parent = read_json(Path(revision["preserved_failed_run"]))
    if protocol != resolve_protocol(revision) or parent["status"] != "source_failed":
        raise ValueError("Resolved revision or preserved parent changed")
    if (
        engine["inherited_cpu_seconds"] != parent["cpu_seconds"]
        or not engine["resource_gate_passed"]
    ):
        raise ValueError("Inherited CPU/resource proof differs from the preserved parent")
    return engine


def _paths(record):
    run_id = record["run_id"]
    return ROOT / run_id, ARTIFACTS / run_id


def _record(path):
    path = Path(path)
    record = read_json(path)
    output, artifacts = _paths(record)
    if path.resolve() != (output / "run.json").resolve():
        raise ValueError("Revision record path/identity mismatch")
    engine = load_revision_engine(record["engine_freeze"])
    if (
        sha256_file(Path(record["engine_freeze"])) != record["engine_sha256"]
        or record["protocol_version"] != engine["protocol_version"]
        or record["review_design_sha256"] != engine["review_design_sha256"]
        or record["artifacts"] != str(artifacts)
        or record["cpu_seconds"] < engine["inherited_cpu_seconds"]
        or record["inherited_cpu_seconds"] != engine["inherited_cpu_seconds"]
        or not np.isfinite(record["cpu_seconds"])
        or record["score_archives"].get("pythia_template") != engine["source_score_reference"]
        or record["source_cutoffs"] != engine["source_cutoffs_reference"]
    ):
        raise ValueError("Revision identity, version or cumulative CPU binding changed")
    claim_path = ROOT / "claims" / f"{engine['review_design_sha256']}.json"
    claim = read_json(claim_path)
    if claim["run_id"] != record["run_id"] or claim["engine_sha256"] != record["engine_sha256"]:
        raise ValueError("Single-attempt revision claim differs")
    highwater = read_json(output / "cpu_highwater.json")
    if (
        highwater["run_id"] != record["run_id"]
        or highwater["engine_sha256"] != record["engine_sha256"]
        or not np.isfinite(highwater["cpu_seconds"])
        or highwater["cpu_seconds"] < engine["inherited_cpu_seconds"]
    ):
        raise ValueError("Cumulative CPU high-water identity changed")
    if record["cpu_seconds"] < highwater["cpu_seconds"]:
        record["cpu_seconds"] = highwater["cpu_seconds"]
        record["cpu_recovered_from_highwater"] = True
    return record, engine, read_json(Path(engine["protocol_path"]))


def _reference(ref):
    verify_hashes({ref["path"]: ref["sha256"]})
    return Path(ref["path"])


def _cutoff_reference(record, stage, revision):
    ref = record["source_cutoffs" if stage == "source" else "simulation_cutoffs"]
    expected = (
        Path(revision["source_cutoffs_to_reuse"])
        if stage == "source"
        else _paths(record)[0] / "simulation_cutoffs.json"
    )
    if _reference(ref).resolve() != expected.resolve():
        raise ValueError("Numerical cutoff provenance differs from the frozen revision")
    return ref


def _valid_profile(row, cutoff, config):
    if not row["success"]:
        if not isinstance(row.get("error"), str):
            raise ValueError("Unresolved fit must retain its error")
        return
    intervals = np.asarray(row["intervals"], dtype=float)
    if (
        intervals.ndim != 2
        or intervals.shape[1] != 2
        or not len(intervals)
        or np.any(~np.isfinite(intervals))
        or np.any((intervals < 0) | (intervals > 1))
        or np.any(intervals[:, 0] > intervals[:, 1])
        or np.any(intervals[1:, 0] <= intervals[:-1, 1])
        or not np.isfinite(row["estimate"])
        or not 0 <= row["estimate"] <= 1
        or row["scan_nodes"] not in config["numerics"]["scan_nodes"][1:]
        or row["cutoff"] != cutoff
        or not np.all(np.isfinite([row["length"], row["span"]]))
        or not any(lo <= row["estimate"] <= hi for lo, hi in intervals)
    ):
        raise ValueError("Invalid numerical profile/set/cutoff evidence")
    length = float(np.sum(intervals[:, 1] - intervals[:, 0]))
    span = float(intervals[-1, 1] - intervals[0, 0])
    if abs(row["length"] - length) > 1e-11 or abs(row["span"] - span) > 1e-11:
        raise ValueError("Confidence length/span inconsistent with its components")


def collect_control(record, engine, config, stage):
    """Rebuild the complete gate from fresh-stream chunks, never from old summaries."""
    prefix = "source" if stage == "source" else "simulation"
    method = SOURCE if stage == "source" else SIMULATION
    methods = [CUT, SOURCE] if stage == "source" else [SIMULATION]
    revision = read_json(REVISION)
    ref = _cutoff_reference(record, stage, revision)
    cutoffs = read_json(Path(ref["path"]))["cutoffs"][method]
    root = _paths(record)[1] / f"{prefix}_validation"
    cells = cell_grid(config, f"{prefix}_validation")
    expected_root = config["rng"][f"{prefix}_validation"]
    evidence = {ref["path"]: ref["sha256"]}
    summaries, references, qualifications = [], [], []
    for number, (size, fraction, alpha) in enumerate(cells):
        directory = root / f"cell_{number:02d}"
        index_path = directory / "index.json"
        index = read_json(index_path)
        identity = {
            "cell": [size, fraction, alpha],
            "kind": "validation",
            "methods": methods,
            "repetitions": 1000,
            "completed": 1000,
            "seed_entropy": expected_root,
            "seed_spawn_key": [number],
        }
        if any(index.get(k) != v for k, v in identity.items()):
            raise ValueError(
                "Control evidence has incomplete cells, old streams or changed methods"
            )
        evidence[str(index_path)] = sha256_file(index_path)
        rows, cursor = [], 0
        for chunk in index["chunks"]:
            start, stop = chunk["start"], chunk["stop"]
            trial_path = directory / f"trials_{start:04d}_{stop:04d}.json.gz"
            counts_path = directory / f"counts_{start:04d}_{stop:04d}.npz"
            if (
                start != cursor
                or stop != min(1000, start + 25)
                or Path(chunk["rows"]).resolve() != trial_path.resolve()
                or set(map(lambda p: Path(p).resolve(), chunk["sha256"]))
                != {trial_path.resolve(), counts_path.resolve()}
            ):
                raise ValueError("Control chunks must be contiguous local committed evidence")
            verify_hashes(chunk["sha256"])
            evidence.update(chunk["sha256"])
            section = json.loads(gzip.decompress(trial_path.read_bytes()))
            expected = {
                (rep, key, m) for rep in range(start, stop) for key in CHECKPOINTS for m in methods
            }
            if (
                len(section) != len(expected)
                or {(r["replicate"], r["key"], r["method"]) for r in section} != expected
                or any(type(r["success"]) is not bool for r in section)
            ):
                raise ValueError("Every attempted checkpoint/method/trial must appear exactly once")
            with np.load(counts_path, allow_pickle=False) as counts:
                k = 2 if stage == "source" else 4
                if (
                    counts["auxiliary"].shape != (stop - start, 6, k, 8)
                    or not np.all(counts["auxiliary"].sum(axis=-1) == 25000)
                    or counts["observed"].shape != (stop - start, 6, 8)
                    or not np.all(counts["observed"].sum(axis=-1) == size)
                    or not np.all(counts["membership_counts"].sum(axis=-1) == size)
                    or np.any((counts["accepted"] < 0) | (counts["accepted"] > size))
                ):
                    raise ValueError("Control sufficient statistics violate frozen denominators")
            rows.extend(section)
            cursor = stop
        if cursor != 1000:
            raise ValueError("Incomplete control chunk range")
        for key in CHECKPOINTS:
            for m in methods:
                group = [r for r in rows if r["key"] == key and r["method"] == m]
                if m == method:
                    for r in group:
                        _valid_profile(r, cutoffs[key][str(size)], config)
                summary = {
                    "key": key,
                    "sample_size": size,
                    "fraction": fraction,
                    "alpha": alpha,
                    "method": m,
                    **summarize_trials(group, fraction),
                }
                if m == method:
                    summaries.append(summary)
                    qualifications.append(
                        {k: summary[k] for k in ("key", "sample_size", "fraction", "alpha")}
                        | assess_proposed_cell(
                            summary["coverage_hits"],
                            summary["attempts"],
                            summary["failures"],
                            summary["median_span"],
                            config["qualification"],
                        )
                    )
                else:
                    references.append(summary)
    expected_count = config["qualification"][
        "source_cells" if stage == "source" else "modelling_cells"
    ]
    if len(summaries) != expected_count:
        raise ValueError("Full declared control family is required")
    return {
        "scope": "revision_full_grid",
        "smoke_only": False,
        "cells": expected_count,
        "repetitions": 1000,
        "stage": stage,
        "method": method,
        "protocol_version": record["protocol_version"],
        "run_id": record["run_id"],
        "engine_sha256": record["engine_sha256"],
        "review_design_sha256": record["review_design_sha256"],
        "root_seed": expected_root,
        "cutoff_reference": ref,
        "evidence_sha256": evidence,
        "summaries": summaries,
        "reference_summaries": references,
        "qualifications": qualifications,
        "passed": all(r["qualified"] for r in qualifications),
    }


def checked_revision_gate(record, engine, config, stage):
    name = f"{stage}_gate"
    ref = record.get(name)
    if not ref:
        raise ValueError(f"Required revision {name} is absent")
    expected = _paths(record)[0] / f"{stage}_gate.json"
    if _reference(ref).resolve() != expected.resolve():
        raise ValueError("A gate from another study/version is not a revision access proof")
    gate = read_json(expected)
    if gate.get("passed") is not True:
        raise ValueError("Required revision control did not qualify")
    rebuilt = collect_control(record, engine, config, stage)
    if any(gate.get(k) != v for k, v in rebuilt.items()) or not rebuilt["passed"]:
        raise ValueError("Revision gate does not match complete fresh evidence/qualification")
    return gate


def authorize_revision_role(record, engine, config, generator, role):
    if (generator, role) not in [
        (g, r) for g in ("pythia", "herwig") for r in ("template", "evaluation")
    ]:
        raise ValueError("Unknown reserved data role")
    status = record["status"]
    if status == "source_running" and (generator, role) == ("pythia", "template"):
        return
    if status == "model_running" and role == "template":
        checked_revision_gate(record, engine, config, "source")
        return
    if status == "evaluation_running":
        checked_revision_gate(record, engine, config, "source")
        checked_revision_gate(record, engine, config, "model")
        return
    raise ValueError("Reserved role is not authorized by the revision stage")


def create_revision_study(run_id, engine_path):
    output, artifacts = ROOT / run_id, ARTIFACTS / run_id
    if run_id == "claims" or output.exists() or artifacts.exists():
        raise FileExistsError("Revision study ID exists; evidence is never overwritten")
    engine_path = Path(engine_path)
    engine = load_revision_engine(engine_path)
    revision = read_json(REVISION)
    parent = read_json(Path(revision["preserved_failed_run"]))
    ref = parent["score_archives"]["pythia_template"]
    _reference(ref)
    if ref["path"] != revision["source_score_reference"]:
        raise ValueError("Only the original verified Pythia template scores may be reused")
    claim_path = ROOT / "claims" / f"{engine['review_design_sha256']}.json"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim = {"run_id": run_id, "engine_sha256": sha256_file(engine_path)}
    if claim_path.exists():
        if read_json(claim_path) != claim:
            raise ValueError("This prospective design already has its one validation attempt")
    else:
        with claim_path.open("x") as handle:
            json.dump(claim, handle, sort_keys=True)
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    record = {
        "run_id": run_id,
        "status": "ready",
        "started_utc": now(),
        "engine_freeze": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "review_design_sha256": engine["review_design_sha256"],
        "protocol_version": engine["protocol_version"],
        "artifacts": str(artifacts),
        "inherited_cpu_seconds": engine["inherited_cpu_seconds"],
        "cpu_seconds": engine["inherited_cpu_seconds"],
        "score_archives": {"pythia_template": ref},
        "source_cutoffs": parent["source_cutoffs"],
        "parent_failed_run": revision["preserved_failed_run"],
    }
    atomic_json(output / "run.json", record)
    atomic_json(
        output / "cpu_highwater.json",
        {
            "run_id": run_id,
            "engine_sha256": record["engine_sha256"],
            "cpu_seconds": record["cpu_seconds"],
        },
    )
    return record


class RevisionBudget(CampaignBudget):
    def save(self):
        super().save()
        atomic_json(
            self.path.parent / "cpu_highwater.json",
            {
                "run_id": self.record["run_id"],
                "engine_sha256": self.record["engine_sha256"],
                "cpu_seconds": self.record["cpu_seconds"],
            },
        )


def execute_revision_stage(run_id, stage):
    path = ROOT / run_id / "run.json"
    record, engine, config = _record(path)
    expected = {
        "source": ("ready", "source_running"),
        "model": ("source_validated", "model_running"),
        "evaluate": ("model_validated", "evaluation_running"),
    }
    if stage not in expected or record["status"] not in expected[stage]:
        raise ValueError(
            "Revision stage requires its preceding successful gate; terminal runs reject reentry"
        )
    if stage != "source":
        checked_revision_gate(record, engine, config, "source")
    if stage == "evaluate":
        checked_revision_gate(record, engine, config, "model")
    record["status"] = {
        "source": "source_running",
        "model": "model_running",
        "evaluate": "evaluation_running",
    }[stage]
    atomic_json(path, record)
    budget = RevisionBudget(
        record, path, config["resources"]["full_machine_cpu_hour_ceiling"] * 3600
    )
    artifacts = Path(record["artifacts"])

    def pools(generator, role):
        authorize_revision_role(record, engine, config, generator, role)
        name = f"{generator}_{role}"
        if name not in record["score_archives"]:
            from parallasse.template_revision_scoring import score_revision_role

            ref = score_revision_role(path, generator, role, artifacts / f"{name}_scores.npz")
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
            if stage == "model" and "simulation_cutoffs" not in record:
                cutoffs = calibrate_stage(
                    artifacts / "simulation_calibration",
                    model_pools,
                    config,
                    "simulation_calibration",
                    SIMULATION,
                    budget,
                )
                cutoff_path = path.parent / "simulation_cutoffs.json"
                if cutoff_path.exists():
                    raise ValueError("Uncommitted numerical cutoffs require an audit")
                atomic_json(
                    cutoff_path,
                    {
                        "cutoffs": cutoffs,
                        "frozen_utc": now(),
                        "protocol_version": record["protocol_version"],
                    },
                )
                record["simulation_cutoffs"] = {
                    "path": str(cutoff_path),
                    "sha256": sha256_file(cutoff_path),
                }
                budget.save()
            ref = _cutoff_reference(record, stage, read_json(REVISION))
            cutoffs = read_json(Path(ref["path"]))["cutoffs"]
            methods = [CUT, SOURCE] if stage == "source" else [method]
            run_grid(
                artifacts / f"{prefix}_validation",
                model_pools,
                config,
                f"{prefix}_validation",
                methods,
                "validation",
                cutoffs,
                budget,
            )
            gate = {**collect_control(record, engine, config, stage), "validated_utc": now()}
            gate_path = path.parent / f"{stage}_gate.json"
            atomic_json(gate_path, gate)
            record[f"{stage}_gate"] = {"path": str(gate_path), "sha256": sha256_file(gate_path)}
            record["status"] = (
                ("source_validated" if stage == "source" else "model_validated")
                if gate["passed"]
                else f"{stage}_failed"
            )
            record[f"{stage}_completed_utc"] = now()
        else:
            cutoffs = {}
            for name in ("source_cutoffs", "simulation_cutoffs"):
                ref = record[name]
                _reference(ref)
                cutoffs.update(read_json(Path(ref["path"]))["cutoffs"])
            summaries, contrasts = [], []
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
            passed = len(primary) == 6 and all(
                r["coverage"] >= criteria["coverage_floor"]
                and r["failures"] / r["attempts"] <= criteria["maximum_failure_rate"]
                and r["median_span"] is not None
                and r["median_span"] <= criteria["maximum_median_set_span"]
                and delta[r["key"]]["paired_mc95"][0] > 0
                for r in primary
            )
            atomic_json(
                path.parent / "evaluation_summary.json",
                {
                    "summaries": summaries,
                    "contrasts": contrasts,
                    "primary_criterion_met": bool(passed),
                    "scope": "conditional_on_observed_pools",
                    "protocol_version": record["protocol_version"],
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
