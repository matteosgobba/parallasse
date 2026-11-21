"""Versioned gate evidence, reserved-role isolation and non-resetting budget."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import parallasse.template_revision as revision
import parallasse.template_revision_scoring as scoring
from parallasse.download import sha256_file
from parallasse.score_template_design import CHECKPOINTS
from parallasse.template_campaign import CUT, SOURCE, ResourceCeiling
from parallasse.template_workflow import atomic_json


def config():
    return revision.resolve_protocol(
        json.loads(Path("configs/template_validation_revision.json").read_text())
    )


def test_resolved_revision_changes_only_declared_fields():
    design = json.loads(Path("configs/template_validation_revision.json").read_text())
    cfg = revision.resolve_protocol(design)
    base = json.loads(Path(design["parent_protocol"]).read_text())
    for key in [
        "models",
        "seeds",
        "likelihood",
        "numerics",
        "grid",
        "binning",
        "primary_decision",
        "pairing_and_uncertainty",
    ]:
        assert cfg[key] == base[key]
    assert cfg["rng"]["source_validation"] == 2026111521
    assert cfg["rng"]["simulation_validation"] == 2026111523
    assert cfg["rng"]["simulation_calibration"] == base["rng"]["simulation_calibration"]
    assert cfg["rng"]["pythia_evaluation"] == base["rng"]["pythia_evaluation"]
    assert cfg["resources"]["source_calibration_lr_pairs"] == 0


def fixture_control(tmp_path, monkeypatch, hits=957, all_cells=True):
    cfg = config()
    output = tmp_path / "results/task0013_revision/fixture"
    artifacts = tmp_path / "artifacts/task0013_revision/fixture"
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    cutoff = tmp_path / "cutoffs.json"
    atomic_json(
        cutoff, {"cutoffs": {SOURCE: {key: {"1000": 4.0, "10000": 4.0} for key in CHECKPOINTS}}}
    )
    overlay = tmp_path / "revision.json"
    atomic_json(overlay, {"source_cutoffs_to_reuse": str(cutoff)})
    monkeypatch.setattr(revision, "REVISION", overlay)
    record = {
        "run_id": "fixture",
        "status": "source_running",
        "artifacts": str(artifacts),
        "protocol_version": "test-v010",
        "engine_sha256": "engine-binding",
        "review_design_sha256": "design-binding",
        "source_cutoffs": {"path": str(cutoff), "sha256": sha256_file(cutoff)},
    }
    monkeypatch.setattr(revision, "ROOT", tmp_path / "results/task0013_revision")
    monkeypatch.setattr(revision, "ARTIFACTS", tmp_path / "artifacts/task0013_revision")
    cells = revision.cell_grid(cfg, "source_validation")
    for number, (n, f, a) in enumerate(cells if all_cells else cells[:1]):
        folder = artifacts / "source_validation" / f"cell_{number:02d}"
        folder.mkdir(parents=True)
        index = {
            "cell": [n, f, a],
            "kind": "validation",
            "methods": [CUT, SOURCE],
            "repetitions": 1000,
            "completed": 1000,
            "seed_entropy": 2026111521,
            "seed_spawn_key": [number],
            "chunks": [],
        }
        for start in range(0, 1000, 25):
            stop = start + 25
            rows = []
            for rep in range(start, stop):
                interval = [f - 0.04, f + 0.04] if rep < hits else [0.01, 0.10]
                for key in CHECKPOINTS:
                    for method in [CUT, SOURCE]:
                        rows.append(
                            {
                                "replicate": rep,
                                "key": key,
                                "method": method,
                                "success": True,
                                "estimate": sum(interval) / 2,
                                "intervals": [interval],
                                "length": interval[1] - interval[0],
                                "span": interval[1] - interval[0],
                                "scan_nodes": 33,
                                "cutoff": 4.0,
                            }
                        )
            trials = folder / f"trials_{start:04d}_{stop:04d}.json.gz"
            trials.write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
            counts = folder / f"counts_{start:04d}_{stop:04d}.npz"
            aux = np.zeros((25, 6, 2, 8), dtype=np.int64)
            aux[:, :, :, 0] = 25000
            obs = np.zeros((25, 6, 8), dtype=np.int64)
            obs[:, :, 0] = n
            np.savez(
                counts,
                auxiliary=aux,
                observed=obs,
                membership_counts=np.tile([n, 0], (25, 1)),
                accepted=np.zeros((25, 6), dtype=np.int64),
            )
            index["chunks"].append(
                {
                    "start": start,
                    "stop": stop,
                    "rows": str(trials),
                    "sha256": {str(trials): sha256_file(trials), str(counts): sha256_file(counts)},
                }
            )
        atomic_json(folder / "index.json", index)
    return record, {}, cfg, output, artifacts


def test_complete_new_evidence_qualifies_and_is_version_bound(tmp_path, monkeypatch):
    record, engine, cfg, output, _ = fixture_control(tmp_path, monkeypatch)
    gate = revision.collect_control(record, engine, cfg, "source")
    assert gate["passed"] and gate["cells"] == 36 and gate["root_seed"] == 2026111521
    assert len(gate["summaries"]) == len(gate["reference_summaries"]) == 36
    assert all(r["lower_coverage_bound"] >= 0.93 for r in gate["qualifications"])
    path = output / "source_gate.json"
    atomic_json(path, {**gate, "validated_utc": "synthetic fixture"})
    record["source_gate"] = {"path": str(path), "sha256": sha256_file(path)}
    revision.checked_revision_gate(record, engine, cfg, "source")
    for name, value in [
        ("engine_sha256", "other-engine"),
        ("protocol_version", "v0.9"),
        ("root_seed", 2026111311),
    ]:
        changed = copy.deepcopy(gate)
        changed[name] = value
        atomic_json(path, changed)
        record["source_gate"]["sha256"] = sha256_file(path)
        with pytest.raises(ValueError, match="fresh evidence"):
            revision.checked_revision_gate(record, engine, cfg, "source")


def test_incomplete_or_old_stream_evidence_cannot_qualify(tmp_path, monkeypatch):
    record, engine, cfg, _, artifacts = fixture_control(tmp_path, monkeypatch, all_cells=False)
    path = artifacts / "source_validation/cell_00/index.json"
    original = json.loads(path.read_text())
    for key, value in [
        ("seed_entropy", 2026111311),
        ("completed", 999),
        ("seed_spawn_key", [1]),
        ("methods", [SOURCE]),
    ]:
        index = copy.deepcopy(original)
        index[key] = value
        atomic_json(path, index)
        with pytest.raises(ValueError, match="incomplete cells, old streams"):
            revision.collect_control(record, engine, cfg, "source")


def test_tampered_chunks_and_duplicate_attempts_are_not_coverage_misses(tmp_path, monkeypatch):
    record, engine, cfg, _, artifacts = fixture_control(tmp_path, monkeypatch, all_cells=False)
    path = artifacts / "source_validation/cell_00/index.json"
    index = json.loads(path.read_text())
    trial = Path(index["chunks"][0]["rows"])
    raw = trial.read_bytes()
    trial.write_bytes(raw + b"altered")
    with pytest.raises(ValueError, match="Frozen bytes"):
        revision.collect_control(record, engine, cfg, "source")
    rows = json.loads(gzip.decompress(raw))
    rows[1] = rows[0]
    trial.write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
    index["chunks"][0]["sha256"][str(trial)] = sha256_file(trial)
    atomic_json(path, index)
    with pytest.raises(ValueError, match="exactly once"):
        revision.collect_control(record, engine, cfg, "source")


def test_old_or_unqualified_gate_is_not_a_revision_access_proof(tmp_path, monkeypatch):
    record, engine, cfg, output, _ = fixture_control(tmp_path, monkeypatch, all_cells=False)
    old = tmp_path / "old-source-gate.json"
    atomic_json(old, {"passed": True, "scope": "frozen_full_grid", "cells": 36})
    record["source_gate"] = {"path": str(old), "sha256": sha256_file(old)}
    with pytest.raises(ValueError, match="another study/version"):
        revision.checked_revision_gate(record, engine, cfg, "source")
    path = output / "source_gate.json"
    atomic_json(path, {"passed": False})
    record["source_gate"] = {"path": str(path), "sha256": sha256_file(path)}
    with pytest.raises(ValueError, match="did not qualify"):
        revision.checked_revision_gate(record, engine, cfg, "source")


def test_profile_validity_includes_components_span_mle_and_scan_checks():
    cfg = config()
    good = {
        "success": True,
        "estimate": 0.5,
        "intervals": [[0.4, 0.6]],
        "scan_nodes": 33,
        "cutoff": 4.0,
        "length": 0.2,
        "span": 0.2,
    }
    revision._valid_profile(good, 4.0, cfg)
    for name, value in [
        ("intervals", [[0.1, 0.2], [0.15, 0.3]]),
        ("span", np.nan),
        ("length", 0.3),
        ("estimate", 0.9),
        ("scan_nodes", 17),
        ("cutoff", 3.84),
    ]:
        with pytest.raises(ValueError):
            revision._valid_profile({**good, name: value}, 4.0, cfg)
    revision._valid_profile({"success": False, "error": "preserved failure"}, 4.0, cfg)


def test_worker_denies_herwig_before_opening_inputs(monkeypatch):
    cfg = config()
    record = {"status": "source_running"}
    monkeypatch.setattr(scoring, "_record", lambda path: (record, {}, cfg))
    with pytest.raises(ValueError, match="not authorized"):
        scoring.check_revision_score_job(
            {
                "mode": "revision_reserved_role",
                "study_record": "fixture",
                "generator": "herwig",
                "role": "template",
            }
        )
    with pytest.raises(ValueError, match="must not access reserved"):
        scoring.check_revision_score_job(
            {"mode": "old_source_smoke", "raw_file": "data/raw/QG_jets_3.npz"}
        )


def test_role_order_requires_both_full_gates_before_evaluation(monkeypatch):
    cfg = config()
    seen = []

    def check(record, engine, config, stage):
        seen.append(stage)

    monkeypatch.setattr(revision, "checked_revision_gate", check)
    revision.authorize_revision_role({"status": "model_running"}, {}, cfg, "herwig", "template")
    assert seen == ["source"]
    seen.clear()
    revision.authorize_revision_role(
        {"status": "evaluation_running"}, {}, cfg, "pythia", "evaluation"
    )
    assert seen == ["source", "model"]
    with pytest.raises(ValueError, match="not authorized"):
        revision.authorize_revision_role({"status": "source_failed"}, {}, cfg, "herwig", "template")


def test_single_attempt_claim_and_cpu_highwater_cannot_be_reset(tmp_path, monkeypatch):
    cfg = config()
    root = tmp_path / "results/task0013_revision"
    art = tmp_path / "artifacts/task0013_revision"
    monkeypatch.setattr(revision, "ROOT", root)
    monkeypatch.setattr(revision, "ARTIFACTS", art)
    score = tmp_path / "scores.npz"
    score.write_bytes(b"explicit synthetic fixture")
    ref = {
        "path": str(score),
        "sha256": sha256_file(score),
        "generator": "pythia",
        "role": "template",
        "rows": 50000,
    }
    cut = tmp_path / "cutoff.json"
    atomic_json(cut, {"cutoffs": {}})
    cutref = {"path": str(cut), "sha256": sha256_file(cut)}
    parent = tmp_path / "parent.json"
    atomic_json(parent, {"score_archives": {"pythia_template": ref}, "source_cutoffs": cutref})
    overlay = tmp_path / "revision.json"
    atomic_json(
        overlay, {"preserved_failed_run": str(parent), "source_score_reference": str(score)}
    )
    monkeypatch.setattr(revision, "REVISION", overlay)
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, cfg)
    engine_path = tmp_path / "engine.json"
    atomic_json(engine_path, {"synthetic_fixture": True})
    engine = {
        "protocol_version": "test-v010",
        "review_design_sha256": "design",
        "inherited_cpu_seconds": 1245.143441,
        "source_score_reference": ref,
        "source_cutoffs_reference": cutref,
        "protocol_path": str(protocol),
    }
    monkeypatch.setattr(revision, "load_revision_engine", lambda path: engine)
    record = revision.create_revision_study("one", engine_path)
    assert record["cpu_seconds"] == 1245.143441
    with pytest.raises(ValueError, match="one validation attempt"):
        revision.create_revision_study("other", engine_path)
    path = root / "one/run.json"
    guard = root / "one/cpu_highwater.json"
    atomic_json(
        guard, {"run_id": "one", "engine_sha256": record["engine_sha256"], "cpu_seconds": 1500.0}
    )
    recovered, _, _ = revision._record(path)
    assert recovered["cpu_seconds"] == 1500.0 and recovered["cpu_recovered_from_highwater"]
    with pytest.raises(ResourceCeiling):
        revision.RevisionBudget(recovered, path, 1499).check()
    assert json.loads(path.read_text())["cpu_seconds"] >= 1500.0
    changed = json.loads(path.read_text())
    changed["cpu_seconds"] = 0
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="cumulative CPU"):
        revision._record(path)


@pytest.mark.parametrize(
    "terminal", ["source_failed", "model_failed", "completed", "source_validated"]
)
def test_source_reentry_after_terminal_source_decision_is_rejected(tmp_path, monkeypatch, terminal):
    monkeypatch.setattr(revision, "_record", lambda p: ({"status": terminal}, {}, {}))
    with pytest.raises(ValueError, match="preceding successful gate"):
        revision.execute_revision_stage("fixture", "source")


def test_design_only_review_cannot_be_loaded_as_executable():
    with pytest.raises(ValueError, match="revision executable"):
        revision.load_revision_engine(revision.REVIEW)


@pytest.mark.parametrize("source_pass,model_pass", [(False, False), (True, False), (True, True)])
def test_full_revision_orchestration_is_source_first_and_never_recalibrates_source(
    tmp_path, monkeypatch, source_pass, model_pass
):
    """Explicit synthetic scoring/numeric doubles; never real reserved records."""
    from parallasse.template_sampling import class_pools

    cfg = config()
    root = tmp_path / "results/task0013_revision"
    art = tmp_path / "artifacts/task0013_revision"
    monkeypatch.setattr(revision, "ROOT", root)
    monkeypatch.setattr(revision, "ARTIFACTS", art)
    score = tmp_path / "source-scores.npz"
    score.write_bytes(b"synthetic cached probabilities")
    score_ref = {
        "path": str(score),
        "sha256": sha256_file(score),
        "generator": "pythia",
        "role": "template",
        "rows": 50000,
    }
    cut = tmp_path / "source-cutoffs.json"
    values = {
        SOURCE: {key: {str(n): 4.0 for n in cfg["grid"]["sample_sizes"]} for key in CHECKPOINTS}
    }
    atomic_json(cut, {"cutoffs": values})
    cutref = {"path": str(cut), "sha256": sha256_file(cut)}
    parent = tmp_path / "failed-parent.json"
    atomic_json(
        parent, {"source_cutoffs": cutref, "score_archives": {"pythia_template": score_ref}}
    )
    overlay = tmp_path / "revision.json"
    atomic_json(
        overlay,
        {
            "preserved_failed_run": str(parent),
            "source_score_reference": str(score),
            "source_cutoffs_to_reuse": str(cut),
        },
    )
    monkeypatch.setattr(revision, "REVISION", overlay)
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, cfg)
    engine_path = tmp_path / "engine.json"
    atomic_json(engine_path, {"synthetic_fixture": True})
    engine = {
        "protocol_version": "test-v010",
        "review_design_sha256": "test-design",
        "inherited_cpu_seconds": 1245.143441,
        "source_score_reference": score_ref,
        "source_cutoffs_reference": cutref,
        "protocol_path": str(protocol),
    }
    monkeypatch.setattr(revision, "load_revision_engine", lambda path: engine)
    labels = np.r_[np.zeros(400), np.ones(400)]
    v = np.tile(np.linspace(0.001, 0.999, 400), 2)
    pools = class_pools(
        np.repeat(v[:, None], 6, axis=1),
        labels,
        np.arange(800),
        np.repeat(np.linspace(0, 1, 9)[None, :], 6, axis=0),
        np.full(6, 0.5),
    )
    monkeypatch.setattr(revision, "load_score_pools", lambda ref, engine: pools)
    stages = []
    scored = []
    calibrated = []

    def score_role(path, generator, role, destination):
        r = json.loads(path.read_text())
        revision.authorize_revision_role(r, engine, cfg, generator, role)
        scored.append((generator, role))
        destination.write_bytes(b"synthetic worker double")
        return {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "generator": generator,
            "role": role,
        }

    monkeypatch.setattr(scoring, "score_revision_role", score_role)

    def grid(root, pools, config, stage, methods, kind, cutoffs, budget):
        stages.append(stage)
        assert config["rng"][stage] == (2026111521 if stage == "source_validation" else 2026111523)

    monkeypatch.setattr(revision, "run_grid", grid)

    def collect(record, engine, config, stage):
        return {
            "passed": source_pass if stage == "source" else model_pass,
            "stage": stage,
            "protocol_version": record["protocol_version"],
            "run_id": record["run_id"],
            "engine_sha256": record["engine_sha256"],
            "scope": "explicit_synthetic_double",
        }

    monkeypatch.setattr(revision, "collect_control", collect)

    def calibrate(root, pools, config, stage, method, budget):
        calibrated.append(stage)
        assert stage == "simulation_calibration"
        return {
            method: {key: {str(n): 4.0 for n in cfg["grid"]["sample_sizes"]} for key in CHECKPOINTS}
        }

    monkeypatch.setattr(revision, "calibrate_stage", calibrate)

    def evaluate(root, model_pools, population, config, generator, cutoffs, budget):
        from parallasse.template_campaign import SIMULATION

        summaries = [
            {
                "generator": generator,
                "sample_size": 10000,
                "fraction": 0.5,
                "method": SIMULATION,
                "key": key,
                "coverage": 0.99,
                "failures": 0,
                "attempts": 1000,
                "median_span": 0.1,
            }
            for key in CHECKPOINTS
        ]
        contrasts = [
            {
                "generator": generator,
                "sample_size": 10000,
                "fraction": 0.5,
                "key": key,
                "contrasts": {SIMULATION: {"paired_mc95": [0.01, 0.03]}},
            }
            for key in CHECKPOINTS
        ]
        return summaries, contrasts

    monkeypatch.setattr(revision, "evaluate_population", evaluate)
    revision.create_revision_study("flow", engine_path)
    result = revision.execute_revision_stage("flow", "source")
    assert not scored and not calibrated and stages == ["source_validation"]
    if not source_pass:
        assert result["status"] == "source_failed"
        with pytest.raises(ValueError, match="preceding successful gate"):
            revision.execute_revision_stage("flow", "model")
        return
    assert result["status"] == "source_validated"
    result = revision.execute_revision_stage("flow", "model")
    assert scored == [("herwig", "template")] and calibrated == ["simulation_calibration"]
    if not model_pass:
        assert result["status"] == "model_failed"
        with pytest.raises(ValueError, match="preceding successful gate"):
            revision.execute_revision_stage("flow", "evaluate")
        return
    assert revision.execute_revision_stage("flow", "evaluate")["status"] == "completed"
    assert scored == [("herwig", "template"), ("pythia", "evaluation"), ("herwig", "evaluation")]
    assert json.loads(cut.read_text())["cutoffs"] == values
