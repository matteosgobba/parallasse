import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from parallasse.download import sha256_file
from parallasse.score_template_design import CHECKPOINTS
from parallasse.template_campaign import (
    CUT,
    SIMULATION,
    SOURCE,
    CampaignBudget,
    ResourceCeiling,
    cell_grid,
    create_study,
    execute_stage,
    load_score_pools,
    read_trials,
    run_cell,
)
from parallasse.template_sampling import class_pools
from parallasse.template_workflow import atomic_json, authorize_role, check_score_job


def config():
    return json.loads(Path("configs/score_templates.json").read_text())


def gate(tmp_path, name, cells, smoke=False):
    path = tmp_path / f"{name}.json"
    atomic_json(
        path,
        {
            "passed": True,
            "scope": "frozen_full_grid",
            "smoke_only": smoke,
            "cells": cells,
            "repetitions": 1000,
        },
    )
    return {"path": str(path), "sha256": sha256_file(path)}


def test_role_gates_keep_smoke_source_model_and_final_access_separate(tmp_path):
    cfg = config()
    with pytest.raises(ValueError, match="not authorized"):
        authorize_role({"status": "source_running"}, "herwig", "template", cfg)
    with pytest.raises(ValueError, match="not authorized"):
        authorize_role({"status": "source_running"}, "pythia", "evaluation", cfg)
    source = gate(tmp_path, "source", 36, smoke=True)
    record = {"status": "model_running", "source_gate": source}
    with pytest.raises(ValueError, match="not a complete"):
        authorize_role(record, "herwig", "template", cfg)
    record["source_gate"] = gate(tmp_path, "source", 36)
    authorize_role(record, "herwig", "template", cfg)
    record["status"] = "evaluation_running"
    with pytest.raises(ValueError, match="model_gate is absent"):
        authorize_role(record, "herwig", "evaluation", cfg)
    record["model_gate"] = gate(tmp_path, "model", 108)
    authorize_role(record, "herwig", "evaluation", cfg)
    Path(record["source_gate"]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="Frozen bytes"):
        authorize_role(record, "pythia", "evaluation", cfg)


def test_worker_rejects_premature_access_before_encoded_input_open(tmp_path, monkeypatch):
    import parallasse.template_workflow as workflow

    cfg = tmp_path / "protocol.json"
    atomic_json(cfg, config())
    record = tmp_path / "run.json"
    atomic_json(record, {"status": "source_running"})
    monkeypatch.setattr(workflow, "load_engine", lambda _: {"protocol_path": str(cfg)})
    with pytest.raises(ValueError, match="not authorized"):
        check_score_job(
            {
                "mode": "reserved_role",
                "engine_freeze": "fixture",
                "study_record": str(record),
                "generator": "herwig",
                "role": "template",
            }
        )
    with pytest.raises(ValueError, match="must not access reserved"):
        check_score_job({"mode": "old_source_smoke", "raw_file": "data/raw/QG_jets_3.npz"})


def pools_fixture():
    labels = np.r_[np.zeros(50), np.ones(50)]
    values = np.r_[np.linspace(0.01, 0.8, 50), np.linspace(0.2, 0.99, 50)]
    return class_pools(
        np.repeat(values[:, None], 6, axis=1),
        labels,
        np.arange(100),
        np.repeat(np.linspace(0, 1, 9)[None, :], 6, axis=0),
        np.full(6, 0.5),
    )


def test_four_template_fixed_endpoint_uses_actual_population_and_all_three_methods(tmp_path):
    cfg = config()
    pools = pools_fixture()
    auxiliary = pools + pools
    target = pools_fixture()
    cutoffs = {
        method: {key: {"1000": 3.841458820694124} for key in CHECKPOINTS}
        for method in (SOURCE, SIMULATION)
    }
    record = {"status": "smoke_only", "cpu_seconds": 0.0}
    budget = CampaignBudget(record, tmp_path / "run.json", 60)
    with threadpool_limits(1):
        index = run_cell(
            tmp_path / "cell",
            auxiliary,
            cfg,
            (1000, 0.5, None),
            np.random.SeedSequence(51),
            [CUT, SOURCE, SIMULATION],
            "evaluation",
            cutoffs,
            budget,
            population=target,
            repetitions=2,
            chunk_size=1,
        )
    rows = read_trials(index)
    assert len(rows) == 2 * 6 * 3 and {r["method"] for r in rows} == {CUT, SOURCE, SIMULATION}
    assert index["completed"] == 2 and len(index["chunks"]) == 2
    for chunk in index["chunks"]:
        path = next(p for p in chunk["sha256"] if p.endswith(".npz"))
        with np.load(path, allow_pickle=False) as a:
            assert a["auxiliary"].shape == (1, 6, 4, 8)
            assert np.all(a["observed"].sum(axis=2) == 1000)
            assert a["row_draw_sha256"].shape == (1,)


def test_cell_resume_keeps_exact_committed_draws_and_detects_tampering(tmp_path, monkeypatch):
    import parallasse.template_campaign as campaign

    cfg = config()
    pools = pools_fixture()
    record = {"status": "smoke_only", "cpu_seconds": 0.0}
    budget = CampaignBudget(record, tmp_path / "run.json", 60)
    stream = np.random.SeedSequence(71)
    first = run_cell(
        tmp_path / "cell",
        pools,
        cfg,
        (1000, 0.5, None),
        stream,
        [SOURCE],
        "calibration",
        {},
        budget,
        repetitions=2,
        chunk_size=1,
    )
    monkeypatch.setattr(
        campaign, "evaluate_draw", lambda *a, **k: pytest.fail("Completed draws repeated")
    )
    second = run_cell(
        tmp_path / "cell",
        pools,
        cfg,
        (1000, 0.5, None),
        stream,
        [SOURCE],
        "calibration",
        {},
        budget,
        repetitions=2,
        chunk_size=1,
    )
    assert first == second
    path = Path(second["chunks"][0]["rows"])
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="Frozen bytes"):
        run_cell(
            tmp_path / "cell",
            pools,
            cfg,
            (1000, 0.5, None),
            stream,
            [SOURCE],
            "calibration",
            {},
            budget,
            repetitions=2,
            chunk_size=1,
        )


def test_resource_stop_preserves_state_without_changing_design(tmp_path):
    record = {"status": "source_running", "cpu_seconds": 0.0}
    budget = CampaignBudget(record, tmp_path / "run.json", 0)
    with pytest.raises(ResourceCeiling):
        budget.check()
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["resource_ceiling_reached"] and saved["status"] == "source_running"
    assert len(cell_grid(config(), "source_validation")) == 6
    assert len(cell_grid(config(), "simulation_validation")) == 18


def test_completed_study_and_engine_ids_reject_reentry_before_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/task0013/completed").mkdir(parents=True)
    atomic_json(tmp_path / "results/task0013/completed/run.json", {"status": "completed"})
    with pytest.raises(FileExistsError):
        create_study("completed", "missing-engine")
    with pytest.raises(ValueError, match="completed/failed"):
        execute_stage("completed", "source")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("freeze_template_engine")
    (tmp_path / "results/task0012/completed").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        runner.freeze_engine("completed")


def test_cached_score_rows_must_equal_the_exact_reserved_role(tmp_path, monkeypatch):
    import parallasse.template_campaign as campaign

    cohort = tmp_path / "cohorts.npz"
    bins = tmp_path / "bins.npz"
    np.savez(
        cohort,
        pythia_template_rows=np.array([1, 2, 3, 4]),
        pythia_template_labels=np.array([0, 1, 0, 1]),
    )
    np.savez(bins, edges=np.repeat(np.linspace(0, 1, 9)[None, :], 6, axis=0))
    freeze = {"archives": {"cohorts": {"path": str(cohort)}, "bin_edges": {"path": str(bins)}}}
    monkeypatch.setattr(campaign, "verify_reservation", lambda _: freeze)
    atomic_json(
        tmp_path / "model_cut_contract.json", {"cuts": {k: {"threshold": 0.5} for k in CHECKPOINTS}}
    )
    scores = tmp_path / "scores.npz"
    np.savez(
        scores,
        keys=np.array(CHECKPOINTS),
        generator=np.asarray("pythia"),
        role=np.asarray("template"),
        rows=np.array([10, 20, 30, 40]),
        labels=np.array([0, 1, 0, 1]),
        scores=np.full((4, 6), 0.5),
    )
    with pytest.raises(ValueError, match="row IDs/labels"):
        load_score_pools(
            {
                "path": str(scores),
                "sha256": sha256_file(scores),
                "generator": "pythia",
                "role": "template",
            },
            {"reservation": "fixture", "protocol_path": str(tmp_path / "protocol.json")},
        )


@pytest.mark.parametrize("gate_passes", [True, False])
def test_stage_orchestration_keeps_role_order_and_failed_gates(tmp_path, monkeypatch, gate_passes):
    """Numerical procedures have separate tests; here use explicit synthetic gate doubles."""
    import parallasse.template_campaign as campaign

    cfg = config()
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, cfg)
    engine_path = tmp_path / "engine.json"
    atomic_json(engine_path, {"unit_fixture": True})
    engine = {
        "protocol_path": str(protocol),
        "reservation": "unit fixture, never real reserved data",
    }
    monkeypatch.setattr(campaign, "load_engine", lambda _: engine)
    labels = np.r_[np.zeros(400), np.ones(400)]
    values = np.tile(np.linspace(0.001, 0.999, 400), 2)
    pools = class_pools(
        np.repeat(values[:, None], 6, axis=1),
        labels,
        np.arange(800),
        np.repeat(np.linspace(0, 1, 9)[None, :], 6, axis=0),
        np.full(6, 0.5),
    )
    scored = []

    def score(record_path, generator, role, engine_path, destination):
        record = json.loads(Path(record_path).read_text())
        authorize_role(record, generator, role, cfg)
        scored.append((generator, role))
        destination.write_bytes(b"explicit synthetic scoring double")
        return {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "generator": generator,
            "role": role,
        }

    monkeypatch.setattr(campaign, "score_reserved_role", score)
    monkeypatch.setattr(campaign, "load_score_pools", lambda reference, engine: pools)

    def calibration(root, pools, config, stage, method, budget):
        assert len(pools) == (2 if stage.startswith("source") else 4)
        return {
            method: {key: {str(n): 4.0 for n in cfg["grid"]["sample_sizes"]} for key in CHECKPOINTS}
        }

    monkeypatch.setattr(campaign, "calibrate_stage", calibration)

    def validation(root, pools, config, stage, methods, cutoffs, budget):
        return {
            "passed": gate_passes,
            "scope": "frozen_full_grid",
            "smoke_only": False,
            "cells": 36 if stage.startswith("source") else 108,
            "repetitions": 1000,
        }

    monkeypatch.setattr(campaign, "validate_stage", validation)

    def evaluation(root, model_pools, population, config, generator, cutoffs, budget):
        rows = [
            {
                "generator": generator,
                "sample_size": 10000,
                "fraction": 0.5,
                "key": key,
                "method": SIMULATION,
                "coverage": 0.99,
                "attempts": 1000,
                "failures": 0,
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
                "contrasts": {SIMULATION: {"paired_mc95": [0.1, 0.2]}},
            }
            for key in CHECKPOINTS
        ]
        return rows, contrasts

    monkeypatch.setattr(campaign, "evaluate_population", evaluation)
    monkeypatch.chdir(tmp_path)
    create_study("synthetic", engine_path)
    first = execute_stage("synthetic", "source")
    if not gate_passes:
        assert first["status"] == "source_failed"
        with pytest.raises(ValueError, match="preceding successful gate"):
            execute_stage("synthetic", "model")
        assert scored == [("pythia", "template")]
    else:
        assert first["status"] == "source_validated"
        assert execute_stage("synthetic", "model")["status"] == "model_validated"
        assert execute_stage("synthetic", "evaluate")["status"] == "completed"
        assert scored == [
            ("pythia", "template"),
            ("herwig", "template"),
            ("pythia", "evaluation"),
            ("herwig", "evaluation"),
        ]
