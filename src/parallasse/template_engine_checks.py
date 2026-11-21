"""Bounded source/synthetic checks used before creating an executable freeze."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from parallasse.heldout import read_json, verify_hashes
from parallasse.score_template_design import CHECKPOINTS, verify_reservation
from parallasse.template_campaign import (
    CUT,
    SOURCE,
    CampaignBudget,
    evaluate_draw,
    read_trials,
    run_cell,
)
from parallasse.template_feasibility import mixture_weights, synthetic_templates
from parallasse.template_inference import ProfileLikelihood, calibrated_cutoff, summarize_trials
from parallasse.template_sampling import class_pools, draw_experiment, histograms
from parallasse.template_workflow import (
    RESERVATION,
    atomic_json,
    score_rows,
    selected_model_contract,
)


def old_source_inputs():
    freeze = verify_reservation(RESERVATION)
    config = read_json(Path(freeze["protocol_path"]))
    core = read_json(Path(config["core_evaluation"]))
    archive = core["score_artifacts"]["test"]
    verify_hashes({archive["path"]: archive["sha256"]})
    original = read_json(Path(config["model_freeze"]))
    cohort = original["dataset_contract"]["cohorts"]["test"]
    hashes = {
        archive["path"]: archive["sha256"],
        cohort["row_ids_path"]: cohort["row_ids_sha256"],
        "data/raw/QG_jets.npz": original["dataset_contract"]["raw_sha256"]["source"],
    }
    verify_hashes(hashes)
    with np.load(archive["path"], allow_pickle=False) as a:
        columns = [list(a["keys"]).index(key) for key in CHECKPOINTS]
        rows, labels, scores = a["rows"], a["labels"], a["scores"][:, columns]
    np.testing.assert_array_equal(rows, np.load(cohort["row_ids_path"], allow_pickle=False))
    with np.load(freeze["archives"]["bin_edges"]["path"], allow_pickle=False) as a:
        edges = a["edges"]
    cuts = read_json(Path(core["cut_freeze"]))["cuts"]
    pools = class_pools(
        scores, labels, rows, edges, np.array([cuts[k]["threshold"] for k in CHECKPOINTS])
    )
    return freeze, config, hashes, rows, labels, scores, pools


def replay_source_scores(artifacts, settings, inputs):
    freeze, _, proof, rows, labels, scores, _ = inputs
    count = settings["source_score_rows"]
    actual = score_rows(
        "data/raw/QG_jets.npz",
        rows[:count],
        labels[:count],
        selected_model_contract(freeze),
        Path(artifacts) / "old_source_replay",
        {"mode": "old_source_smoke", "source_proof_sha256": proof, "reservation": str(RESERVATION)},
    )
    difference = float(np.max(np.abs(actual - scores[:count])))
    if not np.array_equal(actual, scores[:count]):
        raise RuntimeError(f"Original PFN source scores do not replay exactly: {difference}")
    return {
        "rows": count,
        "keys": list(CHECKPOINTS),
        "maximum_absolute_probability_difference": difference,
        "exact_replay": True,
        "scope": "old Pythia test prefix only; no new-record scores",
    }


def synthetic_checks(config):
    t = synthetic_templates()
    cases = []
    for alpha in (0.0, 0.5, 1.0):
        model = ProfileLikelihood(
            1000 * (mixture_weights(0.35, alpha)[0] @ t), 25000 * t, config["numerics"]
        )
        result = model.confidence_set()
        check = model._solve(independent=True)
        if (
            abs(result["estimate"] - 0.35) > 1e-4
            or abs(model.best.alpha - alpha) > 1e-4
            or abs(check.nll - model.best.nll) > 1e-4
        ):
            raise RuntimeError("Noiseless recovery/optimizer comparison failed")
        cases.append({"truth_alpha": alpha, **result})
    swap = np.array([[0.2, 0.8], [0.8, 0.2], [0.8, 0.2], [0.2, 0.8]])
    separated = ProfileLikelihood(
        10000 * (mixture_weights(0.9, 0)[0] @ swap), 5000 * swap, config["numerics"], False
    ).confidence_set()
    if len(separated["intervals"]) != 2 or separated["span"] != 1:
        raise RuntimeError("Separated confidence components were lost")
    flat = ProfileLikelihood(
        1000 * t[0], np.tile(25000 * t[0], (4, 1)), config["numerics"], False
    ).confidence_set()
    if flat["intervals"] != [[0.0, 1.0]]:
        raise RuntimeError("Non-identification was hidden")
    return {"noiseless": cases, "disconnected": separated, "non_identified": flat}


def source_smoke(artifacts, settings, inputs):
    _, config, _, _, _, _, pools = inputs
    size = settings["source_smoke_sample_size"]
    fraction = settings["source_smoke_fraction"]
    rng = np.random.default_rng(settings["source_smoke_seed"])
    calrows = {key: [] for key in CHECKPOINTS}
    for _ in range(settings["source_smoke_calibration_repetitions"]):
        draw = draw_experiment(pools, fraction, size, rng)
        for row in evaluate_draw(draw, config, [SOURCE], {}, size, fraction, calibration=True):
            calrows[row["key"]].append(row)
    cutoffs = {
        SOURCE: {key: {str(size): calibrated_cutoff([calrows[key]], config)} for key in CHECKPOINTS}
    }
    atomic_json(
        Path(artifacts) / "smoke_cutoffs.json",
        {"scope": "smoke_subset_not_an_official_gate", "cutoffs": cutoffs},
    )
    with (Path(artifacts) / "smoke_calibration.json.gz").open("xb") as handle:
        handle.write(gzip.compress(json.dumps(calrows, allow_nan=False).encode(), mtime=0))
    # Exercise the same chunk/resume path used in campaigns, at a bounded smoke size.
    record = {"status": "smoke_only", "cpu_seconds": 0.0}
    budget = CampaignBudget(record, Path(artifacts) / "smoke_record.json", 240)
    stream = np.random.SeedSequence(settings["source_smoke_seed"] + 1)
    index = run_cell(
        Path(artifacts) / "smoke_cell",
        pools,
        config,
        (size, fraction, None),
        stream,
        [CUT, SOURCE],
        "validation",
        cutoffs,
        budget,
        repetitions=settings["source_smoke_validation_repetitions"],
        chunk_size=4,
    )
    replay = run_cell(
        Path(artifacts) / "smoke_cell",
        pools,
        config,
        (size, fraction, None),
        stream,
        [CUT, SOURCE],
        "validation",
        cutoffs,
        budget,
        repetitions=settings["source_smoke_validation_repetitions"],
        chunk_size=4,
    )
    if replay != index:
        raise RuntimeError("Completed smoke cell changed on reentry")
    rows = read_trials(index)
    if any(not r["success"] for r in rows):
        raise RuntimeError("Source smoke has numerical failures")
    summaries = [
        {
            "key": key,
            "method": method,
            **summarize_trials(
                [r for r in rows if r["key"] == key and r["method"] == method], fraction
            ),
        }
        for key in CHECKPOINTS
        for method in (CUT, SOURCE)
    ]
    return {
        "scope": "smoke_only; no nominal coverage or access gate claim",
        "smoke_only": True,
        "calibration_repetitions": settings["source_smoke_calibration_repetitions"],
        "validation_repetitions": settings["source_smoke_validation_repetitions"],
        "cutoffs": cutoffs,
        "summaries": summaries,
        "resume_exact": True,
    }


def runtime_probe(settings, inputs, deadline):
    _, config, _, _, _, _, pools = inputs
    source = histograms(pools).astype(float)
    source /= source.sum(axis=2, keepdims=True)
    rng = np.random.default_rng(settings["runtime_seed"])
    tilt = np.exp(
        settings["synthetic_variation_tilt"] * np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    )
    results = []
    for model, key in enumerate(CHECKPOINTS):
        p = source[model]
        h = p * tilt
        h /= h.sum(axis=1, keepdims=True)
        for size in config["grid"]["sample_sizes"]:
            for fraction in config["grid"]["fractions"]:
                for kind, variations in (
                    ("source", [None]),
                    ("synthetic_simulation", [0.0, 0.5, 1.0]),
                ):
                    for alpha in variations:
                        if time.perf_counter() > deadline:
                            raise TimeoutError("Bounded engine benchmark ceiling")
                        t = p if alpha is None else np.vstack([p, h])
                        auxiliary = np.array(
                            [
                                rng.multinomial(settings["benchmark_template_rows_per_class"], law)
                                for law in t
                            ]
                        )
                        data = rng.multinomial(size, mixture_weights(fraction, alpha)[0] @ t)
                        started = time.process_time()
                        calibration = ProfileLikelihood(data, auxiliary, config["numerics"])
                        lr = calibration.lr_at(fraction)
                        pair_time = time.process_time() - started
                        started = time.process_time()
                        result = ProfileLikelihood(
                            data, auxiliary, config["numerics"]
                        ).confidence_set()
                        interval_time = time.process_time() - started
                        results.append(
                            {
                                "key": key,
                                "kind": kind,
                                "sample_size": size,
                                "fraction": fraction,
                                "alpha": alpha,
                                "lr": lr,
                                "calibration_pair_cpu_seconds": pair_time,
                                "interval_cpu_seconds": interval_time,
                                "scan_nodes": result["scan_nodes"],
                                "profile_evaluations": result["profile_evaluations"],
                                "span": result["span"],
                            }
                        )
    times = {}
    for kind in ("source", "synthetic_simulation"):
        rows = [r for r in results if r["kind"] == kind]
        times[kind] = {
            metric: float(np.quantile([r[metric] for r in rows], 0.95))
            for metric in ("calibration_pair_cpu_seconds", "interval_cpu_seconds")
        }
    counts = config["resources"]
    seconds = (
        counts["source_confidence_sets"] * times["source"]["interval_cpu_seconds"]
        + counts["simulation_confidence_sets"]
        * times["synthetic_simulation"]["interval_cpu_seconds"]
        + counts["source_calibration_lr_pairs"] * times["source"]["calibration_pair_cpu_seconds"]
        + counts["simulation_calibration_lr_pairs"]
        * times["synthetic_simulation"]["calibration_pair_cpu_seconds"]
    )
    projection = seconds / 3600 * settings["projection_safety_factor"]
    return {
        "scope": (
            "source/synthetic timing only; actual generator-family runtime remains conditional"
        ),
        "cases": results,
        "p95_cpu_seconds": times,
        "safety_factor": settings["projection_safety_factor"],
        "projected_cpu_hours_with_margin": projection,
        "cpu_hour_ceiling": counts["full_machine_cpu_hour_ceiling"],
        "resource_gate_passed": bool(projection <= counts["full_machine_cpu_hour_ceiling"]),
    }
