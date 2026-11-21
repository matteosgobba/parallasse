"""Run the bounded TASK-0010 pilot, preserving all existing experiment evidence."""

import argparse
import fcntl
import importlib.metadata
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from evaluate_heldout import save_arrays, write_json
from run_source_closure import source_inputs
from threadpoolctl import threadpool_limits

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.heldout_metrics import KEYS
from parallasse.mixture_uncertainty import fraction_interval
from parallasse.template_feasibility import (
    MultinomialTemplates,
    identification,
    mixture_weights,
    source_bins,
    synthetic_templates,
    trial_summary,
)

CONFIG = Path("configs/template_feasibility.json")
SELECTED = tuple(f"{model}-{seed}" for model in ("PFN-K", "PFN-ID") for seed in (17, 29, 43))


def attempt(model, nodes):
    started = time.perf_counter()
    try:
        result = {"success": True, **model.confidence_set(grid_size=nodes)}
    except RuntimeError as error:
        result = {"success": False, "error": str(error)}
    return {**result, "wall_seconds": time.perf_counter() - started}


def synthetic_pilot(config, deadline):
    templates = synthetic_templates()
    noiseless = []
    for alpha in config["synthetic_variations"]:
        data = config["sample_size"] * (mixture_weights(0.35, alpha)[0] @ templates)
        model = MultinomialTemplates(data, 5000 * templates)
        fit, independent = model.fit(), model.fit(method="SLSQP")
        noiseless.append(
            {
                "truth_fraction": 0.35,
                "truth_alpha": alpha,
                "fraction": fit.fraction,
                "alpha": fit.alpha,
                "both_optimizers_succeeded": fit.success and independent.success,
                "optimizer_nll_difference": abs(fit.nll - independent.nll),
                "profile": attempt(model, config["confidence_scan_nodes"]),
            }
        )
    confounded = np.array(
        [templates[0], templates[1], 0.8 * templates[0] + 0.2 * templates[1], templates[1]]
    )
    identical = np.tile(templates[0], (4, 1))
    diagnostics = {
        "identified": identification(templates),
        "confounded": identification(confounded),
        "identical_classes": identification(identical),
        "two_bins": identification(templates[:, :2] / templates[:, :2].sum(axis=1)[:, None]),
    }
    flat = MultinomialTemplates(10000 * templates[0], 5000 * identical, False)
    diagnostics["flat_confidence_set"] = flat.confidence_set()
    rows = []
    rng = np.random.default_rng(config["sampling_seed"])
    draws = []
    for alpha in config["synthetic_variations"]:
        for replicate in range(config["synthetic_replicates"]):
            if time.perf_counter() > deadline:
                raise TimeoutError("Bounded pilot exceeded its runtime ceiling")
            auxiliary = np.array([rng.multinomial(5000, t) for t in templates])
            data = rng.multinomial(
                config["sample_size"], mixture_weights(0.5, alpha)[0] @ templates
            )
            result = attempt(MultinomialTemplates(data, auxiliary), config["confidence_scan_nodes"])
            rows.append({"truth_alpha": alpha, "replicate": replicate, **result})
            draws.append(np.r_[data, auxiliary.ravel()])
    tolerance = config["numerical_tolerances"]
    checks = {
        "noiseless_recovery_and_optimizer_agreement": all(
            r["both_optimizers_succeeded"]
            and abs(r["fraction"] - 0.35) < tolerance["noiseless_fraction_absolute"]
            and abs(r["alpha"] - r["truth_alpha"]) < tolerance["noiseless_variation_absolute"]
            and r["optimizer_nll_difference"] < tolerance["optimizer_nll_agreement"]
            for r in noiseless
        ),
        "rank_identified_case": diagnostics["identified"]["locally_identified"],
        "rank_rejects_confounded_identical_and_two_bin_cases": all(
            not diagnostics[name]["locally_identified"]
            for name in ("confounded", "identical_classes", "two_bins")
        ),
        "flat_case_retains_full_fraction_set": diagnostics["flat_confidence_set"]["intervals"]
        == [[0.0, 1.0]],
        "synthetic_profiles_succeeded": all(r["success"] for r in rows),
    }
    summary = {
        "noiseless": noiseless,
        "identification": diagnostics,
        "checks": checks,
        "toy_summary": {
            str(alpha): trial_summary([r for r in rows if r["truth_alpha"] == alpha])
            for alpha in config["synthetic_variations"]
        },
    }
    return summary, rows, np.array(draws)


def source_pilot(config, output, artifacts, record, deadline):
    rows, labels, _, hashes = source_inputs(read_json(Path("configs/mixtures.json")))
    evaluation = read_json(Path(config["source_evaluation_run"]))
    freeze_path = Path(evaluation["model_freeze"])
    verify_hashes({str(freeze_path): evaluation["model_freeze_sha256"]})
    freeze = read_json(freeze_path)
    for model in freeze["models"]:
        if f"{model['model']}-{model['seed']}" in SELECTED:
            hashes[model["checkpoint"]] = model["checkpoint_sha256"]
    hashes[str(freeze_path)] = evaluation["model_freeze_sha256"]
    response_path = Path(config["source_partition_record"])
    old_validation = read_json(Path("results/task0006/validation.json"))
    hashes[str(response_path)] = old_validation["evidence_sha256"][str(response_path)]
    verify_hashes(hashes)
    partition_record = read_json(response_path)["partition"]
    verify_hashes({partition_record["path"]: partition_record["sha256"]})
    hashes[partition_record["path"]] = partition_record["sha256"]
    with np.load(partition_record["path"], allow_pickle=False) as archive:
        response, population = archive["response_indices"], archive["population_indices"]
        np.testing.assert_array_equal(archive["source_rows"], rows)
    assert not np.intersect1d(response, population).size
    with np.load(evaluation["score_artifacts"]["test"]["path"], allow_pickle=False) as archive:
        scores = archive["scores"][:, [KEYS.index(key) for key in SELECTED]]
    cuts_record = read_json(Path(evaluation["cut_freeze"]))
    cuts = np.array([cuts_record["cuts"][key]["threshold"] for key in SELECTED])
    edges = np.array(
        [source_bins(scores[response, i], config["source_bins"]) for i in range(len(SELECTED))]
    )
    pools = {
        name: [indices[labels[indices] == label] for label in (1, 0)]
        for name, indices in (("response", response), ("population", population))
    }
    source_laws = np.array(
        [
            [np.histogram(scores[indices, i], edges[i])[0] for indices in pools["response"]]
            for i in range(len(SELECTED))
        ]
    )
    assert np.all(source_laws > 0)
    binning = save_arrays(
        artifacts / "source_binning.npz",
        keys=np.array(SELECTED),
        edges=edges,
        response_histograms=source_laws,
        response_rows=rows[response],
        population_rows=rows[population],
    )
    record.update(
        source_input_sha256=hashes,
        source_binning=binning,
        source_binning_frozen_utc=datetime.now(UTC).isoformat(),
    )
    write_json(output / "run.json", record, "w")
    trial_rows, draw_arrays = [], {}
    streams = np.random.SeedSequence(config["sampling_seed"] + 1).spawn(2)
    for design, stream in zip(config["source_designs"], streams, strict=True):
        rng = np.random.default_rng(stream)
        response_draws, pseudo_draws = [], []
        for replicate in range(config["source_replicates_per_design"]):
            if time.perf_counter() > deadline:
                raise TimeoutError("Bounded pilot exceeded its runtime ceiling")
            if design == "repeated_template_common_law":
                template_rows = [
                    rng.choice(indices, 5000, replace=True) for indices in pools["population"]
                ]
            else:
                template_rows = pools["response"]
            count = rng.binomial(config["sample_size"], config["fraction"])
            pseudo = np.r_[
                rng.choice(pools["population"][0], count, replace=True),
                rng.choice(pools["population"][1], config["sample_size"] - count, replace=True),
            ]
            response_draws.append(np.concatenate(template_rows))
            pseudo_draws.append(pseudo)
            for i, key in enumerate(SELECTED):
                auxiliary = np.array(
                    [np.histogram(scores[idx, i], edges[i])[0] for idx in template_rows]
                )
                data = np.histogram(scores[pseudo, i], edges[i])[0]
                result = attempt(
                    MultinomialTemplates(data, auxiliary), config["confidence_scan_nodes"]
                )
                common = {"design": design, "replicate": replicate, "key": key}
                trial_rows.append({**common, "method": "finite_multinomial_profile", **result})
                eq, eg = [float(np.mean(scores[idx, i] > cuts[i])) for idx in template_rows]
                accepted = np.array([np.sum(scores[pseudo, i] > cuts[i])], dtype=int)
                estimate, lo, hi, _ = fraction_interval(
                    accepted, config["sample_size"], eq, eg, 5000, 5000
                )
                trial_rows.append(
                    {
                        **common,
                        "method": "original_cut_delta",
                        "success": True,
                        "estimate": float(estimate[0]),
                        "intervals": [[float(lo[0]), float(hi[0])]],
                        "length": float(hi[0] - lo[0]),
                        "wall_seconds": 0.0,
                    }
                )
        draw_arrays[f"{design}_template_indices"] = np.array(response_draws)
        draw_arrays[f"{design}_pseudo_indices"] = np.array(pseudo_draws)
    summaries = [
        {
            "design": design,
            "key": key,
            "method": method,
            **trial_summary(
                [
                    r
                    for r in trial_rows
                    if r["design"] == design and r["key"] == key and r["method"] == method
                ]
            ),
        }
        for design in config["source_designs"]
        for key in SELECTED
        for method in ("original_cut_delta", "finite_multinomial_profile")
    ]
    draws = save_arrays(artifacts / "source_draws.npz", **draw_arrays)
    return summaries, trial_rows, draws, scores, edges, source_laws, hashes


def opened_target_diagnostic(config, edges, source_laws):
    """Only old, already opened TASK-0004 scores; never a reserved new file."""
    evaluation = read_json(Path(config["source_evaluation_run"]))
    target = evaluation["score_artifacts"]["target"]
    verify_hashes({target["path"]: target["sha256"]})
    with np.load(target["path"], allow_pickle=False) as archive:
        if tuple(archive["keys"]) != KEYS or len(archive["rows"]) != 100000:
            raise ValueError("Unexpected old Herwig archive")
        labels = archive["labels"]
        scores = archive["scores"][:, [KEYS.index(key) for key in SELECTED]]
    if any(np.sum(labels == label) != 50000 for label in (0, 1)):
        raise ValueError("Old Herwig labels do not match their recorded cohort")
    diagnostics, laws = [], []
    for i, key in enumerate(SELECTED):
        auxiliary = np.array(
            [
                *source_laws[i],
                *[np.histogram(scores[labels == label, i], edges[i])[0] for label in (1, 0)],
            ],
            dtype=float,
        )
        t = auxiliary / auxiliary.sum(axis=1, keepdims=True)
        laws.append(t)
        geometries = {
            str(alpha): identification(
                t, alpha=alpha, auxiliary_size=config["prospective_template_rows_per_class"]
            )
            for alpha in config["synthetic_variations"]
        }
        # Fractional Asimov counts project a possible 25k-per-class design.
        data = config["sample_size"] * (mixture_weights(0.5, 0.5)[0] @ t)
        profile = attempt(
            MultinomialTemplates(data, config["prospective_template_rows_per_class"] * t),
            config["confidence_scan_nodes"],
        )
        diagnostics.append(
            {
                "key": key,
                "geometry": geometries,
                "asimov_profile": profile,
                "minimum_observed_template_bin_count": int(auxiliary.min()),
            }
        )
    return diagnostics, np.array(laws), target


def run_pilot(run_id):
    output, artifacts = Path("results/task0010") / run_id, Path("artifacts/task0010") / run_id
    if output.exists() or artifacts.exists():
        raise FileExistsError("Pilot run already exists; evidence is never overwritten")
    lock_path = Path("work/task0010-locks") / f"{run_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or artifacts.exists():
            raise FileExistsError("Pilot run already exists; evidence is never overwritten")
        output.mkdir(parents=True)
        artifacts.mkdir(parents=True)
        config = read_json(CONFIG)
        started = time.perf_counter()
        record = {
            "task": "TASK-0010",
            "run_id": run_id,
            "status": "preparing",
            "started_utc": datetime.now(UTC).isoformat(),
            "fresh_records_accessed": False,
            "new_classifier_fits": 0,
            "new_classifier_inference": 0,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "threadpoolctl")
            },
            "python": sys.version,
        }
        paths = [
            CONFIG,
            Path(__file__).relative_to(Path.cwd()),
            Path("src/parallasse/template_feasibility.py"),
            Path("tests/test_template_feasibility.py"),
            Path("scripts/run_source_closure.py"),
            Path("scripts/evaluate_heldout.py"),
            Path("src/parallasse/heldout.py"),
            Path("src/parallasse/mixture_uncertainty.py"),
            Path("src/parallasse/mixtures.py"),
            Path("src/parallasse/heldout_metrics.py"),
            Path("configs/mixtures.json"),
            Path("requirements-lock.txt"),
            Path("pyproject.toml"),
        ]
        snapshot = artifacts / "source_snapshot"
        for path in paths:
            (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot / path)
        record.update(
            source_files_sha256={str(p): sha256_file(p) for p in paths},
            source_snapshot=str(snapshot),
        )
        write_json(output / "protocol.json", config)
        write_json(output / "run.json", record)
        deadline = started + config["runtime_ceiling_seconds"]
        try:
            with threadpool_limits(limits=config["threads"]):
                print(
                    "Checking synthetic identification, boundaries and finite templates", flush=True
                )
                synthetic, toy_rows, toy_draws = synthetic_pilot(config, deadline)
                write_json(output / "synthetic.json", synthetic)
                write_json(output / "synthetic_trials.json", {"rows": toy_rows})
                save_arrays(artifacts / "synthetic_draws.npz", counts=toy_draws)
                if not all(synthetic["checks"].values()):
                    raise RuntimeError("Synthetic feasibility checks failed")
                print("Running paired Pythia pilot on the persisted source partition", flush=True)
                summaries, trials, draws, _, edges, source_laws, _ = source_pilot(
                    config, output, artifacts, record, deadline
                )
                write_json(output / "source_summary.json", {"rows": summaries})
                write_json(output / "source_trials.json", {"rows": trials})
                if any(r["failures"] for r in summaries):
                    raise RuntimeError("Source pilot has unresolved fit failures")
                repeated = [r for r in summaries if r["design"] == "repeated_template_common_law"]
                mcse = np.sqrt(0.95 * 0.05 / config["source_replicates_per_design"])
                if any(abs(r["coverage"] - 0.95) > 5 * mcse for r in repeated):
                    raise RuntimeError("Small source common-law diagnostic failed")
                record.update(
                    status="source_pilot_passed",
                    source_draws=draws,
                    source_pilot_passed_utc=datetime.now(UTC).isoformat(),
                )
                write_json(output / "run.json", record, "w")
                print(
                    "Inspecting already opened Herwig scores for exploratory geometry only",
                    flush=True,
                )
                record["old_target_diagnostic_started_utc"] = datetime.now(UTC).isoformat()
                diagnostics, laws, target = opened_target_diagnostic(config, edges, source_laws)
                write_json(
                    output / "opened_target_diagnostic.json",
                    {
                        "rows": diagnostics,
                        "scope": (
                            "Exploratory old-record geometry/Asimov projection, "
                            "not reserved validation"
                        ),
                    },
                )
                save_arrays(
                    artifacts / "diagnostic_templates.npz", keys=np.array(SELECTED), templates=laws
                )
                widths = [r["asimov_profile"].get("length") for r in diagnostics]
                feasible = all(
                    r["geometry"]["0.5"]["locally_identified"]
                    and r["asimov_profile"]["success"]
                    and r["asimov_profile"]["length"]
                    <= config["prospective_scientific_criteria"][
                        "primary_median_confidence_length_maximum"
                    ]
                    for r in diagnostics
                )
                source_times = [
                    r["wall_seconds"] for r in trials if r["method"] == "finite_multinomial_profile"
                ]
                full_times = [r["wall_seconds"] for r in toy_rows] + [
                    r["asimov_profile"]["wall_seconds"] for r in diagnostics
                ]
                source_p95, full_p95 = (
                    float(np.quantile(source_times, 0.95)),
                    float(np.quantile(full_times, 0.95)),
                )
                summary = {
                    "decision": "go_to_TASK-0011" if feasible else "revise_or_no_go",
                    "source_checks_passed": True,
                    "synthetic_checks": synthetic["checks"],
                    "old_herwig_midpoint_profile_lengths": dict(zip(SELECTED, widths, strict=True)),
                    "source_interval_seconds_p95": source_p95,
                    "four_template_interval_seconds_p95": full_p95,
                    "full_design_projection": {
                        "four_template_intervals": 180000,
                        "source_template_intervals": 72000,
                        "definition": (
                            "6 checkpoints * 6 f/N cells * (3 repeated variation laws "
                            "+ 2 disjoint endpoint laws) * 1000 repetitions; "
                            "source-only template comparator on 2 endpoints"
                        ),
                        "serial_hours_at_p95": (180000 * full_p95 + 72000 * source_p95) / 3600,
                        "scope": (
                            "Projection only, excludes calibration-toy campaigns and "
                            "verification; not a measured full run"
                        ),
                    },
                    "coverage_validation": (
                        "Pending TASK-0013; this small pilot does not establish 95% coverage"
                    ),
                    "fresh_evaluation_performance": "Not accessed",
                }
                write_json(output / "summary.json", summary)
                record.update(
                    status="completed",
                    decision=summary["decision"],
                    old_target_archive=target,
                    wall_seconds=time.perf_counter() - started,
                    completed_utc=datetime.now(UTC).isoformat(),
                )
                record["output_sha256"] = {
                    str(p): sha256_file(p)
                    for p in output.iterdir()
                    if p.is_file() and p.name != "run.json"
                }
                record["artifact_sha256"] = {
                    str(p): sha256_file(p) for p in artifacts.glob("*.npz")
                }
                write_json(output / "run.json", record, "w")
                print(
                    f"Pilot complete: {summary['decision']}; {record['wall_seconds']:.2f} s",
                    flush=True,
                )
        except Exception as error:
            record.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                wall_seconds=time.perf_counter() - started,
            )
            write_json(output / "run.json", record, "w")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("Use a simple unique run ID")
    if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        parser.error("Run from the canonical project root")
    run_pilot(args.run_id)
