"""Frozen EFN source fitting, calibration and fresh five-family evaluation."""

import argparse
import csv
import fcntl
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from evaluate_heldout import save_arrays, write_json
from evaluate_mixture_transfer import write_metrics

from parallasse.data import fingerprints, integrity_summary, load_jets
from parallasse.development import load_source_development
from parallasse.download import download_declared_file, sha256_file
from parallasse.efn_metrics import (
    KEYS,
    MODELS,
    bias_summary,
    classification,
    classification_summary,
    distribution_summary,
    mixture_rows,
)
from parallasse.heldout import (
    INPUT_FREEZE,
    prepare_cohort,
    read_json,
    verify_hashes,
    verify_input_freeze,
)
from parallasse.heldout_metrics import calibration_cut, checked_scores
from parallasse.mixture_uncertainty import bootstrap_acceptance, expected_bias
from parallasse.mixtures import infer_fraction, mixture_counts
from parallasse.preprocessing import kinematics
from parallasse.replication import prepare_replication_inputs, replication_cohorts

CONFIG = Path("configs/efn.json")
PREVIOUS = Path("results/task0008/20261111-replication-01/run.json")
CORE = Path("results/task0004/20261110-heldout-01/run.json")


def now():
    return datetime.now(UTC).isoformat()


def save_record(output, record):
    write_json(output / "run.json", record, "w")


def record_outputs(output, record, names):
    record["output_sha256"].update(
        {str(output / name): sha256_file(output / name) for name in names}
    )


def verify_stage(output, status):
    record = read_json(output / "run.json")
    if record["status"] != status:
        raise ValueError(f"EFN stage requires state {status}")
    verify_hashes(record["source_files_sha256"])
    verify_hashes(record["source_files_sha256"], Path(record["source_snapshot"]))
    verify_hashes(record["input_sha256"])
    verify_hashes(record["output_sha256"])
    for ref in record.get("archives", {}).values():
        verify_hashes({ref["path"]: ref["sha256"]})
    return record


def prepare_stage(run_id, output, artifacts):
    if output.exists() or artifacts.exists():
        raise FileExistsError("EFN run already exists; prior evidence is never overwritten")
    config = read_json(CONFIG)
    if config["protocol_version"] != "0.8.1-efn-study" or config["with_bc"] is not False:
        raise ValueError("Unsupported EFN protocol")
    validation = read_json(Path("results/task0008/validation.json"))
    verify_hashes(validation["evidence_sha256"])
    verify_hashes(
        {
            p: h
            for p, h in validation["implementation_sha256"].items()
            if p not in ("README.md", "AGENTS.md")
        }
    )
    verify_input_freeze()
    previous = read_json(PREVIOUS)
    verify_hashes(previous["input_sha256"])
    output.mkdir(parents=True)
    record = {
        "task": "TASK-0009",
        "run_id": run_id,
        "status": "preparing",
        "started_utc": now(),
        "protocol_version": config["protocol_version"],
        "config": config,
        "artifacts": str(artifacts),
        "python": sys.version,
        "numpy_version": np.__version__,
        "target_predictions_opened": False,
        "source_predictions_opened": False,
        "output_sha256": {},
        "archives": {},
        "input_sha256": {
            str(PREVIOUS): sha256_file(PREVIOUS),
            str(INPUT_FREEZE): sha256_file(INPUT_FREEZE),
            str(CORE): sha256_file(CORE),
            "results/task0008/validation.json": sha256_file(
                Path("results/task0008/validation.json")
            ),
        },
    }
    core = read_json(CORE)
    verify_hashes({core["cut_freeze"]: core["cut_freeze_sha256"]})
    record["input_sha256"][core["cut_freeze"]] = core["cut_freeze_sha256"]
    write_json(output / "run.json", record)
    started = time.perf_counter()
    try:
        artifacts.mkdir(parents=True)
        paths = [
            *sorted(Path("src/parallasse").glob("*.py")),
            CONFIG,
            Path("scripts/run_efn_study.py"),
            Path("scripts/efn_worker.py"),
            Path("scripts/heldout_worker.py"),
            Path("scripts/evaluate_heldout.py"),
            Path("scripts/evaluate_mixture_transfer.py"),
            Path("scripts/run_source_closure.py"),
            Path("tests/test_efn.py"),
            Path("requirements-lock.txt"),
        ]
        snapshot = artifacts / "source_snapshot"
        for p in paths:
            (snapshot / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, snapshot / p)
        record.update(
            source_snapshot=str(snapshot),
            source_files_sha256={str(p): sha256_file(p) for p in paths},
        )
        write_json(output / "protocol.json", config)
        prior_ref = previous["identities"]
        verify_hashes({prior_ref["path"]: prior_ref["sha256"]})
        with np.load(prior_ref["path"], allow_pickle=False) as hashes:
            prior_hashes = np.concatenate(
                [hashes[k] for k in ("old_source", "old_target", "source", "target")]
            )
        old_specs = (
            read_json(Path("configs/data.json"))["files"]
            + read_json(Path("configs/replication.json"))["files"]
        )
        prior_labels = []
        for spec in old_specs:
            path = Path("data/raw") / spec["filename"]
            expected = previous["input_sha256"][str(path)]
            verify_hashes({str(path): expected})
            record["input_sha256"][str(path)] = expected
            with np.load(path, allow_pickle=False) as data:
                prior_labels.append(data["y"])
        labels, identities, summaries = {}, {}, {}
        for spec in config["files"]:
            provenance = download_declared_file(spec, Path("data/raw"), output)
            raw = Path("data/raw") / spec["filename"]
            record["input_sha256"][str(raw)] = provenance["sha256"]
            particles, y = load_jets(raw)
            if len(y) != spec["expected_rows"]:
                raise ValueError("Unexpected new-file row count")
            h = fingerprints(particles)
            labels[spec["role"]], identities[spec["role"]] = y, h
            summaries[spec["role"]] = integrity_summary(particles, y, h)
            del particles
        cohorts, counts = replication_cohorts(
            labels["source"],
            identities["source"],
            labels["target"],
            identities["target"],
            np.concatenate(prior_labels),
            prior_hashes,
            config["partition_seed"],
        )
        record["archives"]["cohorts"] = save_arrays(artifacts / "cohorts.npz", **cohorts)
        record["archives"]["identities"] = save_arrays(artifacts / "identities.npz", **identities)
        record["input_sha256"][prior_ref["path"]] = prior_ref["sha256"]
        write_json(
            output / "audit.json",
            {
                "previous_identity_records": len(prior_hashes),
                "new_datasets": summaries,
                "retained_counts": counts,
                "previous_identities": prior_ref,
                "scope": (
                    "Exact constituent identities and file/row IDs. "
                    "Original collision IDs are unavailable."
                ),
            },
        )
        record_outputs(
            output,
            record,
            ("protocol.json", "audit.json", "pythia_provenance.json", "herwig_provenance.json"),
        )
        record.update(
            status="prepared",
            audit_completed_utc=now(),
            prepare_wall_seconds=time.perf_counter() - started,
        )
        save_record(output, record)
        print("Reserved and audited fresh EFN comparison events.", flush=True)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_record(output, record)
        raise


def development_inputs(schedule, path):
    data = load_source_development(Path.cwd(), schedule["train_rows"], schedule["tune_rows"])
    arrays = {}
    for split in ("train", "tune"):
        particles = getattr(data, f"{split}_particles")
        coordinates, masks = [], []
        for start in range(0, len(particles), 4096):
            c, m = kinematics(particles[start : start + 4096])
            coordinates.append(c)
            masks.append(m)
        arrays[f"{split}_coordinates"] = np.concatenate(coordinates)
        arrays[f"{split}_mask"] = np.concatenate(masks)
        arrays[f"{split}_labels"] = getattr(data, f"{split}_labels")
        arrays[f"{split}_rows"] = getattr(data, f"{split}_rows")
    arrays["probe_particles"] = data.train_particles[:64]
    ref = save_arrays(path, **arrays)
    return {**ref, "provenance": data.provenance}


def fit_stage(run_id, output, artifacts):
    record = verify_stage(output, "prepared")
    config = record["config"]
    started = time.perf_counter()
    record.update(status="fitting", fit_started_utc=now(), development={})
    save_record(output, record)
    try:
        results = {}
        for mode in ("pilot", "full"):
            schedule = config["training"][mode]
            ref = development_inputs(schedule, artifacts / f"development_{mode}.npz")
            record["development"][mode] = ref
            record["archives"][f"development_{mode}"] = {k: ref[k] for k in ("path", "sha256")}
            save_record(output, record)
            results[mode] = []
            for seed in schedule["seeds"]:
                print(f"Fitting EFN {mode} seed {seed}", flush=True)
                with (artifacts / f"{mode}-{seed}.log").open("x") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/efn_worker.py",
                            "fit",
                            str(output / "run.json"),
                            mode,
                            str(seed),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                result = read_json(artifacts / f"{mode}-EFN-{seed}" / "fit.json")
                results[mode].append(result)
                print(
                    f"EFN {mode} seed {seed}: tuning AUC {result['tune']['roc_auc']:.5f}",
                    flush=True,
                )
            if mode == "pilot" and not all(r["checkpoint_replay_exact"] for r in results[mode]):
                raise ValueError("EFN pilot failed checkpoint feasibility")
        original = read_json(Path("results/task0003/20261110-source-fits-01/run.json"))
        pairing = {}
        for item in results["full"]:
            comparator = next(
                m for m in original["models"] if m["model"] == "PFN-K" and m["seed"] == item["seed"]
            )
            epochs = list(zip(item["history"], comparator["history"], strict=False))
            if any(a["train_order_sha256"] != b["train_order_sha256"] for a, b in epochs):
                raise ValueError("EFN and PFN source minibatch order differ")
            pairing[str(item["seed"])] = {"common_epochs": len(epochs), "same_order": True}
        write_json(
            output / "source_fits.json",
            {
                "pilot": results["pilot"],
                "full": results["full"],
                "matched_pfn_batch_order": pairing,
            },
        )
        manifest = {
            "models": results["full"],
            "cpu_threads": config["training"]["cpu_threads"],
            "batch_size": config["training"]["batch_size"],
            "frozen_utc": now(),
            "training_source_only": True,
            "source_files_sha256": record["source_files_sha256"],
        }
        write_json(output / "efn_models.json", manifest)
        record["input_sha256"].update(
            {m["checkpoint"]: m["checkpoint_sha256"] for m in results["full"]}
        )
        record_outputs(output, record, ("source_fits.json", "efn_models.json"))
        record.update(
            status="models_frozen",
            models_frozen_utc=manifest["frozen_utc"],
            fit_wall_seconds=time.perf_counter() - started,
            new_full_fits=3,
            pilot_fits=1,
        )
        save_record(output, record)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_record(output, record)
        raise


def score_cohort(role, record, output, artifacts, frozen, rows=None, labels=None):
    config = record["config"]
    manifest = output / "efn_models.json"
    with tempfile.TemporaryDirectory(prefix=f"{role}-", dir=artifacts) as folder:
        scratch = Path(folder)
        inputs = scratch / "inputs.npz"
        if role == "calibration":
            rows, labels = prepare_cohort(frozen, "calibration", inputs)
        else:
            spec = next(s for s in config["files"] if s["role"] == role)
            actual = prepare_replication_inputs(
                Path("data/raw") / spec["filename"], rows, frozen, inputs
            )
            if not np.array_equal(actual, labels):
                raise ValueError("Audited and raw inference labels differ")
        scores = {}
        families = ("efn",) if role == "calibration" else ("xgb", "neural", "efn")
        for family in families:
            target = scratch / f"{family}.npz"
            if family == "efn":
                cmd = [
                    sys.executable,
                    "scripts/efn_worker.py",
                    "predict",
                    str(manifest),
                    sha256_file(manifest),
                    str(inputs),
                    str(target),
                ]
            else:
                cmd = [
                    sys.executable,
                    "scripts/heldout_worker.py",
                    family,
                    str(INPUT_FREEZE),
                    sha256_file(INPUT_FREEZE),
                    str(inputs),
                    str(target),
                ]
            with (artifacts / f"{role}-{family}.log").open("x") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            with np.load(target, allow_pickle=False) as values:
                if set(scores) & set(values.files):
                    raise ValueError("Repeated checkpoint predictions")
                scores.update({k: checked_scores(values[k]) for k in values.files})
        keys = KEYS[-3:] if role == "calibration" else KEYS
        if set(scores) != set(keys) or any(s.shape != labels.shape for s in scores.values()):
            raise ValueError("Incomplete checkpoint scores")
        matrix = np.column_stack([scores[k] for k in keys])
        ref = save_arrays(
            artifacts / f"{role}_scores.npz",
            rows=rows,
            labels=labels,
            keys=np.asarray(keys),
            scores=matrix,
        )
    return matrix, ref


def calibrate_stage(run_id, output, artifacts):
    record = verify_stage(output, "models_frozen")
    frozen = verify_input_freeze()
    record.update(status="calibrating", calibration_started_utc=now())
    save_record(output, record)
    try:
        scores, ref = score_cohort("calibration", record, output, artifacts, frozen)
        core = read_json(CORE)
        lock = read_json(Path(core["cut_freeze"]))
        if lock["operator"] != ">":
            raise ValueError("Original cuts must be strict")
        cuts = {k: v["threshold"] for k, v in lock["cuts"].items()}
        efn = {k: calibration_cut(scores[:, i]) for i, k in enumerate(KEYS[-3:])}
        cuts.update({k: v["threshold"] for k, v in efn.items()})
        write_json(
            output / "cut_freeze.json",
            {
                "operator": ">",
                "efn_cuts": efn,
                "cuts": cuts,
                "calibration_scores": ref,
                "models_sha256": sha256_file(output / "efn_models.json"),
                "baseline_cut_sha256": core["cut_freeze_sha256"],
                "frozen_utc": now(),
            },
        )
        record["archives"]["calibration_scores"] = ref
        record_outputs(output, record, ("cut_freeze.json",))
        record.update(status="cuts_frozen", cuts_frozen_utc=now())
        save_record(output, record)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_record(output, record)
        raise


def source_stage(run_id, output, artifacts):
    record = verify_stage(output, "cuts_frozen")
    config = record["config"]
    frozen = verify_input_freeze()
    cuts = read_json(output / "cut_freeze.json")["cuts"]
    started = time.perf_counter()
    record.update(status="source_scoring", source_predictions_opened=True, source_opening_utc=now())
    save_record(output, record)
    try:
        with np.load(record["archives"]["cohorts"]["path"], allow_pickle=False) as data:
            rows, labels = data["source_rows"], data["source_labels"]
            response, population = data["response_indices"], data["population_indices"]
        scores, ref = score_cohort("source", record, output, artifacts, frozen, rows, labels)
        record["archives"]["source_scores"] = ref
        passed = scores > np.array([cuts[k] for k in KEYS])
        rq, rg = [passed[response][labels[response] == k] for k in (1, 0)]
        q, g = [passed[population][labels[population] == k] for k in (1, 0)]
        nq, ng = len(rq), len(rg)
        eq, eg = rq.mean(axis=0), rg.mean(axis=0)
        infer_fraction(0.5, eq, eg)
        cells = [(f, n) for f in config["fractions"] for n in config["sample_sizes"]]
        streams = np.random.SeedSequence(config["source_control_seed"]).spawn(len(cells))
        fixed_streams = np.random.SeedSequence(config["source_pseudo_seed"]).spawn(len(cells))
        metrics, arrays = [], {"keys": np.asarray(KEYS)}
        for i, ((f, n), stream, fixed_stream) in enumerate(
            zip(cells, streams, fixed_streams, strict=True)
        ):
            children = stream.spawn(3)
            rngs = [np.random.Generator(np.random.PCG64(s)) for s in children]
            cq = bootstrap_acceptance(q, nq, config["replicates"], rngs[0])
            cg = bootstrap_acceptance(g, ng, config["replicates"], rngs[1])
            counts, truth = mixture_counts(q, g, f, n, config["replicates"], rngs[2])
            fixed_counts, fixed_truth = mixture_counts(
                q, g, f, n, config["replicates"], np.random.Generator(np.random.PCG64(fixed_stream))
            )
            metrics.extend(
                mixture_rows("source_repeated_response", f, n, counts, cq / nq, cg / ng, nq, ng)
            )
            metrics.extend(
                mixture_rows("pythia_fixed_response", f, n, fixed_counts, eq, eg, nq, ng)
            )
            arrays.update(
                {
                    f"response_quarks_{i}": cq,
                    f"response_gluons_{i}": cg,
                    f"control_accepted_{i}": counts,
                    f"control_quark_counts_{i}": truth,
                    f"fixed_accepted_{i}": fixed_counts,
                    f"fixed_quark_counts_{i}": fixed_truth,
                }
            )
        record["archives"]["source_mixtures"] = save_arrays(
            artifacts / "source_mixtures.npz", **arrays
        )
        write_metrics(output / "source_mixtures.csv", metrics)
        control = [r["coverage_95"] for r in metrics if r["mode"] == "source_repeated_response"]
        deviation = max(
            abs(v - 0.95) / np.sqrt(0.95 * 0.05 / config["replicates"]) for v in control
        )
        class_seed = np.random.SeedSequence(config["classification_bootstrap_seed"]).spawn(2)[0]
        classification_rows, draws = classification(
            labels,
            scores,
            cuts,
            np.random.Generator(np.random.PCG64(class_seed)),
            config["bootstrap_replicates"],
        )
        write_metrics(output / "source_classification.csv", classification_rows)
        record["archives"]["source_classification_bootstrap"] = save_arrays(
            artifacts / "source_classification_bootstrap.npz", draws=draws, keys=np.asarray(KEYS)
        )
        write_json(
            output / "source_summary.json",
            {
                "source_gate_passed": bool(deviation <= 5),
                "control_coverage_min": min(control),
                "control_coverage_max": max(control),
                "max_deviation_in_nominal_mcse": float(deviation),
                "response_sizes": {"quarks": nq, "gluons": ng},
                "response_quark_acceptance": eq.tolist(),
                "response_gluon_acceptance": eg.tolist(),
                "classification": classification_summary(classification_rows, draws),
            },
        )
        record_outputs(
            output,
            record,
            ("source_mixtures.csv", "source_classification.csv", "source_summary.json"),
        )
        record.update(
            status="source_validated" if deviation <= 5 else "source_diagnostic_failed",
            source_completed_utc=now(),
            source_wall_seconds=time.perf_counter() - started,
        )
        save_record(output, record)
        print(
            f"Source coverage {min(control):.3f} to {max(control):.3f}. State {record['status']}.",
            flush=True,
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_record(output, record)
        raise


def target_stage(run_id, output, artifacts):
    record = verify_stage(output, "source_validated")
    if not read_json(output / "source_summary.json")["source_gate_passed"]:
        raise ValueError("Source coverage gate failed")
    config = record["config"]
    frozen = verify_input_freeze()
    cuts = read_json(output / "cut_freeze.json")["cuts"]
    started = time.perf_counter()
    record.update(status="target_scoring", target_predictions_opened=True, target_opening_utc=now())
    save_record(output, record)
    try:
        with np.load(record["archives"]["cohorts"]["path"], allow_pickle=False) as data:
            rows, labels = data["target_rows"], data["target_labels"]
            response, source_labels = data["response_indices"], data["source_labels"]
        with np.load(record["archives"]["source_scores"]["path"], allow_pickle=False) as s:
            source_pass = s["scores"] > np.array([cuts[k] for k in KEYS])
        rq, rg = [source_pass[response][source_labels[response] == k] for k in (1, 0)]
        nq, ng = len(rq), len(rg)
        eq, eg = rq.mean(axis=0), rg.mean(axis=0)
        scores, ref = score_cohort("target", record, output, artifacts, frozen, rows, labels)
        record["archives"]["target_scores"] = ref
        passed = scores > np.array([cuts[k] for k in KEYS])
        q, g = [passed[labels == k] for k in (1, 0)]
        hq, hg = q.mean(axis=0), g.mean(axis=0)
        cells = [(f, n) for f in config["fractions"] for n in config["sample_sizes"]]
        streams = np.random.SeedSequence(config["target_pseudo_seed"]).spawn(len(cells))
        metrics, arrays = [], {"keys": np.asarray(KEYS)}
        for i, ((f, n), stream) in enumerate(zip(cells, streams, strict=True)):
            counts, truth = mixture_counts(
                q, g, f, n, config["replicates"], np.random.Generator(np.random.PCG64(stream))
            )
            metrics.extend(mixture_rows("herwig_fixed_response", f, n, counts, eq, eg, nq, ng))
            arrays.update({f"accepted_{i}": counts, f"quark_counts_{i}": truth})
        write_metrics(output / "target_mixtures.csv", metrics)
        record["archives"]["target_mixtures"] = save_arrays(
            artifacts / "target_mixtures.npz", **arrays
        )
        class_seed = np.random.SeedSequence(config["classification_bootstrap_seed"]).spawn(2)[1]
        class_rows, class_draws = classification(
            labels,
            scores,
            cuts,
            np.random.Generator(np.random.PCG64(class_seed)),
            config["bootstrap_replicates"],
        )
        write_metrics(output / "target_classification.csv", class_rows)
        record["archives"]["target_classification_bootstrap"] = save_arrays(
            artifacts / "target_classification_bootstrap.npz",
            draws=class_draws,
            keys=np.asarray(KEYS),
        )
        children = np.random.SeedSequence(config["bias_bootstrap_seed"]).spawn(4)
        boot = {"keys": np.asarray(KEYS)}
        for name, pool, child in zip(
            ("source_q", "source_g", "target_q", "target_g"), (rq, rg, q, g), children, strict=True
        ):
            boot[name] = bootstrap_acceptance(
                pool,
                len(pool),
                config["bootstrap_replicates"],
                np.random.Generator(np.random.PCG64(child)),
            )
        comparisons = {}
        for i, f in enumerate(config["fractions"]):
            point = expected_bias(f, eq, eg, hq, hg)
            draws = expected_bias(
                f,
                boot["source_q"] / nq,
                boot["source_g"] / ng,
                boot["target_q"] / len(q),
                boot["target_g"] / len(g),
            )
            boot[f"expected_bias_{i}"] = draws
            comparisons[str(f)] = bias_summary(point, draws)
        record["archives"]["bias_bootstrap"] = save_arrays(artifacts / "bias_bootstrap.npz", **boot)
        source_class_rows = []
        with (output / "source_classification.csv").open(newline="") as handle:
            source_class_rows = list(csv.DictReader(handle))
        with np.load(
            record["archives"]["source_classification_bootstrap"]["path"], allow_pickle=False
        ) as s:
            source_draws = s["draws"]
        gaps = {}
        source_points = np.array([float(r["auc"]) for r in source_class_rows]).reshape(5, 3)
        target_points = np.array([r["auc"] for r in class_rows]).reshape(5, 3)
        source_auc_draws = source_draws[:, :, 0].reshape(-1, 5, 3).mean(axis=2)
        target_auc_draws = class_draws[:, :, 0].reshape(-1, 5, 3).mean(axis=2)
        for i, model in enumerate(MODELS):
            gaps[model] = distribution_summary(
                source_points[i] - target_points[i], source_auc_draws[:, i] - target_auc_draws[:, i]
            )
        primary = comparisons[str(config["primary_fraction"])]["contrasts"]["EFN_minus_PFN-K"]
        write_json(
            output / "summary.json",
            {
                "classification": {
                    "source": read_json(output / "source_summary.json")["classification"],
                    "target": classification_summary(class_rows, class_draws),
                    "auc_gap_source_minus_target": gaps,
                },
                "by_fraction": comparisons,
                "primary_fraction": config["primary_fraction"],
                "primary_sample_size": config["primary_sample_size"],
                "improvement_criterion_met": bool(
                    primary["mean"] < 0 and primary["bootstrap_95_hi"] < 0
                ),
                "response_sizes": {"quarks": nq, "gluons": ng},
                "target_sizes": {"quarks": len(q), "gluons": len(g)},
                "scope": config["interpretation"],
            },
        )
        verify_hashes(record["input_sha256"])
        verify_hashes(record["output_sha256"])
        record_outputs(
            output, record, ("target_mixtures.csv", "target_classification.csv", "summary.json")
        )
        record.update(
            status="completed",
            completed_utc=now(),
            target_wall_seconds=time.perf_counter() - started,
        )
        save_record(output, record)
        print(
            f"EFN absolute-bias contrast {primary['mean']:+.6f} "
            f"[{primary['bootstrap_95_lo']:+.6f}, {primary['bootstrap_95_hi']:+.6f}].",
            flush=True,
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_record(output, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "fit", "calibrate", "source", "target"), required=True
    )
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        raise ValueError("Use a simple, unique run ID")
    output = Path("results/task0009") / args.run_id
    artifacts = Path("artifacts/task0009") / args.run_id
    lock_dir = Path("work/efn-locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    operation = {
        "prepare": prepare_stage,
        "fit": fit_stage,
        "calibrate": calibrate_stage,
        "source": source_stage,
        "target": target_stage,
    }[args.stage]
    with (lock_dir / f"{args.run_id}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        operation(args.run_id, output, artifacts)


if __name__ == "__main__":
    main()
