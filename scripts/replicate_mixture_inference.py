"""Audit and freeze new events, validate source coverage, then replicate on Herwig."""

import argparse
import platform
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
from evaluate_mixture_transfer import metric_rows, write_metrics

from parallasse.data import CANONICAL_VERSION, fingerprints, integrity_summary, load_jets
from parallasse.download import download_declared_file, sha256_file
from parallasse.heldout import INPUT_FREEZE, read_json, verify_hashes, verify_input_freeze
from parallasse.heldout_metrics import KEYS, checked_scores, cohort_metrics
from parallasse.mixture_uncertainty import bias_contrasts, bootstrap_acceptance, expected_bias
from parallasse.mixtures import mixture_counts
from parallasse.replication import prepare_replication_inputs, replication_cohorts

CONFIG = Path("configs/replication.json")
CORE = Path("results/task0004/20261110-heldout-01/run.json")


def audit_stage(run_id, output, artifacts):
    if output.exists() or artifacts.exists():
        raise FileExistsError("Replication run already exists; evidence is never overwritten")
    config = read_json(CONFIG)
    if (
        config["protocol_version"] != "0.7-independent-event-replication"
        or config["with_bc"] is not False
    ):
        raise ValueError("Unsupported replication protocol")
    previous = read_json(Path("results/task0007/validation.json"))
    verify_hashes(
        {
            p: h
            for p, h in previous["implementation_sha256"].items()
            if p not in ("README.md", "AGENTS.md")
        }
    )
    first = read_json(Path("results/task0001/validation.json"))
    verify_hashes(first["evidence_sha256"])
    verify_input_freeze()
    core_validation = read_json(Path("results/task0004/validation.json"))
    verify_hashes({str(CORE): core_validation["evidence_sha256"][str(CORE)]})
    core = read_json(CORE)
    verify_hashes({core["cut_freeze"]: core["cut_freeze_sha256"]})
    output.mkdir(parents=True)
    record = {
        "task": "TASK-0008",
        "run_id": run_id,
        "status": "auditing",
        "protocol_version": config["protocol_version"],
        "started_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "new_fits": 0,
        "source_predictions_opened": False,
        "target_predictions_opened": False,
        "model_freeze_sha256": sha256_file(INPUT_FREEZE),
        "input_sha256": {
            str(CORE): sha256_file(CORE),
            str(INPUT_FREEZE): sha256_file(INPUT_FREEZE),
            core["cut_freeze"]: core["cut_freeze_sha256"],
            "results/task0007/validation.json": sha256_file(
                Path("results/task0007/validation.json")
            ),
        },
    }
    write_json(output / "run.json", record)
    started = time.perf_counter()
    try:
        artifacts.mkdir(parents=True)
        paths = [
            *sorted(Path("src/parallasse").glob("*.py")),
            CONFIG,
            Path("configs/mixture_uncertainty.json"),
            Path("configs/mixtures.json"),
            Path("scripts/replicate_mixture_inference.py"),
            Path("scripts/heldout_worker.py"),
            Path("scripts/evaluate_heldout.py"),
            Path("scripts/evaluate_mixture_transfer.py"),
            Path("scripts/run_source_closure.py"),
            Path("tests/test_replication.py"),
            Path("requirements-lock.txt"),
        ]
        snapshot = artifacts / "source_snapshot"
        for path in paths:
            (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot / path)
        record.update(
            source_snapshot=str(snapshot),
            source_files_sha256={str(p): sha256_file(p) for p in paths},
        )
        verify_hashes(record["source_files_sha256"], snapshot)
        write_json(output / "protocol.json", config)
        original = read_json(Path("configs/data.json"))
        all_hashes, all_labels, summaries = {}, {}, {}
        for kind, spec in [(f"old_{s['role']}", s) for s in original["files"]] + [
            (s["role"], s) for s in config["files"]
        ]:
            raw = Path("data/raw") / spec["filename"]
            if not kind.startswith("old_"):
                provenance = download_declared_file(spec, raw.parent, output)
                record["input_sha256"][str(raw)] = provenance["sha256"]
            else:
                expected = read_json(Path(f"results/task0001/{spec['generator']}_provenance.json"))[
                    "sha256"
                ]
                verify_hashes({str(raw): expected})
                record["input_sha256"][str(raw)] = expected
            particles, labels = load_jets(raw)
            if len(labels) != spec["expected_rows"]:
                raise ValueError("Unexpected file row count")
            hashes = fingerprints(particles)
            summaries[kind] = integrity_summary(particles, labels, hashes)
            all_hashes[kind], all_labels[kind] = hashes, labels
            del particles
            print(f"Audited {kind} {spec['filename']}: {len(labels)} jets", flush=True)
        arrays, counts = replication_cohorts(
            all_labels["source"],
            all_hashes["source"],
            all_labels["target"],
            all_hashes["target"],
            np.r_[all_labels["old_source"], all_labels["old_target"]],
            np.r_[all_hashes["old_source"], all_hashes["old_target"]],
            config["partition_seed"],
        )
        identities = save_arrays(artifacts / "identity_hashes.npz", **all_hashes)
        cohorts = save_arrays(artifacts / "cohorts.npz", **arrays)
        write_json(
            output / "audit.json",
            {
                "canonical_definition": CANONICAL_VERSION,
                "datasets": summaries,
                "retained_counts": counts,
                "identities": identities,
                "cohorts": cohorts,
                "old_identity_overlap_excluded": True,
                "new_source_target_disjoint": True,
            },
        )
        record.update(
            status="audited",
            audit_completed_utc=datetime.now(UTC).isoformat(),
            audit_wall_seconds=time.perf_counter() - started,
            cohorts=cohorts,
            identities=identities,
            output_sha256={
                str(output / name): sha256_file(output / name)
                for name in (
                    "protocol.json",
                    "audit.json",
                    "pythia_provenance.json",
                    "herwig_provenance.json",
                )
            },
        )
        write_json(output / "run.json", record, "w")
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", record, "w")
        raise


def verified_stage(output, expected):
    record = read_json(output / "run.json")
    if record["status"] != expected:
        raise ValueError(f"Replication stage requires state {expected}")
    verify_hashes(record["input_sha256"])
    verify_hashes(record["output_sha256"])
    verify_hashes(record["source_files_sha256"])
    verify_hashes(record["source_files_sha256"], Path(record["source_snapshot"]))
    for key in ("cohorts", "identities"):
        verify_hashes({record[key]["path"]: record[key]["sha256"]})
    if "source_scores" in record:
        verify_hashes({record["source_scores"]["path"]: record["source_scores"]["sha256"]})
    if "source_draws" in record:
        verify_hashes({record["source_draws"]["path"]: record["source_draws"]["sha256"]})
    if (
        expected == "source_validated"
        and not read_json(output / "source_summary.json")["source_gate_passed"]
    ):
        raise ValueError("Source interval coverage gate did not pass")
    return record


def score_new_cohort(config, frozen, role, rows, expected_labels, artifacts):
    spec = next(s for s in config["files"] if s["role"] == role)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"{role}-", dir=artifacts) as folder:
        scratch = Path(folder)
        inputs = scratch / "inputs.npz"
        labels = prepare_replication_inputs(
            Path("data/raw") / spec["filename"], rows, frozen, inputs
        )
        if not np.array_equal(labels, expected_labels):
            raise ValueError("Audited row IDs and raw labels differ")
        scores = {}
        for family in ("xgb", "neural"):
            destination = scratch / f"{family}.npz"
            with (artifacts / f"{role}-{family}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/heldout_worker.py",
                        family,
                        str(INPUT_FREEZE),
                        sha256_file(INPUT_FREEZE),
                        str(inputs),
                        str(destination),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            with np.load(destination, allow_pickle=False) as values:
                if set(scores) & set(values.files):
                    raise ValueError("Repeated inference key")
                scores.update({key: checked_scores(values[key]) for key in values.files})
        if set(scores) != set(KEYS) or any(v.shape != labels.shape for v in scores.values()):
            raise ValueError("Incomplete frozen-checkpoint inference")
        predictions = np.column_stack([scores[key] for key in KEYS])
        ref = save_arrays(
            artifacts / f"{role}_scores.npz",
            rows=rows,
            labels=labels,
            keys=np.asarray(KEYS),
            scores=predictions,
        )
    ref["wall_seconds"] = time.perf_counter() - started
    return predictions, ref


def fixed_cuts():
    core = read_json(CORE)
    lock = read_json(Path(core["cut_freeze"]))
    if lock["operator"] != ">" or set(lock["cuts"]) != set(KEYS):
        raise ValueError("Expected the original strict frozen cuts")
    return {key: lock["cuts"][key]["threshold"] for key in KEYS}


def source_stage(run_id, output, artifacts):
    record = verified_stage(output, "audited")
    if record["run_id"] != run_id:
        raise ValueError("Run ID mismatch")
    config = read_json(CONFIG)
    frozen = verify_input_freeze()
    cuts = fixed_cuts()
    started = time.perf_counter()
    record.update(
        status="source_scoring",
        source_predictions_opened=True,
        numerical_freeze_utc=datetime.now(UTC).isoformat(),
    )
    write_json(output / "run.json", record, "w")
    try:
        with np.load(record["cohorts"]["path"], allow_pickle=False) as data:
            rows, labels = data["source_rows"], data["source_labels"]
            response, population = data["response_indices"], data["population_indices"]
        scores, ref = score_new_cohort(config, frozen, "source", rows, labels, artifacts)
        passed = scores > np.array([cuts[key] for key in KEYS])
        rq, rg = [passed[response][labels[response] == k] for k in (1, 0)]
        q, g = [passed[population][labels[population] == k] for k in (1, 0)]
        nq, ng = len(rq), len(rg)
        eq, eg = rq.mean(axis=0), rg.mean(axis=0)
        cells = [(f, n) for f in config["fractions"] for n in config["sample_sizes"]]
        controls = np.random.SeedSequence(config["source_control_seed"]).spawn(len(cells))
        fixed = np.random.SeedSequence(config["source_pseudo_data_seed"]).spawn(len(cells))
        metrics, arrays = [], {"keys": np.asarray(KEYS)}
        for i, ((f, n), child, fixed_child) in enumerate(zip(cells, controls, fixed, strict=True)):
            children = child.spawn(3)
            rngs = [np.random.Generator(np.random.PCG64(s)) for s in children]
            cq = bootstrap_acceptance(q, nq, config["replicates"], rngs[0])
            cg = bootstrap_acceptance(g, ng, config["replicates"], rngs[1])
            accepted, truth = mixture_counts(q, g, f, n, config["replicates"], rngs[2])
            fixed_counts, fixed_truth = mixture_counts(
                q, g, f, n, config["replicates"], np.random.Generator(np.random.PCG64(fixed_child))
            )
            metrics.extend(
                metric_rows("source_repeated_response", f, n, accepted, cq / nq, cg / ng, nq, ng)
            )
            metrics.extend(metric_rows("pythia_fixed_response", f, n, fixed_counts, eq, eg, nq, ng))
            arrays.update(
                {
                    f"response_quarks_{i}": cq,
                    f"response_gluons_{i}": cg,
                    f"control_accepted_{i}": accepted,
                    f"control_quark_counts_{i}": truth,
                    f"fixed_accepted_{i}": fixed_counts,
                    f"fixed_quark_counts_{i}": fixed_truth,
                }
            )
        draws = save_arrays(artifacts / "source_draws.npz", **arrays)
        write_metrics(output / "source_metrics.csv", metrics)
        control = [r["coverage_95"] for r in metrics if r["mode"] == "source_repeated_response"]
        conditional = [r["coverage_95"] for r in metrics if r["mode"] == "pythia_fixed_response"]
        deviation = max(
            abs(v - 0.95) / np.sqrt(0.95 * 0.05 / config["replicates"]) for v in control
        )
        point_rows, _ = cohort_metrics(labels, scores, cuts, np.random.default_rng(0), 0)
        write_metrics(output / "source_classification.csv", point_rows)
        write_json(
            output / "source_summary.json",
            {
                "source_gate_passed": bool(deviation <= 5),
                "control_coverage_min": min(control),
                "control_coverage_max": max(control),
                "fixed_pool_coverage_min": min(conditional),
                "fixed_pool_coverage_max": max(conditional),
                "max_deviation_in_nominal_mcse": float(deviation),
                "response_quarks": nq,
                "response_gluons": ng,
                "population_quarks": len(q),
                "population_gluons": len(g),
                "response_quark_acceptance": eq.tolist(),
                "response_gluon_acceptance": eg.tolist(),
            },
        )
        record.update(
            status="source_validated" if deviation <= 5 else "source_diagnostic_failed",
            source_completed_utc=datetime.now(UTC).isoformat(),
            source_wall_seconds=time.perf_counter() - started,
            source_scores=ref,
            source_draws=draws,
        )
        record["output_sha256"].update(
            {
                str(output / name): sha256_file(output / name)
                for name in (
                    "source_metrics.csv",
                    "source_summary.json",
                    "source_classification.csv",
                )
            }
        )
        write_json(output / "run.json", record, "w")
        print(
            f"Source coverage {min(control):.3f} to {max(control):.3f}. State {record['status']}."
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", record, "w")
        raise


def target_stage(run_id, output, artifacts):
    record = verified_stage(output, "source_validated")
    if record["run_id"] != run_id:
        raise ValueError("Run ID mismatch")
    config = read_json(CONFIG)
    frozen = verify_input_freeze()
    cuts = fixed_cuts()
    started = time.perf_counter()
    record.update(
        status="target_scoring",
        target_predictions_opened=True,
        target_opening_utc=datetime.now(UTC).isoformat(),
    )
    write_json(output / "run.json", record, "w")
    try:
        with np.load(record["cohorts"]["path"], allow_pickle=False) as c:
            target_rows, target_labels = c["target_rows"], c["target_labels"]
            response, source_labels = c["response_indices"], c["source_labels"]
        with np.load(record["source_scores"]["path"], allow_pickle=False) as s:
            spass = s["scores"] > np.array([cuts[k] for k in KEYS])
        rq, rg = [spass[response][source_labels[response] == k] for k in (1, 0)]
        nq, ng = len(rq), len(rg)
        eq, eg = rq.mean(axis=0), rg.mean(axis=0)
        scores, ref = score_new_cohort(
            config, frozen, "target", target_rows, target_labels, artifacts
        )
        passed = scores > np.array([cuts[k] for k in KEYS])
        q, g = [passed[target_labels == k] for k in (1, 0)]
        hq, hg = q.mean(axis=0), g.mean(axis=0)
        cells = [(f, n) for f in config["fractions"] for n in config["sample_sizes"]]
        streams = np.random.SeedSequence(config["target_pseudo_data_seed"]).spawn(len(cells))
        metrics, arrays = [], {"keys": np.asarray(KEYS)}
        for i, ((f, n), stream) in enumerate(zip(cells, streams, strict=True)):
            counts, truth = mixture_counts(
                q, g, f, n, config["replicates"], np.random.Generator(np.random.PCG64(stream))
            )
            metrics.extend(metric_rows("herwig_fixed_response", f, n, counts, eq, eg, nq, ng))
            arrays.update({f"accepted_{i}": counts, f"quark_counts_{i}": truth})
        draws = save_arrays(artifacts / "target_draws.npz", **arrays)
        write_metrics(output / "target_metrics.csv", metrics)
        point_rows, _ = cohort_metrics(target_labels, scores, cuts, np.random.default_rng(0), 0)
        write_metrics(output / "target_classification.csv", point_rows)
        boot = {"keys": np.asarray(KEYS)}
        children = np.random.SeedSequence(config["bootstrap_seed"]).spawn(4)
        for name, pool, child in zip(
            ("source_q", "source_g", "target_q", "target_g"), (rq, rg, q, g), children, strict=True
        ):
            boot[name] = bootstrap_acceptance(
                pool,
                len(pool),
                config["bootstrap_replicates"],
                np.random.Generator(np.random.PCG64(child)),
            )
        bias = {}
        for i, f in enumerate(config["fractions"]):
            point = expected_bias(f, eq, eg, hq, hg)
            samples = expected_bias(
                f,
                boot["source_q"] / nq,
                boot["source_g"] / ng,
                boot["target_q"] / len(q),
                boot["target_g"] / len(g),
            )
            boot[f"expected_bias_{i}"] = samples
            bias[str(f)] = bias_contrasts(point, samples)
        boot_ref = save_arrays(artifacts / "bias_bootstrap.npz", **boot)
        primary = bias[str(config["primary_fraction"])]["contrasts"]["PFN-ID_minus_PFN-K"]
        write_json(
            output / "summary.json",
            {
                "by_fraction": bias,
                "primary_fraction": config["primary_fraction"],
                "primary_sample_size": config["primary_sample_size"],
                "replication_criterion_met": bool(
                    primary["mean"] > 0 and primary["bootstrap_95_lo"] > 0
                ),
                "scope": config["scope"],
                "response_sizes": {"quarks": nq, "gluons": ng},
                "target_sizes": {"quarks": len(q), "gluons": len(g)},
                "target_quark_acceptance": hq.tolist(),
                "target_gluon_acceptance": hg.tolist(),
                "source_response_quark_acceptance": eq.tolist(),
                "source_response_gluon_acceptance": eg.tolist(),
            },
        )
        verify_hashes(record["input_sha256"])
        verify_hashes(record["output_sha256"])
        record.update(
            status="completed",
            completed_utc=datetime.now(UTC).isoformat(),
            target_wall_seconds=time.perf_counter() - started,
            target_scores=ref,
            target_draws=draws,
            bias_bootstrap=boot_ref,
        )
        record["output_sha256"].update(
            {
                str(output / name): sha256_file(output / name)
                for name in ("target_metrics.csv", "target_classification.csv", "summary.json")
            }
        )
        write_json(output / "run.json", record, "w")
        print(
            f"Primary PID absolute-bias contrast {primary['mean']:+.6f}. "
            f"Replica criterion {primary['mean'] > 0 and primary['bootstrap_95_lo'] > 0}."
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", record, "w")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "source", "target"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        raise ValueError("Use a simple, unique run ID")
    output = Path("results/task0008") / args.run_id
    artifacts = Path("artifacts/task0008") / args.run_id
    operation = {"audit": audit_stage, "source": source_stage, "target": target_stage}[args.stage]
    operation(args.run_id, output, artifacts)


if __name__ == "__main__":
    main()
