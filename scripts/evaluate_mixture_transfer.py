"""Validate source interval coverage before exploratory Herwig mixture inference."""

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
from run_source_closure import source_inputs

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.heldout_metrics import KEYS, checked_scores
from parallasse.mixture_uncertainty import (
    bias_contrasts,
    bootstrap_acceptance,
    expected_bias,
    inference_summary,
)
from parallasse.mixtures import mixture_counts

CONFIG = Path("configs/mixture_uncertainty.json")


def write_metrics(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(mode, fraction, size, accepted, eq, eg, nq, ng):
    stats = inference_summary(accepted, size, fraction, eq, eg, nq, ng)
    rows = []
    for key, values in zip(KEYS, stats, strict=True):
        model, seed = key.rsplit("-", 1)
        rows.append(
            {
                "mode": mode,
                "fraction": fraction,
                "sample_size": size,
                "model": model,
                "seed": int(seed),
                **values,
            }
        )
    return rows


def load_source_pools(config):
    previous = read_json(Path("results/task0006/validation.json"))
    run_path = Path(config["source_closure_run"])
    verify_hashes({str(run_path): previous["evidence_sha256"][str(run_path)]})
    run = read_json(run_path)
    verify_hashes(run["input_sha256"])
    verify_hashes(run["output_sha256"])
    verify_hashes(run["source_files_sha256"])
    base = read_json(Path(config["parent_protocol"]))
    if base != read_json(run_path.parent / "protocol.json") or run["status"] != "completed":
        raise ValueError("Source mixture design or completed run changed")
    rows, labels, passed, inputs = source_inputs(base)
    partition = run["partition"]
    verify_hashes({partition["path"]: partition["sha256"]})
    pools = {}
    with np.load(partition["path"], allow_pickle=False) as archive:
        for name in ("response", "population"):
            indices = archive[f"{name}_indices"]
            if not np.array_equal(rows[indices], archive[f"{name}_rows"]):
                raise ValueError("Saved source partition is misaligned")
            pools[name] = [passed[indices][labels[indices] == k] for k in (1, 0)]
    inputs.update(
        {
            str(run_path): sha256_file(run_path),
            **run["output_sha256"],
            partition["path"]: partition["sha256"],
            run["draws"]["path"]: run["draws"]["sha256"],
        }
    )
    verify_hashes(inputs)
    return base, run, pools, inputs


def source_stage(run_id, output, artifacts):
    if output.exists() or artifacts.exists():
        raise FileExistsError("Mixture run already exists; evidence is never overwritten")
    config = read_json(CONFIG)
    if (
        config["protocol_version"] != "0.6-mixture-uncertainty"
        or config["interval"]["level"] != 0.95
    ):
        raise ValueError("Unsupported uncertainty protocol")
    base, source_run, pools, inputs = load_source_pools(config)
    output.mkdir(parents=True)
    record = {
        "task": "TASK-0007",
        "run_id": run_id,
        "status": "preparing",
        "protocol_version": config["protocol_version"],
        "started_utc": datetime.now(UTC).isoformat(),
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
            *sorted(Path("src/parallasse").glob("*.py")),
            CONFIG,
            Path(config["parent_protocol"]),
            Path("scripts/evaluate_mixture_transfer.py"),
            Path("scripts/run_source_closure.py"),
            Path("scripts/evaluate_heldout.py"),
            Path("tests/test_mixture_uncertainty.py"),
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
        record.update(design_frozen_utc=datetime.now(UTC).isoformat(), status="source_sampling")
        write_json(output / "run.json", record, "w")
        q, g = pools["population"]
        fixed_q, fixed_g = [v.mean(axis=0) for v in pools["response"]]
        nq, ng = [len(v) for v in pools["response"]]
        cells = [(f, n) for f in base["fractions"] for n in base["sample_sizes"]]
        streams = np.random.SeedSequence(config["source_coverage_control"]["rng_seed"]).spawn(
            len(cells)
        )
        rows, arrays = [], {"keys": np.asarray(KEYS)}
        with np.load(source_run["draws"]["path"], allow_pickle=False) as prior:
            for i, ((fraction, size), stream) in enumerate(zip(cells, streams, strict=True)):
                seeds = stream.spawn(3)
                rngs = [np.random.Generator(np.random.PCG64(s)) for s in seeds]
                cq = bootstrap_acceptance(q, nq, base["replicates"], rngs[0])
                cg = bootstrap_acceptance(g, ng, base["replicates"], rngs[1])
                accepted, truth = mixture_counts(q, g, fraction, size, base["replicates"], rngs[2])
                arrays.update(
                    {
                        f"response_quarks_{i}": cq,
                        f"response_gluons_{i}": cg,
                        f"accepted_{i}": accepted,
                        f"quark_counts_{i}": truth,
                    }
                )
                rows.extend(
                    metric_rows(
                        "source_repeated_response",
                        fraction,
                        size,
                        accepted,
                        cq / nq,
                        cg / ng,
                        nq,
                        ng,
                    )
                )
                rows.extend(
                    metric_rows(
                        "pythia_fixed_response",
                        fraction,
                        size,
                        prior[f"accepted_{i}"],
                        fixed_q,
                        fixed_g,
                        nq,
                        ng,
                    )
                )
        archive = save_arrays(artifacts / "source_control.npz", **arrays)
        write_metrics(output / "source_metrics.csv", rows)
        control = [r["coverage_95"] for r in rows if r["mode"] == "source_repeated_response"]
        fixed = [r["coverage_95"] for r in rows if r["mode"] == "pythia_fixed_response"]
        nominal_mcse = np.sqrt(0.95 * 0.05 / base["replicates"])
        deviation = max(abs(v - 0.95) / nominal_mcse for v in control)
        summary = {
            "control_coverage_min": min(control),
            "control_coverage_max": max(control),
            "fixed_pool_coverage_min": min(fixed),
            "fixed_pool_coverage_max": max(fixed),
            "max_control_deviation_in_nominal_mcse": float(deviation),
            "source_gate_passed": bool(deviation <= 5),
            "replicates_per_cell": base["replicates"],
            "interpretation": (
                "Repeated-response control tests nominal coverage under a common empirical law. "
                "Fixed-pool coverage conditions on the original response and population."
            ),
        }
        write_json(output / "source_summary.json", summary)
        record.update(
            status="source_validated" if deviation <= 5 else "source_diagnostic_failed",
            source_completed_utc=datetime.now(UTC).isoformat(),
            source_wall_seconds=time.perf_counter() - started,
            source_archive=archive,
            output_sha256={
                str(output / name): sha256_file(output / name)
                for name in ("protocol.json", "source_metrics.csv", "source_summary.json")
            },
        )
        write_json(output / "run.json", record, "w")
        print(
            f"Source coverage range {min(control):.3f} to {max(control):.3f}. "
            f"Gate {record['status']}."
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", record, "w")
        raise


def verified_source_stage(output):
    record = read_json(output / "run.json")
    if record["status"] != "source_validated":
        raise ValueError("Target mixtures require the completed source coverage gate")
    verify_hashes(record["source_files_sha256"])
    verify_hashes(record["source_files_sha256"], Path(record["source_snapshot"]))
    verify_hashes(record["input_sha256"])
    verify_hashes(record["output_sha256"])
    verify_hashes({record["source_archive"]["path"]: record["source_archive"]["sha256"]})
    if not read_json(output / "source_summary.json")["source_gate_passed"]:
        raise ValueError("Source coverage gate did not pass")
    return record


def target_stage(run_id, output, artifacts):
    record = verified_source_stage(output)
    if record["run_id"] != run_id:
        raise ValueError("Run ID differs from the validated source run")
    config = read_json(CONFIG)
    base, _, pools, _ = load_source_pools(config)
    core = read_json(Path(base["source_evaluation_run"]))
    source_q, source_g = pools["response"]
    eq, eg = [v.mean(axis=0) for v in (source_q, source_g)]
    nq, ng = len(source_q), len(source_g)
    started = time.perf_counter()
    record.update(
        status="target_sampling",
        target_opening_utc=datetime.now(UTC).isoformat(),
        target_performance_opened=True,
    )
    write_json(output / "run.json", record, "w")
    try:
        ref = core["score_artifacts"]["target"]
        cohort = read_json(Path("results/task0001/cohorts.json"))["cohorts"]["target"]
        target_inputs = {
            ref["path"]: ref["sha256"],
            cohort["row_ids_path"]: cohort["row_ids_sha256"],
        }
        verify_hashes(target_inputs)
        with np.load(ref["path"], allow_pickle=False) as data:
            labels, scores = data["labels"], data["scores"]
            if (
                tuple(data["keys"]) != KEYS
                or scores.shape != (100000, len(KEYS))
                or labels.shape != (100000,)
                or not np.array_equal(
                    data["rows"], np.load(cohort["row_ids_path"], allow_pickle=False)
                )
                or any(np.sum(labels == k) != 50000 for k in (0, 1))
            ):
                raise ValueError("Target scores, keys, labels or persisted IDs disagree")
        for column in scores.T:
            checked_scores(column)
        lock = read_json(Path(core["cut_freeze"]))
        passed = scores > np.array([lock["cuts"][k]["threshold"] for k in KEYS])
        q, g = [passed[labels == k] for k in (1, 0)]
        hq, hg = q.mean(axis=0), g.mean(axis=0)
        cells = [(f, n) for f in base["fractions"] for n in base["sample_sizes"]]
        streams = np.random.SeedSequence(config["fixed_response_modes"]["rng_seed"]).spawn(
            len(cells)
        )
        rows, arrays = [], {"keys": np.asarray(KEYS)}
        for i, ((fraction, size), stream) in enumerate(zip(cells, streams, strict=True)):
            accepted, truth = mixture_counts(
                q,
                g,
                fraction,
                size,
                base["replicates"],
                np.random.Generator(np.random.PCG64(stream)),
            )
            arrays.update({f"accepted_{i}": accepted, f"quark_counts_{i}": truth})
            rows.extend(
                metric_rows("herwig_fixed_response", fraction, size, accepted, eq, eg, nq, ng)
            )
        draws = save_arrays(artifacts / "target_pseudoexperiments.npz", **arrays)
        write_metrics(output / "target_metrics.csv", rows)
        seeds = np.random.SeedSequence(config["bias_comparison"]["rng_seed"]).spawn(4)
        boot = {"keys": np.asarray(KEYS)}
        for name, pool, seed in zip(
            ("source_q", "source_g", "target_q", "target_g"),
            (source_q, source_g, q, g),
            seeds,
            strict=True,
        ):
            boot[name] = bootstrap_acceptance(
                pool,
                len(pool),
                config["bias_comparison"]["replicates"],
                np.random.Generator(np.random.PCG64(seed)),
            )
        bias = {}
        for i, fraction in enumerate(base["fractions"]):
            point = expected_bias(fraction, eq, eg, hq, hg)
            samples = expected_bias(
                fraction,
                boot["source_q"] / nq,
                boot["source_g"] / ng,
                boot["target_q"] / len(q),
                boot["target_g"] / len(g),
            )
            boot[f"expected_bias_{i}"] = samples
            bias[str(fraction)] = bias_contrasts(point, samples)
        boot_archive = save_arrays(artifacts / "bias_bootstrap.npz", **boot)
        summary = {
            "interpretation": "Exploratory inference with the fixed original Pythia response.",
            "primary_fraction": base["primary_followup_cell"]["fraction"],
            "primary_pseudo_data_size": base["primary_followup_cell"]["sample_size"],
            "by_fraction": bias,
            "scope": (
                "Bias intervals resample finite source-response and Herwig class pools. "
                "Per-pseudoexperiment intervals propagate counts and source-response precision. "
                "Models and numerical cuts are fixed."
            ),
            "coverage_scope": (
                "Herwig coverage conditions on the observed response and population, "
                "as does the original-pool Pythia diagnostic."
            ),
            "target_quark_acceptance": hq.tolist(),
            "target_gluon_acceptance": hg.tolist(),
            "source_response_quark_acceptance": eq.tolist(),
            "source_response_gluon_acceptance": eg.tolist(),
            "draws": draws,
            "bias_bootstrap": boot_archive,
        }
        write_json(output / "summary.json", summary)
        verify_hashes({**record["input_sha256"], **target_inputs})
        verify_hashes(record["output_sha256"])
        record["output_sha256"].update(
            {
                str(output / name): sha256_file(output / name)
                for name in ("target_metrics.csv", "summary.json")
            }
        )
        record["input_sha256"].update(target_inputs)
        record.update(
            status="completed",
            completed_utc=datetime.now(UTC).isoformat(),
            target_wall_seconds=time.perf_counter() - started,
            target_draws=draws,
            bias_bootstrap=boot_archive,
        )
        write_json(output / "run.json", record, "w")
        print(f"Saved {len(rows)} Herwig metric rows and the paired absolute-bias contrasts.")
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", record, "w")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("source", "target"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        raise ValueError("Use a simple, unique run ID")
    output = Path("results/task0007") / args.run_id
    artifacts = Path("artifacts/task0007") / args.run_id
    operation = source_stage if args.stage == "source" else target_stage
    operation(args.run_id, output, artifacts)


if __name__ == "__main__":
    main()
