"""Source-only executable freeze and measured resource gate for revision v0.10."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.score_template_design import CHECKPOINTS
from parallasse.template_campaign import CUT, SOURCE, CampaignBudget, read_trials, run_cell
from parallasse.template_engine_checks import old_source_inputs, synthetic_checks
from parallasse.template_feasibility import mixture_weights
from parallasse.template_gate_review import assess_proposed_cell
from parallasse.template_inference import ProfileLikelihood
from parallasse.template_revision import REVIEW, REVISION, resolve_protocol
from parallasse.template_revision_scoring import score_rows
from parallasse.template_sampling import histograms
from parallasse.template_workflow import atomic_json, load_engine, now, selected_model_contract


def benchmark_revision(settings, config, inputs, cutoffs, inherited_cpu, deadline):
    laws = histograms(inputs[-1]).astype(float)
    laws /= laws.sum(axis=-1, keepdims=True)
    rng = np.random.default_rng(settings["runtime_seed"])
    tilt = np.exp(
        settings["synthetic_variation_tilt"] * np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    )
    simulation_cutoffs = [
        config["calibration"]["minimum_lr_cutoff"],
        max(v for row in cutoffs[SOURCE].values() for v in row.values()),
    ]
    cases = []
    for i, key in enumerate(CHECKPOINTS):
        p = laws[i]
        h = p * tilt
        h /= h.sum(axis=-1, keepdims=True)
        for size in config["grid"]["sample_sizes"]:
            for f in config["grid"]["fractions"]:
                for kind, variations in [
                    ("source", [None]),
                    ("synthetic_simulation", [0.0, 0.5, 1.0]),
                ]:
                    for alpha in variations:
                        if time.perf_counter() > deadline:
                            raise TimeoutError("Revision resource pilot ceiling")
                        t = p if alpha is None else np.vstack([p, h])
                        aux = np.array(
                            [
                                rng.multinomial(settings["benchmark_template_rows_per_class"], law)
                                for law in t
                            ]
                        )
                        data = rng.multinomial(size, mixture_weights(f, alpha)[0] @ t)
                        start = time.process_time()
                        model = ProfileLikelihood(data, aux, config["numerics"])
                        model.lr_at(f)
                        pair_cpu = time.process_time() - start
                        for cutoff in (
                            [cutoffs[SOURCE][key][str(size)]]
                            if alpha is None
                            else simulation_cutoffs
                        ):
                            start = time.process_time()
                            interval = ProfileLikelihood(
                                data, aux, config["numerics"]
                            ).confidence_set(cutoff)
                            cases.append(
                                {
                                    "key": key,
                                    "kind": kind,
                                    "sample_size": size,
                                    "fraction": f,
                                    "alpha": alpha,
                                    "cutoff": cutoff,
                                    "interval_cpu_seconds": time.process_time() - start,
                                    "calibration_pair_cpu_seconds": pair_cpu,
                                    "scan_nodes": interval["scan_nodes"],
                                }
                            )
    p95 = {
        kind: {
            name: float(np.quantile([r[name] for r in cases if r["kind"] == kind], 0.95))
            for name in ["interval_cpu_seconds", "calibration_pair_cpu_seconds"]
        }
        for kind in ["source", "synthetic_simulation"]
    }
    resources = config["resources"]
    seconds = (
        resources["source_confidence_sets"] * p95["source"]["interval_cpu_seconds"]
        + resources["simulation_confidence_sets"]
        * p95["synthetic_simulation"]["interval_cpu_seconds"]
        + resources["simulation_calibration_lr_pairs"]
        * p95["synthetic_simulation"]["calibration_pair_cpu_seconds"]
    )
    remaining = seconds / 3600 * settings["projection_safety_factor"]
    cumulative = remaining + inherited_cpu / 3600
    return {
        "scope": (
            "Old source laws and declared synthetic tilts only; "
            "actual Herwig cutoffs/runtime unknown"
        ),
        "cases": cases,
        "profile_case_count": len(cases),
        "calibration_pair_count": 144,
        "synthetic_cutoff_proxies": simulation_cutoffs,
        "p95_cpu_seconds": p95,
        "safety_factor": settings["projection_safety_factor"],
        "projected_remaining_cpu_hours": remaining,
        "inherited_cpu_seconds": inherited_cpu,
        "projected_cumulative_cpu_hours": cumulative,
        "cpu_hour_ceiling": 24,
        "resource_gate_passed": bool(cumulative <= 24),
    }


def freeze_revision_engine(run_id):
    output = Path("results/task0013_revision_engine") / run_id
    artifacts = Path("artifacts/task0013_revision_engine") / run_id
    if output.exists() or artifacts.exists():
        raise FileExistsError("Revision engine ID exists; evidence is never overwritten")
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    start = time.perf_counter()
    paths = [
        *sorted(Path("src/parallasse").glob("*.py")),
        *sorted(Path("scripts").glob("*.py")),
        *sorted(Path("tests").glob("*.py")),
        *sorted(Path("tests/fixtures").glob("*.json")),
        Path("requirements-lock.txt"),
        Path("pyproject.toml"),
        REVISION,
        Path("configs/template_revision_checks.json"),
    ]
    snapshot = artifacts / "source_snapshot"
    for path in paths:
        (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot / path)
    implementation = {str(p): sha256_file(p) for p in paths}
    record = {
        "run_id": run_id,
        "status": "checking",
        "started_utc": now(),
        "new_reserved_scores": 0,
        "reserved_validation_roots_drawn": False,
        "source_snapshot": str(snapshot),
        "source_files_sha256": implementation,
    }
    atomic_json(output / "run.json", record)
    try:
        revision = read_json(REVISION)
        review = read_json(REVIEW)
        review_run = read_json(REVIEW.parent / "run.json")
        if (
            review["configuration_sha256"] != sha256_file(REVISION)
            or review_run["status"] != "completed_review"
        ):
            raise ValueError("The selected completed prospective design is required")
        base = load_engine(revision["parent_engine"])
        parent = read_json(Path(revision["preserved_failed_run"]))
        if (
            parent["status"] != "source_failed"
            or read_json(Path(revision["preserved_failed_gate"]))["passed"] is not False
        ):
            raise ValueError("Preserve the failed parent, do not relabel it")
        config = resolve_protocol(revision)
        atomic_json(output / "protocol.json", config)
        contract_path = Path(revision["parent_protocol"]).parent / "model_cut_contract.json"
        shutil.copy2(contract_path, output / "model_cut_contract.json")
        settings = read_json(Path("configs/template_revision_checks.json"))
        inputs = old_source_inputs()
        cutoffs = read_json(Path(revision["source_cutoffs_to_reuse"]))["cutoffs"]
        with threadpool_limits(1):
            print("Replaying old-source PFNs through the revised scoring worker", flush=True)
            n = settings["source_score_rows"]
            actual = score_rows(
                "data/raw/QG_jets.npz",
                inputs[3][:n],
                inputs[4][:n],
                selected_model_contract(inputs[0]),
                artifacts / "old_source_replay",
                {
                    "mode": "old_source_smoke",
                    "source_proof_sha256": inputs[2],
                    "reservation": base["reservation"],
                },
            )
            if not np.array_equal(actual, inputs[5][:n]):
                raise ValueError(
                    "Revised worker does not replay the frozen source predictions exactly"
                )
            replay = {
                "rows": n,
                "probabilities": n * 6,
                "maximum_absolute_difference": 0.0,
                "exact": True,
                "scope": "Old Pythia test prefix only; no reserved-role scoring",
            }
            atomic_json(output / "source_replay.json", replay)
            synthetic = synthetic_checks(config)
            atomic_json(output / "synthetic.json", synthetic)
            smoke_record = {"status": "engineering_smoke", "smoke_only": True, "cpu_seconds": 0.0}
            budget = CampaignBudget(smoke_record, artifacts / "smoke_run.json", 240)
            kwargs = {
                "directory": artifacts / "smoke_cell",
                "pools": inputs[-1],
                "config": config,
                "cell": (1000, 0.5, None),
                "stream": np.random.SeedSequence(settings["smoke_root"]),
                "methods": [CUT, SOURCE],
                "kind": "validation",
                "cutoffs": cutoffs,
                "budget": budget,
                "repetitions": settings["smoke_repetitions"],
                "chunk_size": 2,
            }
            index = run_cell(**kwargs)
            trials = read_trials(index)
            if not all(r["success"] for r in trials):
                raise ValueError("Engineering source smoke contains unresolved fits")
            if index != run_cell(**kwargs):
                raise ValueError("Engineering resume changed committed draws")
            q = config["qualification"]
            toy = {
                "boundary_957_accepted": assess_proposed_cell(957, 1000, 0, 0.1, q)["qualified"],
                "boundary_956_rejected": not assess_proposed_cell(956, 1000, 0, 0.1, q)[
                    "qualified"
                ],
                "uninformative_full_span_rejected": not assess_proposed_cell(1000, 1000, 0, 1.0, q)[
                    "qualified"
                ],
            }
            if not all(toy.values()):
                raise ValueError("Qualification fixtures disagree with prospective design")
            atomic_json(
                output / "source_smoke.json",
                {
                    "smoke_only": True,
                    "repetitions": settings["smoke_repetitions"],
                    "profile_sets": settings["smoke_repetitions"] * 6,
                    "cut_intervals": settings["smoke_repetitions"] * 6,
                    "root": settings["smoke_root"],
                    "all_resolved": True,
                    "resume_exact": True,
                    "synthetic_summary_fixtures": toy,
                    "scope": "Engineering only; cannot authorize any reserved role",
                },
            )
            print(
                "Benchmarking frozen numerical engine and the inherited cumulative budget",
                flush=True,
            )
            runtime = benchmark_revision(
                settings,
                config,
                inputs,
                cutoffs,
                parent["cpu_seconds"],
                start + settings["runtime_ceiling_seconds"],
            )
            atomic_json(output / "runtime.json", runtime)
        if not runtime["resource_gate_passed"]:
            raise RuntimeError("Cumulative resource projection exceeds 24 CPU hours")
        verify_hashes(implementation)
        verify_hashes(implementation, snapshot)
        input_paths = [
            REVISION,
            REVIEW,
            REVIEW.parent / "run.json",
            Path(revision["parent_protocol"]),
            Path(revision["parent_engine"]),
            Path(revision["preserved_failed_run"]),
            Path(revision["preserved_failed_gate"]),
            Path(revision["source_cutoffs_to_reuse"]),
            Path(revision["source_score_reference"]),
            output / "protocol.json",
            output / "model_cut_contract.json",
            *[
                output / name
                for name in [
                    "source_replay.json",
                    "synthetic.json",
                    "source_smoke.json",
                    "runtime.json",
                ]
            ],
        ]
        hashes = {**inputs[2], **{str(p): sha256_file(p) for p in input_paths}}
        engine = {
            "state": "revision_executable_frozen",
            "smoke_only": False,
            "frozen_utc": now(),
            "base_engine": revision["parent_engine"],
            "reservation": base["reservation"],
            "protocol_path": str(output / "protocol.json"),
            "protocol_version": config["protocol_version"],
            "review_design_sha256": sha256_file(REVIEW),
            "inherited_cpu_seconds": parent["cpu_seconds"],
            "source_score_reference": parent["score_archives"]["pythia_template"],
            "source_cutoffs_reference": parent["source_cutoffs"],
            "resource_gate_passed": True,
            "implementation_sha256": implementation,
            "source_snapshot": str(snapshot),
            "input_sha256": hashes,
            "source_and_model_gate_status": (
                "pending fresh independent controls; no evaluation access"
            ),
        }
        path = output / "revision_engine_freeze.json"
        atomic_json(path, engine)
        record.update(
            status="completed",
            completed_utc=now(),
            wall_seconds=time.perf_counter() - start,
            input_sha256=hashes,
            engine_freeze_sha256=sha256_file(path),
        )
        atomic_json(output / "run.json", record)
        return record
    except Exception as error:
        record.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            wall_seconds=time.perf_counter() - start,
        )
        atomic_json(output / "run.json", record)
        raise
