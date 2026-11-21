"""Check and freeze the executable template engine using only old source/synthetic data."""

import argparse
import fcntl
import importlib.metadata
import re
import shutil
import sys
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.template_engine_checks import (
    old_source_inputs,
    replay_source_scores,
    runtime_probe,
    source_smoke,
    synthetic_checks,
)
from parallasse.template_workflow import RESERVATION, atomic_json, now


def freeze_engine(run_id):
    output, artifacts = Path("results/task0012") / run_id, Path("artifacts/task0012") / run_id
    if output.exists() or artifacts.exists():
        raise FileExistsError("Engine ID exists; evidence is never overwritten")
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    settings = read_json(Path("configs/template_engine_checks.json"))
    started = time.perf_counter()
    record = {
        "task": "TASK-0012",
        "run_id": run_id,
        "status": "checking",
        "started_utc": now(),
        "fresh_scores_computed": False,
        "new_classifier_fits": 0,
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "scipy", "torch", "xgboost", "threadpoolctl")
        },
        "python": sys.version,
        "scope": settings["scope"],
        "source_only_repair": settings.get("source_only_repair"),
    }
    paths = [
        *sorted(Path("src/parallasse").glob("*.py")),
        *[
            Path("scripts") / name
            for name in (
                "freeze_template_engine.py",
                "run_template_study.py",
                "template_score_worker.py",
                "heldout_worker.py",
            )
        ],
        Path("tests/test_template_engine.py"),
        Path("tests/test_template_workflow.py"),
        Path("tests/fixtures/template_optimizer_source.json"),
        Path("configs/template_engine_checks.json"),
        Path("requirements-lock.txt"),
        Path("pyproject.toml"),
    ]
    snapshot = artifacts / "source_snapshot"
    for p in paths:
        (snapshot / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, snapshot / p)
    implementation = {str(p): sha256_file(p) for p in paths}
    verify_hashes(implementation, snapshot)
    record.update(source_files_sha256=implementation, source_snapshot=str(snapshot))
    atomic_json(output / "run.json", record)
    try:
        inputs = old_source_inputs()
        reservation, config, source_hashes, *_ = inputs
        with threadpool_limits(settings["numerical_threads"]):
            print("Replaying six frozen PFNs on the original Pythia prefix", flush=True)
            replay = replay_source_scores(artifacts, settings, inputs)
            atomic_json(output / "source_replay.json", replay)
            print("Checking boundaries, separated sets and non-identification", flush=True)
            synthetic = synthetic_checks(config)
            atomic_json(output / "synthetic.json", synthetic)
            print("Running bounded source calibration/interval/resume smoke", flush=True)
            smoke = source_smoke(artifacts, settings, inputs)
            atomic_json(output / "source_smoke.json", smoke)
            print("Timing the complete source/synthetic parameter grid", flush=True)
            benchmark = runtime_probe(
                settings, inputs, started + settings["runtime_ceiling_seconds"]
            )
            atomic_json(output / "runtime.json", benchmark)
        if not benchmark["resource_gate_passed"]:
            raise RuntimeError("Measured full-grid CPU projection exceeds the frozen ceiling")
        verify_hashes(implementation)
        hashes = {**source_hashes, str(RESERVATION): sha256_file(RESERVATION)}
        if settings.get("source_only_repair"):
            superseded = (
                Path("results/task0012") / settings["source_only_repair"]["supersedes"] / "run.json"
            )
            hashes[str(superseded)] = sha256_file(superseded)
        hashes.update(
            {
                str(output / name): sha256_file(output / name)
                for name in (
                    "source_replay.json",
                    "source_smoke.json",
                    "synthetic.json",
                    "runtime.json",
                )
            }
        )
        engine = {
            "state": "executable_frozen",
            "smoke_only": False,
            "frozen_utc": now(),
            "reservation": str(RESERVATION),
            "protocol_path": reservation["protocol_path"],
            "implementation_sha256": implementation,
            "source_snapshot": str(snapshot),
            "input_sha256": hashes,
            "resource_gate_passed": True,
            "calibrated_lr_cutoffs": "pending_TASK-0013; smoke cutoffs are not valid gates",
            "source_and_model_gate_status": "pending; neither final evaluation role authorized",
        }
        atomic_json(output / "executable_freeze.json", engine)
        record.update(
            status="completed",
            completed_utc=now(),
            wall_seconds=time.perf_counter() - started,
            engine_freeze_sha256=sha256_file(output / "executable_freeze.json"),
            source_files_sha256=implementation,
            source_snapshot=str(snapshot),
            input_sha256=hashes,
            output_sha256={
                str(output / "executable_freeze.json"): sha256_file(
                    output / "executable_freeze.json"
                )
            },
        )
        atomic_json(output / "run.json", record)
        print(
            f"Executable frozen; CPU projection "
            f"{benchmark['projected_cpu_hours_with_margin']:.2f} h; no reserved scores",
            flush=True,
        )
    except Exception as error:
        record.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            wall_seconds=time.perf_counter() - started,
        )
        atomic_json(output / "run.json", record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("Simple unique run ID required")
    if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        parser.error("Run from canonical project root")
    lock_path = Path("work/template-engine-locks") / f"{args.run_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_engine(args.run_id)
