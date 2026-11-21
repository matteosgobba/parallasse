"""Freeze a deterministic calibration/gate review, without scoring or validation draws."""

import argparse
import fcntl
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

from scipy.stats import binom

from parallasse.download import sha256_file
from parallasse.heldout import read_json, verify_hashes
from parallasse.template_gate_review import ideal_operating_characteristics, required_hits
from parallasse.template_workflow import atomic_json, load_engine, now


def review(run_id):
    output = Path("results/task0013_review") / run_id
    artifacts = Path("artifacts/task0013_review") / run_id
    if output.exists() or artifacts.exists():
        raise FileExistsError("Review ID exists; prior evidence is never overwritten")
    config_path = Path("configs/template_validation_revision.json")
    config = read_json(config_path)
    load_engine(config["parent_engine"])
    parent = read_json(Path(config["preserved_failed_run"]))
    gate = read_json(Path(config["preserved_failed_gate"]))
    if parent["status"] != "source_failed" or gate["passed"] is not False:
        raise ValueError("The unchanged failed parent is required")
    if set(parent["score_archives"]) != {"pythia_template"}:
        raise ValueError("Review boundary requires modelling/final roles unscored")
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    started = time.perf_counter()
    sources = list(
        map(
            Path,
            [
                str(config_path),
                "src/parallasse/template_gate_review.py",
                "scripts/review_template_validation.py",
                "tests/test_template_gate_review.py",
                "requirements-lock.txt",
                "pyproject.toml",
            ],
        )
    )
    snapshot = artifacts / "source_snapshot"
    for p in sources:
        (snapshot / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, snapshot / p)
    hashes = {str(p): sha256_file(p) for p in sources}
    inputs = {
        config[key]: sha256_file(Path(config[key]))
        for key in (
            "parent_protocol",
            "parent_engine",
            "preserved_failed_run",
            "preserved_failed_gate",
            "source_cutoffs_to_reuse",
            "source_score_reference",
        )
    }
    record = {
        "run_id": run_id,
        "status": "checking",
        "started_utc": now(),
        "scope": config["scope"],
        "source_files_sha256": hashes,
        "input_sha256": inputs,
        "source_snapshot": str(snapshot),
        "new_scores": 0,
        "new_validation_draws": 0,
        "access_granted": False,
    }
    atomic_json(output / "run.json", record)
    try:
        characteristic = [
            ideal_operating_characteristics(1000, 973, m, 1000, 916, 984) for m in (1, 3, 9)
        ]
        candidates = []
        for name, error in [
            ("marginal_exact_lower_93", 0.05),
            ("family_exact_lower_93", 0.05 / 144),
        ]:
            hits = required_hits(1000, 0.93, error)
            candidates.append(
                {
                    "name": name,
                    "minimum_hits": hits,
                    "per_cell_error": error,
                    "power": {
                        str(p): float(binom.sf(hits - 1, 1000, p))
                        for p in (0.93, 0.95, 0.96, 0.965, 0.97, 0.973, 0.98, 1.0)
                    },
                }
            )
        # Independent exact integer arithmetic for the 93/100 binomial boundary.
        denominator = 100**1000
        tails = {}
        for minimum in (956, 957):
            numerator = sum(
                math.comb(1000, h) * 93**h * 7 ** (1000 - h) for h in range(minimum, 1001)
            )
            tails[str(minimum)] = {
                "probability": numerator / denominator,
                "within_one_over_2880": numerator * 2880 <= denominator,
            }
        if tails["956"]["within_one_over_2880"] or not tails["957"]["within_one_over_2880"]:
            raise ArithmeticError("Exact integer threshold verification failed")
        result = {
            "ideal_continuous_pivot": characteristic,
            "candidate_power": candidates,
            "exact_integer_boundary": tails,
            "limitations": [
                "Illustrative law, not a measured source/Herwig false-veto rate",
                "Shared checkpoints prevent an independent-global-group extrapolation",
                "No retrospective passing gate assigned to v0.9",
                "Proposed qualification can fail a genuinely 95%-covering procedure at n=1000",
            ],
        }
        atomic_json(output / "operating_characteristics.json", result)
        design = {
            "state": "design_only_not_executable",
            "frozen_utc": now(),
            "protocol_version": config["protocol_version"],
            "configuration": str(config_path),
            "configuration_sha256": sha256_file(config_path),
            "input_sha256": inputs,
            "source_files_sha256": hashes,
            "new_validation_streams_drawn": False,
            "implementation_required": True,
            "scoring_or_stage_access_granted": False,
        }
        atomic_json(output / "revision_design_freeze.json", design)
        verify_hashes(hashes)
        verify_hashes(hashes, snapshot)
        verify_hashes(inputs)
        record.update(
            status="completed_review",
            completed_utc=now(),
            wall_seconds=time.perf_counter() - started,
            output_sha256={
                str(p): sha256_file(p)
                for p in (
                    output / "operating_characteristics.json",
                    output / "revision_design_freeze.json",
                )
            },
        )
        atomic_json(output / "run.json", record)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(output / "run.json", record)
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("Simple unique run ID required")
    if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        parser.error("Run from canonical project root")
    lock_path = Path("work/template-review-locks") / f"{args.run_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = review(args.run_id)
    print(
        json.dumps(
            {k: result[k] for k in ("run_id", "status", "new_scores", "new_validation_draws")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
