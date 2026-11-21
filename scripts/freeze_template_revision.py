"""Freeze the complete v0.10 executable after old-source/synthetic checks."""

import argparse
import fcntl
import re
from pathlib import Path

from parallasse.template_revision_freeze import freeze_revision_engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("Simple unique ID required")
    if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        parser.error("Run from canonical root")
    path = Path("work/template-revision-engine-locks") / f"{args.run_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = freeze_revision_engine(args.run_id)
    print(f"Revision executable {args.run_id}: {result['status']}", flush=True)
