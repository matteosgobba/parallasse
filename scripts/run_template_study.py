"""Run a frozen score-template study through its source/model/evaluation gates."""

import argparse
import fcntl
import re
from pathlib import Path

from threadpoolctl import threadpool_limits

from parallasse.template_campaign import create_study, execute_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "source", "model", "evaluate")
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--engine-freeze", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("Simple unique run ID required")
    if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        parser.error("Run from canonical project root")
    lock_path = Path("work/template-study-locks") / f"{args.run_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock, threadpool_limits(1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            if args.engine_freeze is None:
                parser.error("prepare requires --engine-freeze")
            result = create_study(args.run_id, args.engine_freeze)
        else:
            result = execute_stage(args.run_id, args.stage)
    print(f"Study {args.run_id}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
