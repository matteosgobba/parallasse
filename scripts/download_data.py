"""Acquire the two declared files; no model or target quality measurements."""

import argparse
import json
from pathlib import Path

from parallasse.download import download_declared_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("results/task0001"))
    parser.add_argument("--generator", choices=["pythia", "herwig"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for spec in config["files"]:
        if args.generator and args.generator != spec["generator"]:
            continue
        result = download_declared_file(spec, args.raw_dir, args.metadata_dir)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
