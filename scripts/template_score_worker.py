"""Internal Torch-only inference worker with independently checked role authorization."""

import argparse
from pathlib import Path

import numpy as np
from heldout_worker import neural_scores_for_cohort

from parallasse.heldout import read_json, verify_hashes
from parallasse.score_template_design import CHECKPOINTS
from parallasse.template_workflow import check_score_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("job_sha256")
    args = parser.parse_args()
    verify_hashes({str(args.job): args.job_sha256})
    job = read_json(args.job)
    contract = check_score_job(job)
    with np.load(job["inputs"], allow_pickle=False) as inputs:
        predictions = neural_scores_for_cohort(contract, inputs)
    with Path(job["destination"]).open("xb") as handle:
        np.savez(
            handle,
            keys=np.array(CHECKPOINTS),
            scores=np.column_stack([predictions[k] for k in CHECKPOINTS]),
        )


if __name__ == "__main__":
    main()
