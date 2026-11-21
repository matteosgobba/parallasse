"""Offline source-training preprocessing smoke check; no classifier training."""

import json
from pathlib import Path

import numpy as np

from parallasse.download import sha256_file
from parallasse.preprocessing import encode_pid, kinematics


def main() -> None:
    root = Path.cwd()
    output = root / "results/task0001"
    cohorts = json.loads((output / "cohorts.json").read_text())
    preprocessing = json.loads((output / "preprocessing.json").read_text())
    train = cohorts["cohorts"]["train"]
    rows_path = root / train["row_ids_path"]
    assert sha256_file(rows_path) == train["row_ids_sha256"]
    path = root / "data/raw" / train["original_file"]
    assert sha256_file(path) == cohorts["raw_sha256"]["source"]
    rows = np.load(rows_path, allow_pickle=False)[:128]
    with np.load(path, allow_pickle=False) as data:
        x = data["X"][rows]
    encoded, mask = kinematics(x)
    tokens = encode_pid(x, preprocessing["pid_vocabulary"])
    assert len(x) == 128
    assert np.all(np.isfinite(encoded)) and np.all(encoded[~mask] == 0)
    assert np.all(tokens[~mask] == 0) and np.all(tokens[mask] >= 2)
    np.testing.assert_allclose(encoded[..., 0].sum(axis=1), 1, atol=2e-7)
    np.testing.assert_allclose(np.sum(encoded[..., 0] * encoded[..., 1], axis=1), 0, atol=2e-7)
    result = {
        "status": "passed",
        "cohort": "source_train",
        "rows_checked": len(rows),
        "encoded_shape": list(encoded.shape),
        "all_coordinates_finite": True,
        "padding_inert_in_preprocessing": True,
        "source_train_row_ids_sha256": train["row_ids_sha256"],
        "preprocessing_sha256": sha256_file(output / "preprocessing.json"),
        "target_opened_by_smoke": False,
        "models_trained": 0,
    }
    (output / "smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
