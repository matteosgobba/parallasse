import json

import numpy as np
import pytest

from parallasse.development import development_rows, load_source_development
from parallasse.download import sha256_file


@pytest.mark.parametrize("name", ["calibration", "test", "target"])
def test_heldout_cohort_is_rejected_before_any_file_is_opened(tmp_path, name):
    with pytest.raises(ValueError, match="only source train and tune"):
        development_rows(tmp_path, name, 1)


def test_development_loader_checks_hashes_and_only_opens_train_tune_and_source(
    tmp_path, monkeypatch
):
    out = tmp_path / "results/task0001"
    out.mkdir(parents=True)
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    processed = tmp_path / "data/processed"
    processed.mkdir()
    particles = np.zeros((10, 2, 4))
    particles[:, 0] = [1, 0, 0, 22]
    labels = np.arange(10) % 2
    source = raw / "QG_jets.npz"
    np.savez(source, X=particles, y=labels)
    manifest = {"cohorts": {}, "raw_sha256": {"source": sha256_file(source)}}
    for name, rows in (("train", [0, 1, 2, 3]), ("tune", [4, 5, 6, 7])):
        path = processed / f"{name}_rows.npy"
        np.save(path, np.array(rows))
        manifest["cohorts"][name] = {
            "generator": "pythia",
            "original_file": "QG_jets.npz",
            "row_ids_path": str(path.relative_to(tmp_path)),
            "row_ids_sha256": sha256_file(path),
        }
    (out / "cohorts.json").write_text(json.dumps(manifest))
    (out / "preprocessing.json").write_text("{}")
    validation = {
        "evidence_sha256": {
            str(p.relative_to(tmp_path)): sha256_file(p)
            for p in (out / "cohorts.json", out / "preprocessing.json")
        }
    }
    (out / "validation.json").write_text(json.dumps(validation))
    actual_load = np.load
    opened = []

    def recording_load(path, *args, **kwargs):
        opened.append(path.name)
        return actual_load(path, *args, **kwargs)

    monkeypatch.setattr("parallasse.development.np.load", recording_load)
    data = load_source_development(tmp_path, 2, 2)
    np.testing.assert_array_equal(data.train_rows, [0, 1])
    np.testing.assert_array_equal(data.tune_rows, [4, 5])
    assert opened == ["train_rows.npy", "tune_rows.npy", "QG_jets.npz"]
    np.save(processed / "train_rows.npy", np.array([3, 2, 1, 0]))
    with pytest.raises(ValueError, match="row IDs changed"):
        development_rows(tmp_path, "train", 2)
