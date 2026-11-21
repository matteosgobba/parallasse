import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from parallasse.download import sha256_file
from parallasse.heldout import derive_cuts, verified_cuts
from parallasse.heldout_metrics import (
    KEYS,
    BinaryMetrics,
    calibration_cut,
    cohort_metrics,
    summarize,
)


def test_calibration_strict_order_statistic_and_ties():
    unique = np.linspace(0, 1, 10)
    cut = calibration_cut(unique)
    assert cut["threshold"] == unique[-2]
    assert cut["accepted_gluons"] == 1
    tied = unique.copy()
    tied[-1] = tied[-2]
    cut = calibration_cut(tied)
    assert cut["accepted_gluons"] == 0 and cut["tied_at_threshold"] == 2


def test_weighted_auc_and_fixed_cut_match_expanded_rows_with_ties():
    quarks = np.array([0.1, 0.4, 0.7, 0.8])
    gluons = np.array([0.1, 0.4, 0.4, 0.9, 1.0])
    wq = np.array([1, 0, 1, 2])
    wg = np.array([0, 2, 1, 0, 2])
    q, g = np.repeat(quarks, wq), np.repeat(gluons, wg)
    auc, tq, fg = BinaryMetrics(quarks, gluons, 0.4).weighted(wq, wg)
    assert auc == pytest.approx(roc_auc_score([1] * len(q) + [0] * len(g), np.r_[q, g]))
    assert tq == (q > 0.4).sum() and fg == (g > 0.4).sum()


def test_paired_draws_identical_models_and_independent_cohort_streams():
    y = np.tile([0, 1], 10)
    p = np.linspace(0, 1, len(y))
    scores = np.tile(p[:, None], (1, 12))
    streams = np.random.SeedSequence(20261110).spawn(2)
    cuts = dict.fromkeys(KEYS, 0.7)
    rows, draws = {}, {}
    for cohort, stream in zip(("test", "target"), streams, strict=True):
        rows[cohort], draws[cohort] = cohort_metrics(
            y, scores, cuts, np.random.Generator(np.random.PCG64(stream)), 30
        )
        np.testing.assert_array_equal(draws[cohort], np.repeat(draws[cohort][:, :1], 12, axis=1))
    assert not np.array_equal(draws["test"], draws["target"])
    report = summarize(rows, draws)
    assert report["auc_contrasts"]["test"]["PFN-K_minus_XGB"]["bootstrap_95_lo"] == 0.0


def test_pid_interaction_is_difference_of_cohort_gains():
    y = np.tile([0, 1], 4)
    test = np.full((len(y), 12), 0.5)
    target = test.copy()
    test[:, 9:] = y[:, None]
    rows, draws = {}, {}
    for cohort, scores in (("test", test), ("target", target)):
        rows[cohort], draws[cohort] = cohort_metrics(
            y, scores, dict.fromkeys(KEYS, 0.5), np.random.default_rng(13), 10
        )
    effect = summarize(rows, draws)["pid_auc_gain_pythia_minus_herwig"]
    assert set(effect.values()) == {0.5}


def test_cut_freeze_rejects_missing_state_and_tampered_bytes(tmp_path):
    lock = tmp_path / "cuts.json"
    with pytest.raises(ValueError, match="prior numerical cut freeze"):
        verified_cuts({"status": "calibrating"}, lock, "demo")
    calibration = tmp_path / "calibration.npz"
    np.savez(
        calibration,
        labels=np.zeros(5000),
        keys=KEYS,
        scores=np.tile(np.linspace(0, 1, 5000)[:, None], (1, 12)),
    )
    cuts = derive_cuts(calibration)
    lock.write_text(
        json.dumps(
            {
                "run_id": "demo",
                "operator": ">",
                "model_freeze_sha256": "model",
                "calibration_cohort": "source calibration gluons only",
                "background_acceptance_target": 0.1,
                "calibration_scores": {
                    "path": str(calibration),
                    "sha256": sha256_file(calibration),
                },
                "cuts": cuts,
            }
        )
    )
    record = {
        "status": "cuts_frozen",
        "cut_freeze_sha256": sha256_file(lock),
        "model_freeze_sha256": "model",
    }
    assert verified_cuts(record, lock, "demo") == {k: v["threshold"] for k, v in cuts.items()}
    lock.write_text(lock.read_text().replace('"operator": ">"', '"operator": ">="'))
    with pytest.raises(ValueError, match="Frozen bytes changed"):
        verified_cuts(record, lock, "demo")
