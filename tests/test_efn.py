import fcntl
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from parallasse.efn import EnergyFlowNetwork, efn_prediction_checks
from parallasse.efn_metrics import KEYS, bias_summary, classification, classification_summary


def test_energy_weighted_direction_encoder_and_exact_collinear_pooling():
    model = EnergyFlowNetwork([1], [])
    with torch.no_grad():
        model.particle_encoder[0].weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.particle_encoder[0].bias.fill_(3)
        model.jet_head[0].weight.fill_(1)
        model.jet_head[0].bias.zero_()
    original = torch.tensor([[[0.6, 0.1, 0.2], [0.4, -0.2, 0.1]]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    torch.testing.assert_close(model(original, mask), torch.tensor([3.3]), atol=1e-6, rtol=0)
    split = torch.tensor([[[0.3, 0.1, 0.2], [0.3, 0.1, 0.2], [0.4, -0.2, 0.1], [0.0, 50.0, 20.0]]])
    torch.testing.assert_close(
        model(split, torch.ones((1, 4), dtype=torch.bool)), model(original, mask), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        model(original, mask, torch.full((1, 2), -991)), model(original, mask), atol=0, rtol=0
    )


def test_padding_values_and_gradients_are_inert():
    torch.manual_seed(7)
    model = EnergyFlowNetwork([4], [4])
    x = torch.tensor([[[1.0, 0.1, 0.2], [float("nan")] * 3]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    logits = model(x, mask)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert torch.all(x.grad[~mask] == 0)
    assert torch.isfinite(x.grad[mask]).all()
    torch.testing.assert_close(logits, model(x[:, :1], mask[:, :1]), atol=0, rtol=0)
    with pytest.raises(ValueError, match="Boolean mask"):
        model(x, mask.long())


def test_irc_preprocessing_checks_at_phi_seam_and_parameter_count():
    torch.manual_seed(17)
    model = EnergyFlowNetwork([32, 32, 64], [64, 32])
    assert sum(p.numel() for p in model.parameters()) == 9537
    assert all(
        torch.count_nonzero(layer.bias) == 0
        for layer in model.modules()
        if isinstance(layer, torch.nn.Linear)
    )
    raw = np.array(
        [
            [[3, 0.1, 3.12, 211], [1, 0.2, -3.12, 22], [0, 0, 0, 0]],
            [[2, -0.2, 0.3, -211], [2, 0.1, 0.4, 130], [0, 0, 0, 0]],
        ],
        dtype=float,
    )
    checks = efn_prediction_checks(
        model,
        raw,
        {
            "probability_atol": 2e-6,
            "probability_rtol": 1e-5,
            "soft_limit_atol": 2e-5,
            "soft_relative_pt": [0.01, 0.0001, 0.000001, 0.00000001],
        },
    )
    assert checks["extra_padding_max_abs_probability_delta"] == 0
    assert checks["collinear_max_abs_probability_delta"] < 2e-6
    assert checks["soft_limit"][-1]["max_abs_probability_delta"] < 2e-5
    assert checks["pid_change_probability_delta"] == 0


def test_five_family_event_bootstrap_pairs_all_fifteen_columns():
    y = np.tile([0, 1], 20)
    scores = np.repeat(np.linspace(0.01, 0.99, len(y))[:, None], 15, axis=1)
    rows, draws = classification(y, scores, dict.fromkeys(KEYS, 0.5), np.random.default_rng(31), 20)
    assert len(rows) == 15 and draws.shape == (20, 15, 3)
    np.testing.assert_array_equal(draws, np.repeat(draws[:, :1, :], 15, axis=1))
    contrast = classification_summary(rows, draws)["EFN_minus_PFN-K_auc"]
    assert set(contrast.values()) == {0.0}
    with pytest.raises(ValueError, match="align"):
        classification(y, scores[:, :12], dict.fromkeys(KEYS, 0.5), np.random.default_rng(31), 20)


def test_efn_bias_contrast_uses_absolute_values_per_seed():
    point = np.zeros((5, 3))
    point[2] = [-0.1, 0.2, 0.3]
    point[4] = [0.02, -0.03, 0.04]
    samples = np.tile(point.reshape(1, 15), (30, 1))
    result = bias_summary(point.ravel(), samples)
    effect = result["contrasts"]["EFN_minus_PFN-K"]
    expected = np.mean(np.abs(point[4]) - np.abs(point[2]))
    assert effect["mean"] == pytest.approx(expected)
    assert effect["bootstrap_95_lo"] == pytest.approx(expected)
    assert effect["bootstrap_95_hi"] == pytest.approx(expected)


def test_public_entry_rejects_concurrent_run_before_creating_results(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_efn_study.py"
    lock_dir = tmp_path / "work/efn-locks"
    lock_dir.mkdir(parents=True)
    lock_path = lock_dir / "busy.lock"
    lock_path.write_text("reserved")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [sys.executable, str(script), "--stage", "prepare", "--run-id", "busy"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode != 0 and "BlockingIOError" in result.stderr
        assert not (tmp_path / "results").exists()
    assert lock_path.read_text() == "reserved"


def test_stage_guard_rejects_early_evaluation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_efn_study")
    (tmp_path / "run.json").write_text(json.dumps({"status": "models_frozen"}))
    with pytest.raises(ValueError, match="cuts_frozen"):
        runner.verify_stage(tmp_path, "cuts_frozen")
