import subprocess
import sys

import numpy as np
import pytest
import torch

from parallasse.baselines import (
    classification_metrics,
    fit_training_scaler,
    scaler_from_record,
    scaler_record,
)
from parallasse.mlp_baseline import AggregateMLP, fit_mlp_pilot, mlp_quark_scores


def synthetic_data():
    rng = np.random.default_rng(31)
    y = np.tile([0, 1], 64)
    x = rng.normal(0, 0.1, size=(len(y), 6)).astype(np.float32)
    x[:, 0] = y * 4 - 2
    return x[:96], y[:96], x[96:], y[96:]


def test_scaler_uses_training_statistics_and_can_be_restored():
    train = np.array([[0] * 6, [2] * 6], dtype=np.float32)
    tune = np.full((2, 6), 1000, dtype=np.float32)
    scaler = fit_training_scaler(train)
    transformed = scaler.transform(tune)
    np.testing.assert_array_equal(scaler.mean_, np.ones(6))
    assert scaler.n_samples_seen_ == 2
    np.testing.assert_array_equal(transformed, np.full((2, 6), 999))
    restored = scaler_from_record(scaler_record(scaler))
    np.testing.assert_array_equal(restored.transform(tune), transformed)


def test_score_direction_and_invalid_scores():
    assert (
        classification_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))["roc_auc"]
        == 1
    )
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics(np.array([0, 1]), np.array([-0.1, 1.2]))
    with pytest.raises(ValueError, match="Both classes"):
        classification_metrics(np.array([0, 0]), np.array([0.1, 0.2]))


def test_xgb_uses_correct_class_direction_and_saved_best_round(tmp_path):
    code = """
import sys
from pathlib import Path
import numpy as np
import xgboost as xgb
from parallasse.xgb_baseline import fit_xgb_pilot, xgb_quark_scores
assert 'torch' not in sys.modules
rng=np.random.default_rng(31)
y=np.tile([0,1],64)
x=rng.normal(0,.1,size=(len(y),6)).astype(np.float32)
x[:,0]=y*4-2
config=dict(n_estimators=20,learning_rate=.2,tree_method='hist',objective='binary:logistic',
            eval_metric='auc',early_stopping_rounds=3,max_depth_candidates=[2])
model,scores,record=fit_xgb_pilot(x[:96],y[:96],x[96:],y[96:],config,2,17,1)
assert record['tune']['roc_auc']==1
assert scores[y[96:]==1].mean()>scores[y[96:]==0].mean()
assert record['trees_used']<=record['rounds_trained']<=20
path=Path(sys.argv[1])/'model.ubj'
model.save_model(path)
restored=xgb.XGBClassifier(n_jobs=1)
restored.load_model(path)
np.testing.assert_array_equal(xgb_quark_scores(restored,x[96:]),scores)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_baseline_utilities_do_not_import_either_model_runtime():
    code = "import sys; import parallasse.baselines; "
    code += "assert 'torch' not in sys.modules and 'xgboost' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_mlp_training_checkpoint_selection_and_replay_are_consistent(tmp_path):
    assert "xgboost" not in sys.modules
    x, y, t, v = synthetic_data()
    config = dict(
        hidden_sizes=[64, 32], learning_rate=0.01, weight_decay=0.0001, max_epochs=3, batch_size=16
    )
    model, scaler, scores, record = fit_mlp_pilot(x, y, t, v, config, 17, 1)
    assert record["epochs_trained"] == 3
    assert record["scaler"]["n_samples_seen"] == len(x)
    assert record["tune"]["roc_auc"] == max(item["tune"]["roc_auc"] for item in record["history"])
    assert record["history"][-1]["train_bce"] < record["history"][0]["train_bce"]
    assert scores[v == 1].mean() > scores[v == 0].mean()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = AggregateMLP(config["hidden_sizes"])
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    np.testing.assert_array_equal(mlp_quark_scores(restored, scaler, t), scores)
