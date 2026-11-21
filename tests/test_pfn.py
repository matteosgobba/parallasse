import pytest
import torch

from parallasse.pfn import ParticleFlowNetwork
from parallasse.source_neural import BestCheckpoint, pfn_prediction_checks, weight_digest


def inputs():
    coordinates = torch.randn((3, 9, 3), generator=torch.Generator().manual_seed(7))
    mask = torch.arange(9)[None, :] < torch.tensor([[3], [9], [5]])
    tokens = torch.arange(9).expand(3, -1).remainder(5) + 1
    return coordinates, mask, tokens


@pytest.mark.parametrize("use_pid", [False, True])
def test_prediction_invariance_to_order_and_arbitrary_padding(use_pid):
    torch.manual_seed(17)
    model = ParticleFlowNetwork(14, 8, [32, 32, 64], [64, 32], use_pid)
    result = pfn_prediction_checks(model, inputs())
    assert result["jets_checked"] == 3
    assert result["extra_padding_max_abs_probability_delta"] == 0.0


def test_mask_excludes_encoder_bias_and_gradients_from_padding():
    model = ParticleFlowNetwork(14, 8, [1], [], True)
    with torch.no_grad():
        model.particle_encoder[0].weight.fill_(1)
        model.particle_encoder[0].weight[:, 3:] = 0
        model.particle_encoder[0].bias.fill_(1)
        model.jet_head[0].weight.fill_(1)
        model.jet_head[0].bias.zero_()
    coordinates = torch.zeros((2, 4, 3), requires_grad=True)
    mask = torch.tensor([[True, False, True, False], [False, True, False, False]])
    tokens = torch.where(mask, 1, -990)
    logits = model(coordinates, mask, tokens)
    torch.testing.assert_close(logits, torch.tensor([2.0, 1.0]), atol=0, rtol=0)
    logits.sum().backward()
    assert torch.all(coordinates.grad[mask] == 1)
    assert torch.all(coordinates.grad[~mask] == 0)


def test_matched_initialization_and_pid_switch():
    torch.manual_seed(29)
    k = ParticleFlowNetwork(14, 8, [32, 32, 64], [64, 32], False)
    torch.manual_seed(29)
    pid = ParticleFlowNetwork(14, 8, [32, 32, 64], [64, 32], True)
    assert weight_digest(k) == weight_digest(pid)
    coordinates, mask, tokens = inputs()
    unknown = torch.ones_like(tokens)
    torch.testing.assert_close(
        k(coordinates, mask, tokens), k(coordinates, mask, unknown), atol=0, rtol=0
    )
    torch.testing.assert_close(
        k(coordinates, mask, tokens), pid(coordinates, mask, unknown), atol=0, rtol=0
    )
    assert not torch.equal(pid(coordinates, mask, tokens), pid(coordinates, mask, unknown))


def test_tuning_checkpoint_preserves_first_best_and_uses_patience():
    model = torch.nn.Linear(1, 1, bias=False)
    selection = BestCheckpoint(patience=2)
    for epoch, auc in enumerate([0.7, 0.8, 0.8, 0.79], 1):
        with torch.no_grad():
            model.weight.fill_(epoch)
        assert selection.update(epoch, auc, model) == (epoch == 4)
    assert selection.epoch == 2
    assert selection.weights["weight"].item() == 2
