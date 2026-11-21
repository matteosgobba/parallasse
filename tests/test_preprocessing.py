import numpy as np
import pytest

from parallasse.preprocessing import encode_pid, fit_pid_vocabulary, kinematics


def sample():
    return np.array(
        [
            [[3, 0.1, 3.12, 211], [1, 0.2, -3.12, 22], [0, 0, 0, 0]],
            [[2, -0.2, 0.3, -211], [2, 0.1, 0.4, 130], [0, 0, 0, 0]],
        ],
        dtype=np.float64,
    )


def test_wrapping_and_weighted_centers():
    encoded, mask = kinematics(sample())
    assert encoded.dtype == np.float32
    np.testing.assert_allclose(encoded[..., 0].sum(axis=1), 1)
    np.testing.assert_allclose((encoded[..., 0] * encoded[..., 1]).sum(axis=1), 0, atol=1e-8)
    np.testing.assert_allclose(
        (encoded[..., 0] * np.sin(encoded[..., 2])).sum(axis=1), 0, atol=1e-8
    )
    assert np.abs(encoded[0, :, 2]).max() < 0.1  # The +/-pi seam is not a wide jet.
    assert np.all(encoded[~mask] == 0)


def test_padding_and_particle_order_do_not_change_coordinates():
    x = sample()
    encoded, mask = kinematics(x)
    permutation = [2, 0, 1]
    shuffled, shuffled_mask = kinematics(x[:, permutation])
    np.testing.assert_allclose(shuffled, encoded[:, permutation], atol=1e-7)
    np.testing.assert_array_equal(shuffled_mask, mask[:, permutation])
    padded, _ = kinematics(np.pad(x, ((0, 0), (0, 5), (0, 0))))
    np.testing.assert_array_equal(padded[:, :3], encoded)
    assert np.all(padded[:, 3:] == 0)


def test_periodic_shift_does_not_change_coordinates():
    x = sample()
    expected, _ = kinematics(x)
    x[0, 0, 2] += 2 * np.pi
    actual, _ = kinematics(x)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_training_only_vocabulary_unknown_pid_and_identity_ablation():
    x = sample()
    vocabulary = fit_pid_vocabulary(x[:1])
    assert vocabulary == [22, 211]
    tokens = encode_pid(x, vocabulary)
    np.testing.assert_array_equal(tokens, [[3, 2, 0], [1, 1, 0]])
    np.testing.assert_array_equal(
        encode_pid(x, vocabulary, include_identity=False), [[1, 1, 0], [1, 1, 0]]
    )
    with pytest.raises(ValueError, match="unique and sorted"):
        encode_pid(x, [211, 22])


def test_ambiguous_circular_center_is_rejected():
    x = np.array([[[1, 0, 0, 22], [1, 0, np.pi, 22]]])
    with pytest.raises(ValueError, match="Ambiguous circular"):
        kinematics(x)
