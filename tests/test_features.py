import numpy as np

from parallasse.features import FEATURE_NAMES, aggregate_features


def test_six_features_match_an_unequal_weight_analytic_example():
    particles = np.array([[[3, 0, 0, 211], [1, 2, 0, 22], [0, 0, 0, 0]]], dtype=float)
    expected = [2, np.sqrt(0.625), 0.75 * np.sqrt(0.5) + 0.25 * np.sqrt(1.5), 0.75, 0.75, 0.75]
    actual = aggregate_features(particles)
    assert actual.dtype == np.float32 and len(FEATURE_NAMES) == 6
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6)


def test_single_particle_and_equal_weight_limits():
    single = np.array([[[4, 0.4, -2.8, 211], [0, 0, 0, 0]]])
    np.testing.assert_allclose(aggregate_features(single)[0], [1, 1, 0, 0, 0, 1], atol=1e-7)
    equal = np.array([[[1, -1, 0, 211], [1, 1, 0, 22]]])
    np.testing.assert_allclose(aggregate_features(equal)[0], [2, np.sqrt(0.5), 1, 1, 1, 0.5])


def test_features_are_pid_blind_and_invariant_to_padding_order_and_global_pt_scale():
    particles = np.array([[[3, 0.1, 3.12, 211], [1, 0.2, -3.12, 22], [0, 0, 0, 0]]])
    expected = aggregate_features(particles)
    changed = np.pad(particles[:, [2, 0, 1]], ((0, 0), (0, 2), (0, 0)))
    real = changed[..., 0] > 0
    changed[..., 0] *= 7
    changed[..., 3][real] = 13
    changed[..., 2][real] += 2 * np.pi
    np.testing.assert_allclose(aggregate_features(changed), expected, rtol=1e-6, atol=1e-7)
