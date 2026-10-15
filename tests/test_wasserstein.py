import numpy as np
import pytest
from silhouette_tools.metrics import wasserstein as ws


def test_wasserstein_basic_identity():
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 1.0, 1.0])
    assert ws.wasserstein_1d(x, y) == 0.0


def test_wasserstein_constant_shift():
    x = np.array([0.0, 0.0, 0.0])
    y = np.array([1.0, 1.0, 1.0])
    assert ws.wasserstein_1d(x, y) == 1.0


def test_wasserstein_permutation_invariance():
    x = np.array([3.0, 2.0, 1.0])
    y = np.array([1.0, 2.0, 3.0])
    assert ws.wasserstein_1d(x, y) == 0.0


def test_wasserstein_shifted_distribution():
    x = np.array([1.0, 2.0, 3.0])
    y = x + 1.0
    assert ws.wasserstein_1d(x, y) == 1.0


def test_wasserstein_single_point():
    x = np.array([0.0])
    y = np.array([5.0])
    assert ws.wasserstein_1d(x, y) == 5.0


def test_wasserstein_symmetry():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([3.0, 4.0, 5.0])
    assert ws.wasserstein_1d(x, y) == ws.wasserstein_1d(y, x)


def test_wasserstein_linear_scaling():
    x = np.array([1.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    assert ws.wasserstein_1d(2 * x, 2 * y) == 2 * ws.wasserstein_1d(x, y)


########## Tests after adding weighting ##########
# ---------------------------------------------------------------------
# Invariance / consistency tests
# ---------------------------------------------------------------------

def test_wasserstein_uniform_weights_equivalent():
    x = np.array([0.0, 2.0, 4.0])
    y = np.array([1.0, 3.0])
    w_x = np.ones_like(x)
    w_y = np.ones_like(y)

    assert np.isclose(
        ws.wasserstein_1d(x, y),
        ws.wasserstein_1d(x, y, w_x=w_x, w_y=w_y),
    )


def test_wasserstein_uniform_weight_scaling():
    x = np.array([1.0, 1.0, 2.0, 3.5])
    y = np.array([1.0, 2.0, 3.0])

    w_x = np.ones_like(x)
    w_y = np.ones_like(y)

    base = ws.wasserstein_1d(x, y)

    assert np.isclose(base, ws.wasserstein_1d(x, y, w_x=w_x))
    assert np.isclose(base, ws.wasserstein_1d(x, y, w_x=w_x, w_y=2.0 * w_y))


def test_wasserstein_weight_normalization_invariance():
    x = np.array([0.0, 2.0])
    y = np.array([1.0])
    w_x = np.array([1.0, 3.0])

    assert np.isclose(
        ws.wasserstein_1d(x, y, w_x=w_x),
        ws.wasserstein_1d(x, y, w_x=10.0 * w_x),
    )


# ---------------------------------------------------------------------
# Semantic behavior tests
# ---------------------------------------------------------------------

def test_wasserstein_zero_weights_ignored():
    x = np.array([0.0, 10.0])
    y = np.array([1.0])
    w_x = np.array([1.0, 0.0])

    assert np.isclose(ws.wasserstein_1d(x, y, w_x=w_x), 1.0)


def test_wasserstein_weighted_difference():
    x = np.array([0.0, 10.0, 2.0, 3.5])
    y = np.array([1.0, 2.0, 3.0])

    w_x = np.array([1.0, 1.0, 1.0, 2.0])
    w_y = np.array([1.0, 1.0, 2.0])

    base = ws.wasserstein_1d(x, y)

    assert not np.isclose(base, ws.wasserstein_1d(x, y, w_x=w_x))
    assert not np.isclose(base, ws.wasserstein_1d(x, y, w_y=w_y))
    assert np.isclose(
        ws.wasserstein_1d(x, y, w_y=w_y),
        ws.wasserstein_1d(x, y, w_y=2.0 * w_y),
    )


def test_wasserstein_weighted_exact():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.0])
    w_x = np.array([1.0, 2.0, 1.0])

    expected = (1.0 * 1 + 2.0 * 2 + 1.0 * 3) / 4.0
    assert np.isclose(ws.wasserstein_1d(x, y, w_x=w_x), expected)


# ---------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------

def test_wasserstein_negative_weights_error():
    x = np.array([0.0, 1.0])
    y = np.array([0.0])
    w_x = np.array([1.0, -1.0])

    with pytest.raises(ValueError):
        ws.wasserstein_1d(x, y, w_x=w_x)


def test_wasserstein_weight_length_mismatch():
    x = np.array([0.0, 1.0])
    y = np.array([0.0])
    w_x = np.array([1.0])

    with pytest.raises(ValueError):
        ws.wasserstein_1d(x, y, w_x=w_x)
