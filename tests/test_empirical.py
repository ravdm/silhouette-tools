import numpy as np

from silhouette_tools.models.empirical import EmpiricalMeasure


def test_balanced_sampling_exact():
    """
    Balanced sampling with n == dataset size should return
    exactly the original data points (up to permutation).
    """
    X = np.arange(20).reshape(10, 2)   # 10 samples, dim=2
    emp = EmpiricalMeasure(X)

    samples = emp.sample(10, strategy="balanced", shuffle=False)

    assert samples.shape == X.shape
    assert np.array_equal(samples, X)


def test_iid_sampling_is_approximately_uniform():
    """
    IID sampling should approximately reproduce the uniform
    empirical distribution over indices.
    """
    rng = np.random.default_rng(0)

    X = np.arange(10).reshape(10, 1)
    emp = EmpiricalMeasure(X)

    n_samples = 10_000
    samples = emp.sample(n_samples, strategy="iid", rng=rng)

    # Count how often each original point appears
    unique, counts = np.unique(samples[:, 0], return_counts=True)

    freqs = counts / n_samples
    expected = 1.0 / len(X)

    # Allow small deviation (Monte Carlo noise)
    assert np.allclose(freqs, expected, atol=0.01)
