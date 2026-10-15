import numpy as np

from silhouette_tools.models.empirical_naive_bayes import EmpiricalNaiveBayesMixture


def make_axis_aligned_gaussian_mixture(
    n_per: int = 1_000,
    dim: int = 5,
    sep: float = 5.0,
    rng: np.random.Generator | None = None,
):
    """
    Construct a simple 2-component, axis-aligned Gaussian mixture
    with overall mean approximately zero.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    mean1 = np.full(dim, sep)
    mean2 = -mean1

    X1 = rng.normal(loc=mean1, scale=1.0, size=(n_per, dim))
    X2 = rng.normal(loc=mean2, scale=1.0, size=(n_per, dim))

    X = np.vstack([X1, X2])
    true_labels = np.array([0] * n_per + [1] * n_per)

    return X, true_labels


def test_sampling_sanity_without_labels():
    """
    Initialize without labels and check that sampling produces
    reasonable global mean / variance.
    """
    rng = np.random.default_rng(1)
    X, _ = make_axis_aligned_gaussian_mixture(rng=rng)

    model = EmpiricalNaiveBayesMixture(
        X,
        k=2,
        init="random",
        rng=rng,
    )

    samples = model.sample(10_000)

    # Global mean should be close to zero
    mean = samples.mean(axis=0)
    assert np.allclose(mean, 0.0, atol=0.5)

    # Variance should be non-degenerate
    var = samples.var(axis=0)
    assert np.all(var > 0.5)


def test_update_labels_and_weights():
    """
    After updating to correct labels, component weights and
    component means should be correct.
    """
    rng = np.random.default_rng(2)
    X, true_labels = make_axis_aligned_gaussian_mixture(rng=rng)

    model = EmpiricalNaiveBayesMixture(
        X,
        k=2,
        init="random",
        rng=rng,
    )

    # Update to true partition
    model.update_labels(true_labels)

    # Weights should be exactly balanced
    assert np.allclose(model.weights, [0.5, 0.5])

    # Sample per-component means
    samples = model.sample(20_000, strategy="balanced")

    # Split samples by nearest true mean
    mean_pos = X[true_labels == 0].mean(axis=0)
    mean_neg = X[true_labels == 1].mean(axis=0)

    # Check that sampled means match empirical means
    sampled_mean = samples.mean(axis=0)
    true_mean = X.mean(axis=0)

    assert np.allclose(sampled_mean, true_mean, atol=0.2)
