from numpy.testing import assert_allclose
from scipy.stats import gaussian_kde,norm
import numpy as np
from silhouette_tools.models.naive_bayes_kde import (
    fit_bandwidth_likelihood_cv,
    NaiveBayesKDEMixture,
)


def test_bandwidth_shrinks_with_sample_size():
    rng = np.random.default_rng(0)

    grid = np.logspace(-3, 0, 40)

    ns = [10, 100, 1000]
    bws = []

    for n in ns:
        x = rng.uniform(0.0, 1.0, size=n)
        bw = fit_bandwidth_likelihood_cv(x, grid, n_folds=5)
        print(f"n={n}, bw={bw}")
        bws.append(bw)

    # Check monotonic decrease (allowing for small CV noise)
    assert bws[0] > bws[1] > bws[2], f"Bandwidths not decreasing: {bws}"

def test_bandwidth_scales_with_component_count():
    rng = np.random.default_rng(0)

    N = 1000
    d = 2

    # Generate identical distribution
    X = rng.normal(loc=0.0, scale=1.0, size=(N, d))

    grid = np.logspace(-2, 0.5, 40)

    component_counts = [1, 4, 20]
    mean_bandwidths = []

    for k in component_counts:
        # Even split of identical data
        labels = np.repeat(np.arange(k), N // k)

        mix = NaiveBayesKDEMixture(X, labels)
        mix.fit_bandwidths(
            method="likelihood",
            grid=grid,
            n_folds=5,
        )

        mean_bw = mix.bandwidths.mean()
        mean_bandwidths.append(mean_bw)

        print(f"k={k}, mean bandwidth={mean_bw:.6f}")

    # Assert monotonic increase
    assert mean_bandwidths[0] < mean_bandwidths[1] < mean_bandwidths[2], \
        f"Bandwidths not increasing with component count: {mean_bandwidths}"

def test_manual_bandwidth_equivalence():
    rng = np.random.default_rng(0)

    n = 200
    d = 2

    # Two clearly separated blocks
    X0 = rng.uniform(0, 1, size=(n, d))
    X1 = rng.uniform(1, 2, size=(n, d))

    X = np.vstack([X0, X1])
    labels = np.array([0] * n + [1] * n)

    bw_value = 0.2

    # --- Standalone NB fits ---
    nb0 = NaiveBayesKDEMixture(X0, np.zeros(n))
    nb1 = NaiveBayesKDEMixture(X1, np.zeros(n))

    nb0.bandwidths[:] = bw_value
    nb1.bandwidths[:] = bw_value
    nb0._bandwidths_ready = True
    nb1._bandwidths_ready = True

    # --- Mixture fit ---
    mix = NaiveBayesKDEMixture(X, labels)
    mix.bandwidths[:] = bw_value
    mix._bandwidths_ready = True

    # Test points
    X_test = rng.normal(size=(50, d))

    log0 = nb0.log_density(X_test)
    log1 = nb1.log_density(X_test)
    log_mix = mix.log_density(X_test)

    # mixture weights
    pi0 = mix.weights[0]
    pi1 = mix.weights[1]

    log_manual = np.log(pi0 * np.exp(log0) + pi1 * np.exp(log1))

    assert np.allclose(log_mix, log_manual, atol=1e-12)

def test_1d_kde_matches_scipy():
    rng = np.random.default_rng(0)

    n = 200
    X = rng.normal(size=(n, 1))

    # Use fixed bandwidth
    bw = 0.3

    # --- Your implementation ---
    model = NaiveBayesKDEMixture(X, np.zeros(n))
    model.bandwidths[:] = bw
    model._bandwidths_ready = True

    X_test = rng.normal(size=(50, 1))
    log_model = model.log_density(X_test)

    # --- SciPy implementation ---
    # gaussian_kde uses a bandwidth factor that multiplies data std
    # We force absolute bandwidth = bw

    kde = gaussian_kde(X[:, 0])
    kde.set_bandwidth(bw_method=bw / np.std(X[:, 0], ddof=1))

    log_scipy = np.log(kde.evaluate(X_test[:, 0]))

    # Compare
    assert np.allclose(log_model, log_scipy, atol=1e-6)

def test_mixture_equivalence_fixed_bandwidth():
    rng = np.random.default_rng(0)

    # ---- generate two simple 2D blocks ----
    n0 = 100
    n1 = 200

    X0 = rng.uniform(0.0, 1.0, size=(n0, 2))
    X1 = rng.uniform(1.0, 2.0, size=(n1, 2))

    X = np.vstack([X0, X1])
    labels = np.array([0] * n0 + [1] * n1)

    # ---- fixed bandwidth ----
    bw_value = 0.25

    # =====================================================
    # Separate models
    # =====================================================

    model0 = NaiveBayesKDEMixture(X0, np.zeros(n0))
    model0.bandwidths[:] = bw_value
    model0._bandwidths_ready = True

    model1 = NaiveBayesKDEMixture(X1, np.zeros(n1))
    model1.bandwidths[:] = bw_value
    model1._bandwidths_ready = True

    # =====================================================
    # Joint mixture model
    # =====================================================

    mix = NaiveBayesKDEMixture(X, labels)
    mix.bandwidths[:] = bw_value
    mix._bandwidths_ready = True

    # =====================================================
    # Evaluation points
    # =====================================================

    X_eval = rng.uniform(-0.5, 2.5, size=(50, 2))

    log_f0 = model0.log_density(X_eval)
    log_f1 = model1.log_density(X_eval)

    # mixture weights should match empirical proportions
    w0 = n0 / (n0 + n1)
    w1 = n1 / (n0 + n1)

    # manual log mixture
    stacked = np.vstack([
        np.log(w0) + log_f0,
        np.log(w1) + log_f1,
    ]).T

    m = np.max(stacked, axis=1)
    log_manual = m + np.log(np.sum(np.exp(stacked - m[:, None]), axis=1))

    log_mix = mix.log_density(X_eval)

    assert_allclose(log_mix, log_manual, atol=1e-6)

def test_2d_product_structure():
    rng = np.random.default_rng(0)

    n = 100
    X = rng.normal(0, 1, size=(n, 2))
    labels = np.zeros(n)

    bw = np.array([[0.3, 0.7]])

    model = NaiveBayesKDEMixture(X, labels)
    model.bandwidths[:] = bw
    model._bandwidths_ready = True

    X_test = rng.normal(0, 1, size=(20, 2))

    log_p_model = model.log_density(X_test)
    p_model = np.exp(log_p_model)

    # Manual product KDE
    p_true = []
    for x in X_test:
        px = 1.0
        for j in range(2):
            diffs = x[j] - X[:, j]
            px *= np.mean(norm.pdf(diffs, scale=bw[0, j]))
        p_true.append(px)

    p_true = np.array(p_true)

    assert np.allclose(p_model, p_true, atol=1e-10)

def test_mixture_matches_manual_formula():
    rng = np.random.default_rng(0)

    n0 = 80
    n1 = 120

    X0 = rng.normal(-2, 0.5, size=(n0, 1))
    X1 = rng.normal(2, 0.8, size=(n1, 1))

    X = np.vstack([X0, X1])
    labels = np.array([0]*n0 + [1]*n1)

    model = NaiveBayesKDEMixture(X, labels)

    model.bandwidths[0, 0] = 0.4
    model.bandwidths[1, 0] = 0.6
    model._bandwidths_ready = True

    X_test = np.linspace(-5, 5, 200)[:, None]

    p_model = np.exp(model.log_density(X_test))

    # Manual mixture
    p0 = np.mean(norm.pdf(X_test - X0.T, scale=0.4), axis=1)
    p1 = np.mean(norm.pdf(X_test - X1.T, scale=0.6), axis=1)

    w0 = n0 / (n0 + n1)
    w1 = n1 / (n0 + n1)

    p_true = w0 * p0 + w1 * p1

    assert np.allclose(p_model, p_true, atol=1e-10)




