import numpy as np
import pytest
from silhouette_tools.silhouettes.objectives.pairwise_nb import (
    convolve_points_1d,
    evaluate_projection_batch,
    PairwiseSilhouetteNBMixObjective
)
from silhouette_tools.metrics.wasserstein import wasserstein_1d as ws1d


def test_convolve_points_1d_size():
    x = np.ones(5)
    y = np.ones(6)
    assert len(convolve_points_1d(x, y)) == 5 * 6


def test_convolve_points_1d_values_simple():
    x = np.array([1.0, 2.0])
    y = np.array([10.0, 20.0])
    out = convolve_points_1d(x, y)
    expected = np.array([11.0, 21.0, 12.0, 22.0])
    assert np.allclose(np.sort(out), np.sort(expected))


def test_convolve_points_1d_commutativity():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([-1.0, 4.0])
    assert np.allclose(
        np.sort(convolve_points_1d(x, y)),
        np.sort(convolve_points_1d(y, x)),
    )


def test_convolve_points_1d_singleton():
    x = np.array([2.5])
    y = np.array([1.5])
    out = convolve_points_1d(x, y)
    assert out.shape == (1,)
    assert out[0] == 4.0


def test_convolve_points_1d_zero_vector():
    x = np.zeros(3)
    y = np.array([1.0, 2.0])
    out = convolve_points_1d(x, y)
    assert np.allclose(out, np.tile(y, 3))


def test_convolve_points_1d_negative_values():
    x = np.array([-1.0, 2.0])
    y = np.array([-3.0, 4.0])
    out = convolve_points_1d(x, y)
    expected = np.array([-4.0, 3.0, -1.0, 6.0])
    assert np.allclose(np.sort(out), np.sort(expected))


def test_convolve_points_1d_dtype_preserved():
    x = np.array([1, 2], dtype=np.int64)
    y = np.array([3, 4], dtype=np.int64)
    out = convolve_points_1d(x, y)
    assert out.dtype.kind in {"i", "f"}


def test_convolve_points_1d_empty_input_error():
    x = np.array([])
    y = np.array([1.0])
    with pytest.raises(ValueError):
        convolve_points_1d(x, y)


def test_convolve_points_1d_non_1d_error():
    x = np.ones((2, 2))
    y = np.ones(3)
    with pytest.raises(ValueError):
        convolve_points_1d(x, y)


# =============================================================================
# Batch evaluator
# =============================================================================

#If we give every sample its own label it is equal to the plain old empirical distribution

def test_evaluate_projection_batch_identity_vs_perturbed():
    """
    If every sample has its own label, the Naive Bayes mixture coincides
    with the empirical distribution, so all projection scores should be zero.
    Changing a label should break this equality.
    """
    d = 3
    n = 10
    k = n

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    labels = np.arange(n)

    proj_specs = [
        (0, 1, 1.0, 1.0),
        (0, 2, 0.5, 0.1),
    ]
    B = len(proj_specs)

    empirical_projections = np.empty((B, n), dtype=float)
    for p, (i, j, a, b) in enumerate(proj_specs):
        empirical_projections[p] = a * X[:, i] + b * X[:, j]

    # --- Identity case: exact match ---
    results = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=k,
        metric=ws1d,
    )
    assert np.allclose(results, 0.0)

    # --- Perturbed case: change one label ---
    labels = labels.copy()
    labels[1] = 0

    results = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=k,
        metric=ws1d,
    )
    assert (~np.isclose(results, 0.0)).all()

def test_projection_batch_label_permutation_invariance():
    d = 3
    n = 20
    k = 4
    X = np.random.randn(n, d)
    labels = np.repeat(np.arange(k), n // k)
    perm = np.random.permutation(n)

    proj_specs = ((0, 1, 1.0, 1.0),)
    B = len(proj_specs)

    empirical_projections = np.zeros((B, n))
    for idx, (i, j, a, b) in enumerate(proj_specs):
        empirical_projections[idx] = a * X[:, i] + b * X[:, j]

    r1 = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=k,
        metric=ws1d,
    )

    r2 = evaluate_projection_batch(
        X=X[perm],
        labels=labels[perm],
        empirical_projections=empirical_projections[:, perm],
        proj_specs=proj_specs,
        k=k,
        metric=ws1d,
    )

    assert np.allclose(r1, r2)
def test_empty_cluster_handled():
    n, d = 10, 3
    X = np.random.randn(n, d)

    labels = np.zeros(n, dtype=int)
    labels[:5] = 1  # cluster 2 empty

    proj_specs = ((0, 1, 1.0, 1.0),)
    empirical_projections = np.array([
        X[:, 0] + X[:, 1]
    ])

    results = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=3,
        metric=ws1d,
    )

    assert np.isfinite(results).all()

def test_evaluate_projection_batch_axis_projections_zero():
    """
    No matter what projections onto the axes should have zero distance always
    """
    rng = np.random.default_rng(0)

    n = 100
    d = 4
    k = 7

    # Random data, arbitrary labels
    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    # Axis-aligned projections only
    # x_i projections implemented as (a=1, b=0) or (a=0, b=1)
    graph = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    ]
    coefs = [
        (2.0, 0.0),
        (0.0, 1.0),
    ]

    # Build proj_specs and empirical projections
    proj_specs = []
    empirical_projections = []

    for i, j in graph:
        for a, b in coefs:
            proj_specs.append((i, j, a, b))
            empirical_projections.append(a * X[:, i] + b * X[:, j])

    empirical_projections = np.asarray(empirical_projections)

    results = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=k,
        metric=ws1d,
    )

    # Every projection must match exactly
    assert np.allclose(results, 0.0)

def test_evaluate_projection_batch_axis_projections_perturbed_nonzero():
    """
    Negative test for the previous
    """
    rng = np.random.default_rng(0)

    n = 100
    d = 4
    k = 7

    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    graph = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    ]

    # Slightly perturbed coefficients (NOT axis-aligned)
    eps = 1e-2
    coefs = [
        (1.0, eps),
        (eps, 1.0),
    ]

    proj_specs = []
    empirical_projections = []

    for i, j in graph:
        for a, b in coefs:
            proj_specs.append((i, j, a, b))
            empirical_projections.append(a * X[:, i] + b * X[:, j])

    empirical_projections = np.asarray(empirical_projections)

    results = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=k,
        metric=ws1d,
    )

    # At least one projection must now differ
    assert (~np.isclose(results, 0.0)).all()

def test_evaluate_projection_batch_checkerboard_zero():
    # 4-point checkerboard in [0,1]^2
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    n, d = X.shape

    # Labels: opposite corners grouped
    labels = np.array([0, 1, 1, 0])
    k = 2

    # Several nontrivial projections
    proj_specs = (
        (0, 1, 1.0, 1.0),
        (0, 1, 1.0, -1.0),
        (0, 1, 0.5, 0.5),
    )
    B = len(proj_specs)

    empirical_projections = np.zeros((B, n))
    for p, (i, j, a, b) in enumerate(proj_specs):
        empirical_projections[p] = a * X[:, i] + b * X[:, j]

    results = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=k,
        metric=ws1d,
    )

    # All projections should match exactly
    assert np.isclose(results, 0.0).all()

def test_evaluate_projection_batch_two_checkerboards_zero():
    """
    Two checkerboards:
      - One at the origin, split across two labels (0, 1, 1, 0)
      - One translated by +2 in every coordinate, all in a single label

    Even though the translated checkerboard is *not* split, the pairwise
    Naive Bayes construction still reproduces the correct 1D marginals
    for every projection. Therefore all Wasserstein distances should be zero.
    """

    # First checkerboard (split)
    X0 = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    labels0 = np.array([0, 1, 1, 0])

    # Second checkerboard (shifted, unsplit)
    X1 = X0 + 2.0
    labels1 = np.array([2, 2, 2, 2])

    # Combine
    X = np.vstack([X0, X1])
    labels = np.concatenate([labels0, labels1])

    n, d = X.shape
    k = 3  # labels {0,1,2}

    # Nontrivial projections
    proj_specs = (
        (0, 1, 1.0, 1.0),
        (0, 1, 1.0, -1.0),
        (0, 1, 0.3, 0.7),
        (0, 1, -0.6, 0.2),
    )
    B = len(proj_specs)

    empirical_projections = np.zeros((B, n))
    for p, (i, j, a, b) in enumerate(proj_specs):
        empirical_projections[p] = a * X[:, i] + b * X[:, j]

    results = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=k,
        metric=ws1d,
    )

    # All projections should match exactly
    assert np.isclose(results, 0.0).all()

def test_evaluate_projection_batch_two_checkerboards_nonzero_after_label_flip():
    """
    Negative of `test_evaluate_projection_batch_two_checkerboards_zero`.

    Same setup: two checkerboards (one split, one translated).
    We flip a single label in the translated checkerboard.

    This breaks the exact Naive Bayes reconstruction, so *all*
    pairwise projection distances should now be non-zero.
    """

    # First checkerboard (split)
    X0 = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    labels0 = np.array([0, 1, 1, 0])

    # Second checkerboard (shifted)
    X1 = X0 + 2.0
    labels1 = np.array([2, 2, 2, 2])

    # Flip a single label (minimal perturbation)
    labels1 = labels1.copy()
    labels1[0] = 0

    # Combine
    X = np.vstack([X0, X1])
    labels = np.concatenate([labels0, labels1])

    n, d = X.shape
    k = 3

    proj_specs = (
        (0, 1, 1.0, 1.0),
        (0, 1, 1.0, -1.0),
        (0, 1, 0.3, 0.7),
        (0, 1, -0.6, 0.2),
    )
    B = len(proj_specs)

    empirical_projections = np.zeros((B, n))
    for p, (i, j, a, b) in enumerate(proj_specs):
        empirical_projections[p] = a * X[:, i] + b * X[:, j]

    results = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=k,
        metric=ws1d,
    )

    # Strong assertion: every projection must be non-zero
    assert (~np.isclose(results, 0.0)).all()


def test_object_matches_batch_evaluator():
    """
    Ensure that PairwiseSilhouetteNBMixObjective.evaluate_labels
    produces exactly the same scores as a direct call to
    evaluate_projection_batch over all projections.

    This checks that no information is lost or reordered incorrectly
    when going through the object wrapper.
    """

    rng = np.random.default_rng(123)

    # ---- Problem setup ----
    n = 50
    d = 5
    k = 4

    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    graph = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    coefs = [
        (1.0, 1.0),
        (1.0, -0.5),
        (0.3, 0.7),
    ]

    # ---- Build object ----
    obj = PairwiseSilhouetteNBMixObjective(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    # ---- Object evaluation ----
    scores_obj = obj.evaluate_labels(
        labels=labels,
        batch_size=None,
        n_workers=1,
    )

    # ---- Direct batch evaluation (single batch = all projections) ----
    scores_batch = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=obj.empirical_projections,
        proj_specs=obj.full_proj_specs,
        k=k,
        metric=ws1d,
    )

    # ---- Compare ----
    assert scores_obj.shape == scores_batch.shape
    assert np.allclose(
        scores_obj,
        scores_batch,
    )

def test_pairwise_silhouette_evaluate_labels_parallel_matches_serial():
    """
    Ensure that evaluate_labels produces identical results when run
    serially and with multiprocessing enabled.

    This test explicitly exercises the multiprocessing code path.
    """

    # ---- Dataset ----
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [3.0, 3.0],
    ])

    labels = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    k = 3

    graph = [(0, 1)]
    coefs = [
        (1.0, 1.0),
        (1.0, -1.0),
        (0.3, 0.7),
        (-0.6, 0.2),
    ]

    obj = PairwiseSilhouetteNBMixObjective(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    # ---- Serial run (ground truth) ----
    scores_serial = obj.evaluate_labels(
        labels=labels,
        batch_size=2,
        n_workers=1,
    )

    # ---- Parallel run ----
    scores_parallel = obj.evaluate_labels(
        labels=labels,
        batch_size=2,
        n_workers=2,
    )

    # ---- Assertions ----
    assert scores_serial.shape == scores_parallel.shape
    assert np.allclose(scores_serial, scores_parallel)


# ----------------------------------------------------------------------
# IMPORTANT:
# This guard is REQUIRED for multiprocessing safety on macOS / Windows.
# Pytest will still discover and run the test normally.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__])
