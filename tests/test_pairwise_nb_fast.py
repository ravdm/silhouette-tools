import time
import subprocess,sys
import numpy as np
import pytest

from silhouette_tools.metrics.wasserstein import wasserstein_1d as ws1d
from silhouette_tools.silhouettes.objectives.pairwise_nb import (
    PairwiseSilhouetteNBMixObjective,
)
from silhouette_tools.silhouettes.objectives.pairwise_nb_fast import (
    _PairwiseNBBatchWorker,
)

from silhouette_tools.silhouettes.objectives.pairwise_nb_fast import (
    CachedPairwiseSilhouetteNBMixObjective,
)

###################################################
########## TEST _PairwiseNBBatchWorker ##########
###################################################


def test_cached_object_matches_stateless_baseline():
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

    # ---- Build test for chached worker ----

    test_obj = _PairwiseNBBatchWorker(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    scores_test = test_obj.evaluate_labels(labels=labels)

    # ---- Build baseline ----

    true_obj = PairwiseSilhouetteNBMixObjective(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    scores_obj = true_obj.evaluate_labels(labels=labels)

    assert np.allclose(scores_test, scores_obj)
    np.testing.assert_array_equal(scores_test, scores_obj)


def test_cached_worker_label_flip_and_restore():
    """
    Flipping a single label and then restoring it should
    return the cached worker to the exact same state.
    """

    rng = np.random.default_rng(42)

    n = 50
    d = 6
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

    test_obj = _PairwiseNBBatchWorker(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )
    true_obj = PairwiseSilhouetteNBMixObjective(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    
    # Initial evaluation
    scores_initial = test_obj.evaluate_labels(labels=labels).copy()
    true_scores_initial = true_obj.evaluate_labels(labels)
    #check to make sure match
    assert np.allclose(scores_initial, true_scores_initial)
    np.testing.assert_array_equal(scores_initial, true_scores_initial)

    # Flip one label
    labels_flipped = labels.copy()
    flip_idx = 2
    labels_flipped[flip_idx] = (labels_flipped[flip_idx] + 1) % k

    scores_flipped = test_obj.evaluate_labels(labels=labels_flipped)
    true_scores_flipped = true_obj.evaluate_labels(labels=labels_flipped)

    assert np.allclose(scores_flipped, true_scores_flipped)
    np.testing.assert_array_equal(scores_flipped, true_scores_flipped)

    # Restore the label
    scores_restored = test_obj.evaluate_labels(labels=labels)

    # ---- Assertions ----
    assert not np.allclose(scores_flipped, scores_initial, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(scores_restored, scores_initial)


def test_cached_worker_second_call_faster():
    """
    The second call to evaluate_labels should be faster than the first
    due to cache reuse. Prints timings for inspection.
    """

    rng = np.random.default_rng(0)

    n = 80
    d = 6
    k = 5

    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    graph = [(0, 1), (1, 2), (2, 3), (3, 4)]
    coefs = [(1.0, 1.0), (0.7, -0.2), (-0.3, 0.9)]

    worker = _PairwiseNBBatchWorker(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    # First call (cache build)
    t0 = time.perf_counter()
    worker.evaluate_labels(labels)
    t_build = time.perf_counter() - t0

    # Second call (cache reuse)
    t0 = time.perf_counter()
    worker.evaluate_labels(labels)
    t_cached = time.perf_counter() - t0

    print(f"\nCache build time:   {t_build:.6f}s")
    print(f"Cached eval time:   {t_cached:.6f}s")
    print(f"Speedup factor:     {t_build / t_cached:.2f}x")

    assert t_cached <  t_build


def test_cached_worker_subgraph_missing_initial_vertices_and_label_swap():
    """
    Graph does not include dimensions 0 or 1.
    Cached worker must match baseline both before and after
    a single label swap.
    """

    rng = np.random.default_rng(2024)

    n = 60
    d = 8
    k = 4

    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    # Graph deliberately skips dimensions 0 and 1
    graph = [
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    coefs = [
        (1.0, 1.0),
        (0.5, -0.3),
    ]

    cached = _PairwiseNBBatchWorker(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    baseline = PairwiseSilhouetteNBMixObjective(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    # ---- First evaluation ----
    scores_cached_1 = cached.evaluate_labels(labels)
    scores_base_1 = baseline.evaluate_labels(labels)

    assert scores_cached_1.shape == scores_base_1.shape
    assert np.allclose(scores_cached_1, scores_base_1)

    # ---- Swap a single label ----
    labels2 = labels.copy()
    idx = 0
    labels2[idx] = (labels2[idx] + 1) % k

    scores_cached_2 = cached.evaluate_labels(labels2)
    scores_base_2 = baseline.evaluate_labels(labels2)

    assert scores_cached_2.shape == scores_base_2.shape
    assert np.allclose(scores_cached_2, scores_base_2)


def test_cached_multiple_label_changes():
    rng = np.random.default_rng(1)
    n, d, k = 60, 4, 4
    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    graph = [(0, 1)]
    coefs = [(1.0, 0.5)]

    cached = _PairwiseNBBatchWorker(X=X, k=k, metric=ws1d, graph=graph, coefs=coefs)
    baseline = PairwiseSilhouetteNBMixObjective(
        X=X, k=k, metric=ws1d, graph=graph, coefs=coefs
    )

    cached.evaluate_labels(labels)

    labels2 = labels.copy()
    labels2[[1, 5, 9]] = (labels2[[1, 5, 9]] + 1) % k

    s = cached.evaluate_labels(labels2)
    b = baseline.evaluate_labels(labels2)
    assert np.allclose(s, b)


def test_cached_permutation_invariance():
    rng = np.random.default_rng(2)
    n, d, k = 40, 3, 3
    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    perm = rng.permutation(n)

    graph = [(0, 1)]
    coefs = [(1.0, 1.0)]

    w1 = _PairwiseNBBatchWorker(X=X, k=k, metric=ws1d, graph=graph, coefs=coefs)
    w2 = _PairwiseNBBatchWorker(X=X[perm], k=k, metric=ws1d, graph=graph, coefs=coefs)

    s1 = w1.evaluate_labels(labels)
    s2 = w2.evaluate_labels(labels[perm])

    assert np.allclose(s1, s2)


def test_cached_empty_components():
    rng = np.random.default_rng(3)
    n, d, k = 30, 4, 5
    X = rng.normal(size=(n, d))
    labels = np.zeros(n, dtype=int)  # only one active component

    graph = [(0, 1)]
    coefs = [(1.0, 1.0)]

    cached = _PairwiseNBBatchWorker(X=X, k=k, metric=ws1d, graph=graph, coefs=coefs)
    scores = cached.evaluate_labels(labels)

    assert np.isfinite(scores).all()


def test_cached_worker_single_flip_stress_vs_baseline():
    """
    Stress test:
      - large dataset
      - cache build
      - single-label update
      - baseline recomputation
      - correctness + timing comparison
    """

    import time
    import numpy as np
    from silhouette_tools.silhouettes.projections import graphs
    from silhouette_tools.metrics.wasserstein import wasserstein_1d as ws1d

    rng = np.random.default_rng(0)

    # ---- Large problem ----
    n = 300
    d = 100
    k = 10

    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    graph = graphs.random_tree(d)
    coefs = [
        (1.0, 1.0),
        (1.0, -0.5),
    ]

    # ---- Cached worker ----
    cached = _PairwiseNBBatchWorker(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    # Initial cache build
    t0 = time.perf_counter()
    cached.evaluate_labels(labels)
    t1 = time.perf_counter()

    # Single label flip
    labels_flipped = labels.copy()
    labels_flipped[0] = (labels_flipped[0] + 1) % k

    t2 = time.perf_counter()
    scores_cached = cached.evaluate_labels(labels_flipped)
    t3 = time.perf_counter()

    # ---- Baseline (stateless) ----
    baseline = PairwiseSilhouetteNBMixObjective(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )

    t4 = time.perf_counter()
    scores_baseline = baseline.evaluate_labels(labels_flipped)
    t5 = time.perf_counter()

    # ---- Correctness ----
    assert np.allclose(scores_cached, scores_baseline)

    # ---- Timing diagnostics ----
    build_time = t1 - t0
    update_time = t3 - t2
    baseline_time = t5 - t4

    print("\n[Pairwise NB Cached vs Baseline Stress Test]")
    print(f"Cached initial build: {build_time:.3f} s")
    print(f"Cached single update: {update_time:.3f} s")
    print(f"Baseline recompute:   {baseline_time:.3f} s")
    print(f"Speedup vs baseline:  {baseline_time / update_time:.1f}x")

def test_cached_manager_matches_single_worker_serial():
    rng = np.random.default_rng(0)

    # ---- Problem setup ----
    n = 40
    d = 6
    k = 3

    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    graph = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    coefs = [
        (1.0, 1.0),
        (1.0, -0.5),
        (0.3, 0.7),
    ]

    # ---- Single worker baseline ----
    worker = _PairwiseNBBatchWorker(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )
    scores_worker = worker.evaluate_labels(labels)

    # ---- Cached manager, serial path ----
    manager = CachedPairwiseSilhouetteNBMixObjective(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
        n_workers=1,
    )
    scores_manager = manager.evaluate_labels(labels)

    # ---- Exact equality ----
    np.testing.assert_array_equal(scores_manager, scores_worker)


def test_cached_manager_parallel_agrees():

    rng = np.random.default_rng(0)

    # ---- Problem setup ----
    n = 40
    d = 6
    k = 3

    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    graph = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    coefs = [
        (1.0, 1.0),
        (1.0, -0.5),
        (0.3, 0.7),
    ]

    # ---- Single worker baseline ----
    worker = _PairwiseNBBatchWorker(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
    )
    scores_worker = worker.evaluate_labels(labels)

    # ---- Cached manager, parallel path ----
    manager = CachedPairwiseSilhouetteNBMixObjective(
        X=X,
        k=k,
        metric=ws1d,
        graph=graph,
        coefs=coefs,
        n_workers=2,  # force parallel path
        mp_context= "fork",
    )
    scores_parallel = manager.evaluate_labels(labels)

    np.testing.assert_allclose(
        np.sort(scores_parallel),
        np.sort(scores_worker),
    )
    manager.close()
def test_cached_worker_many_sequential_label_flips_vs_baseline_tracked():
    rng = np.random.default_rng(1234)

    n = 60
    d = 5
    k = 4

    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, size=n)

    graph = [(0, 1), (1, 2), (2, 3)]
    coefs = [(1.0, 1.0), (0.5, -0.7)]

    cached = _PairwiseNBBatchWorker(
        X=X, k=k, metric=ws1d, graph=graph, coefs=coefs
    )
    baseline = PairwiseSilhouetteNBMixObjective(
        X=X, k=k, metric=ws1d, graph=graph, coefs=coefs
    )

    cached.evaluate_labels(labels)
    baseline.evaluate_labels(labels)

    current_labels = labels.copy()

    max_abs_errors = []
    max_rel_errors = []

    for t in range(1000):
        idx = rng.integers(0, n)
        old_label = current_labels[idx]

        new_label = rng.integers(0, k - 1)
        if new_label >= old_label:
            new_label += 1

        current_labels[idx] = new_label

        s_cached = cached.evaluate_labels(current_labels)
        s_base = baseline.evaluate_labels(current_labels)

        abs_err = np.max(np.abs(s_cached - s_base))
        rel_err = np.max(
            np.abs(s_cached - s_base) / np.maximum(1.0, np.abs(s_base))
        )

        max_abs_errors.append(abs_err)
        max_rel_errors.append(rel_err)

    max_abs = max(max_abs_errors)
    max_rel = max(max_rel_errors)

    print(f"\nMax abs error over 1000 steps: {max_abs:.3e}")
    print(f"Max rel error over 1000 steps: {max_rel:.3e}")

    # Hard numerical guarantees
    assert max_abs < 1e-12
    assert max_rel < 1e-12
