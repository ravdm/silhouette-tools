import numpy as np
from silhouette_tools.silhouettes.objectives.pairwise_nb_cached import CachedPairwiseSilhouetteNBMixObjective,_PairwiseNBBatchWorker
from silhouette_tools.metrics.wasserstein import wasserstein_1d as ws1d
def main():
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
        mp_context= "fork", #need for testing
    )
    scores_parallel = manager.evaluate_labels(labels)

    try:
        scores_parallel = manager.evaluate_labels(labels)
        np.testing.assert_allclose(
            np.sort(scores_parallel),
            np.sort(scores_worker),
        )
    finally:
        manager.close()
if __name__ =="__main__":
    main()
