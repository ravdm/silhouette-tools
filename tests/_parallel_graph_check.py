"""Run real worker processes behind the regression tests' time limit."""

import argparse
import importlib
import os

import numpy as np

from silhouette_tools.metrics.wasserstein import wasserstein_1d
from silhouette_tools.silhouettes.objectives.pairwise_nb import (
    PairwiseSilhouetteNBMixObjective,
)
from silhouette_tools.silhouettes.projections import graphs, pairwise


def failing_metric(**kwargs):
    raise ValueError("deliberate metric failure")


def exiting_metric(**kwargs):
    os._exit(7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["cached", "fast"], required=True)
    parser.add_argument("--kind", choices=["chain", "regular"], default="chain")
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--context", choices=["fork", "spawn"], default="fork")
    parser.add_argument("--failure", choices=["metric", "exit", "initialization"])
    args = parser.parse_args()

    implementation = importlib.import_module(
        f"silhouette_tools.silhouettes.objectives.pairwise_nb_{args.variant}"
    )
    rng = np.random.default_rng(42)
    X = np.round(rng.normal(size=(24, args.dimension)), 1)
    labels = np.repeat(np.arange(3), 8)
    if args.kind == "regular":
        graph = graphs.random_regular_graph(args.dimension, 4, rng)
    else:
        graph = graphs.chain_graph(args.dimension)
    coefs = pairwise.silhouette_coefficients(a=-2, b1=1, b2=1, k=2)

    worker_graph = graph
    metric = wasserstein_1d
    if args.failure == "metric":
        metric = failing_metric
    elif args.failure == "exit":
        metric = exiting_metric
    elif args.failure == "initialization":
        worker_graph = [(0, args.dimension), (1, 2)]

    manager = implementation.CachedPairwiseSilhouetteNBMixObjective(
        X, 3, metric=metric, graph=worker_graph, coefs=coefs,
        n_workers=9, mp_context=args.context,
    )
    processes = []
    try:
        if args.failure:
            try:
                manager.evaluate_labels(labels)
            except RuntimeError as error:
                expected = {
                    "metric": "deliberate metric failure",
                    "exit": "exited before returning results",
                    "initialization": "IndexError",
                }[args.failure]
                assert expected in str(error), str(error)
            else:
                raise AssertionError("Worker failure was not reported")
            assert not manager.workers_initialized
            assert manager._processes == []
            # A failure must release the old workers and allow a fresh start.
            manager.metric = wasserstein_1d
            manager.graph = graph

        baseline = PairwiseSilhouetteNBMixObjective(
            X, 3, metric=wasserstein_1d, graph=graph, coefs=coefs,
        )
        changed = labels.copy()
        changed[0] = 1
        changed_again = changed.copy()
        changed_again[8] = 2
        for assignments in (labels, changed, changed_again, labels):
            actual = manager.evaluate_labels(assignments)
            expected = baseline.evaluate_labels(assignments)
            np.testing.assert_allclose(
                np.sort(actual), np.sort(expected), rtol=0, atol=1e-12,
            )
            assert actual.size == len(graph) * len(coefs)
            assert len(manager._processes) == min(9, len(graph))
        processes = manager._processes.copy()
    finally:
        manager.close()
        manager.close()  # Closing an already closed objective is harmless.
    assert all(not process.is_alive() for process in processes)


if __name__ == "__main__":
    main()
