"""Shared graph partitioning and communication for cached objectives."""

from collections.abc import Sequence
from queue import Empty
from traceback import format_exc

import numpy as np


def partition_graph_edges(
    edges: Sequence[tuple[int, int]], n_batches: int
) -> list[list[tuple[int, int]]]:
    """Balance edge counts, then greedily minimize vertices within each batch.

    Each edge is assigned exactly once. Batch sizes differ by at most one;
    empty batches occur only when there are fewer edges than batches.
    """
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")

    size, remainder = divmod(len(edges), n_batches)
    capacities = [size + (i < remainder) for i in range(n_batches)]
    batches: list[list[tuple[int, int]]] = [[] for _ in range(n_batches)]
    vertices = [set() for _ in range(n_batches)]

    for u, v in edges:
        edge_vertices = {u, v}
        available = (
            i for i in range(n_batches) if len(batches[i]) < capacities[i]
        )
        batch = min(
            available,
            key=lambda i: (len(edge_vertices - vertices[i]), len(batches[i])),
        )
        batches[batch].append((u, v))
        vertices[batch].update(edge_vertices)

    return batches


def run_worker(in_queue, out_queue, worker_type, X, k, *, metric, graph, coefs):
    """Keep a worker's cache alive and send failures back to its caller."""
    try:
        worker = worker_type(X, k, metric=metric, graph=graph, coefs=coefs)
        while True:
            labels = in_queue.get()
            if labels is None:
                return
            out_queue.put((worker.evaluate_labels(labels), None))
    except Exception:
        # A traceback string remains transferable even for exceptions that
        # cannot themselves be pickled by multiprocessing.
        out_queue.put((None, format_exc()))


def collect_worker_scores(out_queue, processes) -> np.ndarray:
    """Collect one result per active worker, detecting failures and exits."""
    results = []
    while len(results) < len(processes):
        try:
            scores, error = out_queue.get(timeout=0.1)
        except Empty:
            exited = [p for p in processes if p.exitcode is not None]
            if exited:
                details = ", ".join(
                    f"pid={p.pid}, exitcode={p.exitcode}" for p in exited
                )
                raise RuntimeError(
                    f"Silhouette worker exited before returning results ({details})"
                ) from None
            continue
        if error is not None:
            raise RuntimeError(f"Silhouette worker failed:\n{error}")
        results.append(scores)

    return np.concatenate(results)
