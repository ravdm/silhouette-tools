import heapq
import multiprocessing as mp
from collections.abc import Sequence
from typing import Callable, List, Optional, Sequence, Tuple, Iterable

import numpy as np
from numpy.typing import NDArray

from .pairwise_nb import convolve_points_1d
from ._parallel import collect_worker_scores, partition_graph_edges, run_worker
import time 

def _build_component(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a weighted 1D empirical measure for a single mixture component
    under a pairwise (2-sparse) projection.

    For component samples x and y (same length m), construct all pairwise
    sums a*x_i + b*y_j with total mass m. Identical values are consolidated
    by summing weights.

    Empty components return empty arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")

    m = x.shape[0]
    if m == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    raw_values = convolve_points_1d(a * x, b * y)

    values, counts = np.unique(raw_values, return_counts=True) #NOTE slowest step right now
    weights = counts / m

    return values, weights



def _merge_sorted_values_weights( #TODO TEST THIS XXX
    values_list: Iterable[np.ndarray],
    weights_list: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple sorted (values, weights) arrays into one sorted array,
    collapsing duplicate values by summing weights.

    All input value arrays must be sorted ascending.
    """
    # Build iterators of (value, weight) pairs
    iterables = (
        zip(v, w) for v, w in zip(values_list, weights_list)
    )

    merged_iter = heapq.merge(*iterables)

    out_values = []
    out_weights = []

    last_v = None
    acc_w = 0.0

    for v, w in merged_iter:
        if last_v is None or v != last_v: #new v seen 
            if last_v is not None:
                out_values.append(last_v)
                out_weights.append(acc_w)
            last_v = v
            acc_w = w
        else: #not a new v, so accumulate more weight
            acc_w += w

    if last_v is not None:
        out_values.append(last_v)
        out_weights.append(acc_w)

    return (
        np.asarray(out_values, dtype=float),
        np.asarray(out_weights, dtype=float),
    )

class _PairwiseNBBatchWorker:
    """
    Stateful worker for evaluating a subset of pairwise Naive Bayes silhouette
    projections with incremental caching.

    This object owns a fixed batch of pairwise (2-sparse) linear projections,
    defined by a subset of graph edges and projection coefficients. For each
    projection, it maintains cached per-component empirical measures that can
    be incrementally updated when cluster labels change.

    Responsibilities
    ----------------
    - Hold a reduced view of the data matrix `X`, restricted to vertices
      induced by the worker's assigned graph edges.
    - Precompute and store the empirical (data) projection measures for all
      edge–coefficient combinations in the batch.
    - Cache per-component projected measures (values and weights) for each
      projection.
    - Efficiently update only those cached components whose membership changes
      between successive labelings.
    - Assemble the full model projection from cached components and evaluate
      a 1D discrepancy metric against the empirical projection.

    Caching Semantics
    -----------------
    - On the first call to `evaluate_labels`, all component caches are built.
    - On subsequent calls, only components whose labels have changed are
      recomputed.
    - Cached component weights are normalized so that the total mass of each
      component equals its sample count; consequently, the total model mass
      equals `n`.

    Invariants
    ----------
    - The graph structure, projection coefficients, and number of projections
      (`B`) are fixed at construction time and must not change.
    - The worker is stateful: `evaluate_labels` mutates internal caches.
    - This class assumes labels are integer-valued in `[0, k)` and enforces
      this at runtime.

    Intended Usage
    --------------
    This class is not user-facing. It is designed to be owned and orchestrated
    by a higher-level manager that partitions projections across multiple
    workers and aggregates their outputs, optionally in parallel.

    Notes
    -----
    - The leading underscore indicates this is an internal implementation
      detail, not part of the public API.
    - The worker is fully functional on its own for small problem sizes, but
      is primarily intended to operate as part of a larger batched system.
    """

    X: np.ndarray
    n: int
    k: int
    B: int
    metric: Callable[..., float]

    graph: list[tuple[int, int]]
    coefs: list[tuple[float, float]]

    edge_proj_specs: list[tuple[int, int, float, float]]
    empirical_values: list[np.ndarray]
    empirical_weights: list[np.ndarray]

    cache_points: list[list[Optional[np.ndarray]]]
    cache_weights: list[list[Optional[np.ndarray]]]
    cache_ready: bool

    def __init__(
        self,
        X: np.ndarray,
        k: int,
        *,
        metric: Callable[..., float],
        graph: list[tuple[int, int]],
        coefs: list[tuple[float, float]],
    ) -> None:

        self.n = X.shape[0]
        self.k = k
        self.B = len(graph) * len(coefs)
        self.coefs = coefs
        self.metric = metric

        relevant_vertices, inverse = np.unique(graph, return_inverse=True)
        self.X = X[:, relevant_vertices]
        self.graph = [tuple(map(int, row)) for row in inverse]

        self.edge_proj_specs = [
            (i, j, a, b) for (i, j) in self.graph for (a, b) in self.coefs
        ]
        self.empirical_values = []
        self.empirical_weights = []
        for i, j, a, b in self.edge_proj_specs:
            proj = a * self.X[:, i] + b * self.X[:, j]
            values, counts = np.unique(proj, return_counts=True)
            weights = counts

            self.empirical_values.append(values)
            self.empirical_weights.append(weights)

        self.cache_points = [[None for _ in range(self.k)] for _ in range(self.B)]
        self.cache_weights = [[None for _ in range(self.k)] for _ in range(self.B)]
        self.cache_labels = np.empty(self.n, dtype=int)
        self.cache_ready = False

    def _update_components(
        self,
        component_idxs: int | Sequence[int] | NDArray[np.integer],
    ) -> None:
        """
        Update cached (values, weights) for one or more mixture components
        across all edge–projection specs.
        """
        if not self.cache_ready:
            raise RuntimeError(
                "_update_components called before cache initialization. "
                "Call evaluate_labels() at least once first."
            )
        # Normalize to iterable
        if isinstance(component_idxs, int):
            component_idxs = (component_idxs,)

        for component_idx in component_idxs:
            mask = self.cache_labels == component_idx

            for edge_proj_idx, (i, j, a, b) in enumerate(self.edge_proj_specs):
                x = self.X[mask, i]
                y = self.X[mask, j]


                values, weights = _build_component(x, y, a, b)

                self.cache_points[edge_proj_idx][component_idx] = values
                self.cache_weights[edge_proj_idx][component_idx] = weights
    def evaluate_labels(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)

        if labels.shape != self.cache_labels.shape:
            raise ValueError(
                "labels must have the same shape as cached labels: "
                f"{labels.shape} vs {self.cache_labels.shape}"
            )
        if np.any((labels < 0) | (labels >= self.k)):
            raise ValueError(
                f"labels must be integers in [0, {self.k}), got out-of-range values"
            )

        if not self.cache_ready:
            self.cache_ready = True
            self.cache_labels = labels.copy()
            all_idxs = np.arange(self.k)
            self._update_components(all_idxs)
        else:
            diff = self.cache_labels != labels
            changed = np.unique(np.concatenate([self.cache_labels[diff], labels[diff]]))
            self.cache_labels = labels.copy()

            if changed.size > 0:
                self._update_components(changed)

            # TODO Check from here out. I have not gone over this.
        output_scores = np.empty(self.B, dtype=float)
        for edge_proj_idx in range(self.B):
            acc_values = []
            acc_weights = []

            for component_idx in range(self.k):
                v = self.cache_points[edge_proj_idx][component_idx]
                w = self.cache_weights[edge_proj_idx][component_idx]

                if v is None or v.size == 0:
                    continue

                acc_values.append(v)
                acc_weights.append(w)

            if acc_values:
                #There are two options, one that sorts everything and one that doesnt 
                #sorting seems to be slightly faster

                #do not sort
                model_values = np.concatenate(acc_values)
                model_weights = np.concatenate(acc_weights)

                # #sort
                # model_values, model_weights = _merge_sorted_values_weights(
                #         acc_values,
                #         acc_weights,
                #         )
            else:
                model_values = np.empty(0, dtype=float)
                model_weights = np.empty(0, dtype=float)
            output_scores[edge_proj_idx] = self.metric(
                x=self.empirical_values[edge_proj_idx],
                y=model_values,
                w_x=self.empirical_weights[edge_proj_idx],
                w_y=model_weights,
            )

        return output_scores


#######################################################
######  Main object for running label evaluations #####
#######################################################

def _worker_process_main(
    in_q,
    out_q,
    X: np.ndarray,
    k: int,
    *,
    metric: Callable[..., float],
    graph: list[tuple[int, int]],
    coefs: list[tuple[float, float]],
):
    run_worker(
        in_q, out_q, _PairwiseNBBatchWorker, X, k,
        metric=metric, graph=graph, coefs=coefs,
    )


class CachedPairwiseSilhouetteNBMixObjective:
    n: int
    d: int
    k: int

    def __init__(
        self,
        X: np.ndarray,
        k: int,
        *,
        metric: Callable[..., float],
        graph: list[tuple[int, int]],
        coefs: list[tuple[float, float]],
        n_workers: int = 1,
        mp_context: str = "fork",
    ) -> None:
        self.X = X
        self.k = k
        self.metric = metric
        self.graph = graph
        self.coefs = coefs
        self.n_workers = n_workers
        self.mp_context = mp_context

        self.n, self.d = X.shape
        self.workers_initialized = False
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        if len(graph) == 0:
            raise ValueError("graph must contain at least one edge")
        if len(coefs) == 0:
            raise ValueError("coefs must contain at least one projection")

        if n_workers == 1:
            # Single-process path: one worker object
            self._worker = _PairwiseNBBatchWorker(
                X=X,
                k=k,
                metric=metric,
                graph=graph,
                coefs=coefs,
            )

    def evaluate_labels(self, labels: np.ndarray) -> np.ndarray:
        if self.n_workers == 1:
            return self._worker.evaluate_labels(labels)

        try:
            if not self.workers_initialized:
                ctx = mp.get_context(self.mp_context)
                self._in_queues = []
                self._processes = []
                self._out_queue = ctx.Queue()
                self.workers_initialized = True

                # The requested count is an upper limit: every active worker
                # must have at least one edge to evaluate.
                graph_batches = partition_graph_edges(
                    self.graph, min(self.n_workers, len(self.graph))
                )
                for graph_batch in graph_batches:
                    in_q = ctx.Queue()
                    self._in_queues.append(in_q)
                    process = ctx.Process(
                        target=_worker_process_main,
                        args=(in_q, self._out_queue, self.X, self.k),
                        kwargs=dict(
                            metric=self.metric,
                            graph=graph_batch,
                            coefs=self.coefs,
                        ),
                    )
                    process.start()
                    self._processes.append(process)

            for queue in self._in_queues:
                queue.put(labels)

            return collect_worker_scores(self._out_queue, self._processes)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not getattr(self, "workers_initialized", False):
            return

        # 1. Ask workers to exit cleanly
        for q in self._in_queues:
            try:
                q.put(None)
            except Exception:
                pass

        # 2. Give them time to exit normally
        for p in self._processes:
            p.join(timeout=2.0)

        # 3. Force-kill any stragglers (safety net)
        for p in self._processes:
            if p.is_alive():
                p.terminate()

        for p in self._processes:
            p.join()

        # 4. Close queues and their feeder threads
        for q in self._in_queues:
            try:
                # A failed worker may leave unread labels in its queue.
                # Do not wait for a feeder thread that has no consumer.
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass

        try:
            self._out_queue.close()
            self._out_queue.join_thread()
        except Exception:
            pass

        # 5. Reset state
        self._in_queues.clear()
        self._processes.clear()
        self.workers_initialized = False
