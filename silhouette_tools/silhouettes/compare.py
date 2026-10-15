from concurrent.futures import ProcessPoolExecutor
from math import ceil
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from silhouette_tools.models.base import SampleableModel
from scipy.sparse import issparse, spmatrix


def _compare_batch(
    X1: np.ndarray,
    X2: np.ndarray,
    P_batch: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    metric_kwargs: dict,
) -> np.ndarray:
    """
    Worker function: compute silhouette distances for a batch of projections.
    """
    # Project samples: shape (n_samples, batch_size)
    Y1 = X1 @ P_batch
    Y2 = X2 @ P_batch

    out = np.empty(P_batch.shape[1], dtype=float)
    for j in range(P_batch.shape[1]):
        out[j] = metric(Y1[:, j], Y2[:, j], **metric_kwargs)
    return out


def compare(
    model1: SampleableModel,
    model2: SampleableModel,
    projections: spmatrix | np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_samples1: int,
    n_samples2: int,
    strategy1: Literal["iid", "balanced", "stratified"] = "iid",
    strategy2: Literal["iid", "balanced", "stratified"] = "iid",
    n_workers: int = 1,
    batch_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    **metric_kwargs: Any,
) -> np.ndarray:
    """
    Compare two sampleable models along a collection of silhouette projections.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample once
    X1 = model1.sample(n_samples1, strategy=strategy1)
    X2 = model2.sample(n_samples2, strategy=strategy2)

    if X1.shape[1] != X2.shape[1]:
        raise ValueError("model1 and model2 must have the same ambient dimension")

    d, m = projections.shape
    if X1.shape[1] != d:
        raise ValueError("Projection matrix dimension does not match sample dimension")

    # Choose batch size
    if batch_size is None:
        batch_size = ceil(m / n_workers)

    # Prepare projection access
    def get_projection_block(start: int, end: int) -> np.ndarray:
        if issparse(projections):
            return projections[:, start:end].toarray()
        else:
            return projections[:, start:end]

    # Serial case
    if n_workers == 1:
        blocks = []
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            P_batch = get_projection_block(start, end)
            blocks.append(_compare_batch(X1, X2, P_batch, metric, metric_kwargs))
        return np.concatenate(blocks)

    # Parallel case
    tasks = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            P_batch = get_projection_block(start, end)
            tasks.append(
                pool.submit(
                    _compare_batch,
                    X1,
                    X2,
                    P_batch,
                    metric,
                    metric_kwargs,
                )
            )

        results = [t.result() for t in tasks]

    return np.concatenate(results)
