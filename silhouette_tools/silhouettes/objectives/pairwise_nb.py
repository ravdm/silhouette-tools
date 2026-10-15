import multiprocessing as mp
from math import ceil
from typing import Callable, Sequence

import numpy as np


def convolve_points_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return all pairwise sums x_i + y_j.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        raise ValueError("Inputs must be non-empty")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")

    return (x[:, None] + y[None, :]).ravel()


def evaluate_projection_batch(
    *,
    X: np.ndarray,  # shape (n, d)
    labels: np.ndarray,  # shape (n,)
    empirical_projections: np.ndarray,  # shape (B, n)
    proj_specs: Sequence[tuple[int, int, float, float]],
    k: int,
    metric: Callable[..., float],
) -> np.ndarray:
    """
    Evaluate a batch of pairwise silhouette projections.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Full dataset.

    labels : np.ndarray, shape (n,)
        Cluster assignments.

    empirical_projections : np.ndarray, shape (B, n)
        Precomputed empirical projections aligned with proj_specs.

    proj_specs : sequence of (i, j, a, b)
        Projection definitions corresponding to each row in
        empirical_projections.

    k : int
        Number of mixture components.

    metric : callable
        1D discrepancy function.

    Returns
    -------
    scores : np.ndarray, shape (B,)
        One score per projection.
    """
    if len(proj_specs) != empirical_projections.shape[0]:
        raise ValueError(
            "proj_specs and empirical_projections must have the same length"
        )
    B = empirical_projections.shape[0]
    scores = np.empty(B, dtype=float)

    # --- STUB: logic will go here ---
    for p in range(B):
        i, j, a, b = proj_specs[p]

        model_points = []
        model_weights = []

        for label_idx in range(k):
            mask = labels == label_idx
            component_size = np.sum(mask)

            if component_size == 0:
                continue

            tmp_points_i = X[mask, i] * a
            tmp_points_j = X[mask, j] * b

            tmp_points = convolve_points_1d(tmp_points_i, tmp_points_j)
            values, counts = np.unique(tmp_points, return_counts=True)

            component_weight = 1.0 / component_size
            model_points.append(values)
            model_weights.append(counts * component_weight)
        x_model = np.concatenate(model_points)
        w_model = np.concatenate(model_weights)
        scores[p] = metric(x=x_model, y=empirical_projections[p], w_x=w_model)
    return scores

def _evaluate_batch_worker(args):
    X, labels, empirical_projections, proj_specs, k, metric, s = args
    scores = evaluate_projection_batch(
        X=X,
        labels=labels,
        empirical_projections=empirical_projections,
        proj_specs=proj_specs,
        k=k,
        metric=metric,
    )
    return s, scores

class PairwiseSilhouetteNBMixObjective:
    """
    Objective function for silhouette-based evaluation of a Naive Bayes mixture
    using pairwise (2-sparse) linear projections.

    This object encapsulates all data and structural information needed to
    evaluate a partition (label assignment) of the dataset under a fixed
    silhouette construction:
      - a fixed dataset X
      - a fixed number of mixture components k
      - a fixed graph defining pairwise projections
      - fixed 2D projection coefficients
      - a 1D discrepancy metric (e.g. Wasserstein-1)

    The intent is that this object is *callable repeatedly* with different
    labelings during partition optimization, without re-allocating or
    re-constructing projection structure.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int,
        *,
        metric: Callable[..., float],
        graph: list[tuple[int, int]],
        coefs: list[tuple[float, float]],
    ) -> None:
        """
        Initialize the silhouette objective.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            The full dataset. Rows correspond to samples, columns to features.

        k : int
            Number of mixture components (clusters). Labels are assumed to take
            values in {0, ..., k-1}.

        metric : callable
            A function taking two 1D numpy arrays and returning a nonnegative
            scalar discrepancy. Typically a 1D Wasserstein distance.

        graph : list of (int, int)
            Edge list defining which coordinate pairs are used for pairwise
            projections. Each edge (i, j) corresponds to a 2D subspace.

        coefs : list of (float, float)
            List of 2D projection coefficient pairs. For each (i, j) in `graph`,
            each coefficient (a, b) defines a projection
                x ↦ a * x_i + b * x_j.
        """
        self.X = X
        self.n, self.d = X.shape
        self.k = k

        self.metric = metric
        self.graph = graph
        self.coefs = coefs

        # ---- Precompute empirical projections (ONCE) ----
        # Total number of projections
        self.P = len(graph) * len(coefs)

        # Shape: (P, n)
        self.empirical_projections = np.empty((self.P, self.n), dtype=float)
        self.full_proj_specs = []
        p = 0
        for i, j in self.graph:
            Xi = X[:, i]
            Xj = X[:, j]
            for a, b in self.coefs:
                self.empirical_projections[p] = a * Xi + b * Xj
                self.full_proj_specs.append((i, j, a, b))
                p += 1

    def evaluate_labels(
        self,
        labels: np.ndarray,
        batch_size: int | None = None,
        n_workers: int = 1,
    ) -> np.ndarray:
        """
        Evaluate the silhouette objective for a given label assignment.
        """

        labels = np.asarray(labels)
        if labels.shape != (self.n,):
            raise ValueError(f"labels must have shape ({self.n},), got {labels.shape}")

        if batch_size is None:
            batch_size = ceil(self.P / n_workers)

        # Build batch index ranges
        batch_ranges = []
        start = 0
        while start < self.P:
            end = min(start + batch_size, self.P)
            batch_ranges.append((start, end))
            start = end

        scores = np.empty(self.P, dtype=float)

        if n_workers == 1:
            for s, e in batch_ranges:
                scores[s:e] = evaluate_projection_batch(
                    X=self.X,
                    labels=labels,
                    empirical_projections=self.empirical_projections[s:e],
                    proj_specs=self.full_proj_specs[s:e],
                    k=self.k,
                    metric=self.metric,
                )
        else:
            with mp.Pool(processes=n_workers) as pool:
                tasks = [
                    (
                        self.X,
                        labels,
                        self.empirical_projections[s:e],
                        self.full_proj_specs[s:e],
                        self.k,
                        self.metric,
                        s,
                    )
                    for (s, e) in batch_ranges
                ]

                for s, batch_scores in pool.map(_evaluate_batch_worker, tasks):
                    scores[s : s + batch_scores.size] = batch_scores
        return scores
