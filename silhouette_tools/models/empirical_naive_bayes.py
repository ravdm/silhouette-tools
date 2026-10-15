import numpy as np
from typing import Optional, Literal

from .base import SampleableModel


class EmpiricalNaiveBayesMixture(SampleableModel):
    """
    Mixture of empirical naive Bayes measure

    This model represents a mixture of k components, where each component
    is defined by assuming independence across dimensions and using
    empirical (coordinate-wise) marginals within each component.

    Concretely:
        - Data points are partitioned into k components.
        - Within each component, the joint distribution is approximated
          as a product of 1D empirical marginals.
        - Sampling proceeds by first selecting a component, then sampling
          each coordinate independently from that component's empirical
          marginal.

    This class is intended as a structured, sampleable model for use in
    evaluation (e.g. silhouettes), not as a fully optimized density model.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int,
        *,
        labels: Optional[np.ndarray] = None,
        init: Literal["random", "knn"] = "random",
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Input data defining the empirical distribution.

        k : int
            Number of components (rank of the model).

        labels : np.ndarray of shape (n,), optional
            Initial component assignments. If provided, must contain
            integers in {0, ..., k-1}. If None, labels are initialized
            according to `init`.

        init : {"random", "knn"}, default="random"
            Initialization strategy used when `labels` is None.

        rng : np.random.Generator, optional
            Random number generator used for initialization and sampling.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n, d)")
        if k <= 0:
            raise ValueError("k must be a positive integer")

        self.X = X
        self.n, self._dim = X.shape
        self.k = k

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        if labels is not None:
            labels = np.asarray(labels)
            if labels.shape != (self.n,):
                raise ValueError("labels must have shape (n,)")
            if labels.min() < 0 or labels.max() >= k:
                raise ValueError("labels must be in {0, ..., k-1}")
            self.labels = labels.copy()
        else:
            self.labels = self._initialize_labels(init)

        # Precompute indices per component for fast sampling
        self._build_component_index()

    @property
    def dim(self) -> int:
        """Ambient data dimension."""
        return self._dim

    def _initialize_labels(self, init: str) -> np.ndarray:
        """
        Initialize component labels according to the chosen strategy.
        """
        if init == "random":
            return self.rng.integers(0, self.k, size=self.n)

        elif init == "knn":
            # Very simple placeholder: pick k random centers and assign
            # by nearest center (Euclidean).
            centers = self.X[self.rng.choice(self.n, size=self.k, replace=False)]
            dists = np.linalg.norm(self.X[:, None, :] - centers[None, :, :], axis=2)
            return np.argmin(dists, axis=1)

        else:
            raise NotImplementedError(f"Unknown init strategy: {init}")

    def _build_component_index(self) -> None:
        """
        Build index lists for each component.
        """
        self.component_indices = []
        for j in range(self.k):
            idx = np.nonzero(self.labels == j)[0]
            if len(idx) == 0:
                raise ValueError(f"Component {j} has no assigned samples")
            self.component_indices.append(idx)

        # Component weights (empirical mixture weights)
        self.weights = np.array(
            [len(idx) for idx in self.component_indices], dtype=float
        )
        self.weights /= self.weights.sum()

    def update_labels(self, new_labels: np.ndarray) -> None:
        """
        Update component assignments and rebuild internal structures.

        This method is intended to be called by external optimization
        routines (e.g. silhouette-guided refinements).
        """
        new_labels = np.asarray(new_labels)
        if new_labels.shape != (self.n,):
            raise ValueError("new_labels must have shape (n,)")
        if new_labels.min() < 0 or new_labels.max() >= self.k:
            raise ValueError("new_labels must be in {0, ..., k-1}")

        self.labels = new_labels.copy()
        self._build_component_index()

    def sample(
        self,
        n: int,
        *,
        strategy: Literal["iid", "balanced"] = "iid",
        shuffle: bool = True,
    ) -> np.ndarray:
        """
        Draw samples from the low-rank empirical model.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        strategy : {"iid", "balanced"}, default="iid"
            Component sampling strategy.

            - "iid":
                Sample components according to empirical mixture weights.
            - "balanced":
                Enforce approximately equal sampling from each component.

        shuffle : bool, default=True
            Whether to randomly permute the returned samples.

        Returns
        -------
        samples : np.ndarray of shape (n, d)
            Generated samples.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        # Choose components
        if strategy == "iid":
            comp_ids = self.rng.choice(self.k, size=n, p=self.weights)

        elif strategy == "balanced":
            reps = n // self.k
            rem = n % self.k
            comp_ids = np.concatenate([
                np.repeat(np.arange(self.k), reps),
                np.arange(rem),
            ])
            if shuffle:
                self.rng.shuffle(comp_ids)

        else:
            raise NotImplementedError(f"Unknown sampling strategy: {strategy}")

        # Sample coordinates independently within each component
        samples = np.empty((n, self._dim), dtype=self.X.dtype)

        for i, j in enumerate(comp_ids):
            idx = self.component_indices[j]
            # For each coordinate, sample independently from empirical marginal
            coord_indices = self.rng.integers(0, len(idx), size=self._dim)
            samples[i] = self.X[idx[coord_indices], np.arange(self._dim)]

        return samples
    def linf_depth(
            self,
            X_test,
            ) -> np.ndarray:
        raise NotImplementedError


