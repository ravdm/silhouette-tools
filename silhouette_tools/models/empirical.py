import numpy as np
from typing import Literal, Optional

from .base import SampleableModel


class EmpiricalMeasure(SampleableModel):
    """
    Empirical probability measure supported on a finite dataset.

    This class represents the empirical distribution

        μ̂_n = (1/n) ∑_{i=1}^n δ_{x_i},

    where {x_i} are the observed data points. It is primarily intended
    for evaluation and comparison against learned models (e.g. via
    silhouettes, Wasserstein distances, etc.).

    Sampling from the empirical measure may be i.i.d. (with replacement)
    or deterministic / variance-reduced, depending on the chosen strategy.
    """

    def __init__(self, X: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Dataset defining the empirical measure.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n, d)")

        self._X = X
        self._n, self._dim = X.shape

    @property
    def dim(self) -> int:
        """Ambient data dimension."""
        return self._dim

    @property
    def n(self) -> int:
        """Number of empirical samples."""
        return self._n

    def sample(
        self,
        n: int,
        *,
        strategy: Literal["iid", "balanced"] = "iid",
        shuffle: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Draw samples from the empirical measure.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        strategy : {"iid", "balanced"}, default="iid"
            Sampling strategy to use.

            - "iid":
                Sample with replacement from the empirical distribution.
            - "balanced":
                Deterministically cycle through the dataset to produce
                an approximately uniform subsample without replacement
                when possible. This is intended for variance reduction
                in numerical evaluation.

        shuffle : bool, default=True
            Whether to randomly permute the returned samples. This is
            typically useful for balanced sampling.

        rng : np.random.Generator, optional
            Random number generator to use. If None, a default generator
            is created.

        Returns
        -------
        samples : np.ndarray of shape (n, d)
            Sampled observations.
        """
        if rng is None:
            rng = np.random.default_rng()

        if strategy == "iid":
            indices = rng.integers(0, self._n, size=n)

        elif strategy == "balanced":
            if n <= self._n:
                indices = np.arange(n)
            else:
                reps = n // self._n
                rem = n % self._n
                indices = np.concatenate([
                    np.tile(np.arange(self._n), reps),
                    np.arange(rem),
                ])

            if shuffle:
                rng.shuffle(indices)

        else:
            raise NotImplementedError(f"Unknown sampling strategy: {strategy}")

        return self._X[indices]
