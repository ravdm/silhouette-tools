import numpy as np
from abc import ABC, abstractmethod
from typing import Literal

class SampleableModel(ABC):
    """
    Abstract base class for models that define a distribution from which
    samples can be drawn.

    A `SampleableModel` represents any object that can generate samples
    in ℝ^d. By default, samples are assumed to be i.i.d., but implementations
    may optionally support alternative sampling strategies intended for
    variance reduction or numerical stability (e.g. stratified or balanced
    sampling).

    Importantly, non-i.i.d. sampling strategies are meant for *numerical
    evaluation* (e.g. Monte Carlo integration, discrepancy estimation) and
    should not be interpreted as faithful generative sampling from the
    underlying joint distribution.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Ambient dimension of the data space.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        n: int,
        *,
        strategy: Literal["iid", "balanced", "stratified"] = "iid",
        shuffle: bool = True,
    ) -> np.ndarray:
        """
        Draw samples from the model.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        strategy : {"iid", "balanced", "stratified"}, default="iid"
            Sampling strategy to use.

            - "iid":
                Draw i.i.d. samples from the model.
            - "balanced":
                Enforce deterministic or approximately equal representation
                across mixture components or partitions, when applicable.
            - "stratified":
                Use stratified sampling to reduce Monte Carlo variance.

            Implementations may support only a subset of these strategies
            and should raise `NotImplementedError` for unsupported options.

        shuffle : bool, default=True
            Whether to randomly permute the returned samples. This is
            typically desirable for balanced or stratified sampling schemes.

        Returns
        -------
        samples : np.ndarray
            Array of shape (n, d) containing sampled observations, where
            d is the ambient data dimension.
        """
        raise NotImplementedError


class DensityModel(SampleableModel):
    """
    Abstract base class for models that define a probability density
    over R^d.

    A DensityModel provides evaluation of the probability density and/or
    its logarithm. Subclasses must implement `log_density`. The `density`
    method is provided as a convenience and may be overridden for
    efficiency.
    """

    @abstractmethod
    def log_density(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the log-density log p(X).

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Input points.

        Returns
        -------
        log_p : np.ndarray of shape (n,)
            Log-density evaluated at each input point.
        """
        raise NotImplementedError

    def density(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the density p(X).

        This default implementation computes exp(log_density(X)).
        Subclasses may override this method if density evaluation is
        available directly and is more efficient or numerically stable.
        """
        return np.exp(self.log_density(X))
