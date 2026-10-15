import warnings
from typing import Literal

import numpy as np
from numpy._core.numerictypes import floating
from numpy.typing import NDArray
from silhouette_tools.models.base import DensityModel
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def _log_gaussian_kernel(x: NDArray[np.floating], bw: float) -> NDArray[np.floating]:
    """
    Compute log N(x | 0, bw^2) elementwise.
    """
    return -0.5 * (x / bw) ** 2 - np.log(bw) - 0.5 * np.log(2 * np.pi)


def fit_bandwidth_likelihood_cv(
    x: np.ndarray,
    grid: np.ndarray,
    n_folds: int = 5,
) -> float:
    x = x[:, None]  # make 2D

    kde = KernelDensity(kernel="gaussian")

    search = GridSearchCV(
        kde,
        {"bandwidth": grid},
        cv=n_folds,
        n_jobs=1,  # careful with nested parallelism
    )

    search.fit(x)
    return search.best_params_["bandwidth"]


class NaiveBayesKDEMixture(DensityModel):
    def __init__(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        weights: np.ndarray | None = None,
    ):
        # ---- basic shape checks ----

        if X.ndim != 2:
            raise ValueError("X must be 2D (n, d)")
        if labels.ndim != 1:
            raise ValueError("labels must be 1D")
        if X.shape[0] != labels.shape[0]:
            raise ValueError("X and labels must have same length")
        self.X = np.asarray(X, dtype=float)
        self._n_samples, self._dim = X.shape

        labels = np.asarray(labels)
        unique, relabeled = np.unique(labels, return_inverse=True)
        self.label_map = unique  # original → canonical
        self.labels = relabeled  # 0,…,k−1
        self._k = unique.size

        self.bandwidths = np.empty((self._k, self._dim), dtype=float)
        self._bandwidths_ready = False

        if weights is None:
            counts = np.bincount(self.labels, minlength=self._k)
            weights = counts / counts.sum()
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (self._k,):
                raise ValueError(f"weights must have shape ({self._k},)")
            if np.any(weights < 0):
                raise ValueError("weights must be nonnegative")
            weights = weights / weights.sum()
        self.weights = weights

    @property
    def dim(self) -> int:
        return self._dim

    def fit_bandwidths(
        self,
        method: Literal["silverman", "likelihood"],
        *,
        grid: np.ndarray | None = None,
        n_folds: int = 5,
    ) -> None:
        if self._bandwidths_ready:
            raise RuntimeError("bandwidths already fitted")

        elif method not in {"silverman","likelihood"}:
            raise ValueError(f"unknown bandwidth method {method!r}")

        for c in range(self._k):
            mask = self.labels == c
            Xc = self.X[mask]  # shape (n_c, d)

            eps = 1e-6

            for j in range(self._dim):
                data_1d = Xc[:, j]

                if np.var(data_1d) == 0.0:
                    warnings.warn(
                        f"Zero-variance feature detected (component={c}, dim={j}); "
                        f"using fallback bandwidth {eps}.",
                        RuntimeWarning,
                    )
                    bw = eps
                elif method == "silverman":
                    kde = KDEUnivariate(data_1d)
                    kde.fit(bw="silverman", fft=False)
                    bw = float(kde.bw)
                elif method == "likelihood":
                    if grid is None:
                        raise ValueError(f"No grid input for likelihood CV search")
                    bw = fit_bandwidth_likelihood_cv(data_1d, grid, n_folds)

                self.bandwidths[c, j] = bw
        self._bandwidths_ready = True

    def log_density(self, X: np.ndarray) -> np.ndarray:
        if not self._bandwidths_ready:
            raise RuntimeError("bandwidths not fitted; call fit_bandwidths() first")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._dim:
            raise ValueError(f"X must have shape (n, {self._dim})")

        n_eval = X.shape[0]
        log_probs = np.empty((n_eval, self._k), dtype=float)

        for c in range(self._k):
            mask = self.labels == c
            Xc = self.X[mask]  # (n_c, d)
            n_c = Xc.shape[0]

            if n_c == 0:
                log_probs[:, c] = -np.inf
                continue
            assert self.weights is not None
            log_pc = np.log(self.weights[c])

            # accumulate log-density per dimension
            log_px_c = np.zeros(n_eval, dtype=float)

            for j in range(self._dim):
                bw = self.bandwidths[c, j]
                diffs = X[:, None, j] - Xc[None, :, j]  # (n_eval, n_c)

                log_k = _log_gaussian_kernel(diffs, bw)  # (n_eval, n_c)

                # log mean exp over samples in component
                m = np.max(log_k, axis=1)
                log_kde = m + np.log(np.mean(np.exp(log_k - m[:, None]), axis=1)) 
                #mean above so that it is (1/n_k) k(...)

                log_px_c += log_kde

            log_probs[:, c] = log_pc + log_px_c

        # log-sum-exp over components
        m = np.max(log_probs, axis=1)
        return m + np.log(np.sum(np.exp(log_probs - m[:, None]), axis=1))

    def sample(
        self,
        n: int,
        *,
        strategy: Literal["iid", "balanced", "stratified"] = "iid",
        shuffle: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError("Sampling not implemented yet")
