import numpy as np
from numpy.typing import NDArray
from silhouette_tools.models.base import SampleableModel
from multiprocessing import get_context

def _pnorm_depth_batch(
    X_batch: NDArray[np.floating],
    Z: NDArray[np.floating],
    mass: float,
    p: float,
) -> NDArray[np.floating]:
    diffs = X_batch[:, None, :] - Z[None, :, :]
    dists = np.linalg.norm(diffs, ord=p, axis=-1)
    return np.quantile(dists, mass, axis=1)


def pnorm_depth(
    model: SampleableModel,
    X_test: NDArray[np.floating],
    *,
    mass: float,
    p: float = 2.0,
    n_samples: int = 1_000,
    batch_size: int | None = None,
    n_workers: int = 1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """
    Monte Carlo p-norm data depth via radius-for-mass.

    For each test point x, this function estimates the smallest radius r such
    that a fraction `mass` of the reference distribution lies within an
    ℓ_p-ball centered at x:

        r(x) = inf{ r : P(||X - x||_p ≤ r) ≥ mass }

    The probability is approximated using Monte Carlo samples drawn from
    `model`.

    Parameters
    ----------
    model : SampleableModel
        Reference distribution to sample from.

    X_test : ndarray of shape (n, d)
        Test points at which depth is evaluated.

    mass : float
        Target probability mass in (0, 1].

    p : float, default=2.0
        p-norm used to measure distance. Must be positive.

    n_samples : int, default=1000
        Number of Monte Carlo samples drawn from the model.

    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    radii : ndarray of shape (n,)
        Estimated radius-for-mass depth values. Smaller values indicate
        greater depth (less anomalous).
    """
    if not (0.0 < mass <= 1.0):
        raise ValueError("mass must lie in (0, 1]")

    if p <= 0:
        raise ValueError("p must be positive")

    X_test = np.asarray(X_test, dtype=float)
    if X_test.ndim != 2:
        raise ValueError("X_test must be a 2D array")

    n_test, d = X_test.shape
    if batch_size is None:
        batch_size = max(1, 10_000 // d)  # heuristic, safe default

    # Draw reference samples
    Z = model.sample(n_samples)
    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 2 or Z.shape[1] != X_test.shape[1]:
        raise ValueError("Sampled data has incompatible shape")

    batches = [X_test[i : i + batch_size] for i in range(0, n_test, batch_size)]


    if n_workers == 1:
        out = []
        for Xb in batches:
            out.append(_pnorm_depth_batch(Xb, Z, mass, p))
        return np.concatenate(out)

    # ---- Multiprocessing over batches ----
    ctx = get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.starmap(
            _pnorm_depth_batch,
            [(Xb, Z, mass, p) for Xb in batches],
        )

    return np.concatenate(results)
