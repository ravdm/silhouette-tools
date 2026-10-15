import numpy as np
from scipy.stats import wasserstein_distance

try:
    import ot  # Python Optimal Transport
except ImportError as e:
    ot = None

def wasserstein_1d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    w_x: np.ndarray | None = None,
    w_y: np.ndarray | None = None
) -> float:
    """
    Compute the 1D Wasserstein-1 (Earth Mover's) distance between two
    empirical distributions.

    Parameters
    ----------
    x : np.ndarray of shape (n,)
        Samples from the first distribution.

    y : np.ndarray of shape (m,)
        Samples from the second distribution.

    Returns
    -------
    float
        The Wasserstein-1 distance between the empirical measures
        induced by x and y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("wasserstein_1d expects 1D input arrays")

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must be non-empty")

    if w_x is None and w_y is None:
        # Unweighted case
        return float(wasserstein_distance(x, y))

    # Weighted / semi-weighted case
    if w_x is not None:
        w_x = np.asarray(w_x, dtype=float)
        if w_x.ndim != 1 or w_x.shape[0] != x.shape[0]:
            raise ValueError("w_x must be a 1D array matching x")
        if np.any(w_x < 0) or not np.any(w_x):
            raise ValueError("w_x must contain nonnegative weights")

    if w_y is not None:
        w_y = np.asarray(w_y, dtype=float)
        if w_y.ndim != 1 or w_y.shape[0] != y.shape[0]:
            raise ValueError("w_y must be a 1D array matching y")
        if np.any(w_y < 0) or not np.any(w_y):
            raise ValueError("w_y must contain nonnegative weights")

    return float(
        wasserstein_distance(
            x,
            y,
            u_weights=w_x,
            v_weights=w_y,
        )
    )

def wasserstein_1d_assume_sorted(
    x: np.ndarray,
    y: np.ndarray,
    *,
    w_x: np.ndarray | None = None,
    w_y: np.ndarray | None = None,
) -> float:
    """
    Compute the 1D Wasserstein-1 distance using POT (Python Optimal Transport).

    Parameters
    ----------
    x_points : np.ndarray of shape (n,)
        Support points of the first distribution.
    y_points : np.ndarray of shape (m,)
        Support points of the second distribution.
    w_x : np.ndarray of shape (n,), optional
        Nonnegative weights for x_points.
    w_y : np.ndarray of shape (m,), optional
        Nonnegative weights for y_points.

    Returns
    -------
    float
        Wasserstein-1 distance.
    """
    if ot is None:
        raise ImportError(
            "POT is required for wasserstein_1d_pot. "
            "Install with `pip install pot`."
        )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("wasserstein_1d_pot expects 1D input arrays")

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must be non-empty")

    if w_x is None:
        w_x = np.ones_like(x, dtype=float)
    else:
        w_x = np.asarray(w_x, dtype=float)
        if w_x.shape != x.shape or np.any(w_x < 0):
            raise ValueError("w_x must be nonnegative and match x")

    if w_y is None:
        w_y = np.ones_like(y, dtype=float)
    else:
        w_y = np.asarray(w_y, dtype=float)
        if w_y.shape != y.shape or np.any(w_y < 0):
            raise ValueError("w_y must be nonnegative and match y")

    # POT expects normalized weights
    w_x = w_x / w_x.sum()
    w_y = w_y / w_y.sum()

    return float(
        ot.wasserstein_1d(
            x,
            y,
            w_x,
            w_y,
            require_sort=False,
        )
    )
