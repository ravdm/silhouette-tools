import numpy as np
import scipy as sp
from typing import Literal, List, Tuple
from scipy.sparse import coo_matrix, spmatrix

def silhouette_coefficients(
    *,
    a: float,
    b1: float,
    b2: float,
    k: int,
    normalize: bool = True,
) -> List[Tuple[float, float]]:
    """
    Generate normalized 2D silhouette coefficient vectors as defined in the paper.

    This implements the families:
        (a^m b2, b1)        for m = 1, ..., k
        (b1, a^m b2)     for m = 0, ..., k-1

    Each returned vector is L2-normalized.

    Parameters
    ----------
    a : float
        Base parameter controlling exponential scaling.

    b1 : float
        First coupling coefficient.

    b2 : float
        Second coupling coefficient.

    k : int
        Number of silhouette scales.

    normalize: bool = True
        whether to normalize each 2d vector to unit norm. For large m, can make sense to not normalize so that vector isn't too close to [1,0]

    Returns
    -------
    coeffs : list of (float, float)
        List of normalized 2D coefficient vectors (c_i, c_j).
    """
    if k <= 0:
        raise ValueError("k must be positive")

    coeffs = []

    # First family: (a^m, b1), m = 1,...,k
    for m in range(1, k + 1):
        v = np.array([a**m * b1, b2], dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Zero vector encountered in silhouette coefficients")
        if normalize:
            coeffs.append((v[0] / norm, v[1] / norm))
        else:
            coeffs.append((v[0] , v[1] ))

    # Second family: (b1, a^m b2), m = 0,...,k-1
    for m in range(0, k):
        v = np.array([b1, (a**m) * b2], dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Zero vector encountered in silhouette coefficients")
        if normalize: 
            coeffs.append((v[0] / norm, v[1] / norm))
        else:
            coeffs.append((v[0] , v[1] ))

    return coeffs


def pairwise_projection_matrix(
    edges,
    d,
    vectors_2d,
    *,
    include_axes: bool = True,
    format: Literal["csc", "coo"] = "csc",
)-> spmatrix:
    """
    Build a sparse projection matrix from pairwise edges and 2D vectors.
    """

    rows = []
    cols = []
    data = []

    n_edges = len(edges)
    n_vecs = len(vectors_2d)

    col = 0
    for i, j in edges:
        for u, v in vectors_2d:
            rows.append(i)
            cols.append(col)
            data.append(u)

            rows.append(j)
            cols.append(col)
            data.append(v)

            col += 1

    # Add axis projections (identity)
    if include_axes:
        for i in range(d):
            rows.append(i)
            cols.append(col)
            data.append(1.0)
            col += 1

    P = coo_matrix(
        (data, (rows, cols)),
        shape=(d, col),
    )

    if format == "csc":
        return P.tocsc()
    else:
        return P
