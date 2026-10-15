import numpy as np
import pytest
from scipy.sparse import csc_matrix, coo_matrix
from silhouette_tools.silhouettes.projections.pairwise import (
    silhouette_coefficients,
    pairwise_projection_matrix,
)

"""
Test coefficient maker
"""
def test_number_of_coefficients():
    k = 5
    coeffs = silhouette_coefficients(a=2.0, b1=1.0, b2=3.0, k=k)
    # Two families, each of size k
    assert len(coeffs) == 2 * k


def test_coefficients_are_normalized():
    coeffs = silhouette_coefficients(a=1.5, b1=2.0, b2=3.0, k=4)
    for c in coeffs:
        norm = np.linalg.norm(c)
        assert np.isclose(norm, 1.0)


def test_coefficients_shape_and_type():
    coeffs = silhouette_coefficients(a=2.0, b1=1.0, b2=1.0, k=3)
    for c in coeffs:
        assert isinstance(c, tuple)
        assert len(c) == 2
        assert all(isinstance(x, float) for x in c)


def test_known_simple_case():
    # a = 1 makes all powers trivial
    a = 1.0
    b1 = 1.0
    b2 = 1.0
    k = 2

    coeffs = silhouette_coefficients(a=a, b1=b1, b2=b2, k=k)

    # All vectors should be (1,1)/sqrt(2)
    expected = (1.0 / np.sqrt(2), 1.0 / np.sqrt(2))
    for c in coeffs:
        assert np.allclose(c, expected)


def test_zero_vector_raises():
    # This choice forces a zero vector
    with pytest.raises(ValueError):
        silhouette_coefficients(a=1.0, b1=0.0, b2=0.0, k=3)


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        silhouette_coefficients(a=2.0, b1=1.0, b2=1.0, k=0)

    with pytest.raises(ValueError):
        silhouette_coefficients(a=2.0, b1=1.0, b2=1.0, k=-1)

"""
test pairwise projection matrix (graph + coefficients)
"""
def test_basic_shape_no_axes():
    edges = [(0, 1)]
    d = 3
    vectors = [(1.0, 0.0), (0.0, 1.0)]

    P = pairwise_projection_matrix(
        edges, d, vectors, include_axes=False, format="coo"
    )

    # One edge × two vectors → two projection columns
    assert P.shape == (d, 2)
    assert isinstance(P, coo_matrix)


def test_basic_shape_with_axes():
    edges = [(0, 1)]
    d = 3
    vectors = [(1.0, 0.0), (0.0, 1.0)]

    P = pairwise_projection_matrix(
        edges, d, vectors, include_axes=True, format="coo"
    )

    # Pairwise columns + d axis columns
    assert P.shape == (d, 2 + d)


def test_axis_columns_are_identity():
    edges = [(0, 1)]
    d = 4
    vectors = [(1.0, 2.0)]

    P = pairwise_projection_matrix(
        edges, d, vectors, include_axes=True, format="coo"
    ).toarray()

    # Axis columns are the last d columns
    axis_block = P[:, -d:]
    assert np.allclose(axis_block, np.eye(d))
    assert P.shape == (d, 1 + d)


def test_single_edge_single_vector_values():
    edges = [(0, 2)]
    d = 4
    vectors = [(3.0, 5.0)]

    P = pairwise_projection_matrix(
        edges, d, vectors, include_axes=False, format="coo"
    ).toarray()

    # Only one projection column
    expected = np.zeros((d, 1))
    expected[0, 0] = 3.0
    expected[2, 0] = 5.0

    assert np.allclose(P, expected)


def test_multiple_edges_multiple_vectors_column_count():
    edges = [(0, 1), (1, 2), (2, 3)]
    d = 4
    vectors = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]

    P = pairwise_projection_matrix(
        edges, d, vectors, include_axes=False, format="coo"
    )

    assert P.shape[1] == len(edges) * len(vectors)


def test_csc_format_requested():
    edges = [(0, 1)]
    d = 2
    vectors = [(1.0, 1.0)]

    P = pairwise_projection_matrix(
        edges, d, vectors, include_axes=False, format="csc"
    )

    assert isinstance(P, csc_matrix)


def test_projection_action_matches_manual():
    edges = [(0, 1)]
    d = 2
    vectors = [(2.0, -1.0)]

    P = pairwise_projection_matrix(
        edges, d, vectors, include_axes=False, format="coo"
    ).toarray()

    x = np.array([3.0, 5.0])
    y = x @ P

    # projection = 2*x[0] - 1*x[1]
    assert np.allclose(y[0], 2.0 * 3.0 - 1.0 * 5.0)
