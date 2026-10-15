
import numpy as np
import pytest

from silhouette_tools.silhouettes.objectives.pairwise_nb_cached import (
    _build_component)
import silhouette_tools.silhouettes.projections.graphs as graphs
from silhouette_tools.silhouettes.objectives.pairwise_nb_cached import partition_graph_edges
###################################################
########## TEST _build_component         ##########
###################################################

# ---------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------


def test_build_component_simple_exact():
    x = np.array([1.0, 2.0])
    y = np.array([10.0, 20.0])
    a, b = 1.0, 1.0

    values, weights = _build_component(x, y, a, b)

    # pairwise sums: [11, 21, 12, 22]
    expected_values = np.array([11.0, 12.0, 21.0, 22.0])
    expected_weights = np.array([0.5, 0.5, 0.5, 0.5])  # m = 2

    assert np.allclose(values, expected_values)
    assert np.allclose(weights, expected_weights)
    assert np.isclose(weights.sum(), 2.0)


def test_build_component_consolidation():
    x = np.array([1.0, 1.0])
    y = np.array([2.0, 2.0])
    a, b = 1.0, 1.0

    values, weights = _build_component(x, y, a, b)

    # all sums are 3 → consolidated
    assert np.allclose(values, np.array([3.0]))
    assert np.allclose(weights, np.array([2.0]))  # m = 2


# ---------------------------------------------------------------------
# Weight structure
# ---------------------------------------------------------------------


def test_build_component_total_mass_equals_m():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5)
    y = rng.normal(size=5)

    values, weights = _build_component(x, y, 1.3, -0.7)

    assert np.isclose(weights.sum(), 5.0)


def test_build_component_nonnegative_weights():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    _, weights = _build_component(x, y, 1.0, 1.0)
    assert np.all(weights >= 0.0)


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


def test_build_component_empty():
    x = np.array([])
    y = np.array([])

    values, weights = _build_component(x, y, 1.0, 1.0)

    assert values.size == 0
    assert weights.size == 0


def test_build_component_singleton():
    x = np.array([2.0])
    y = np.array([3.0])

    values, weights = _build_component(x, y, 2.0, -1.0)

    assert np.allclose(values, np.array([1.0]))  # 2*2 - 1*3
    assert np.allclose(weights, np.array([1.0]))


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------


def test_build_component_length_mismatch_error():
    x = np.array([1.0, 2.0])
    y = np.array([3.0])

    with pytest.raises(ValueError):
        _build_component(x, y, 1.0, 1.0)


def test_build_component_non_1d_error():
    x = np.ones((2, 2))
    y = np.ones((2, 2))

    with pytest.raises(ValueError):
        _build_component(x, y, 1.0, 1.0)


###################################################
########## TEST partition_edges_balanced_min_vertices
###################################################

def flatten(batches):
    return [e for batch in batches for e in batch]

def edge_set(edges):
    return {tuple(sorted(e)) for e in edges}


def test_partition_empty_graph():
    batches = partition_graph_edges([], n_batches=3)
    assert len(batches) == 3
    assert all(len(b) == 0 for b in batches)


def test_partition_chain_graph_balanced():
    edges = graphs.chain_graph(10)   # 9 edges
    batches = partition_graph_edges(edges, n_batches=3)

    # All edges preserved
    assert edge_set(flatten(batches)) == edge_set(edges)

    sizes = [len(b) for b in batches]
    assert max(sizes) - min(sizes) <= 1

def test_partition_star_graph():
    edges = graphs.star_graph(10, root=0)  # 9 edges
    batches = partition_graph_edges(edges, n_batches=3)

    assert edge_set(flatten(batches)) == edge_set(edges)

    sizes = [len(b) for b in batches]
    assert sum(sizes) == len(edges)
    assert max(sizes) - min(sizes) <= 1

def test_partition_complete_graph():
    edges = graphs.complete_graph(6)  # 15 edges
    batches = partition_graph_edges(edges, n_batches=4)

    assert edge_set(flatten(batches)) == edge_set(edges)

    sizes = [len(b) for b in batches]
    assert max(sizes) - min(sizes) <= 1


def test_partition_random_tree():
    rng = np.random.default_rng(0)
    edges = graphs.random_tree(50, rng=rng)

    batches = partition_graph_edges(edges, n_batches=5)

    assert edge_set(flatten(batches)) == edge_set(edges)

    sizes = [len(b) for b in batches]
    assert max(sizes) - min(sizes) <= 1
