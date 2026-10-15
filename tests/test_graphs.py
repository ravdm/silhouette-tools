import numpy as np

from silhouette_tools.silhouettes.projections.graphs import (
    chain_graph,
    star_graph,
    complete_graph,
    random_tree,
    random_regular_graph,
)


def is_tree(edges, d):
    """
    Check whether an undirected graph on d vertices given by an edge list
    is a tree (connected and acyclic).
    """
    if len(edges) != d - 1:
        return False

    parent = list(range(d))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    for i, j in edges:
        if not union(i, j):
            return False

    roots = {find(i) for i in range(d)}
    return len(roots) == 1

def is_connected(edges, d):
    """
    Check whether an undirected graph on d vertices given by an edge list
    is connected.
    """
    if d == 0:
        return True
    if not edges and d > 1:
        return False

    adj = [[] for _ in range(d)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    visited = set()
    stack = [0]

    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        stack.extend(adj[v])

    return len(visited) == d


def test_is_connected_empty_graph():
    # No edges, more than one vertex → disconnected
    d = 5
    edges = []

    assert not is_connected(edges, d)


def test_is_connected_two_components():
    # Two disconnected components: {0,1,2} and {3,4}
    d = 5
    edges = [
        (0, 1),
        (1, 2),
        (3, 4),
    ]

    assert not is_connected(edges, d)

def test_chain_graph():
    d = 10
    edges = chain_graph(d)

    assert len(edges) == d - 1
    assert is_tree(edges, d)
    assert is_connected(edges, d)


def test_star_graph():
    d = 10
    edges = star_graph(d, root=0)

    assert len(edges) == d - 1
    assert is_tree(edges, d)

    degrees = np.zeros(d, dtype=int)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1

    assert degrees[0] == d - 1
    assert np.all(degrees[1:] == 1)
    assert is_connected(edges, d)

def test_complete_graph():
    d = 6
    edges = complete_graph(d)

    expected = d * (d - 1) // 2
    assert len(edges) == expected

    seen = set()
    for i, j in edges:
        assert i != j
        assert 0 <= i < j < d
        assert (i, j) not in seen
        seen.add((i, j))
    assert is_connected(edges, d)

def test_random_tree():
    rng = np.random.default_rng(0)
    d = 10

    edges = random_tree(d, rng)

    assert is_tree(edges, d)
    assert is_connected(edges, d)

def test_random_regular_graph_degree_and_coverage():
    rng = np.random.default_rng(1)
    d = 100
    r = 3

    edges = random_regular_graph(d, r, rng)

    degrees = np.zeros(d, dtype=int)
    for i, j in edges:
        assert i != j
        degrees[i] += 1
        degrees[j] += 1

    # Regularity
    assert np.all(degrees == r)

    # Coverage: all vertices appear
    assert np.all(degrees > 0)

    assert not(is_tree(edges, d))

    # Connected: just see if is or not (should be connected)
    connected = is_connected(edges, d)
    print("random regular graph connected:", connected)

def test_random_regular_graph_r_equals_2_behavior():
    rng = np.random.default_rng(2)
    d = 100
    r = 2

    edges = random_regular_graph(d, r, rng)

    degrees = np.zeros(d, dtype=int)
    for i, j in edges:
        assert i != j
        degrees[i] += 1
        degrees[j] += 1

    # Degree regularity still holds
    assert np.all(degrees == r)

    # r=2 graphs are unions of cycles, not trees
    assert not is_tree(edges, d)

    # Connectivity is not guaranteed; just observe it
    connected = is_connected(edges, d)
    print("r=2 regular graph connected:", connected)
