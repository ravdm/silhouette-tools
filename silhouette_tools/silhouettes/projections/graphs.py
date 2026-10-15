from typing import List, Tuple, Optional
import numpy as np


Edge = Tuple[int, int]


def chain_graph(d: int) -> List[Edge]:
    """
    Construct a chain (path) graph on d vertices.

    Edges:
        (0,1), (1,2), ..., (d-2, d-1)
    """
    if d < 2:
        return []
    return [(i, i + 1) for i in range(d - 1)]


def star_graph(d: int, root: int = 0) -> List[Edge]:
    """
    Construct a star graph on d vertices.

    Edges:
        (root, i) for all i != root
    """
    if d < 2:
        return []

    if not (0 <= root < d):
        raise ValueError("root must be in {0, ..., d-1}")

    return [(root, i) if root < i else (i, root)
            for i in range(d) if i != root]


def complete_graph(d: int) -> List[Edge]:
    """
    Construct a complete graph on d vertices.

    Edges:
        (i, j) for all i < j
    """
    edges: List[Edge] = []
    for i in range(d):
        for j in range(i + 1, d):
            edges.append((i, j))
    return edges

def random_regular_graph(
    d: int,
    r: int,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 1000,
) -> List[Edge]:
    """
    Construct a random r-regular simple graph on d vertices.

    This uses repeated random perfect matchings with rejection.
    Uniformity over all r-regular graphs is not guaranteed, but
    the result has good expansion and degree balance properties,
    which is sufficient for silhouette constructions.

    Parameters
    ----------
    d : int
        Number of vertices.

    r : int
        Degree of each vertex. Must satisfy r < d and d * r even.

    rng : np.random.Generator
        Random number generator.

    max_tries : int, default=1000
        Maximum number of attempts before giving up.

    Returns
    -------
    edges : list of (i, j)
        Edge list with i < j.

    Raises
    ------
    ValueError
        If construction fails or parameters are invalid.
    """
    if r < 0:
        raise ValueError("r must be non-negative")
    if r >= d:
        raise ValueError("r must be strictly less than d")
    if (d * r) % 2 != 0:
        raise ValueError("d * r must be even")

    if r == 0:
        return []

    if rng is None:
            rng = np.random.default_rng()

    for _ in range(max_tries):
        # Degree counters
        degree = np.zeros(d, dtype=int)
        edges: set[Edge] = set()

        success = True

    for _ in range(max_tries):

        # Configuration model: create r copies of each vertex
        stubs = np.repeat(np.arange(d), r)
        rng.shuffle(stubs)

        edges = set()
        success = True

        for i in range(0, len(stubs), 2):
            u = int(stubs[i])
            v = int(stubs[i + 1])

            if u == v:
                success = False
                break

            edge = (u, v) if u < v else (v, u)

            if edge in edges:
                success = False
                break

            edges.add(edge)

        if success:
            return list(edges)

    raise ValueError("Failed to construct r-regular graph")
def random_tree(
        d: int,
        rng: Optional[np.random.Generator] = None,
        ) -> List[Edge]:
    """
    Construct a random tree on d vertices using a simple
    random-parent construction.

    Guarantees a connected, acyclic graph.
    """
    if d < 2:
        return []

    if rng is None:
            rng = np.random.default_rng()

    edges: List[Edge] = []
    for i in range(1, d):
        parent = rng.integers(0, i)
        edge = (parent, i) if parent < i else (i, parent)
        edges.append(edge)

    return edges
