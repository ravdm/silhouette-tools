"""Regression tests for graph partitioning and cached parallel objectives."""

from collections import Counter
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest

from silhouette_tools.metrics.wasserstein import wasserstein_1d
from silhouette_tools.silhouettes.objectives import pairwise_nb_cached, pairwise_nb_fast
from silhouette_tools.silhouettes.projections import graphs


@pytest.fixture(params=[pairwise_nb_cached, pairwise_nb_fast], ids=["cached", "fast"])
def implementation(request):
    return request.param


@pytest.mark.parametrize(
    "kind, d, n_batches",
    [
        ("chain", 21, 9),
        ("chain", 24, 9),
        ("regular", 21, 9),
        ("regular", 22, 9),
        ("regular", 24, 9),
        ("chain", 3, 9),
        ("chain", 1, 9),
    ],
)
def test_partition_batches_are_balanced(implementation, kind, d, n_batches):
    if kind == "regular":
        edges = graphs.random_regular_graph(d, 4, np.random.default_rng(0))
    else:
        edges = graphs.chain_graph(d)

    batches = implementation.partition_graph_edges(edges, n_batches)
    sizes = [len(batch) for batch in batches]

    assert len(batches) == n_batches
    assert Counter(edge for batch in batches for edge in batch) == Counter(edges)
    assert max(sizes) - min(sizes) <= 1
    assert sum(size > 0 for size in sizes) == min(len(edges), n_batches)


def test_partition_preserves_repeated_edges(implementation):
    edges = [(0, 1), (0, 1), (1, 2), (2, 3), (0, 1)]
    batches = implementation.partition_graph_edges(edges, 3)
    assert Counter(edge for batch in batches for edge in batch) == Counter(edges)


@pytest.mark.parametrize("n_batches", [0, -1])
def test_partition_requires_positive_batch_count(implementation, n_batches):
    with pytest.raises(ValueError, match="n_batches must be positive"):
        implementation.partition_graph_edges([(0, 1)], n_batches)


@pytest.mark.parametrize("n_workers", [1, 9])
def test_objective_rejects_empty_graph(implementation, n_workers):
    with pytest.raises(ValueError, match="graph must contain at least one edge"):
        implementation.CachedPairwiseSilhouetteNBMixObjective(
            np.zeros((6, 3)),
            2,
            metric=wasserstein_1d,
            graph=[],
            coefs=[(1.0, 1.0)],
            n_workers=n_workers,
        )


@pytest.mark.parametrize("n_workers", [1, 9])
def test_objective_rejects_empty_coefficients(implementation, n_workers):
    with pytest.raises(ValueError, match="coefs must contain at least one projection"):
        implementation.CachedPairwiseSilhouetteNBMixObjective(
            np.zeros((6, 3)),
            2,
            metric=wasserstein_1d,
            graph=[(0, 1)],
            coefs=[],
            n_workers=n_workers,
        )


def run_parallel_check(implementation, *arguments):
    """Fail within a fixed time if a worker regression would otherwise hang."""
    variant = implementation.__name__.rsplit("_", 1)[-1]
    check_env = os.environ.copy()
    check_env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    process = subprocess.Popen(
        [sys.executable, "-B", "-m", "tests._parallel_graph_check",
         "--variant", variant, *arguments],
        cwd=Path(__file__).resolve().parents[1],
        env=check_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(f"Parallel evaluation did not finish within 30 seconds.\n{stdout}\n{stderr}")
    assert process.returncode == 0, f"{stdout}\n{stderr}"


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="fork unavailable")
@pytest.mark.parametrize(
    "kind, d",
    [("chain", 21), ("chain", 24), ("regular", 21), ("regular", 22),
     ("regular", 24), ("chain", 3), ("chain", 2)],
)
def test_nine_workers_match_serial_after_label_changes(implementation, kind, d):
    run_parallel_check(implementation, "--kind", kind, "--dimension", str(d))


def test_spawn_workers_match_serial(implementation):
    run_parallel_check(
        implementation, "--kind", "regular", "--dimension", "24",
        "--context", "spawn",
    )


@pytest.mark.parametrize("failure", ["metric", "exit", "initialization"])
def test_worker_failures_are_reported_and_cleaned_up(implementation, failure):
    run_parallel_check(
        implementation, "--failure", failure, "--context", "spawn",
    )
