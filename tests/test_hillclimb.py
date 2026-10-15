import numpy as np

from silhouette_tools.optimizers.hillclimb import GreedyClimbOptimizer as optimizer

#
def objective_nonzeros(labels: np.ndarray) -> float:
    """
    Objective: minimize the number of nonzero entries.
    Global optimum = 0 (all labels are zero).
    """
    return float(np.count_nonzero(labels))


def objective_sum(labels: np.ndarray) -> float:
    """
    Objective: minimize the sum of labels.
    Global optimum = 0 (all labels are zero).
    """
    return float(np.sum(labels))


def test_basic_hillclimb_minimizes_nonzeros():
    rng = np.random.default_rng(0)

    n = 100
    k = 5

    labels = rng.integers(0, k, size=n)

    rhc = optimizer(
        labels=labels,
        objective=objective_nonzeros,
        rng=rng,
    )

    initial_score = rhc.score

    improved = True
    steps = 0
    while improved:
        improved = rhc.step()
        steps += 1
        if steps > 10_000:
            break

    final_score = rhc.score
    final_labels = rhc.labels

    print("\nGreedyClimbOptimizer (nonzeros)")
    print("initial score:", initial_score)
    print("final score:", final_score)
    print("steps:", steps)

    assert final_score == 0.0
    assert np.all(final_labels == 0)


def test_greedy_vs_random_on_sum_objective():
    """
    Compare greedy vs random hill climbing on a simple convex objective.
    """
    rng = np.random.default_rng(1)

    n = 50
    k = 5

    labels = rng.integers(0, k, size=n)

    greedy_best = optimizer(
        labels=labels,
        objective=objective_sum,
        improvement_strategy= "best",
    )

    greedy_first = optimizer(
        labels=labels,
        objective=objective_sum,
        improvement_strategy= "first",
    )

    # Run best
    best_steps = 0
    while greedy_best.step():
        best_steps += 1
        if best_steps > 10_000:
            break

    # Run first
    first_steps = 0
    while greedy_first.step():
        first_steps += 1
        if first_steps > 10_000:
            break

    print("\nHillClimbOptimizer (sum)")
    print("score:", greedy_best.score)
    print("steps:", best_steps)


    assert greedy_best.score == 0.0
    assert np.all(greedy_best.labels == 0)

    assert greedy_first.score == 0.0
    assert np.all(greedy_first.labels == 0)
    # Greedy should be no slower than random
    assert best_steps < first_steps

def test_queue_negative_label_sum_deterministic_sequence():
    """
    Queue policy with deterministic label and index order,
    objective = -sum(labels), improvement_strategy='first'

    When starting from [-2, -1, 0], the first-improvement queue
    should update indices in predictable order and produce a list
    of changed indices.
    """

    labels = np.array([-3, -2, -1, 0], dtype=int)

    def objective(labels: np.ndarray) -> float:
        return float(-np.sum(labels))

    opt = optimizer(
        labels=labels,
        objective=objective,
        index_policy="queue",
        improvement_strategy="first",
        shuffle_indices=False,
        shuffle_labels=False,
    )

    changed_order = []

    while True:
        prev_labels = opt.labels.copy()
        improved = opt.step()
        if not improved:
            break

        diff = opt.labels - prev_labels
        idx = np.where(diff > 0)[0]  # label increased
        assert len(idx) == 1
        changed_order.append(int(idx[0]))  # store as integer

    assert changed_order == [0, 1,2,0,1,0]

def test_random_hillclimb_takes_more_steps_than_nonzero_count():
    """
    With index_policy='random', shuffle_indices=True, shuffle_labels=True,
    the number of *attempts until all labels are zero* should exceed
    the initial number of nonzero labels.
    """

    rng = np.random.default_rng(0)

    n = 35
    k = 6

    # random labels 0..5
    labels = rng.integers(0, k, size=n)

    def objective(labels: np.ndarray) -> float:
        return float(np.count_nonzero(labels))

    initial_nonzero = np.count_nonzero(labels)

    random_opt = optimizer(
        labels=labels,
        objective=objective,
        index_policy="random",
        improvement_strategy="first",
        shuffle_indices=True,
        shuffle_labels=True,
        rng=rng,
    )

    steps = 0
    while np.count_nonzero(random_opt.labels) > 0:
        improved = random_opt.step()
        steps += 1

        # safety cap to catch pathological infinite loops
        assert steps < 10000

    # The number of steps taken should be *greater* than the number of non-zero entries
    assert steps > initial_nonzero
