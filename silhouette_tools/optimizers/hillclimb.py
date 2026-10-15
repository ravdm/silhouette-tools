from typing import Callable, Literal, Optional

import numpy as np
from silhouette_tools.optimizers.base import PartitionOptimizer
from collections import deque

class GreedyClimbOptimizer(PartitionOptimizer):
    """
    Deterministic greedy hill-climbing optimizer.

    The objective is assumed to be minimized: lower scores are better.
    """

    def __init__(
        self,
        labels: np.ndarray,
        objective: Callable[[np.ndarray], float],
        rng: Optional[np.random.Generator] = None,
        improvement_strategy: Literal["best", "first"] = "best",
        index_policy: Literal["queue", "random", "sequential"] = "queue",
        shuffle_indices: bool = True,
        shuffle_labels: bool = True,
        on_step: Callable[[int, float, np.ndarray], None] | None = None,
        verbose: bool = False,
    ):
        super().__init__(labels, objective, rng, verbose)
        if improvement_strategy not in ("best", "first"):
            raise ValueError(
                f"improvement_strategy must be 'best' or 'first', got {improvement_strategy!r}"
            )

        if index_policy not in ("queue", "random", "sequential"):
            raise ValueError(
                f"index_policy must be 'queue', 'random', or 'sequential', got {index_policy!r}"
            )
        self.improvement_strategy = improvement_strategy
        self.index_policy = index_policy
        self.shuffle_indices = shuffle_indices
        self.shuffle_labels = shuffle_labels
        self.on_step = on_step

        if index_policy == "queue":
            indices = np.arange(self.d)
            if shuffle_indices:
                indices = self.rng.permutation(indices)
            self._index_queue = deque(indices.tolist())

        elif index_policy == "sequential":
            self._indices = np.arange(self.d)

        elif index_policy == "random":
            self._indices = None

    def _step_impl(self) -> bool:
        unique_labels = np.unique(self.labels)

        # ---- RANDOM: no exhaustion guarantee ----
        if self.index_policy == "random":
            i = self.rng.integers(self.d)
            labels = unique_labels
            if self.shuffle_labels:
                labels = self.rng.permutation(unique_labels)

            current = self.labels[i]
            best_score = self.score
            best_labels = None

            for lbl in labels:
                if lbl == current:
                    continue

                new_labels = self.labels.copy()
                new_labels[i] = lbl
                new_score = self.objective(new_labels)

                if new_score < best_score:
                    if self.improvement_strategy == "first":
                        self.labels = new_labels
                        self.best_labels= self.labels.copy()
                        self.score = new_score
                        self.best_score = self.score
                        if self.on_step is not None:
                            self.on_step(self.n_steps, self.score, self.labels)
                        return True

                    best_score = new_score
                    best_labels = new_labels

            if best_labels is not None:
                self.labels = best_labels
                self.best_labels= self.labels.copy()
                self.score = best_score
                self.best_score = self.score
                if self.on_step is not None:
                    self.on_step(self.n_steps, self.score, self.labels)
                return True

            return False

        # ---- SEQUENTIAL / QUEUE: must exhaust everything ----
        if self.index_policy == "sequential":
            indices = self._indices

        elif self.index_policy == "queue":
            indices = list(self._index_queue)

        else:
            raise RuntimeError("Invalid index_policy")

        best_score = self.score
        best_move = None
        assert indices is not None
        
        for i in indices:
            labels = unique_labels
            if self.shuffle_labels:
                labels = self.rng.permutation(unique_labels)
            current = self.labels[i]

            for lbl in labels:
                if lbl == current:
                    continue

                new_labels = self.labels.copy()
                new_labels[i] = lbl
                new_score = self.objective(new_labels)

                if new_score < best_score:
                    if self.improvement_strategy == "first":
                        self.labels = new_labels
                        self.best_labels = self.labels.copy()
                        self.score = new_score
                        self.best_score = new_score

                        if self.index_policy == "queue":
                            self._index_queue.rotate(-1)
                        if self.on_step is not None:
                            self.on_step(self.n_steps, self.score, self.labels)
                        return True

                    best_score = new_score
                    best_move = new_labels

            if self.index_policy == "queue":
                self._index_queue.rotate(-1)

            if best_move is not None:
                self.labels = best_move
                self.score = best_score
                if self.on_step is not None:
                    self.on_step(self.n_steps, self.score, self.labels)
                return True

        return False
