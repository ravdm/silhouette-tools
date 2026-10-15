import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


class PartitionOptimizer(ABC):
    def __init__(
        self,
        labels: np.ndarray,
        objective: Callable[[np.ndarray], float],
        rng: Optional[np.random.Generator] = None,
        verbose: bool = False,
    ):
        self.labels = np.asarray(labels).copy()
        self.d = len(self.labels)
        self.objective = objective
        self.rng = rng or np.random.default_rng()
        self.verbose = verbose

        self.score = objective(self.labels)
        self.best_labels = self.labels.copy()
        self.best_score = self.score
        self.n_steps = 0

    def step(self) -> bool:
        """Public step with bookkeeping."""
        self.n_steps += 1
        t0 = time.perf_counter()
        old_score = self.score

        improved = self._step_impl()

        dt = time.perf_counter() - t0
        if self.verbose and improved:
            print(
                f"[{self.__class__.__name__}] Improvement accepted: "
                f"{old_score:.6g} → {self.score:.6g} "
                f"(step time: {dt:.3f}s)"
            )
        return improved

    @abstractmethod
    def _step_impl(self) -> bool:
        """Algorithm specific logic"""

    def run(
        self,
        max_iters: int,
        max_run_time: float | None = None,
        max_step_time: float | None = None,
    ) -> None:
        """
        Run hill climbing until convergence or max_iters reached.
        """
        t_start = time.perf_counter()
        t_step_start: float | None = None



        try:
            for _ in range(max_iters):
                if max_run_time is not None:
                    if time.perf_counter() - t_start > max_run_time:
                        break
                if max_step_time is not None:
                    t_step_start = time.perf_counter()
                improved = self.step()
                if max_step_time is not None:
                    assert t_step_start is not None
                    if time.perf_counter() - t_step_start > max_step_time:
                        break
                if not improved:
                    break
        except KeyboardInterrupt:
            print("Optimization Interrupted via Keyboard")
