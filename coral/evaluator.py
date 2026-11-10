"""External evaluator E with a hard call budget (proposal section 3.2.4).

The same evaluator class, task instance data and budget are used for every experimental
condition. It validates candidates, converts crashes into failed evaluations with feedback,
applies a time limit, and records every call for later analysis (section 3.7).
"""
from __future__ import annotations

import time

import numpy as np

from .tasks.base import Feedback, Task


class BudgetExhausted(Exception):
    """Raised when a caller asks for an evaluation after the budget is spent."""


class Evaluator:
    def __init__(self, task: Task, budget: int, time_limit_s: float = 5.0):
        self.task = task
        self.budget = budget
        self.time_limit_s = time_limit_s
        self.calls = 0
        self.best = -np.inf
        self.best_x = None
        self.curve: list[float] = []       # best-so-far after each call
        self.improvements = 0              # number of calls that raised the best-so-far
        self.invalid = 0
        self.rows: list[dict] = []
        self._t0 = time.perf_counter()

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    def __call__(self, x, meta: dict | None = None):
        if self.calls >= self.budget:
            raise BudgetExhausted
        self.calls += 1
        start = time.perf_counter()
        x = np.asarray(x, dtype=float)
        if x.shape != (self.task.dim,) or not np.all(np.isfinite(x)) or x.min() < 0 or x.max() > 1:
            score, fb = self.task.fail_score, Feedback(
                "invalid", [], "candidate rejected: wrong shape, non-finite or outside [0, 1]", False)
        else:
            try:
                score, fb = self.task.evaluate(x)
            except Exception as exc:  # a crashing candidate is a failed attempt, not a crashed run
                score, fb = self.task.fail_score, Feedback("error", [], f"evaluation raised {exc!r}", False)
        elapsed = time.perf_counter() - start
        if elapsed > self.time_limit_s:
            # Checked after the fact: candidates here are parameter vectors, not arbitrary code,
            # so there is nothing to pre-empt. See README, "Limitations".
            score, fb = self.task.fail_score, Feedback(
                "timeout", [], f"evaluation took {elapsed:.2f}s (limit {self.time_limit_s}s)", False)
        if not fb.valid:
            self.invalid += 1
        improved = score > self.best
        if improved:
            self.best, self.best_x = score, x.copy()
            if self.calls > 1:
                self.improvements += 1
        self.curve.append(self.best)
        row = {
            "eval_id": self.calls, "score": score, "best_so_far": self.best, "tag": fb.tag,
            "valid": fb.valid, "new_best": improved,
            "t_ms": round((time.perf_counter() - self._t0) * 1000, 2),
        }
        if meta:
            row.update(meta)
        self.rows.append(row)
        return score, fb, self.calls
