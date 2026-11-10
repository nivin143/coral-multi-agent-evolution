"""Task interface: a problem definition x, candidate solutions y and an evaluator E(x, y) = (s, f).

Every task encodes a candidate as a vector in [0, 1]^dim and is *maximised*.
The evaluator returns a score `s` and structured feedback `f` (proposal section 3.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Feedback:
    """Evaluator feedback `f`: a diagnosis tag, the components it points at, and a message."""

    tag: str                      # short diagnosis, e.g. "latency_slo" - agents condition on this
    focus: list = field(default_factory=list)  # indices of the candidate that the diagnosis blames
    message: str = ""             # human-readable detail (what an LLM agent would read)
    valid: bool = True            # False for rejected / crashed / timed-out candidates


class Task:
    name: str = "task"
    domain: str = ""
    dim: int = 0
    groups: list = []             # index groups that belong together (e.g. one circle's x and y)
    description: str = ""
    fail_score: float = -1.0      # score assigned to invalid candidates

    def evaluate(self, x: np.ndarray) -> tuple[float, Feedback]:
        raise NotImplementedError

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(self.dim)

    def key(self, x: np.ndarray) -> tuple:
        """Canonical identity of a candidate, used to detect redundant re-evaluations."""
        return tuple(np.round(x, 3).tolist())
