"""Candidate-generation toolbox shared by every condition.

The fixed evolutionary baseline and the autonomous agents draw from exactly the same set of
variation operators. What differs between conditions is who decides which operator to use,
on which parent, at what step size, and what is remembered afterwards.
"""
from __future__ import annotations

import numpy as np

OPS = ["gauss_small", "gauss_large", "group", "targeted", "resample_group", "crossover", "de_step"]

DESCRIPTIONS = {
    "gauss_small": "small Gaussian nudge to every component (local refinement)",
    "gauss_large": "large Gaussian move on every component (exploration)",
    "group": "move one related group of components (e.g. one circle, one knob pair)",
    "targeted": "move only the components the evaluator feedback blamed",
    "resample_group": "re-draw one group of components uniformly at random",
    "crossover": "copy groups of components from another good attempt in memory",
    "de_step": "step along the difference between two other good attempts in memory",
}


def _random_group(rng, task):
    return task.groups[int(rng.integers(len(task.groups)))]


def apply(op: str, rng: np.random.Generator, x: np.ndarray, task, scale: float = 1.0,
          focus=None, pool=None) -> np.ndarray:
    """Return a new candidate produced by `op` from parent `x`.

    `focus` are the indices named by the parent's evaluator feedback; `pool` is a list of other
    candidate vectors (the population for the baseline, retrieved memory for agents).
    """
    y = x.copy()
    pool = pool or []
    if op == "gauss_small":
        y += rng.normal(0.0, 0.02 * scale, x.size)
    elif op == "gauss_large":
        y += rng.normal(0.0, 0.12 * scale, x.size)
    elif op == "group":
        g = _random_group(rng, task)
        y[g] += rng.normal(0.0, 0.10 * scale, len(g))
    elif op == "targeted":
        g = list(focus) if focus else _random_group(rng, task)
        y[g] += rng.normal(0.0, 0.08 * scale, len(g))
    elif op == "resample_group":
        g = _random_group(rng, task)
        y[g] = rng.random(len(g))
    elif op == "crossover" and pool:
        other = pool[int(rng.integers(len(pool)))]
        for g in task.groups:
            if rng.random() < 0.3:
                y[g] = other[g]
        y += rng.normal(0.0, 0.005 * scale, x.size)
    elif op == "de_step" and len(pool) >= 2:
        i, j = rng.choice(len(pool), size=2, replace=False)
        y += 0.6 * scale * (pool[int(i)] - pool[int(j)])
    else:  # operator needs memory that is not available yet
        y += rng.normal(0.0, 0.02 * scale, x.size)
    return np.clip(y, 0.0, 1.0)
