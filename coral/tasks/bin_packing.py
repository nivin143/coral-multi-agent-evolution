"""Algorithm optimisation task: evolve a priority heuristic for online bin packing.

A candidate is the weight vector of a priority function. Each arriving item goes into the
feasible bin (or a fresh bin) with the highest priority. The score is how close the heuristic
gets to the volume lower bound on a fixed set of instances.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Feedback, Task

FEATURES = [
    "residual",            # space left in the bin after placing the item
    "residual^2",
    "sqrt(residual)",
    "near-perfect fit",    # residual < 0.03
    "bin capacity left",
    "item size * residual",
    "bin is empty",
    "exp(-12 residual)",
]


class OnlineBinPacking(Task):
    name = "bin_packing"
    domain = "algorithm optimisation"
    fail_score = -1.0

    def __init__(self, n_instances: int = 8, n_items: int = 75, instance_seed: int = 20261110):
        self.dim = len(FEATURES)
        self.groups = [[0, 1, 2], [3, 7], [4, 6], [5]]
        rng = np.random.default_rng(instance_seed)  # instances are fixed, independent of run seed
        self.instances = [
            np.clip(rng.weibull(3.0, n_items) * 0.45, 0.02, 1.0) for _ in range(n_instances)
        ]
        self.lower_bounds = [math.ceil(float(s.sum()) - 1e-9) for s in self.instances]
        self.description = (
            "Design a priority function for online bin packing (capacity 1). Each item is placed "
            "in the feasible bin with the highest priority = w . features. A candidate is the "
            f"weight vector w over the features {FEATURES}. Fewer bins is better."
        )

    def pack(self, w: np.ndarray, sizes: np.ndarray):
        caps = np.ones(len(sizes) + 1)
        n_open = 0
        for s in sizes:
            cand = caps[: n_open + 1]          # open bins plus one fresh bin
            res = cand - s
            feas = res >= -1e-12
            res = np.maximum(res, 0.0)
            pr = (
                w[0] * res + w[1] * res * res + w[2] * np.sqrt(res) + w[3] * (res < 0.03)
                + w[4] * cand + w[5] * s * res + w[6] * (cand >= 1.0) + w[7] * np.exp(-12.0 * res)
            )
            pr = np.where(feas, pr, -np.inf)
            j = int(np.argmax(pr))
            caps[j] -= s
            if j == n_open:
                n_open += 1
        return n_open, 1.0 - caps[:n_open]

    def evaluate(self, x: np.ndarray):
        w = 2.0 * x - 1.0
        ratios, loose, used = [], [], []
        for sizes, lb in zip(self.instances, self.lower_bounds):
            n_bins, fill = self.pack(w, sizes)
            ratios.append(lb / n_bins)
            loose.append(float((fill < 0.8).mean()))
            used.append(n_bins)
        score = float(np.mean(ratios))
        loose_frac = float(np.mean(loose))
        if score < 0.87:
            tag, focus = "opens_too_many", [4, 6]
        elif loose_frac > 0.3:
            tag, focus = "loose_bins", [0, 1, 2, 7]
        else:
            tag, focus = "tight", [3, 5, 7]
        msg = (
            f"bins used {used} vs lower bounds {self.lower_bounds}; "
            f"{loose_frac:.0%} of bins are less than 80% full"
        )
        return score, Feedback(tag, focus, msg)
