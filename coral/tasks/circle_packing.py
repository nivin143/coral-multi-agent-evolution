"""Mathematical discovery task: pack n circles in the unit square, maximise the sum of radii.

A candidate is the list of circle centres. The evaluator derives the radii itself with a
deterministic, provably feasible procedure, so every candidate is a valid packing and the
score cannot be gamed by claiming radii that overlap (proposal section 6.2).
"""
from __future__ import annotations

import numpy as np

from .base import Feedback, Task


class CirclePacking(Task):
    name = "circle_packing"
    domain = "mathematical discovery"
    fail_score = -1.0

    def __init__(self, n: int = 26, sweeps: int = 8):
        self.n = n
        self.sweeps = sweeps
        self.dim = 2 * n
        self.groups = [[2 * i, 2 * i + 1] for i in range(n)]
        self.description = (
            f"Place {n} circles inside the unit square so that no two overlap and the sum of "
            "their radii is as large as possible. A candidate gives the centre (x, y) of each "
            "circle; the evaluator assigns the radii."
        )

    def radii(self, centres: np.ndarray) -> np.ndarray:
        """Largest radii this greedy procedure can certify for the given centres."""
        d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        border = np.minimum.reduce([centres[:, 0], centres[:, 1], 1 - centres[:, 0], 1 - centres[:, 1]])
        # Start feasible: r_i <= d_ij / 2 for all j, so r_i + r_j <= d_ij.
        r = np.maximum(np.minimum(border, d.min(axis=1) / 2), 0.0)
        # Grow each radius into the slack its neighbours leave. Each update keeps feasibility.
        for _ in range(self.sweeps):
            for i in range(self.n):
                room = min(border[i], float((d[i] - r).min()))
                if room > r[i]:
                    r[i] = room
        return r

    def verify(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        """Independent check that the packing implied by x is valid."""
        c = np.asarray(x, float).reshape(self.n, 2)
        r = self.radii(c)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        inside = np.all(c - r[:, None] >= -tol) and np.all(c + r[:, None] <= 1 + tol)
        disjoint = np.all(r[:, None] + r[None, :] <= d + tol)
        return bool(inside and disjoint and np.all(r >= 0))

    def evaluate(self, x: np.ndarray):
        c = x.reshape(self.n, 2)
        r = self.radii(c)
        score = float(r.sum())
        order = np.argsort(r)
        small = [int(order[0]), int(order[1])]
        focus = [2 * small[0], 2 * small[0] + 1, 2 * small[1], 2 * small[1] + 1]
        mean = float(r.mean())
        if r[order[0]] < 0.25 * mean:
            tag = "tiny_circle"
        elif float(r.std()) > 0.3 * mean:
            tag = "uneven"
        else:
            tag = "balanced"
        msg = (
            f"sum of radii {score:.4f}; smallest circles are #{small[0]} (r={r[small[0]]:.4f}) "
            f"and #{small[1]} (r={r[small[1]]:.4f}); mean radius {mean:.4f}"
        )
        return score, Feedback(tag, focus, msg)
