"""Paired statistics for runs that share a task and a seed (proposal section 3.7)."""
from __future__ import annotations

from itertools import product

import numpy as np


def paired_summary(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Compare paired samples: returns statistics of d = a - b.

    mean difference, bootstrap 95% CI of the mean, Cohen's d_z (mean / sd of differences),
    wins / ties / losses, and a two-sided sign-flip permutation p-value (exact up to 16 pairs).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    sd = d.std(ddof=1) if n > 1 else 0.0
    observed = abs(d.mean())
    if n <= 16:
        signs = np.array(list(product([1.0, -1.0], repeat=n)))
    else:
        signs = rng.choice([1.0, -1.0], size=(20000, n))
    p = float((np.abs((signs * d).mean(axis=1)) >= observed - 1e-15).mean())
    return {
        "n": n, "mean_diff": float(d.mean()),
        "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
        "cohen_dz": float(d.mean() / sd) if sd > 0 else 0.0,
        "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
        "p_value": p,
    }


def holm(p_values) -> list[float]:
    """Holm step-down adjustment for a family of p-values."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def evals_to_target(curve, target: float) -> tuple[int, bool]:
    """First evaluation at which the best-so-far reaches `target`.

    Runs that never get there are censored at the budget and reported as not reached.
    """
    hit = np.nonzero(np.asarray(curve) >= target)[0]
    return (int(hit[0]) + 1, True) if len(hit) else (len(curve), False)
