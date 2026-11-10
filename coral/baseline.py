"""Baseline A: fixed evolutionary search.

A steady-state evolutionary algorithm whose every decision is a predefined rule: tournament
selection picks the parent, the operator is drawn uniformly from the shared toolbox at a fixed
step size, and the child replaces the worst member of the population if it is better.
"""
from __future__ import annotations

import numpy as np

from . import operators
from .evaluator import Evaluator


def run_fixed_evolution(task, evaluator: Evaluator, rng: np.random.Generator, cfg) -> None:
    pop = []
    for _ in range(cfg.pop_size):
        x = task.random_solution(rng)
        score, fb, eval_id = evaluator(x, {"agent": 0, "op": "init", "source": "own", "skill": False,
                                           "parent": -1, "mem_reads": 0})
        pop.append({"id": eval_id, "x": x, "score": score, "focus": fb.focus})
    while True:  # ends when the evaluator raises BudgetExhausted
        contenders = rng.choice(len(pop), size=cfg.tournament, replace=False)
        parent = max((pop[int(i)] for i in contenders), key=lambda p: p["score"])
        op = operators.OPS[int(rng.integers(len(operators.OPS)))]
        pool = [p["x"] for p in pop if p is not parent]
        x = operators.apply(op, rng, parent["x"], task, 1.0, parent["focus"], pool)
        score, fb, eval_id = evaluator(x, {"agent": 0, "op": op, "source": "own", "skill": False,
                                           "parent": parent["id"], "mem_reads": 0})
        worst = min(range(len(pop)), key=lambda i: pop[i]["score"])
        if score > pop[worst]["score"]:
            pop[worst] = {"id": eval_id, "x": x, "score": score, "focus": fb.focus}
