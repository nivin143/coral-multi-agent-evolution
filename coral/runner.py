"""Run one (task, condition, seed) under a fixed evaluation budget and collect the metrics.

Asynchrony is simulated with an event queue: each agent takes a random, seeded amount of
virtual time per iteration and the agent that finishes first acts next. Agents therefore
interleave irregularly and see each other's results at unpredictable moments, as asynchronous
workers do, while a given seed still reproduces the run exactly.
"""
from __future__ import annotations

import heapq
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .agent import CoralAgent
from .baseline import run_fixed_evolution
from .config import CONFIGS, Config
from .evaluator import BudgetExhausted, Evaluator
from .memory import SharedMemory
from .policy import HeuristicPolicy
from .tasks import TASKS, make_task


def run(task_name: str, config: str | Config, seed: int, budget: int = 1200,
        persist_dir: str | Path | None = None, policy=None, verbose: bool = False) -> dict:
    cfg = CONFIGS[config] if isinstance(config, str) else config
    task = make_task(task_name)
    evaluator = Evaluator(task, budget)
    root = np.random.SeedSequence([seed, list(TASKS).index(task_name)])
    persist = Path(persist_dir) if persist_dir else None
    start = time.perf_counter()
    agents: list[CoralAgent] = []
    memories: list[SharedMemory] = []

    if cfg.kind == "fixed":
        try:
            run_fixed_evolution(task, evaluator, np.random.default_rng(root), cfg)
        except BudgetExhausted:
            pass
    else:
        policy = policy or HeuristicPolicy()
        seeds = root.spawn(cfg.n_agents + 1)
        sched = np.random.default_rng(seeds[-1])
        n = cfg.n_agents
        shared = SharedMemory(persist / "memory" if persist else None) if cfg.shared_memory else None
        for i in range(n):
            mem = shared or SharedMemory(persist / f"memory_agent_{i}" if persist else None)
            if mem not in memories:
                memories.append(mem)
            frac = i / (n - 1) if n > 1 else 0.33
            agents.append(CoralAgent(
                i, task, evaluator, mem, np.random.default_rng(seeds[i]), cfg, policy,
                p_adopt=0.05 + 0.30 * frac,                  # agents differ in how much they borrow
                base_scale=float(0.5 * 5 ** frac) if n > 1 else 1.0,   # and in how boldly they move
                workspace=persist / "workspace" / f"agent_{i}" if persist else None))
        speed = 0.8 + 0.5 * sched.random(n)
        try:
            for a in agents:
                a.bootstrap(max(1, cfg.init_samples_total // n))
            queue = [(float(speed[i] * sched.lognormal(0, 0.4)), i) for i in range(n)]
            heapq.heapify(queue)
            while True:
                t, i = heapq.heappop(queue)
                agents[i].step()
                if verbose and evaluator.calls % 100 == 0:
                    print(f"  eval {evaluator.calls:5d}  best {evaluator.best:.5f}  "
                          f"notes {sum(m.n_notes() for m in memories)}  "
                          f"skills {sum(m.n_skills() for m in memories)}")
                heapq.heappush(queue, (t + float(speed[i] * sched.lognormal(0, 0.4)), i))
        except BudgetExhausted:
            pass
        for m in memories:
            m.dump()
        for a in agents:
            a._save_workspace()

    stats = Counter()
    for a in agents:
        stats.update(a.stats)
    evals_per_agent = Counter(r["agent"] for r in evaluator.rows)
    summary = {
        "task": task_name, "config": cfg.name, "seed": seed, "budget": budget,
        "final_best": evaluator.best,
        "n_improvements": evaluator.improvements,
        "improvement_freq": evaluator.improvements / budget,
        "invalid_evals": evaluator.invalid,
        "runtime_s": round(time.perf_counter() - start, 4),
        "parallel_depth": max(evals_per_agent.values()),   # evaluations on the busiest worker
        "mem_reads": sum(m.reads for m in memories),
        "mem_writes": sum(m.writes for m in memories),
        "n_notes": sum(m.n_notes() for m in memories),
        "n_skills": sum(m.n_skills() for m in memories),
        "llm_calls": getattr(policy, "llm_calls", 0),
        "llm_input_tokens": getattr(policy, "input_tokens", 0),
        "llm_output_tokens": getattr(policy, "output_tokens", 0),
    }
    for k in ("heartbeats", "notes_written", "skill_uses", "skill_successes", "note_informed",
              "note_informed_successes", "cross_agent_uses", "cross_agent_successes",
              "redirections", "redirect_to_other_agent", "duplicates_avoided"):
        summary[k] = stats.get(k, 0)
    return {"summary": summary, "curve": np.asarray(evaluator.curve), "rows": evaluator.rows,
            "best_x": evaluator.best_x, "task": task}
