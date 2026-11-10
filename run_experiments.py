"""Run the full experiment grid: tasks x conditions x seeds, all at the same evaluation budget.

    python run_experiments.py                 # 3 tasks x 7 conditions x 10 seeds, budget 1200
    python run_experiments.py --quick         # small grid for a fast check (about a minute)

Writes results/summary.csv (one row per run), results/curves.npz (best-so-far per evaluation)
and results/evaluations.csv.gz (one row per evaluator call).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from coral import CONFIGS, TASKS, run


def _one(job):
    task, config, seed, budget = job
    out = run(task, config, seed, budget)
    rows = [{"task": task, "config": config, "seed": seed, **r} for r in out["rows"]]
    return out["summary"], out["curve"], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (0..N-1)")
    ap.add_argument("--budget", type=int, default=1200, help="evaluator calls per run")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="3 seeds, budget 300")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.budget = 3, 300

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, c, s, args.budget) for t in args.tasks for c in args.configs for s in range(args.seeds)]
    print(f"{len(jobs)} runs, budget {args.budget}, {args.workers} worker processes")
    start = time.time()
    summaries, curves, n_rows = [], {}, 0
    eval_fields = ["task", "config", "seed", "eval_id", "agent", "op", "source", "skill", "parent",
                   "score", "best_so_far", "new_best", "tag", "valid", "mem_reads", "t_ms"]
    with Pool(args.workers) as pool, gzip.open(out / "evaluations.csv.gz", "wt", newline="",
                                               encoding="utf-8") as gz:
        writer = csv.DictWriter(gz, fieldnames=eval_fields)
        writer.writeheader()
        for i, (summary, curve, rows) in enumerate(pool.imap_unordered(_one, jobs), 1):
            summaries.append(summary)
            curves[(summary["task"], summary["config"], summary["seed"])] = curve
            writer.writerows(rows)
            n_rows += len(rows)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs done ({time.time() - start:.0f}s)")

    summaries.sort(key=lambda s: (s["task"], list(CONFIGS).index(s["config"]), s["seed"]))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    stacked = {}
    for t in args.tasks:
        for c in args.configs:
            stacked[f"{t}|{c}"] = np.stack([curves[(t, c, s)] for s in range(args.seeds)])
    np.savez_compressed(out / "curves.npz", **stacked)
    print(f"wrote {out/'summary.csv'}, {out/'curves.npz'}, {out/'evaluations.csv.gz'} "
          f"({n_rows} evaluations) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
