"""Run one condition on one task and keep everything it produced on disk.

    python run_demo.py                                           # multi-agent on circle packing
    python run_demo.py --task system_tuning --config single_agent --seed 3
    python run_demo.py --policy llm --budget 60                  # Claude makes the decisions

Writes runs/<task>_<config>_seed<N>/ with the shared memory (attempts, notes, skills), one
workspace per agent (context and reflection journal) and the best solution found.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coral import CONFIGS, TASKS, run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="circle_packing", choices=list(TASKS))
    ap.add_argument("--config", default="multi_agent", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--policy", default="heuristic", choices=["heuristic", "llm"])
    ap.add_argument("--llm-model", default="claude-opus-5-5")
    ap.add_argument("--llm-every", type=int, default=1,
                    help="ask the model for every N-th decision; the heuristic makes the others")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    policy = None
    if args.policy == "llm":
        from coral.llm_policy import LLMPolicy
        policy = LLMPolicy(model=args.llm_model, every=args.llm_every)

    cfg = CONFIGS[args.config]
    out = Path(args.out) / f"{args.task}_{args.config}_seed{args.seed}"
    print(f"task={args.task}  condition={cfg.label}  seed={args.seed}  budget={args.budget}  "
          f"policy={args.policy}")
    res = run(args.task, cfg, args.seed, args.budget, persist_dir=out, policy=policy, verbose=True)
    s = res["summary"]
    task = res["task"]
    score, fb = task.evaluate(res["best_x"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_solution.json").write_text(json.dumps(
        {"score": score, "feedback": fb.message, "x": res["best_x"].tolist()}, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")

    print(f"\nbest score        {s['final_best']:.6f}  (re-evaluated: {score:.6f})")
    if hasattr(task, "verify"):
        print(f"packing verified  {task.verify(res['best_x'])}")
    print(f"feedback          {fb.message}")
    print(f"improvements      {s['n_improvements']} of {s['budget']} evaluations")
    if cfg.kind == "agents":
        print(f"memory            {s['mem_reads']} reads, {s['mem_writes']} writes, "
              f"{s['n_notes']} notes, {s['n_skills']} skills")
        print(f"skills applied    {s['skill_uses']} ({s['skill_successes']} improved the parent)")
        print(f"cross-agent reuse {s['cross_agent_uses']} ({s['cross_agent_successes']} improved the parent)")
        print(f"heartbeats        {s['heartbeats']}, redirections {s['redirections']}, "
              f"duplicates avoided {s['duplicates_avoided']}")
        if args.policy == "llm":
            print(f"LLM               {s['llm_calls']} calls, {s['llm_input_tokens']} input / "
                  f"{s['llm_output_tokens']} output tokens, {policy.llm_errors} fell back to the heuristic")
    print(f"runtime           {s['runtime_s']:.2f}s")
    print(f"artifacts in      {out}")


if __name__ == "__main__":
    main()
