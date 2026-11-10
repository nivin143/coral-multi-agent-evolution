"""Turn results/ into tables, figures and results/RESULTS.md.

    python analyze_results.py

Reads results/summary.csv and results/curves.npz written by run_experiments.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coral import ABLATIONS, CONFIGS, MAIN
from coral.stats import evals_to_target, holm, paired_summary

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"fixed_evo": "#2a78d6", "single_agent": "#eb6834", "multi_agent": "#1baf7a"}
SHORT = {"fixed_evo": "A fixed evolution", "single_agent": "B single agent",
         "multi_agent": "Proposed multi-agent", "abl_no_shared_memory": "no shared memory",
         "abl_no_reflection": "no reflection", "abl_no_consolidation": "no consolidation",
         "abl_no_redirection": "no redirection"}
TASK_TITLE = {"circle_packing": "Circle packing (maths)", "bin_packing": "Bin-packing heuristic (algorithm)",
              "system_tuning": "System tuning (systems)"}
COMPARISONS = [  # (label, a, b, hypothesis): difference is a - b
    ("single agent - fixed evolution", "single_agent", "fixed_evo", "H1"),
    ("multi-agent - fixed evolution", "multi_agent", "fixed_evo", "H1/H2"),
    ("multi-agent - single agent", "multi_agent", "single_agent", "H2"),
] + [(f"multi-agent - {SHORT[a]}", "multi_agent", a, "H3") for a in ABLATIONS]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(res / "summary.csv")
    curves = np.load(res / "curves.npz")
    tasks = list(dict.fromkeys(df["task"]))
    configs = [c for c in CONFIGS if c in set(df["config"])]
    n_seeds, budget = df["seed"].nunique(), int(df["budget"].iloc[0])

    # ---- evaluation efficiency: evaluations needed to match Baseline A's median final score
    targets, rows = {}, []
    for t in tasks:
        targets[t] = float(np.median(curves[f"{t}|fixed_evo"][:, -1]))
        for c in configs:
            for s, curve in enumerate(curves[f"{t}|{c}"]):
                n, reached = evals_to_target(curve, targets[t])
                rows.append({"task": t, "config": c, "seed": s, "evals_to_target": n, "reached": reached})
    df = df.merge(pd.DataFrame(rows), on=["task", "config", "seed"])
    df.to_csv(res / "summary_with_efficiency.csv", index=False)

    # ---- Table 1: main metrics
    t1 = []
    for t in tasks:
        for c in configs:
            g = df[(df.task == t) & (df.config == c)]
            t1.append({
                "task": t, "condition": SHORT[c],
                "final best (mean ± sd)": f"{fmt(g.final_best.mean())} ± {fmt(g.final_best.std(ddof=1))}",
                "best seed": fmt(g.final_best.max()),
                "improvements / run": fmt(g.n_improvements.mean(), 1),
                "evals to target (median)": int(g.evals_to_target.median()),
                "reached target": f"{int(g.reached.sum())}/{len(g)}",
                "runtime s": fmt(g.runtime_s.mean(), 2),
                "evals on busiest worker": int(g.parallel_depth.mean()),
                "memory reads": int(g.mem_reads.mean()),
            })
    t1 = pd.DataFrame(t1)
    t1.to_csv(res / "table_main_metrics.csv", index=False)

    # ---- Table 2: paired comparisons on final score; Table 3: on evaluations-to-target
    t2, t3 = [], []
    for t in tasks:
        fam = []
        for label, a, b, hyp in COMPARISONS:
            ga = df[(df.task == t) & (df.config == a)].sort_values("seed")
            gb = df[(df.task == t) & (df.config == b)].sort_values("seed")
            st = paired_summary(ga.final_best.values, gb.final_best.values)
            fam.append({"task": t, "comparison": label, "tests": hyp, **st})
            if b == "fixed_evo":
                # positive = the agent condition needed fewer evaluations than Baseline A
                se = paired_summary(gb.evals_to_target.values, ga.evals_to_target.values)
                t3.append({"task": t, "comparison": label, **se})
        for row, p in zip(fam, holm([r["p_value"] for r in fam])):
            row["p_holm"] = p
        t2 += fam
    t2, t3 = pd.DataFrame(t2), pd.DataFrame(t3)
    t2.to_csv(res / "table_paired_final_score.csv", index=False)
    t3.to_csv(res / "table_paired_eval_efficiency.csv", index=False)

    # ---- Table 4: knowledge reuse and coordination cost (agent conditions)
    t4 = []
    for t in tasks:
        for c in configs:
            if CONFIGS[c].kind != "agents":
                continue
            g = df[(df.task == t) & (df.config == c)]
            rate = lambda s, n: f"{g[s].sum() / g[n].sum():.0%}" if g[n].sum() else "-"
            t4.append({
                "task": t, "condition": SHORT[c],
                "notes kept": fmt(g.n_notes.mean(), 1), "skills": fmt(g.n_skills.mean(), 1),
                "skill uses": fmt(g.skill_uses.mean(), 1), "skill success": rate("skill_successes", "skill_uses"),
                "note-informed steps": fmt(g.note_informed.mean(), 0),
                "note-informed success": rate("note_informed_successes", "note_informed"),
                "cross-agent builds": fmt(g.cross_agent_uses.mean(), 1),
                "cross-agent success": rate("cross_agent_successes", "cross_agent_uses"),
                "redirections": fmt(g.redirections.mean(), 1),
                "duplicates avoided": fmt(g.duplicates_avoided.mean(), 1),
                "memory reads": int(g.mem_reads.mean()), "memory writes": int(g.mem_writes.mean()),
            })
    t4 = pd.DataFrame(t4)
    t4.to_csv(res / "table_knowledge_reuse.csv", index=False)

    # ---- Figure 1: best-so-far against evaluator calls
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.2))
    for ax, t in zip(np.atleast_1d(axes), tasks):
        style(ax)
        x = np.arange(1, budget + 1)
        lo, hi = np.inf, -np.inf
        start = min(60, budget // 4)
        ends = []
        for c in MAIN:
            m, sd = curves[f"{t}|{c}"].mean(0), curves[f"{t}|{c}"].std(0, ddof=1)
            ax.fill_between(x, m - sd, m + sd, color=SERIES[c], alpha=0.12, linewidth=0)
            ax.plot(x, m, color=SERIES[c], linewidth=2, label=SHORT[c])
            ends.append((m[-1], c))
            lo, hi = min(lo, (m - sd)[start]), max(hi, (m + sd)[-1])
        hi += 0.06 * (hi - lo)
        ax.set_ylim(lo, hi)
        gap, prev = 0.055 * (hi - lo), -np.inf
        for val, c in sorted(ends):                 # end labels, nudged apart so they stay legible
            ypos = max(val, prev + gap)
            prev = ypos
            ax.text(budget * 1.015, ypos, fmt(val, 3), va="center", fontsize=8, color=INK)
            ax.plot([budget * 1.004], [ypos], "o", color=SERIES[c], markersize=4, clip_on=False)
        ax.set_xlim(0, budget * 1.12)
        ax.set_title(TASK_TITLE.get(t, t), fontsize=11, color=INK, loc="left")
        ax.set_xlabel("evaluator calls", fontsize=9, color=MUTED)
    np.atleast_1d(axes)[0].set_ylabel("best score so far (higher is better)", fontsize=9, color=MUTED)
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="upper right", ncol=3, labelcolor=INK)
    fig.suptitle(f"Best-so-far score at equal evaluation budget (mean of {n_seeds} seeds, band = ±1 sd)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_dir / "convergence.png", dpi=160)
    plt.close(fig)

    # ---- Figure 2: paired differences with bootstrap confidence intervals
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.6), sharey=True)
    labels = [c[0] for c in COMPARISONS]
    for ax, t in zip(np.atleast_1d(axes), tasks):
        style(ax)
        g = t2[t2.task == t].set_index("comparison").loc[labels]
        y = np.arange(len(labels))[::-1]
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.hlines(y, g.ci_low, g.ci_high, color="#2a78d6", linewidth=2)
        ax.plot(g.mean_diff, y, "o", color="#2a78d6", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9, color=INK)
        ax.set_title(TASK_TITLE.get(t, t), fontsize=11, color=INK, loc="left")
        ax.set_xlabel("difference in final score (right of 0 = first condition better)", fontsize=8, color=MUTED)
        ax.axhline(y[2] - 0.5, color=GRID, linewidth=1.2)
    fig.suptitle(f"Paired differences by seed (mean and 95% bootstrap CI, n = {n_seeds}); "
                 "below the line: full multi-agent minus each ablation",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "paired_differences.png", dpi=160)
    plt.close(fig)

    # ---- RESULTS.md
    show2 = t2.copy()
    show2["mean diff [95% CI]"] = [f"{r.mean_diff:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]" for r in show2.itertuples()]
    show2["d_z"] = show2.cohen_dz.map(lambda v: f"{v:+.2f}")
    show2["W/T/L"] = [f"{r.wins}/{r.ties}/{r.losses}" for r in show2.itertuples()]
    show2["p"] = show2.p_value.map(lambda v: f"{v:.3f}")
    show2["p (Holm)"] = show2.p_holm.map(lambda v: f"{v:.3f}")
    show2 = show2[["task", "comparison", "tests", "mean diff [95% CI]", "d_z", "W/T/L", "p", "p (Holm)"]]
    show3 = t3.copy()
    show3["evals saved [95% CI]"] = [f"{r.mean_diff:+.0f} [{r.ci_low:+.0f}, {r.ci_high:+.0f}]" for r in show3.itertuples()]
    show3["W/T/L"] = [f"{r.wins}/{r.ties}/{r.losses}" for r in show3.itertuples()]
    show3["p"] = show3.p_value.map(lambda v: f"{v:.3f}")
    show3 = show3[["task", "comparison", "evals saved [95% CI]", "W/T/L", "p"]]
    text = f"""# Results

Generated by `analyze_results.py` from `results/summary.csv` and `results/curves.npz`.
Grid: {len(tasks)} tasks x {len(configs)} conditions x {n_seeds} seeds, {budget} evaluator calls per run.
All scores are maximised. Agent conditions use the offline heuristic policy (no LLM calls).

![Best-so-far score against evaluator calls](figures/convergence.png)

## Table 1. Main metrics

"Evals to target" is the number of evaluator calls needed to reach the median final score of
Baseline A on that task (targets: {", ".join(f"{t} {targets[t]:.4f}" for t in tasks)}). Runs that never
reach it are counted at the full budget. "Evals on busiest worker" is the number of evaluations
the busiest agent performed: with four agents the same budget is spread over four workers.

{md_table(t1)}

## Table 2. Paired comparisons of the final score

Pairs share task and seed. Difference = first condition minus second. CI: bootstrap of the mean
(10,000 resamples). d_z: Cohen's d for paired samples. p: two-sided exact sign-flip permutation
test; Holm-adjusted within each task.

![Paired differences](figures/paired_differences.png)

{md_table(show2)}

## Table 3. Evaluation efficiency against Baseline A

Evaluator calls saved in reaching Baseline A's median final score (positive = the agent
condition got there sooner; censored at the budget).

{md_table(show3)}

## Table 4. Knowledge reuse and coordination cost

Means per run. "Success" = the new candidate scored higher than the attempt it was built on.

{md_table(t4)}

## Reproducibility

Standard deviations across seeds are in Table 1. Re-running a (task, condition, seed) triple
reproduces the same evaluation trace exactly; `tests/test_coral.py` checks this.
"""
    (res / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
