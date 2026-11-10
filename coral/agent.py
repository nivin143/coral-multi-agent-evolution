"""A persistent autonomous agent with its own workspace and a heartbeat (proposal 3.2.2, 3.2.3).

Private to the agent (its "context" and workspace): the attempt it is currently working from,
its recent operator statistics, its adapted step sizes, its random stream.
Shared: only what it writes to memory.

The context is bounded on purpose. At every heartbeat the recent statistics and step sizes are
dropped, the way a long-running LLM agent's context is truncated; whatever was not written to
memory as a note is gone. This is the modelling assumption that makes reflection matter.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import operators
from .evaluator import Evaluator
from .memory import SharedMemory


class CoralAgent:
    def __init__(self, agent_id: int, task, evaluator: Evaluator, memory: SharedMemory,
                 rng: np.random.Generator, cfg, policy, p_adopt: float = 0.15,
                 base_scale: float = 1.0, workspace: str | Path | None = None):
        self.id = agent_id
        self.task = task
        self.evaluator = evaluator
        self.memory = memory
        self.rng = rng
        self.cfg = cfg
        self.policy = policy
        self.p_adopt = p_adopt          # how readily this agent builds on attempts from memory
        self.base_scale = base_scale    # this agent's default step size ("temperament")
        self.workspace = Path(workspace) if workspace else None
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)

        self.cur: dict | None = None    # attempt the agent is working from
        self.best: dict | None = None   # agent's personal best
        self.window: dict = {}          # (diagnosis, operator) -> [tries, successes, total gain]
        self.scales: dict = {}          # operator -> adapted step size
        self.steps = 0
        self.stagnation = 0
        self.stats = Counter()
        self.journal: list[str] = []
        self._elites: list[dict] = []

    # -------------------------------------------------------------- retrieval
    def recall_elites(self, k: int = 8) -> list[dict]:
        self._elites = self.memory.top_attempts(k)
        return self._elites

    def recall_knowledge(self, tag: str):
        notes = self.memory.notes_for(tag) if self.cfg.reflection else []
        skills = self.memory.skills_for(tag) if (self.cfg.reflection and self.cfg.consolidation) else []
        return notes, skills

    def scale_for(self, op: str, notes: list[dict]) -> float:
        if op in self.scales:
            return self.scales[op]
        w = s = 0.0
        for n in notes:
            if n["op"] == op:
                ww = n["succ"] + 0.25
                w += ww
                s += ww * n["scale"]
        return s / w if w else self.base_scale

    # ------------------------------------------------------------- evaluation
    def _evaluate(self, x, parent, op, scale=1.0, skill=False, source="own", informed=False) -> dict:
        meta = {"agent": self.id, "op": op, "source": source, "skill": skill,
                "parent": parent["id"] if parent else -1, "mem_reads": self.memory.reads}
        score, fb, eval_id = self.evaluator(x, meta)
        attempt = {"id": eval_id, "agent": self.id, "x": x, "score": score, "tag": fb.tag,
                   "focus": fb.focus, "message": fb.message, "valid": fb.valid,
                   "parent": meta["parent"], "op": op}
        self.memory.add_attempt(attempt, key=self.task.key(x))
        if parent is not None:
            success = score > parent["score"]
            w = self.window.setdefault((parent["tag"], op), [0, 0, 0.0])
            w[0] += 1
            w[1] += int(success)
            w[2] += max(score - parent["score"], 0.0)
            self.scales[op] = float(np.clip(scale * (1.25 if success else 0.94), 0.05, 4.0))
            if skill:
                self.stats["skill_uses"] += 1
                self.stats["skill_successes"] += int(success)
            if informed:
                self.stats["note_informed"] += 1
                self.stats["note_informed_successes"] += int(success)
            if parent["agent"] != self.id:
                self.stats["cross_agent_uses"] += 1
                self.stats["cross_agent_successes"] += int(success)
        if self.best is None or score > self.best["score"]:
            self.best, self.stagnation = attempt, 0
        else:
            self.stagnation += 1
        if self.cur is None or score >= self.cur["score"]:
            self.cur = attempt
        return attempt

    def bootstrap(self, n_samples: int) -> None:
        for _ in range(n_samples):
            self._evaluate(self.task.random_solution(self.rng), None, "init")

    def step(self) -> None:
        """One iteration: retrieve, decide, generate, check for redundancy, evaluate, record."""
        d = self.policy.decide(self)
        pool = [e["x"] for e in self._elites if e["id"] != d.parent["id"]]
        x = operators.apply(d.op, self.rng, d.parent["x"], self.task, d.scale, d.parent["focus"], pool)
        if self.cfg.dedupe:
            for _ in range(3):  # do not spend an evaluation on a candidate memory already holds
                if not self.memory.seen(self.task.key(x)):
                    break
                self.stats["duplicates_avoided"] += 1
                x = operators.apply("gauss_small", self.rng, x, self.task, max(d.scale, 1.0))
        self._evaluate(x, d.parent, d.op, d.scale, d.skill, d.source, d.informed)
        self.steps += 1
        if self.steps % self.cfg.heartbeat == 0:
            self.heartbeat()

    # -------------------------------------------------------------- heartbeat
    def heartbeat(self) -> None:
        self.stats["heartbeats"] += 1
        if self.cfg.reflection:
            self.reflect()
        self.window, self.scales = {}, {}      # bounded context: only memory survives
        if self.cfg.reflection and self.cfg.consolidation:
            self.memory.consolidate()
        if self.cfg.redirection and self.stagnation >= self.cfg.stagnation_limit:
            self.redirect()
        self._save_workspace()

    def reflect(self) -> None:
        """Reflection: write what the last stretch of work showed into memory as notes."""
        lines = []
        for (tag, op), (n, succ, gain) in sorted(self.window.items()):
            scale = self.scales.get(op, self.base_scale)
            text = (f"agent {self.id} @eval {self.evaluator.calls}: under '{tag}', '{op}' improved "
                    f"its parent {succ}/{n} times (step scale {scale:.2f}, total gain {gain:.4g})")
            self.memory.add_note({"tag": tag, "op": op, "n": n, "succ": succ, "gain": gain,
                                  "scale": scale, "agent": self.id, "eval": self.evaluator.calls,
                                  "text": text})
            lines.append(text)
        self.stats["notes_written"] += len(lines)
        summary = "\n".join(lines)
        insight = self.policy.reflect(self, summary)
        if insight:
            self.memory.add_note({"tag": "_insight", "op": "_text", "n": 0, "succ": 0, "gain": 0.0,
                                  "scale": 1.0, "agent": self.id, "eval": self.evaluator.calls,
                                  "text": insight})
        self.journal.append(summary + (("\nlesson: " + insight) if insight else ""))

    def redirect(self) -> None:
        """Redirection: no personal best for a while, so change what the agent is working on."""
        self.stats["redirections"] += 1
        elites = self.recall_elites(8)
        top = elites[0] if elites else None
        if top is not None and top["id"] != self.cur["id"] and top["score"] > self.cur["score"]:
            # Something better is in memory (possibly another agent's): move there.
            if top["agent"] != self.id:
                self.stats["redirect_to_other_agent"] += 1
            self.cur = top
            self.journal.append(f"redirect @eval {self.evaluator.calls}: adopt attempt {top['id']} "
                                f"from agent {top['agent']} (score {top['score']:.5g})")
        else:
            # Already at the best known attempt: restart part of it to leave this basin.
            x = self.best["x"].copy()
            for g in self.task.groups:
                if self.rng.random() < 0.3:
                    x[g] = self.rng.random(len(g))
            self.cur = self._evaluate(x, None, "redirect")
            self.journal.append(f"redirect @eval {self.evaluator.calls}: partial restart from "
                                f"personal best (new score {self.cur['score']:.5g})")
        self.stagnation = 0

    def _save_workspace(self) -> None:
        if not self.workspace:
            return
        state = {"agent": self.id, "steps": self.steps, "p_adopt": self.p_adopt,
                 "base_scale": self.base_scale, "current_score": self.cur["score"],
                 "best_score": self.best["score"], "best_x": self.best["x"].tolist(),
                 "stagnation": self.stagnation, "stats": dict(self.stats)}
        (self.workspace / "context.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
        (self.workspace / "journal.md").write_text("\n\n".join(self.journal), encoding="utf-8")
