"""The agent's decision maker: what to retrieve, which attempt to build on, how to change it.

`HeuristicPolicy` is an offline, seeded stand-in for the LLM that drives a CORAL agent. It makes
the same *kinds* of decisions an LLM agent makes (retrieve, choose a parent, choose a strategy
in light of the evaluator's feedback and of remembered notes and skills) with a contextual
Thompson-sampling rule, so experiments are free, fast and exactly reproducible.
`coral.llm_policy.LLMPolicy` makes the same decisions by asking a Claude model.
"""
from __future__ import annotations

from dataclasses import dataclass

from .operators import OPS


@dataclass
class Decision:
    parent: dict            # the attempt to build on
    op: str                 # variation operator
    scale: float            # step size multiplier
    skill: bool = False     # the choice came from a stored skill
    source: str = "own"     # "own" = agent's current attempt, "memory" = retrieved from memory
    informed: bool = False  # notes from memory were available for this choice


class HeuristicPolicy:
    name = "heuristic"
    llm_calls = 0
    llm_errors = 0
    input_tokens = 0
    output_tokens = 0

    def decide(self, agent) -> Decision:
        rng = agent.rng
        parent, source = agent.cur, "own"

        # 1. Retrieval: look at the best attempts in memory and maybe build on one of them.
        elites = agent.recall_elites(8)
        if elites and rng.random() < agent.p_adopt:
            pick = elites[min(int(rng.geometric(0.4)) - 1, len(elites) - 1)]
            if pick["id"] != agent.cur["id"]:
                parent, source = pick, "memory"

        # 2. Recall what is known about the parent's diagnosis.
        notes, skills = agent.recall_knowledge(parent["tag"])

        # 3. Reuse a skill if one matches.
        if skills and rng.random() < agent.cfg.skill_prob:
            sk = skills[0] if (len(skills) == 1 or rng.random() < 0.7) else skills[1]
            scale = agent.scales.get(sk["op"], sk["scale"])
            return Decision(parent, sk["op"], scale, skill=True, source=source, informed=True)

        # 4. Otherwise choose an operator by Thompson sampling over remembered + recent evidence.
        stats = {}
        for n in notes:
            s = stats.setdefault(n["op"], [0.0, 0.0])
            s[0] += n["n"]
            s[1] += n["succ"]
        for (tag, op), w in agent.window.items():
            if tag == parent["tag"]:
                s = stats.setdefault(op, [0.0, 0.0])
                s[0] += w[0]
                s[1] += w[1]
        best_op, best_draw = OPS[0], -1.0
        for op in OPS:
            n, succ = stats.get(op, (0.0, 0.0))
            draw = rng.beta(1.0 + succ, 1.0 + max(n - succ, 0.0))
            if draw > best_draw:
                best_op, best_draw = op, draw
        return Decision(parent, best_op, agent.scale_for(best_op, notes), source=source,
                        informed=bool(notes))

    def reflect(self, agent, summary: str) -> str:
        """Hook for a free-text lesson at a heartbeat. The offline policy has nothing to add."""
        return ""
