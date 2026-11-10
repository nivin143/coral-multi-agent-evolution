"""Optional LLM-driven policy: a Claude model makes the agent's decisions.

The agent shows the model the task, its current attempt with the evaluator's feedback, the best
attempts in shared memory, and the notes and skills that match the current diagnosis. The model
answers with which attempt to build on, which operator to apply and how boldly. At heartbeats it
also writes a short free-text lesson that is stored in memory as a note.

Needs `pip install anthropic` and credentials (ANTHROPIC_API_KEY, or an `ant auth login` profile).
If a reply cannot be used (refusal, API error, unparseable answer) the offline heuristic decides
that step instead, and the failure is counted in `llm_errors`.
"""
from __future__ import annotations

import json
import re

from .operators import DESCRIPTIONS, OPS
from .policy import Decision, HeuristicPolicy

SYSTEM = (
    "You are an autonomous research agent running an evolutionary search. You do not write the "
    "candidate yourself; you direct the search. Each turn, choose which earlier attempt to build "
    "on, which variation operator to apply to it, and a step scale. Use the evaluator feedback, "
    "the notes and the skills from shared memory: prefer what has worked under the same "
    "diagnosis, and change approach when progress has stalled.\n"
    'Reply with one JSON object and nothing else: {"parent": "current" or an attempt id from the '
    'list, "operator": one of the operator names, "scale": a number from 0.05 to 4, '
    '"rationale": one short sentence}.'
)


class LLMPolicy(HeuristicPolicy):
    name = "llm"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", every: int = 1,
                 client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.every = max(1, every)      # consult the model every N-th decision (cost control)
        self.llm_calls = self.llm_errors = self.input_tokens = self.output_tokens = 0
        self._n = 0
        self.rationales: list[str] = []

    # ------------------------------------------------------------------ calls
    def _ask(self, system: str, prompt: str) -> str | None:
        self.llm_calls += 1
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=4000, system=system,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:   # rate limit, connection or API error: let the heuristic take this step
            self.llm_errors += 1
            return None
        usage = getattr(resp, "usage", None)
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            self.llm_errors += 1
            return None
        return "".join(b.text for b in resp.content if b.type == "text")

    def build_prompt(self, agent, elites, notes, skills) -> str:
        cur = agent.cur
        lines = [
            f"TASK: {agent.task.description}",
            f"You are agent {agent.id}. Evaluations used: {agent.evaluator.calls} of "
            f"{agent.evaluator.budget}. Evaluations since your personal best: {agent.stagnation}.",
            "",
            f"CURRENT ATTEMPT (id {cur['id']}): score {cur['score']:.6g}, diagnosis '{cur['tag']}'.",
            f"Evaluator feedback: {cur['message']}",
            "",
            "BEST ATTEMPTS IN SHARED MEMORY:",
        ]
        lines += [f"- id {e['id']} by agent {e['agent']}: score {e['score']:.6g}, "
                  f"diagnosis '{e['tag']}'" for e in elites] or ["- (none yet)"]
        lines += ["", f"NOTES FOR DIAGNOSIS '{cur['tag']}':"]
        lines += [f"- '{n['op']}' improved {n['succ']:.0f} of {n['n']:.0f} tries at scale "
                  f"{n['scale']:.2f}" for n in notes] or ["- (none yet)"]
        lessons = [n["text"] for n in agent.memory.notes.get("_insight", [])[-3:]]
        if lessons:
            lines += ["", "LESSONS WRITTEN AT EARLIER HEARTBEATS:"] + [f"- {t}" for t in lessons]
        lines += ["", "SKILLS:"] + ([f"- {s['text']}" for s in skills] or ["- (none yet)"])
        lines += ["", "OPERATORS:"] + [f"- {op}: {DESCRIPTIONS[op]}" for op in OPS]
        return "\n".join(lines)

    @staticmethod
    def parse(text: str, agent, elites) -> tuple[dict, str, float] | None:
        match = re.search(r"\{.*\}", text or "", re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            op = obj["operator"]
            scale = float(obj.get("scale", 1.0))
        except (ValueError, KeyError, TypeError):
            return None
        if op not in OPS or not (scale == scale):
            return None
        parent = agent.cur
        wanted = str(obj.get("parent", "current"))
        for e in elites:
            if str(e["id"]) == wanted:
                parent = e
        return parent, op, min(max(scale, 0.05), 4.0)

    # -------------------------------------------------------------- decisions
    def decide(self, agent) -> Decision:
        self._n += 1
        if self._n % self.every:
            return super().decide(agent)
        elites = agent.recall_elites(8)
        notes, skills = agent.recall_knowledge(agent.cur["tag"])
        text = self._ask(SYSTEM, self.build_prompt(agent, elites, notes, skills))
        parsed = self.parse(text, agent, elites) if text else None
        if parsed is None:
            if text is not None:
                self.llm_errors += 1
            return super().decide(agent)
        parent, op, scale = parsed
        return Decision(parent, op, scale, skill=any(s["op"] == op for s in skills),
                        source="own" if parent is agent.cur else "memory",
                        informed=bool(notes or skills))

    def reflect(self, agent, summary: str) -> str:
        if not summary:
            return ""
        text = self._ask(
            "You are an autonomous research agent pausing to reflect. In at most two sentences, "
            "state the most useful lesson from these results for the agents that will read "
            "shared memory next. Plain text only.",
            f"TASK: {agent.task.description}\n\nRESULTS SINCE THE LAST HEARTBEAT:\n{summary}",
        )
        return (text or "").strip()[:500]
