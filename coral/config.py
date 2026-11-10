"""Experimental conditions (proposal section 3.4): two baselines, the proposed approach, ablations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    name: str
    label: str
    kind: str = "agents"            # "fixed" = Baseline A, "agents" = autonomous agent(s)
    n_agents: int = 1
    shared_memory: bool = True      # agents read and write one common memory
    reflection: bool = True         # heartbeat writes notes
    consolidation: bool = True      # heartbeat merges notes and promotes skills
    redirection: bool = True        # heartbeat changes strategy after stagnation
    dedupe: bool = True             # skip candidates memory has already seen
    heartbeat: int = 25             # an agent reflects every `heartbeat` of its own evaluations
    stagnation_limit: int = 50      # evaluations without a personal best before redirection
    init_samples_total: int = 8     # random starting candidates, split across agents
    skill_prob: float = 0.4         # chance to apply a matching skill when one exists
    pop_size: int = 20              # Baseline A population
    tournament: int = 3             # Baseline A tournament size


CONFIGS = {c.name: c for c in [
    Config("fixed_evo", "A: fixed evolutionary search", kind="fixed"),
    Config("single_agent", "B: autonomous single agent", n_agents=1),
    Config("multi_agent", "Proposed: autonomous multi-agent (4)", n_agents=4),
    # Ablations of the proposed approach: switch one mechanism off.
    Config("abl_no_shared_memory", "multi-agent, no shared memory", n_agents=4, shared_memory=False),
    Config("abl_no_reflection", "multi-agent, no reflection", n_agents=4, reflection=False,
           consolidation=False),
    Config("abl_no_consolidation", "multi-agent, no consolidation", n_agents=4, consolidation=False),
    Config("abl_no_redirection", "multi-agent, no redirection", n_agents=4, redirection=False),
]}

MAIN = ["fixed_evo", "single_agent", "multi_agent"]
ABLATIONS = ["abl_no_shared_memory", "abl_no_reflection", "abl_no_consolidation", "abl_no_redirection"]
