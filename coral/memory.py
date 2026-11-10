"""Persistent shared memory: attempts, notes and skills (proposal section 3.2.1).

* Attempts - every evaluated candidate with its score, feedback and lineage.
* Notes    - observations written at heartbeats: how often an operator improved a candidate
             under a given evaluator diagnosis, and at what step size.
* Skills   - reusable procedures promoted from recurring successful notes: "under diagnosis T,
             apply operator O at step size S".

Agents never message each other. This store is the only channel between them (section 3.2.2).
With `persist_dir` set, everything is also written to disk as JSON / JSONL so a run can be
inspected, or loaded again with `SharedMemory.load`.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np


class SharedMemory:
    ELITE_SIZE = 30

    def __init__(self, persist_dir: str | Path | None = None, recent_notes: int = 6):
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.recent_notes = recent_notes
        self.attempts: list[dict] = []
        self._elite: list[dict] = []           # best attempts, best first
        self._elite_keys: list[float] = []     # negated scores, kept sorted for bisect
        self._seen: set = set()
        self.notes: dict[str, list[dict]] = {}  # diagnosis tag -> notes, oldest first
        self.skills: dict[str, list[dict]] = {}
        self.reads = 0
        self.writes = 0
        self.consolidations = 0
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            for name in ("attempts.jsonl", "notes.jsonl"):
                (self.persist_dir / name).write_text("", encoding="utf-8")

    # ---------------------------------------------------------------- attempts
    def add_attempt(self, attempt: dict, key=None) -> None:
        self.writes += 1
        self.attempts.append(attempt)
        if key is not None:
            self._seen.add(key)
        if attempt["valid"]:
            pos = bisect.bisect_right(self._elite_keys, -attempt["score"])
            if pos < self.ELITE_SIZE:
                self._elite_keys.insert(pos, -attempt["score"])
                self._elite.insert(pos, attempt)
                del self._elite_keys[self.ELITE_SIZE:], self._elite[self.ELITE_SIZE:]
        if self.persist_dir:
            rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in attempt.items()}
            with open(self.persist_dir / "attempts.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    def seen(self, key) -> bool:
        self.reads += 1
        return key in self._seen

    def top_attempts(self, k: int = 8) -> list[dict]:
        self.reads += 1
        return self._elite[:k]

    # ------------------------------------------------------------------- notes
    def add_note(self, note: dict) -> None:
        self.writes += 1
        self.notes.setdefault(note["tag"], []).append(note)
        if self.persist_dir:
            with open(self.persist_dir / "notes.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(note) + "\n")

    def notes_for(self, tag: str) -> list[dict]:
        """Consolidated notes for this diagnosis plus the most recent raw ones.

        Retrieval is bounded, like an agent's context: without consolidation only the last
        `recent_notes` raw notes are visible and older observations are effectively forgotten.
        """
        self.reads += 1
        notes = self.notes.get(tag, [])
        merged = [n for n in notes if n.get("consolidated")]
        raw = [n for n in notes if not n.get("consolidated")]
        return merged + raw[-self.recent_notes:]

    def n_notes(self) -> int:
        return sum(len(v) for v in self.notes.values())

    # ------------------------------------------------------------------ skills
    def skills_for(self, tag: str) -> list[dict]:
        self.reads += 1
        return self.skills.get(tag, [])

    def n_skills(self) -> int:
        return sum(len(v) for v in self.skills.values())

    def consolidate(self, min_successes: int = 3, min_rate: float = 0.12, decay: float = 0.85,
                    skills_per_tag: int = 2) -> None:
        """Merge notes per (diagnosis, operator), age old evidence, and promote skills.

        Ageing (`decay`) is the memory clean-up step: statistics that are not refreshed by new
        observations fade, so stale notes stop steering decisions (proposal section 6.4).
        """
        self.writes += 1
        self.consolidations += 1
        for tag, notes in self.notes.items():
            merged: dict[str, dict] = {}
            for n in notes:
                w = decay if n.get("consolidated") else 1.0
                m = merged.setdefault(n["op"], {"tag": tag, "op": n["op"], "n": 0.0, "succ": 0.0,
                                                "gain": 0.0, "_sw": 0.0, "_ss": 0.0,
                                                "consolidated": True})
                m["n"] += w * n["n"]
                m["succ"] += w * n["succ"]
                m["gain"] += w * n["gain"]
                sw = w * (n["succ"] + 0.25)
                m["_sw"] += sw
                m["_ss"] += sw * n["scale"]
            out = []
            for m in merged.values():
                m["scale"] = m.pop("_ss") / m.pop("_sw")
                m["n"], m["succ"] = round(m["n"], 3), round(m["succ"], 3)
                if m["n"] >= 0.5:
                    out.append(m)
            self.notes[tag] = out
            good = [m for m in out if m["succ"] >= min_successes and m["succ"] / m["n"] >= min_rate]
            good.sort(key=lambda m: m["succ"] / m["n"], reverse=True)
            self.skills[tag] = [
                {"tag": tag, "op": m["op"], "scale": round(m["scale"], 4),
                 "success_rate": round(m["succ"] / m["n"], 3), "evidence": m["n"],
                 "text": f"When the evaluator reports '{tag}', apply '{m['op']}' at step scale "
                         f"{m['scale']:.2f} (improved {m['succ']:.0f} of {m['n']:.0f} tries)."}
                for m in good[:skills_per_tag]
            ]
        if self.persist_dir:
            self.dump()

    # ------------------------------------------------------------- persistence
    def dump(self) -> None:
        if not self.persist_dir:
            return
        (self.persist_dir / "notes_consolidated.json").write_text(
            json.dumps(self.notes, indent=1), encoding="utf-8")
        (self.persist_dir / "skills.json").write_text(json.dumps(self.skills, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, persist_dir: str | Path) -> "SharedMemory":
        """Rebuild a memory from a directory written by a previous run."""
        src = Path(persist_dir)
        mem = cls()
        for line in (src / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
            a = json.loads(line)
            a["x"] = np.asarray(a["x"], float)
            mem.add_attempt(a)
        notes = src / "notes_consolidated.json"
        if notes.exists():
            mem.notes = json.loads(notes.read_text(encoding="utf-8"))
        else:
            for line in (src / "notes.jsonl").read_text(encoding="utf-8").splitlines():
                mem.add_note(json.loads(line))
        skills = src / "skills.json"
        if skills.exists():
            mem.skills = json.loads(skills.read_text(encoding="utf-8"))
        mem.reads = mem.writes = 0
        return mem
