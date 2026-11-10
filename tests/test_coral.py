"""Checks that the platform runs and behaves as documented.

    python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coral import CONFIGS, TASKS, make_task, run            # noqa: E402
from coral.evaluator import BudgetExhausted, Evaluator      # noqa: E402
from coral.llm_policy import LLMPolicy                      # noqa: E402
from coral.memory import SharedMemory                       # noqa: E402
from coral.stats import evals_to_target, holm, paired_summary  # noqa: E402


class TaskTests(unittest.TestCase):
    def test_every_task_scores_random_candidates(self):
        rng = np.random.default_rng(0)
        for name in TASKS:
            task = make_task(name)
            for _ in range(5):
                score, fb = task.evaluate(task.random_solution(rng))
                self.assertTrue(np.isfinite(score))
                self.assertTrue(fb.tag and fb.valid)
                self.assertTrue(all(0 <= i < task.dim for i in fb.focus))

    def test_circle_packings_are_always_valid(self):
        task = make_task("circle_packing")
        rng = np.random.default_rng(1)
        for _ in range(20):
            self.assertTrue(task.verify(task.random_solution(rng)))

    def test_evaluation_is_deterministic(self):
        for name in TASKS:
            task = make_task(name)
            x = task.random_solution(np.random.default_rng(2))
            self.assertEqual(task.evaluate(x)[0], make_task(name).evaluate(x)[0])


class EvaluatorTests(unittest.TestCase):
    def test_budget_is_enforced(self):
        task = make_task("system_tuning")
        ev = Evaluator(task, budget=3)
        for _ in range(3):
            ev(np.full(task.dim, 0.5))
        with self.assertRaises(BudgetExhausted):
            ev(np.full(task.dim, 0.5))

    def test_bad_candidates_fail_without_crashing(self):
        task = make_task("system_tuning")
        ev = Evaluator(task, budget=3)
        for bad in (np.full(task.dim, 2.0), np.full(task.dim, np.nan), np.zeros(3)):
            score, fb, _ = ev(bad)
            self.assertEqual(score, task.fail_score)
            self.assertFalse(fb.valid)
        self.assertEqual(ev.invalid, 3)


class RunTests(unittest.TestCase):
    def test_all_conditions_spend_exactly_the_budget(self):
        for name in CONFIGS:
            out = run("system_tuning", name, seed=0, budget=150)
            self.assertEqual(len(out["curve"]), 150, name)
            self.assertTrue(np.all(np.diff(out["curve"]) >= 0), name)

    def test_same_seed_reproduces_the_run(self):
        for name in ("fixed_evo", "single_agent", "multi_agent"):
            a = run("circle_packing", name, seed=4, budget=120)
            b = run("circle_packing", name, seed=4, budget=120)
            np.testing.assert_array_equal(a["curve"], b["curve"])
            c = run("circle_packing", name, seed=5, budget=120)
            self.assertFalse(np.array_equal(a["curve"], c["curve"]))

    def test_mechanisms_fire_and_ablations_switch_them_off(self):
        full = run("system_tuning", "multi_agent", 0, 600)["summary"]
        self.assertGreater(full["n_notes"], 0)
        self.assertGreater(full["n_skills"], 0)
        self.assertGreater(full["skill_uses"], 0)
        self.assertGreater(full["cross_agent_uses"], 0)
        self.assertGreater(full["heartbeats"], 0)
        self.assertEqual(run("system_tuning", "abl_no_reflection", 0, 600)["summary"]["n_notes"], 0)
        self.assertEqual(run("system_tuning", "abl_no_consolidation", 0, 600)["summary"]["n_skills"], 0)
        self.assertEqual(run("system_tuning", "abl_no_redirection", 0, 600)["summary"]["redirections"], 0)
        self.assertEqual(run("system_tuning", "abl_no_shared_memory", 0, 600)["summary"]["cross_agent_uses"], 0)

    def test_memory_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = run("system_tuning", "multi_agent", 1, 300, persist_dir=tmp)
            mem = SharedMemory.load(Path(tmp) / "memory")
            self.assertEqual(len(mem.attempts), 300)
            self.assertAlmostEqual(mem.top_attempts(1)[0]["score"], out["summary"]["final_best"])
            self.assertEqual(mem.n_skills(), out["summary"]["n_skills"])
            for i in range(4):
                self.assertTrue((Path(tmp) / "workspace" / f"agent_{i}" / "context.json").exists())


class FakeClient:
    """Stands in for anthropic.Anthropic so the LLM policy can be tested offline."""

    def __init__(self, reply):
        self.reply, self.calls = reply, 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        self.last = kwargs
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=self.reply)],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


class LLMPolicyTests(unittest.TestCase):
    def test_model_decisions_are_followed(self):
        client = FakeClient('{"parent": "current", "operator": "targeted", "scale": 0.5, "rationale": "x"}')
        policy = LLMPolicy(client=client)
        out = run("system_tuning", "single_agent", 0, 60, policy=policy)
        ops = {r["op"] for r in out["rows"]} - {"init", "redirect"}
        self.assertEqual(ops, {"targeted"})
        self.assertGreater(client.calls, 0)
        self.assertEqual(out["summary"]["llm_input_tokens"], 100 * client.calls)
        self.assertIn("Evaluator feedback", client.last["messages"][0]["content"])

    def test_unusable_reply_falls_back_to_the_heuristic(self):
        policy = LLMPolicy(client=FakeClient("I would try something else."))
        out = run("system_tuning", "single_agent", 0, 60, policy=policy)
        self.assertEqual(len(out["curve"]), 60)
        self.assertGreater(policy.llm_errors, 0)


class StatsTests(unittest.TestCase):
    def test_paired_summary(self):
        a, b = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), np.array([1.0, 2.0, 3.5, 4.0, 5.0, 6.5])
        s = paired_summary(a, b)
        self.assertAlmostEqual(s["mean_diff"], (a - b).mean())
        self.assertEqual((s["wins"], s["losses"]), (6, 0))
        self.assertAlmostEqual(s["p_value"], 2 / 64)      # exact: only the all-plus / all-minus flips
        self.assertLess(s["ci_low"], s["mean_diff"])
        self.assertEqual(paired_summary(a, a)["p_value"], 1.0)

    def test_holm_and_targets(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(evals_to_target([1, 2, 3, 3], 3), (3, True))
        self.assertEqual(evals_to_target([1, 2, 2, 2], 3), (4, False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
