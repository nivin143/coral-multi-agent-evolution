"""System optimisation task: tune the configuration of a simulated request-serving system.

The system is an analytic model (not a real service) with ten interacting knobs, integer
plateaus, a memory budget, a latency objective and a running cost. The evaluator reports the
net goodput and names the binding bottleneck, which is the kind of feedback a benchmark
harness for a real system would return.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Feedback, Task

KNOBS = ["workers", "batch", "cache_gb", "queue", "timeout_s", "compress", "pool", "gc_threshold",
         "prefetch", "replicas"]

MEM_LIMIT_GB = 16.0
LATENCY_SLO_S = 0.30


class SystemTuning(Task):
    name = "system_tuning"
    domain = "system optimisation"
    fail_score = -10.0

    def __init__(self):
        self.dim = len(KNOBS)
        self.groups = [[0, 9], [1, 3], [2, 8], [4], [5], [6], [7]]
        self.description = (
            "Tune a request-serving system to maximise net goodput (requests/s served within the "
            "latency objective, minus running cost). Knobs: " + ", ".join(KNOBS) + ". "
            f"Memory limit {MEM_LIMIT_GB:.0f} GB per replica, p99 latency objective "
            f"{LATENCY_SLO_S:.2f} s."
        )

    @staticmethod
    def decode(x: np.ndarray) -> dict:
        return {
            "workers": 1 + int(round(63 * x[0])),
            "batch": 1 + int(round(127 * x[1])),
            "cache_gb": round(16 * float(x[2]), 1),
            "queue": 10 + 10 * int(round(99 * x[3])),
            "timeout_s": round(0.05 + 1.95 * float(x[4]), 2),
            "compress": int(round(9 * x[5])),
            "pool": 1 + int(round(199 * x[6])),
            "gc_threshold": round(0.30 + 0.65 * float(x[7]), 2),
            "prefetch": round(float(x[8]), 2),
            "replicas": 1 + int(round(7 * x[9])),
        }

    def key(self, x: np.ndarray) -> tuple:
        return tuple(self.decode(x).values())

    @staticmethod
    def simulate(c: dict) -> dict:
        W, B, R, P, G = c["workers"], c["batch"], c["replicas"], c["pool"], c["gc_threshold"]
        hit = min(0.97, 1 - math.exp(-0.35 * c["cache_gb"] * (1 + 0.6 * c["prefetch"])))
        cpu_per_req = 1.0 + 0.12 * c["compress"] + 1.5 * (1 - hit) + 0.4 * c["prefetch"]
        eff_workers = max(1.0, min(W, 16) - 0.015 * max(0, W - 16) ** 2)   # 16 cores, then contention
        batch_gain = B ** 0.3 / (1 + 0.5 * (B / 64) ** 2)
        gc_eff = 1 - 0.6 * (0.95 - G)                                      # low threshold = frequent GC
        cpu = R * 420 * eff_workers * batch_gain * gc_eff / cpu_per_req
        io = 55 * min(P, 120) * (1 - 0.004 * max(0, P - 120)) * (1 + 0.05 * c["compress"]) / (1 - hit + 0.04)
        net = R * 2600 * (1 + 0.22 * c["compress"])
        capacity = min(cpu, io, net)

        mem = c["cache_gb"] + 0.12 * W + 2e-5 * c["queue"] * B + 0.008 * P
        oom = math.exp(-2.0 * max(0.0, mem - MEM_LIMIT_GB))
        gc_pause = 0.02 + 0.5 * max(0.0, G - 0.6) * min(mem, MEM_LIMIT_GB) / MEM_LIMIT_GB
        p99 = 0.01 + 0.0015 * B + 0.5 * c["queue"] / max(capacity, 1.0) + gc_pause + 0.12 * (1 - hit)
        deadline = min(c["timeout_s"], LATENCY_SLO_S)
        on_time = 1.0 if p99 <= deadline else math.exp(-6.0 * (p99 - deadline) / deadline)
        drop = 0.3 * max(0.0, 1 - c["queue"] / (6.0 * B))
        goodput = capacity * on_time * (1 - drop) * oom
        cost = 900 * R + 12 * W * R
        return dict(cpu=cpu, io=io, net=net, capacity=capacity, mem=mem, oom=oom, p99=p99,
                    deadline=deadline, drop=drop, goodput=goodput, cost=cost, hit=hit)

    def evaluate(self, x: np.ndarray):
        c = self.decode(x)
        m = self.simulate(c)
        score = (m["goodput"] - m["cost"]) / 10000.0
        if m["oom"] < 0.99:
            tag, focus = "mem_over", [2, 0, 3, 1]
        elif m["p99"] > m["deadline"]:
            tag, focus = "latency_slo", [1, 3, 7, 4]
        elif m["drop"] > 0.02:
            tag, focus = "queue_drop", [3, 1]
        else:
            name = min(("cpu", "io", "net"), key=lambda k: m[k])
            tag = name + "_bound"
            focus = {"cpu": [0, 1, 5, 7, 9], "io": [6, 2, 8, 5], "net": [5, 9]}[name]
        msg = (
            f"goodput {m['goodput']:.0f} req/s, cost {m['cost']:.0f}; capacity cpu={m['cpu']:.0f} "
            f"io={m['io']:.0f} net={m['net']:.0f}; memory {m['mem']:.1f}/{MEM_LIMIT_GB:.0f} GB; "
            f"p99 {m['p99']:.3f}s vs deadline {m['deadline']:.2f}s; dropped {m['drop']:.0%}; "
            f"cache hit {m['hit']:.0%}; config {c}"
        )
        return float(score), Feedback(tag, focus, msg)
