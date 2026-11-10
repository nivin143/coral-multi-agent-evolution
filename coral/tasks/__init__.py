from .base import Feedback, Task
from .bin_packing import OnlineBinPacking
from .circle_packing import CirclePacking
from .system_tuning import SystemTuning

TASKS = {
    "circle_packing": CirclePacking,
    "bin_packing": OnlineBinPacking,
    "system_tuning": SystemTuning,
}


def make_task(name: str) -> Task:
    return TASKS[name]()


__all__ = ["Feedback", "Task", "TASKS", "make_task", "CirclePacking", "OnlineBinPacking", "SystemTuning"]
