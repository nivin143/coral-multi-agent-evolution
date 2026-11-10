"""CORAL-inspired autonomous multi-agent evolution: an experimental platform."""
from .config import ABLATIONS, CONFIGS, MAIN, Config
from .runner import run
from .tasks import TASKS, make_task

__all__ = ["run", "CONFIGS", "MAIN", "ABLATIONS", "Config", "TASKS", "make_task"]
