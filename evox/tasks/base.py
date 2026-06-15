"""Task: the abstract interface every benchmark task implements.

The engine talks to tasks *only* through this interface, so new tasks can be
added (the paper evaluates five) without touching the two-level loop. A task
owns its candidate representation, its objective, and how it describes itself to
the (mock or real) solution generator.

``candidate_schema`` is an optional hook: it lets the mock generator produce
representation-appropriate candidates without the engine knowing the
representation. A real LLM-backed generator ignores it and uses
``render_task_prompt`` instead.
"""

from __future__ import annotations

import abc
from typing import Any, Dict

from ..eval.result import EvalResult


class Task(abc.ABC):
    #: human-readable, registry-unique task name
    name: str = "task"

    #: True if larger raw scores are better
    is_maximization: bool = True

    @abc.abstractmethod
    def initial_candidate(self) -> Any:
        """Return the seed candidate used to initialise the population."""

    @abc.abstractmethod
    def evaluate(self, candidate: Any) -> EvalResult:
        """Score a candidate."""

    @abc.abstractmethod
    def render_task_prompt(self) -> str:
        """Natural-language description handed to the solution generator."""

    def candidate_schema(self) -> Dict[str, Any]:
        """Optional structured description of the candidate space (mock hook)."""
        return {}

    def metrics(self) -> Dict[str, Any]:
        """Optional static task metrics for logging."""
        return {}

    def artifacts(self) -> Dict[str, Any]:
        """Optional task artifacts (e.g. reference data paths)."""
        return {}
