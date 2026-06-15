"""ToyBlackBoxTask: the single concrete V0 task.

A deterministic, dependency-free black-box optimisation problem. The candidate
is a fixed-length vector of floats; the objective is a shifted sphere
(sum of squared distance to a target), which is minimised. We expose it to the
engine as a *maximisation* of a bounded fitness ``1 / (1 + sphere)`` in (0, 1],
which keeps fitness non-negative — convenient for the ``log(1 + s_start)`` term
in the strategy score.

This task is intentionally tiny and deterministic so that smoke runs are
reproducible and need no API keys. Its representation (a numeric vector) is what
the mock generator knows how to mutate via ``candidate_schema``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from ..eval.result import EvalResult
from .base import Task


class ToyBlackBoxTask(Task):
    is_maximization = True

    def __init__(
        self,
        dim: int = 6,
        low: float = -5.12,
        high: float = 5.12,
        target: float = 0.0,
    ) -> None:
        self.name = "toy_blackbox"
        self.dim = int(dim)
        self.low = float(low)
        self.high = float(high)
        self.target = float(target)

    # ── candidate space ───────────────────────────────────────────────────
    def initial_candidate(self) -> List[float]:
        # Deterministic seed point placed far from the optimum so there is real
        # room to improve (and to stagnate) over a run.
        start = self.low + 0.9 * (self.high - self.low)
        return [start for _ in range(self.dim)]

    def candidate_schema(self) -> Dict[str, Any]:
        return {
            "kind": "vector",
            "dim": self.dim,
            "low": self.low,
            "high": self.high,
        }

    # ── evaluation ────────────────────────────────────────────────────────
    def _sphere(self, x: List[float]) -> float:
        return sum((float(xi) - self.target) ** 2 for xi in x)

    def evaluate(self, candidate: Any) -> EvalResult:
        if not isinstance(candidate, (list, tuple)) or len(candidate) != self.dim:
            return EvalResult(score=0.0, valid=False, feedback="candidate has wrong shape")
        try:
            x = [float(xi) for xi in candidate]
        except (TypeError, ValueError):
            return EvalResult(score=0.0, valid=False, feedback="candidate not numeric")
        if any(not math.isfinite(xi) for xi in x):
            return EvalResult(score=0.0, valid=False, feedback="candidate has non-finite values")
        in_bounds = all(self.low <= xi <= self.high for xi in x)
        sphere = self._sphere(x)
        score = 1.0 / (1.0 + sphere)
        return EvalResult(
            score=score,
            valid=in_bounds,
            feedback=("" if in_bounds else "candidate outside bounds"),
            metrics={"sphere": sphere, "in_bounds": in_bounds},
        )

    def render_task_prompt(self) -> str:
        return (
            "Minimise a shifted sphere objective over a real vector.\n"
            f"- dimension: {self.dim}\n"
            f"- each coordinate in [{self.low}, {self.high}]\n"
            f"- objective: sum((x_i - {self.target})^2), lower is better\n"
            "Return a vector of floats of the given dimension."
        )

    def metrics(self) -> Dict[str, Any]:
        return {"dim": self.dim, "low": self.low, "high": self.high, "target": self.target}
