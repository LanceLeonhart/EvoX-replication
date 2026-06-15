"""Result containers shared by tasks, the engine, and the runner."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class EvalResult:
    """Outcome of evaluating a single candidate (returned by ``Task.evaluate``)."""

    score: float                     # raw task score; direction set by is_maximization
    valid: bool = True
    feedback: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    """Aggregate outcome of one full engine run (written to summary.json)."""

    task: str
    budget_T: int
    window_W: int
    tau: float
    seed: int
    iterations: int
    population_size: int
    best_score: float
    best_fitness: float
    best_node_id: int
    num_windows: int
    num_strategy_switches: int
    strategy_ids: list
    output_dir: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
