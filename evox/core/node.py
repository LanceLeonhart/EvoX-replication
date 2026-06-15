"""Node: a single candidate solution in the population database D.

A Node is the atomic unit stored in ``PopulationDB``. It records the candidate
itself, where it came from (parent + inspirations), which strategy/operator
produced it, and how it scored. Nodes are append-only and never mutated after
creation, which is what lets the population grow monotonically across strategy
switches (EvoX never resets D).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    id: int
    parent_id: Optional[int]
    iteration: int
    strategy_id: str
    operator: str
    candidate: Any
    score: float           # raw task score (direction depends on is_maximization)
    fitness: float         # normalized so that higher is always better
    valid: bool
    inspiration_ids: List[int] = field(default_factory=list)
    feedback: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        return cls(**d)
