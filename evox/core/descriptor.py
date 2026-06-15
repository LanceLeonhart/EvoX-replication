"""PopulationDescriptor: structured phi(D).

A compact, serialisable summary of the population database. EvoX feeds phi(D)
to the outer loop (and, in the real system, to the LLM that proposes new
strategies) so that strategy decisions are conditioned on the *state* of the
search rather than raw node dumps. It is recorded alongside every strategy in
the history H.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from .population import PopulationDB


@dataclass
class PopulationDescriptor:
    size: int
    num_valid: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float
    diversity: float
    operator_counts: Dict[str, int] = field(default_factory=dict)
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    best_node_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def build(cls, db: PopulationDB) -> "PopulationDescriptor":
        nodes = db.all()
        valid = db.valid_nodes()
        fitnesses = [n.fitness for n in valid]

        best_node = db.best()
        if fitnesses:
            best_f = max(fitnesses)
            mean_f = statistics.fmean(fitnesses)
            median_f = statistics.median(fitnesses)
            std_f = statistics.pstdev(fitnesses) if len(fitnesses) > 1 else 0.0
            # diversity proxy: spread of fitness values (representation-agnostic)
            diversity = (max(fitnesses) - min(fitnesses)) if len(fitnesses) > 1 else 0.0
        else:
            best_f = mean_f = median_f = std_f = diversity = 0.0

        return cls(
            size=len(nodes),
            num_valid=len(valid),
            best_fitness=best_f,
            mean_fitness=mean_f,
            median_fitness=median_f,
            std_fitness=std_f,
            diversity=diversity,
            operator_counts=dict(Counter(n.operator for n in nodes)),
            strategy_counts=dict(Counter(n.strategy_id for n in nodes)),
            best_node_id=best_node.id if best_node else None,
        )
