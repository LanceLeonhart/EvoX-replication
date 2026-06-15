"""StrategyInterpreter: maps a Strategy S to concrete search decisions.

This is the indirection that keeps the engine honest. The engine never decides
*which* parent, *which* inspirations, or *which* operator to use — it asks the
interpreter, which reads those decisions out of the active strategy S. Swapping
S therefore changes search behaviour without touching the engine.
"""

from __future__ import annotations

import random
from typing import List

from .node import Node
from .population import PopulationDB
from .strategy import Strategy


class StrategyInterpreter:
    def select_parent(self, db: PopulationDB, strategy: Strategy, rng: random.Random) -> Node:
        mode = strategy.parent_selection
        if mode == "best":
            return db.best()
        if mode == "recent":
            return db.recent(1)[0]
        if mode == "random":
            return db.random_node(rng)
        if mode == "tournament":
            return db.tournament(rng)
        raise ValueError(f"unknown parent_selection {mode!r}")

    def select_operator(self, strategy: Strategy, rng: random.Random) -> str:
        ops = list(strategy.operator_weights.items())
        names = [o for o, _ in ops]
        weights = [float(w) for _, w in ops]
        return rng.choices(names, weights=weights, k=1)[0]

    def select_inspirations(
        self,
        db: PopulationDB,
        strategy: Strategy,
        parent: Node,
        rng: random.Random,
    ) -> List[Node]:
        n = strategy.num_inspirations
        if n <= 0:
            return []
        pool = [node for node in (db.valid_nodes() or db.all()) if node.id != parent.id]
        if not pool:
            return []
        mode = strategy.inspiration_selection
        if mode == "best":
            chosen = sorted(pool, key=lambda x: x.fitness, reverse=True)[:n]
        elif mode == "recent":
            chosen = pool[-n:]
        elif mode == "random":
            chosen = rng.sample(pool, min(n, len(pool)))
        elif mode == "diverse":
            chosen = self._diverse(pool, parent, n)
        else:
            raise ValueError(f"unknown inspiration_selection {mode!r}")
        return chosen

    def _diverse(self, pool: List[Node], parent: Node, n: int) -> List[Node]:
        """Greedy fitness-spread proxy: pick nodes whose fitness is farthest
        from already-selected ones (cheap, representation-agnostic stand-in for
        embedding diversity used in the real system)."""
        selected: List[Node] = []
        remaining = list(pool)
        refs = [parent.fitness]
        while remaining and len(selected) < n:
            nxt = max(remaining, key=lambda x: min(abs(x.fitness - r) for r in refs))
            selected.append(nxt)
            refs.append(nxt.fitness)
            remaining.remove(nxt)
        return selected
