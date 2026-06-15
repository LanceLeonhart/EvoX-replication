"""PopulationDB: the solution database D.

Append-only store of :class:`Node` objects. The engine adds nodes as they are
produced; nothing is ever removed and the database is *never reset* when the
search strategy changes. The interpreter queries it (best / top-k / recent /
random pool) to make parent and inspiration decisions, so the query surface
here is intentionally generic and strategy-agnostic.
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List, Optional

from .node import Node


class PopulationDB:
    def __init__(self) -> None:
        self._nodes: List[Node] = []
        self._by_id: Dict[int, Node] = {}
        self._next_id: int = 0

    # ── construction ──────────────────────────────────────────────────────
    def new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add(self, node: Node) -> Node:
        if node.id in self._by_id:
            raise ValueError(f"duplicate node id {node.id}")
        self._nodes.append(node)
        self._by_id[node.id] = node
        return node

    # ── basic access ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def size(self) -> int:
        return len(self._nodes)

    def all(self) -> List[Node]:
        return list(self._nodes)

    def get(self, node_id: int) -> Node:
        return self._by_id[node_id]

    def valid_nodes(self) -> List[Node]:
        return [n for n in self._nodes if n.valid]

    # ── query surface used by the interpreter ─────────────────────────────
    def best(self) -> Optional[Node]:
        pool = self.valid_nodes() or self._nodes
        if not pool:
            return None
        return max(pool, key=lambda n: n.fitness)

    def best_fitness(self) -> float:
        b = self.best()
        return b.fitness if b is not None else float("-inf")

    def top_k(self, k: int) -> List[Node]:
        pool = self.valid_nodes() or self._nodes
        return sorted(pool, key=lambda n: n.fitness, reverse=True)[:k]

    def recent(self, k: int) -> List[Node]:
        return self._nodes[-k:] if k > 0 else []

    def random_node(self, rng: random.Random) -> Node:
        pool = self.valid_nodes() or self._nodes
        return rng.choice(pool)

    def tournament(self, rng: random.Random, k: int = 3) -> Node:
        pool = self.valid_nodes() or self._nodes
        k = min(k, len(pool))
        contenders = rng.sample(pool, k)
        return max(contenders, key=lambda n: n.fitness)

    # ── statistics ────────────────────────────────────────────────────────
    def fitness_values(self) -> List[float]:
        return [n.fitness for n in self.valid_nodes()]

    def mean_fitness(self) -> float:
        vals = self.fitness_values()
        return statistics.fmean(vals) if vals else 0.0
