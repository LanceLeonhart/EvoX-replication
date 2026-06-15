"""StrategyHistory: the history H = {(S, phi, J)}.

Every closed window contributes one entry binding the strategy that was active,
the population descriptor phi(D) observed, and the score J that strategy earned.
H is the memory the outer loop uses to (a) avoid re-trying strategies that did
not help and (b) — in the real system — give the strategy proposer evidence
about what has and hasn't worked.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from .strategy import Strategy


@dataclass
class HistoryEntry:
    window_index: int
    strategy: Strategy
    descriptor: Dict[str, Any]
    s_start: float
    s_end: float
    delta: float
    J: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["strategy"] = self.strategy.to_dict()
        return d


class StrategyHistory:
    def __init__(self) -> None:
        self._entries: List[HistoryEntry] = []

    def add(self, entry: HistoryEntry) -> HistoryEntry:
        self._entries.append(entry)
        return entry

    def all(self) -> List[HistoryEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def last(self) -> Optional[HistoryEntry]:
        return self._entries[-1] if self._entries else None

    def tried_signatures(self) -> List[Tuple]:
        return [e.strategy.signature() for e in self._entries]

    def best_by_J(self) -> Optional[HistoryEntry]:
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.J)
