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

    def summary(self) -> List[Dict[str, Any]]:
        """Compact, JSON-serialisable view of H = {(S, phi, J)} for prompting.

        Each item carries the strategy (id/name + full compact dict), the score
        ``J`` and improvement ``delta`` it earned, the window bounds, and the
        population descriptor observed at window close (the state ``phi(D)``
        *after* the window). This is what lets a strategy proposer reason about
        which strategies worked and under what population state.
        """
        items: List[Dict[str, Any]] = []
        for e in self._entries:
            items.append(
                {
                    "window_index": e.window_index,
                    "strategy_id": e.strategy.id,
                    "strategy_name": e.strategy.name,
                    "strategy": e.strategy.to_dict(),
                    "J": e.J,
                    "delta": e.delta,
                    "s_start": e.s_start,
                    "s_end": e.s_end,
                    "descriptor": e.descriptor,
                }
            )
        return items

    def best_by_J(self) -> Optional[HistoryEntry]:
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.J)
