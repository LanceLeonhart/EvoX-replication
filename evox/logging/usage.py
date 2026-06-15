"""Usage tracking.

Even though V0 uses a mock client with no real API cost, we account for "calls"
and (estimated) tokens through the same interface a real client would use. This
keeps the budget/usage plumbing in place for when a real LLM is dropped in.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UsageTracker:
    def __init__(self) -> None:
        self.total = Usage()
        self.by_kind: Dict[str, Usage] = {}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # crude word-based proxy; good enough for accounting in the mock path
        return max(1, len(text.split()))

    def record(
        self,
        kind: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        for bucket in (self.total, self.by_kind.setdefault(kind, Usage())):
            bucket.calls += 1
            bucket.prompt_tokens += prompt_tokens
            bucket.completion_tokens += completion_tokens
            bucket.cost_usd += cost_usd

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total.to_dict(),
            "by_kind": {k: v.to_dict() for k, v in self.by_kind.items()},
        }
