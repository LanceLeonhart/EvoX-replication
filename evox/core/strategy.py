"""Strategy: the active search strategy S.

EvoX separates *what to evolve* (the population D) from *how to evolve it* (the
strategy S). ``Strategy`` is the JSON/dataclass schema for S. The outer loop
proposes new strategies; ``VALID`` gates them before they are allowed to become
active. The interpreter (see ``strategy_interpreter.py``) is the only component
that reads these fields to make decisions, so the engine stays agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple

PARENT_SELECTION_MODES = ("best", "recent", "random", "tournament")
INSPIRATION_SELECTION_MODES = ("best", "recent", "random", "diverse")
KNOWN_OPERATORS = ("local_refine", "structural_variation", "free_form")


@dataclass
class Strategy:
    id: str
    name: str
    parent_selection: str
    operator_weights: Dict[str, float]
    num_inspirations: int
    inspiration_selection: str
    exploration: float                 # 0..1 scalar controlling mutation scale
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Strategy":
        return cls(
            id=d["id"],
            name=d["name"],
            parent_selection=d["parent_selection"],
            operator_weights=dict(d["operator_weights"]),
            num_inspirations=int(d["num_inspirations"]),
            inspiration_selection=d["inspiration_selection"],
            exploration=float(d["exploration"]),
            notes=d.get("notes", ""),
        )

    def signature(self) -> Tuple:
        """Structural fingerprint used to detect "different" strategies.

        Two strategies with the same signature behave identically even if their
        ``id``/``name`` differ, so switching to one would be pointless.
        """
        ow = tuple(sorted((k, round(float(v), 4)) for k, v in self.operator_weights.items()))
        return (
            self.parent_selection,
            ow,
            self.num_inspirations,
            self.inspiration_selection,
            round(float(self.exploration), 4),
        )


def validate_strategy(
    strategy: Strategy,
    known_operators: Tuple[str, ...] = KNOWN_OPERATORS,
) -> Tuple[bool, List[str]]:
    """Return ``(ok, reasons)``. ``reasons`` is empty when the strategy is valid."""
    reasons: List[str] = []

    if not strategy.id:
        reasons.append("id is empty")
    if not strategy.name:
        reasons.append("name is empty")

    if strategy.parent_selection not in PARENT_SELECTION_MODES:
        reasons.append(f"parent_selection {strategy.parent_selection!r} not in {PARENT_SELECTION_MODES}")
    if strategy.inspiration_selection not in INSPIRATION_SELECTION_MODES:
        reasons.append(
            f"inspiration_selection {strategy.inspiration_selection!r} not in {INSPIRATION_SELECTION_MODES}"
        )

    if not isinstance(strategy.operator_weights, dict) or not strategy.operator_weights:
        reasons.append("operator_weights must be a non-empty mapping")
    else:
        for op, w in strategy.operator_weights.items():
            if op not in known_operators:
                reasons.append(f"unknown operator {op!r}")
            if not isinstance(w, (int, float)) or w < 0:
                reasons.append(f"operator weight for {op!r} must be >= 0")
        if sum(float(w) for w in strategy.operator_weights.values()) <= 0:
            reasons.append("operator weights sum to 0 (no operator can ever fire)")

    if strategy.num_inspirations < 0:
        reasons.append("num_inspirations must be >= 0")
    if not (0.0 <= strategy.exploration <= 1.0):
        reasons.append("exploration must be in [0, 1]")

    return (len(reasons) == 0, reasons)


def VALID(strategy: Strategy, known_operators: Tuple[str, ...] = KNOWN_OPERATORS) -> bool:
    """Boolean gate matching the paper's ``VALID(S)`` predicate."""
    ok, _ = validate_strategy(strategy, known_operators)
    return ok
