"""Prompt construction for strategy proposal (the outer loop).

When a window stagnates, the outer loop asks for a *new* strategy conditioned on
the population descriptor phi(D), the current strategy, the scored strategy
history H = {(S, phi, J)}, and which strategies have already been tried. The
mock client ignores the string and returns a catalogued strategy, but a real
client would consume this prompt — so it also spells out the legal strategy
vocabulary and a strict JSON output contract.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..core.operators import DEFAULT_OPERATORS
from ..core.strategy import (
    INSPIRATION_SELECTION_MODES,
    KNOWN_OPERATORS,
    PARENT_SELECTION_MODES,
)

if TYPE_CHECKING:
    from .client import StrategyRequest


def _render_history(history_summary) -> str:
    if not history_summary:
        return "(no closed windows yet)"
    rows = []
    for h in history_summary:
        desc = h.get("descriptor", {}) or {}
        rows.append(
            "- window {w}: strategy={sid!r} J={J:.6f} delta={d:.6f} "
            "s_start={ss:.6f} s_end={se:.6f} "
            "phi(D)@close: size={size} best={best:.6f} diversity={div:.6f}".format(
                w=h.get("window_index"),
                sid=h.get("strategy_id"),
                J=float(h.get("J", 0.0)),
                d=float(h.get("delta", 0.0)),
                ss=float(h.get("s_start", 0.0)),
                se=float(h.get("s_end", 0.0)),
                size=desc.get("size"),
                best=float(desc.get("best_fitness", 0.0)),
                div=float(desc.get("diversity", 0.0)),
            )
        )
    return "\n".join(rows)


def build_strategy_prompt(request: "StrategyRequest") -> str:
    lines = []
    lines.append("# Goal")
    lines.append(
        "The search has stagnated under the current strategy. Propose a DIFFERENT, "
        "valid search strategy that is likely to escape the plateau."
    )
    lines.append("")
    lines.append("# Current strategy")
    lines.append(json.dumps(request.current_strategy.to_dict(), indent=2))
    lines.append("")
    lines.append("# Population descriptor phi(D)")
    lines.append(json.dumps(request.descriptor, indent=2, default=str))
    lines.append("")
    lines.append("# Strategy history (S, phi, J)")
    lines.append(
        "Past windows, the strategy that was active, the score J it earned, and "
        "the population state at window close. Prefer strategies that scored well "
        "in similar population states; avoid ones that stagnated."
    )
    lines.append(_render_history(request.history_summary))
    lines.append("")
    lines.append("# Already tried (avoid repeating)")
    lines.append(json.dumps([list(s) for s in request.tried_signatures], default=str))
    lines.append("")
    lines.append("# Legal strategy vocabulary")
    lines.append(f"- parent_selection (one of): {list(PARENT_SELECTION_MODES)}")
    lines.append(f"- inspiration_selection (one of): {list(INSPIRATION_SELECTION_MODES)}")
    lines.append(f"- operator names (weights keys, subset of): {list(KNOWN_OPERATORS)}")
    lines.append("  operator meanings:")
    for name in KNOWN_OPERATORS:
        op = DEFAULT_OPERATORS.get(name)
        if op is not None:
            lines.append(f"    - {name}: {op.description}")
    lines.append("- num_inspirations: integer >= 0")
    lines.append("- exploration: float in [0.0, 1.0]")
    lines.append("- operator_weights: non-empty mapping with weights >= 0 that sum to > 0")
    lines.append("")
    lines.append("# Output contract")
    lines.append(
        "Return ONLY a single JSON object matching the Strategy schema, with keys: "
        "id, name, parent_selection, operator_weights, num_inspirations, "
        "inspiration_selection, exploration, notes. No prose, no markdown outside "
        "the JSON. The object must use only the legal values listed above."
    )
    return "\n".join(lines)
