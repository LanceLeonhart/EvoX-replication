"""Prompt construction for strategy proposal (the outer loop).

When a window stagnates, the outer loop asks for a *new* strategy conditioned on
the population descriptor phi(D), the current strategy, and which strategies
have already been tried. The mock client ignores the string and returns a
catalogued strategy, but a real client would consume this prompt.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import StrategyRequest


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
    lines.append("# Already tried (avoid repeating)")
    lines.append(json.dumps([list(s) for s in request.tried_signatures], default=str))
    lines.append("")
    lines.append("# Output")
    lines.append("Return a strategy JSON object matching the Strategy schema.")
    return "\n".join(lines)
