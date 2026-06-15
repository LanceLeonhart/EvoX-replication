"""Parsers for LLM replies.

Used by the (future) real client to turn raw model text into structured
candidates / strategies. The mock client returns structured objects directly,
but these parsers are kept tested so the real path is ready and the JSON contract
is pinned down.
"""

from __future__ import annotations

import json
import re
from typing import Any, List

from ..core.strategy import Strategy


def _extract_json(text: str) -> str:
    """Pull the first JSON object/array out of a possibly chatty reply."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = min(
        [i for i in (text.find("{"), text.find("[")) if i != -1] or [-1]
    )
    if start == -1:
        raise ValueError("no JSON found in text")
    return text[start:].strip()


def parse_solution_vector(text: str) -> List[float]:
    """Parse a JSON list of floats (the V0 candidate representation)."""
    data = json.loads(_extract_json(text))
    if isinstance(data, dict) and "candidate" in data:
        data = data["candidate"]
    if not isinstance(data, list):
        raise ValueError("expected a JSON list for a vector candidate")
    return [float(x) for x in data]


def parse_strategy(text: str) -> Strategy:
    data = json.loads(_extract_json(text))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object for a strategy")
    return Strategy.from_dict(data)
