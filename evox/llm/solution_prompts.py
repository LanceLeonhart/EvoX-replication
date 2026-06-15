"""Prompt construction for solution (candidate) generation.

The prompt encodes the operator intent, the task description, the parent, and
any inspirations. A real LLM client would send this string and parse the reply;
the V0 mock client realises the edit directly but the prompt is still built and
logged so the seam is exercised and ready for a real backend.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SolutionRequest


_OPERATOR_GUIDANCE = {
    "local_refine": "Make a SMALL, exploitative refinement of the parent candidate.",
    "structural_variation": "Make a LARGER structural change; you may recombine the inspirations.",
    "free_form": "Explore freely; you may ignore the parent and propose a fresh candidate.",
}


def _output_contract(schema: dict) -> str:
    """Explicit, parseable output format derived from the candidate schema."""
    if schema.get("kind") == "vector":
        dim = schema.get("dim")
        low = schema.get("low")
        high = schema.get("high")
        return (
            f"Return ONLY a JSON array of exactly {dim} floats, each within "
            f"[{low}, {high}]. No prose, no markdown, no keys — just the array, "
            "e.g. [0.1, -2.3, ...]."
        )
    return "Return ONLY the candidate in the task's representation, with no extra prose."


def build_solution_prompt(request: "SolutionRequest") -> str:
    schema = request.candidate_schema or {}
    lines = []
    lines.append("# Task")
    lines.append(request.task_prompt)
    lines.append("")
    lines.append("# Operator")
    lines.append(f"{request.operator}: {_OPERATOR_GUIDANCE.get(request.operator, '')}")
    lines.append(f"exploration={request.exploration:.3f}")
    lines.append("")
    lines.append("# Candidate schema")
    lines.append(json.dumps(schema, default=str) if schema else "(unspecified)")
    lines.append("")
    lines.append("# Parent candidate")
    lines.append(repr(request.parent_candidate))
    if request.inspirations:
        lines.append("")
        lines.append("# Inspiration candidates")
        for i, insp in enumerate(request.inspirations):
            lines.append(f"[{i}] {insp!r}")
    lines.append("")
    lines.append("# Output contract")
    lines.append(_output_contract(schema))
    return "\n".join(lines)
