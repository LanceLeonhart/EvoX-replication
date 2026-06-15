"""Prompt construction for solution (candidate) generation.

The prompt encodes the operator intent, the task description, the parent, and
any inspirations. A real LLM client would send this string and parse the reply;
the V0 mock client realises the edit directly but the prompt is still built and
logged so the seam is exercised and ready for a real backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SolutionRequest


_OPERATOR_GUIDANCE = {
    "local_refine": "Make a SMALL, exploitative refinement of the parent candidate.",
    "structural_variation": "Make a LARGER structural change; you may recombine the inspirations.",
    "free_form": "Explore freely; you may ignore the parent and propose a fresh candidate.",
}


def build_solution_prompt(request: "SolutionRequest") -> str:
    lines = []
    lines.append("# Task")
    lines.append(request.task_prompt)
    lines.append("")
    lines.append("# Operator")
    lines.append(f"{request.operator}: {_OPERATOR_GUIDANCE.get(request.operator, '')}")
    lines.append(f"exploration={request.exploration:.3f}")
    lines.append("")
    lines.append("# Parent candidate")
    lines.append(repr(request.parent_candidate))
    if request.inspirations:
        lines.append("")
        lines.append("# Inspiration candidates")
        for i, insp in enumerate(request.inspirations):
            lines.append(f"[{i}] {insp!r}")
    lines.append("")
    lines.append("# Output")
    lines.append("Return a single candidate in the task's representation.")
    return "\n".join(lines)
