"""OperatorRegistry: the variation operators.

EvoX exposes a small, fixed vocabulary of *edit intents*:

  - ``local_refine``        — small exploitative change to the parent
  - ``structural_variation``— larger change, optionally combining inspirations
  - ``free_form``           — unconstrained / restart-style exploration

An ``Operator`` here is just a named intent plus a description. It does not
contain the candidate-mutation logic itself: it builds a solution prompt and
asks the LLM client to realise the edit. In V0 the client is a mock that
produces the concrete candidate; in a later version the same operators drive a
real LLM. This keeps the operator set identical across mock and real runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

from ..llm.solution_prompts import build_solution_prompt

if TYPE_CHECKING:  # avoid import cycle at runtime
    from ..llm.client import LLMClient, SolutionRequest


@dataclass
class Operator:
    name: str
    description: str

    def apply(self, request: "SolutionRequest", client: "LLMClient") -> Any:
        prompt = build_solution_prompt(request)
        response = client.generate_solution(request, prompt)
        return response.candidate


DEFAULT_OPERATORS = {
    "local_refine": Operator(
        "local_refine",
        "Make a small, exploitative change to the parent candidate.",
    ),
    "structural_variation": Operator(
        "structural_variation",
        "Make a larger change, optionally recombining inspiration candidates.",
    ),
    "free_form": Operator(
        "free_form",
        "Explore freely: restart-style, unconstrained candidate generation.",
    ),
}


class OperatorRegistry:
    def __init__(self, operators: Dict[str, Operator] = None) -> None:
        self._ops: Dict[str, Operator] = dict(operators or DEFAULT_OPERATORS)

    def names(self) -> List[str]:
        return list(self._ops.keys())

    def get(self, name: str) -> Operator:
        if name not in self._ops:
            raise KeyError(f"unknown operator {name!r}; known: {self.names()}")
        return self._ops[name]

    def register(self, operator: Operator) -> None:
        self._ops[operator.name] = operator
