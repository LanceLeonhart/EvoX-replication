"""LLM client abstraction + the V0 mock implementation.

Both solution generation (inner loop) and strategy proposal (outer loop) go
through ``LLMClient``. ``MockLLMClient`` implements them deterministically and
offline:

  - ``generate_solution`` mutates a numeric-vector candidate according to the
    operator intent (small / structural / free-form), reading the dimension and
    bounds from the task's ``candidate_schema``.
  - ``propose_strategy`` returns a *different valid* strategy from a fixed
    catalogue, preferring one not yet tried — this is what drives a strategy
    switch after stagnation.

A real client (OpenAI/etc.) would implement the same two methods using the
prompts and parsers, with no change required to the engine or operators.
"""

from __future__ import annotations

import abc
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.strategy import Strategy
from ..logging.usage import UsageTracker


# ── request / response containers ─────────────────────────────────────────
@dataclass
class SolutionRequest:
    operator: str
    parent_candidate: Any
    inspirations: List[Any]
    task_prompt: str
    candidate_schema: Dict[str, Any]
    exploration: float
    strategy_id: str
    seed: int


@dataclass
class SolutionResponse:
    candidate: Any
    raw_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StrategyRequest:
    current_strategy: Strategy
    descriptor: Dict[str, Any]
    tried_signatures: List[Tuple]
    seed: int


@dataclass
class StrategyResponse:
    strategy: Strategy
    raw_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ── abstract interface ────────────────────────────────────────────────────
class LLMClient(abc.ABC):
    @abc.abstractmethod
    def generate_solution(self, request: SolutionRequest, prompt: str) -> SolutionResponse:
        ...

    @abc.abstractmethod
    def propose_strategy(self, request: StrategyRequest, prompt: str) -> StrategyResponse:
        ...


# ── strategy catalogue used by the mock proposer ──────────────────────────
def default_strategy_catalog() -> List[Strategy]:
    """A small set of distinct, valid strategies the mock proposer cycles through.

    They span the exploit -> explore axis so that switching after stagnation
    meaningfully changes behaviour.
    """
    return [
        Strategy(
            id="exploit_local",
            name="Exploit (local refine, best parent)",
            parent_selection="best",
            operator_weights={"local_refine": 0.85, "structural_variation": 0.12, "free_form": 0.03},
            num_inspirations=1,
            inspiration_selection="best",
            exploration=0.15,
            notes="Tight exploitation around the incumbent best.",
        ),
        Strategy(
            id="recombine",
            name="Recombine (structural variation, diverse inspirations)",
            parent_selection="tournament",
            operator_weights={"local_refine": 0.3, "structural_variation": 0.6, "free_form": 0.1},
            num_inspirations=2,
            inspiration_selection="diverse",
            exploration=0.45,
            notes="Mix promising candidates to find new basins.",
        ),
        Strategy(
            id="explore_global",
            name="Explore (free-form restarts, random parents)",
            parent_selection="random",
            operator_weights={"local_refine": 0.2, "structural_variation": 0.3, "free_form": 0.5},
            num_inspirations=0,
            inspiration_selection="random",
            exploration=0.9,
            notes="Heavy exploration to escape plateaus.",
        ),
    ]


# ── mock client ───────────────────────────────────────────────────────────
class MockLLMClient(LLMClient):
    def __init__(
        self,
        usage: Optional[UsageTracker] = None,
        catalog: Optional[List[Strategy]] = None,
    ) -> None:
        self.usage = usage or UsageTracker()
        self.catalog = catalog or default_strategy_catalog()

    # solution generation -------------------------------------------------
    def generate_solution(self, request: SolutionRequest, prompt: str) -> SolutionResponse:
        rng = random.Random(request.seed)
        candidate = self._mutate_vector(request, rng)

        ptoks = UsageTracker.estimate_tokens(prompt)
        ctoks = UsageTracker.estimate_tokens(repr(candidate))
        self.usage.record("generate_solution", prompt_tokens=ptoks, completion_tokens=ctoks)
        return SolutionResponse(
            candidate=candidate,
            raw_text=repr(candidate),
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
        )

    def _mutate_vector(self, request: SolutionRequest, rng: random.Random) -> Any:
        schema = request.candidate_schema or {}
        if schema.get("kind") != "vector":
            # mock only understands vectors; leave non-vector candidates untouched
            return request.parent_candidate

        dim = int(schema["dim"])
        low = float(schema["low"])
        high = float(schema["high"])
        span = high - low
        explore = float(request.exploration)
        parent = [float(v) for v in request.parent_candidate]

        def clip(v: float) -> float:
            return min(high, max(low, v))

        op = request.operator
        if op == "free_form":
            return [rng.uniform(low, high) for _ in range(dim)]

        if op == "structural_variation":
            insps = [list(map(float, c)) for c in request.inspirations if len(c) == dim]
            scale = span * 0.10 * (0.5 + explore)
            child = []
            for i in range(dim):
                base = parent[i]
                if insps and rng.random() < 0.5:
                    base = rng.choice(insps)[i]
                child.append(clip(base + rng.gauss(0.0, scale)))
            return child

        # default: local_refine
        scale = span * 0.03 * (0.5 + explore)
        return [clip(parent[i] + rng.gauss(0.0, scale)) for i in range(dim)]

    # strategy proposal ---------------------------------------------------
    def propose_strategy(self, request: StrategyRequest, prompt: str) -> StrategyResponse:
        rng = random.Random(request.seed)
        current_sig = request.current_strategy.signature()
        tried = set(map(tuple, request.tried_signatures))

        # prefer a catalogued strategy that differs from current AND is untried
        untried = [
            s for s in self.catalog
            if s.signature() != current_sig and tuple(s.signature()) not in tried
        ]
        different = [s for s in self.catalog if s.signature() != current_sig]
        pool = untried or different or self.catalog
        chosen = rng.choice(pool)

        ptoks = UsageTracker.estimate_tokens(prompt)
        ctoks = UsageTracker.estimate_tokens(repr(chosen.to_dict()))
        self.usage.record("propose_strategy", prompt_tokens=ptoks, completion_tokens=ctoks)
        return StrategyResponse(
            strategy=chosen,
            raw_text=repr(chosen.to_dict()),
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
        )
