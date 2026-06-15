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
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.strategy import Strategy, validate_strategy
from ..logging.usage import UsageTracker
from .parsers import parse_solution_vector, parse_strategy


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
    model: str = "mock"
    parse_error: Optional[str] = None
    prompt: str = ""


@dataclass
class StrategyRequest:
    current_strategy: Strategy
    descriptor: Dict[str, Any]
    tried_signatures: List[Tuple]
    seed: int
    history_summary: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StrategyResponse:
    strategy: Strategy
    raw_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "mock"
    prompt: str = ""
    attempts: int = 1
    used_fallback: bool = False
    errors: List[str] = field(default_factory=list)


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


# ── OpenAI client (real inner-loop generation) ─────────────────────────────
def _extract_output_text(response: Any) -> str:
    """Best-effort extraction of text from a Responses API result."""
    txt = getattr(response, "output_text", None)
    if txt:
        return txt
    chunks: List[str] = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            t = getattr(part, "text", None)
            if t:
                chunks.append(t)
    return "".join(chunks)


class OpenAIClient(LLMClient):
    """Real generation via the OpenAI Responses API (default ``gpt-5-mini``).

    As of V2 both methods are real:

      - ``generate_solution`` (G_sol) — inner-loop candidate generation; parse
        failures mark the candidate invalid without crashing the run.
      - ``propose_strategy`` (G_str) — outer-loop strategy generation; the model
        returns one Strategy JSON which is parsed and validated, with configurable
        retries and a fall back to the mock/static proposer on repeated failure,
        so a malformed strategy never crashes the run.

    The OpenAI SDK is imported lazily and the API key is checked *before* the
    import, so a missing key fails with a clear message even when the SDK is not
    installed (e.g. in the test environment).
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        reasoning_effort: str = "minimal",
        verbosity: Optional[str] = "low",
        max_output_tokens: int = 512,
        strategy_retries: int = 1,
        usage: Optional[UsageTracker] = None,
        catalog: Optional[List[Strategy]] = None,
        api_key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.max_output_tokens = int(max_output_tokens)
        self.strategy_retries = max(0, int(strategy_retries))
        self.usage = usage or UsageTracker()
        # fallback proposer used when real strategy generation keeps failing
        self._strategy_fallback = MockLLMClient(usage=self.usage, catalog=catalog)

        if client is not None:
            self._client = client
            return

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set but an 'openai' llm mode was requested. "
                "Export OPENAI_API_KEY or use mock mode. "
                "(No silent fallback to mock.)"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "The 'openai' package is required for openai llm modes. "
                "Install it with: pip install 'evox-replication[openai]' "
                "(or: pip install openai)."
            ) from exc
        self._client = OpenAI(api_key=key)

    # shared request builder ----------------------------------------------
    def _build_kwargs(self, prompt: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.verbosity:
            kwargs["text"] = {"verbosity": self.verbosity}
        return kwargs

    def _call(self, prompt: str):
        """Call the Responses API and return (raw_text, prompt_tokens, completion_tokens)."""
        response = self._client.responses.create(**self._build_kwargs(prompt))
        raw = _extract_output_text(response)
        usage = getattr(response, "usage", None)
        ptoks = int(getattr(usage, "input_tokens", 0) or 0)
        ctoks = int(getattr(usage, "output_tokens", 0) or 0)
        return raw, ptoks, ctoks

    # solution generation -------------------------------------------------
    def generate_solution(self, request: SolutionRequest, prompt: str) -> SolutionResponse:
        # API/transport errors are intentionally NOT swallowed: a bad key or
        # quota problem should fail loudly rather than silently producing a run
        # of all-invalid candidates.
        raw, ptoks, ctoks = self._call(prompt)
        self.usage.record("generate_solution", prompt_tokens=ptoks, completion_tokens=ctoks)

        # Parse failures must NOT crash the run: mark invalid, keep the raw text
        # and the error reason so the engine can log them and continue.
        candidate, parse_error = self._parse_candidate(request, raw)
        return SolutionResponse(
            candidate=candidate,
            raw_text=raw,
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            model=self.model,
            parse_error=parse_error,
        )

    @staticmethod
    def _parse_candidate(request: SolutionRequest, raw: str):
        schema = request.candidate_schema or {}
        try:
            if schema.get("kind") == "vector":
                return parse_solution_vector(raw), None
            return raw, None
        except Exception as exc:  # parse failure -> invalid candidate
            return None, f"{type(exc).__name__}: {exc}"

    # strategy proposal (real in V2, with retry + mock fallback) ----------
    def propose_strategy(self, request: StrategyRequest, prompt: str) -> StrategyResponse:
        errors: List[str] = []
        last_raw = ""
        ptoks_total = 0
        ctoks_total = 0
        max_attempts = 1 + self.strategy_retries

        for attempt in range(1, max_attempts + 1):
            # API/transport errors propagate loudly (consistent with G_sol);
            # only malformed/invalid *content* is retried and falls back.
            raw, ptoks, ctoks = self._call(prompt)
            last_raw = raw
            ptoks_total += ptoks
            ctoks_total += ctoks
            self.usage.record("propose_strategy", prompt_tokens=ptoks, completion_tokens=ctoks)

            try:
                strategy = parse_strategy(raw)
            except Exception as exc:
                errors.append(f"attempt {attempt} parse: {type(exc).__name__}: {exc}")
                continue

            ok, reasons = validate_strategy(strategy)
            if not ok:
                errors.append(f"attempt {attempt} invalid: {reasons}")
                continue

            return StrategyResponse(
                strategy=strategy,
                raw_text=raw,
                prompt_tokens=ptoks_total,
                completion_tokens=ctoks_total,
                model=self.model,
                attempts=attempt,
                used_fallback=False,
                errors=errors,
            )

        # all attempts failed -> fall back to mock/static proposal (always valid)
        fb = self._strategy_fallback.propose_strategy(request, prompt)
        return StrategyResponse(
            strategy=fb.strategy,
            raw_text=last_raw,
            prompt_tokens=ptoks_total,
            completion_tokens=ctoks_total,
            model=self.model,
            attempts=max_attempts,
            used_fallback=True,
            errors=errors,
        )


# ── composite client: independent solution / strategy backends ─────────────
class CompositeLLMClient(LLMClient):
    """Routes ``generate_solution`` and ``propose_strategy`` to (possibly
    different) backends, so solution and strategy generation modes can be chosen
    independently (mock/mock, openai/mock, openai/openai)."""

    def __init__(self, solution_client: LLMClient, strategy_client: LLMClient) -> None:
        self.solution_client = solution_client
        self.strategy_client = strategy_client

    def generate_solution(self, request: SolutionRequest, prompt: str) -> SolutionResponse:
        return self.solution_client.generate_solution(request, prompt)

    def propose_strategy(self, request: StrategyRequest, prompt: str) -> StrategyResponse:
        return self.strategy_client.propose_strategy(request, prompt)
