"""Engine: the two-level EvoX loop.

  Inner loop  — for each of T iterations, produce one new candidate under the
                *current* strategy S and add it to the population D. All "which
                parent / which inspirations / which operator" decisions are
                delegated to the StrategyInterpreter; the engine hard-codes none
                of them.

  Outer loop  — every W iterations, close a window: compute the improvement
                delta and strategy score J, record (S, phi(D), J) in the history
                H, and if the window stagnated (delta <= tau) ask the client for
                a new strategy. A proposed strategy is adopted only if it passes
                VALID and actually differs from the current one. Switching the
                strategy never touches D — the population persists and keeps
                growing across switches.
"""

from __future__ import annotations

import random
from typing import List, Optional

from ..eval.result import RunSummary
from ..llm.client import LLMClient, SolutionRequest, StrategyRequest
from ..llm.strategy_prompts import build_strategy_prompt
from ..logging.event_log import EventLog
from .descriptor import PopulationDescriptor
from .node import Node
from .operators import OperatorRegistry
from .population import PopulationDB
from .progress import is_stagnant, strategy_score, window_delta
from .strategy import Strategy, validate_strategy
from .strategy_history import HistoryEntry, StrategyHistory
from .strategy_interpreter import StrategyInterpreter


class Engine:
    def __init__(
        self,
        task,
        initial_strategy: Strategy,
        client: LLMClient,
        budget_T: int,
        window_W: int,
        tau: float,
        seed: int = 0,
        event_log: Optional[EventLog] = None,
        interpreter: Optional[StrategyInterpreter] = None,
        registry: Optional[OperatorRegistry] = None,
        output_dir: str = "runs",
    ) -> None:
        self.task = task
        self.current = initial_strategy
        self.client = client
        self.T = int(budget_T)
        self.W = int(window_W)
        self.tau = float(tau)
        self.seed = int(seed)
        self.output_dir = output_dir

        self.log = event_log or EventLog()
        self.interpreter = interpreter or StrategyInterpreter()
        self.registry = registry or OperatorRegistry()

        self.db = PopulationDB()
        self.history = StrategyHistory()
        self.rng = random.Random(seed)
        self.num_switches = 0
        self.strategy_ids: List[str] = [initial_strategy.id]

    # ── helpers ───────────────────────────────────────────────────────────
    def _fitness(self, score: float) -> float:
        return score if self.task.is_maximization else -score

    def _gen_seed(self, iteration: int) -> int:
        return (self.seed * 1_000_003 + iteration) & 0x7FFFFFFF

    def _strategy_seed(self, window_index: int) -> int:
        return (self.seed * 7919 + window_index * 131 + 17) & 0x7FFFFFFF

    # ── main loop ───────────────────────────────────────────────────────────
    def run(self) -> RunSummary:
        self.log.log(
            "run_start",
            task=self.task.name,
            budget_T=self.T,
            window_W=self.W,
            tau=self.tau,
            seed=self.seed,
            initial_strategy=self.current.to_dict(),
        )

        self._seed_population()

        window_start_best = self.db.best_fitness()
        window_index = 0

        for it in range(1, self.T + 1):
            self._inner_step(it)

            if it % self.W == 0:
                window_start_best = self._close_window(window_index, window_start_best)
                window_index += 1

        # close a trailing partial window so the last few iterations are still
        # scored and can still trigger a strategy switch
        leftover = self.T % self.W
        if leftover:
            self._close_window(window_index, window_start_best, window_size=leftover)
            window_index += 1

        best = self.db.best()
        summary = RunSummary(
            task=self.task.name,
            budget_T=self.T,
            window_W=self.W,
            tau=self.tau,
            seed=self.seed,
            iterations=self.T,
            population_size=self.db.size,
            best_score=best.score,
            best_fitness=best.fitness,
            best_node_id=best.id,
            num_windows=len(self.history),
            num_strategy_switches=self.num_switches,
            strategy_ids=list(self.strategy_ids),
            output_dir=self.output_dir,
            metrics=self.task.metrics(),
        )
        self.log.log("run_end", **summary.to_dict())
        return summary

    # ── inner loop ──────────────────────────────────────────────────────────
    def _seed_population(self) -> None:
        candidate = self.task.initial_candidate()
        result = self.task.evaluate(candidate)
        node = Node(
            id=self.db.new_id(),
            parent_id=None,
            iteration=0,
            strategy_id=self.current.id,
            operator="seed",
            candidate=candidate,
            score=result.score,
            fitness=self._fitness(result.score),
            valid=result.valid,
            feedback=result.feedback,
            metrics=result.metrics,
        )
        self.db.add(node)
        self.log.log(
            "iteration",
            iteration=0,
            node_id=node.id,
            operator="seed",
            strategy_id=node.strategy_id,
            parent_id=None,
            score=node.score,
            fitness=node.fitness,
            valid=node.valid,
            best_fitness=self.db.best_fitness(),
            population_size=self.db.size,
        )

    def _inner_step(self, iteration: int) -> Node:
        s = self.current
        parent = self.interpreter.select_parent(self.db, s, self.rng)
        operator_name = self.interpreter.select_operator(s, self.rng)
        inspirations = self.interpreter.select_inspirations(self.db, s, parent, self.rng)

        request = SolutionRequest(
            operator=operator_name,
            parent_candidate=parent.candidate,
            inspirations=[n.candidate for n in inspirations],
            task_prompt=self.task.render_task_prompt(),
            candidate_schema=self.task.candidate_schema(),
            exploration=s.exploration,
            strategy_id=s.id,
            seed=self._gen_seed(iteration),
        )
        operator = self.registry.get(operator_name)
        response = operator.apply(request, self.client)
        candidate = response.candidate

        result = self.task.evaluate(candidate)
        # a parse failure (candidate is None / unparseable) is surfaced as an
        # invalid node rather than crashing the run
        feedback = result.feedback
        if response.parse_error:
            feedback = (feedback + " | " if feedback else "") + f"generation: {response.parse_error}"

        node = Node(
            id=self.db.new_id(),
            parent_id=parent.id,
            iteration=iteration,
            strategy_id=s.id,
            operator=operator_name,
            candidate=candidate,
            score=result.score,
            fitness=self._fitness(result.score),
            valid=result.valid,
            inspiration_ids=[n.id for n in inspirations],
            feedback=feedback,
            metrics=result.metrics,
        )
        self.db.add(node)
        self.log.log(
            "iteration",
            iteration=iteration,
            node_id=node.id,
            operator=operator_name,
            strategy_id=s.id,
            parent_id=parent.id,
            inspiration_ids=node.inspiration_ids,
            score=node.score,
            fitness=node.fitness,
            valid=node.valid,
            best_fitness=self.db.best_fitness(),
            population_size=self.db.size,
            generation=self._generation_meta(response),
        )
        return node

    @staticmethod
    def _generation_meta(response, cap: int = 2000) -> dict:
        """Debugging metadata for one generation call: model, token usage, the
        raw model output, a prompt preview, and any parse error."""
        def _trunc(text: str) -> str:
            text = text or ""
            return text if len(text) <= cap else text[:cap] + f"...<+{len(text) - cap} chars>"

        return {
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "parse_error": response.parse_error,
            "raw_output": _trunc(response.raw_text),
            "prompt_preview": _trunc(response.prompt),
        }

    # ── outer loop ──────────────────────────────────────────────────────────
    def _close_window(
        self,
        window_index: int,
        window_start_best: float,
        window_size: Optional[int] = None,
    ) -> float:
        w = self.W if window_size is None else int(window_size)
        s_start = window_start_best
        s_end = self.db.best_fitness()
        delta = window_delta(s_start, s_end)
        J = strategy_score(delta, s_start, w)

        descriptor = PopulationDescriptor.build(self.db)
        self.history.add(
            HistoryEntry(
                window_index=window_index,
                strategy=self.current,
                descriptor=descriptor.to_dict(),
                s_start=s_start,
                s_end=s_end,
                delta=delta,
                J=J,
            )
        )
        self.log.log(
            "window_summary",
            window_index=window_index,
            strategy_id=self.current.id,
            window_size=w,
            s_start=s_start,
            s_end=s_end,
            delta=delta,
            J=J,
            stagnant=is_stagnant(delta, self.tau),
            population_size=self.db.size,
            descriptor=descriptor.to_dict(),
        )

        if is_stagnant(delta, self.tau):
            self._maybe_switch_strategy(window_index, delta)

        return self.db.best_fitness()

    def _maybe_switch_strategy(self, window_index: int, delta: float) -> None:
        descriptor = PopulationDescriptor.build(self.db)
        request = StrategyRequest(
            current_strategy=self.current,
            descriptor=descriptor.to_dict(),
            tried_signatures=self.history.tried_signatures(),
            seed=self._strategy_seed(window_index),
            history_summary=self.history.summary(),
        )
        prompt = build_strategy_prompt(request)
        response = self.client.propose_strategy(request, prompt)
        response.prompt = prompt
        proposed = response.strategy
        gen_meta = self._strategy_gen_meta(response)

        ok, reasons = validate_strategy(proposed, tuple(self.registry.names()))
        is_different = proposed.signature() != self.current.signature()

        if ok and is_different:
            pop_before = self.db.size
            old_id = self.current.id
            self.current = proposed
            self.num_switches += 1
            self.strategy_ids.append(proposed.id)
            self.log.log(
                "strategy_switch",
                window_index=window_index,
                reason="stagnation",
                delta=delta,
                tau=self.tau,
                from_strategy=old_id,
                to_strategy=proposed.id,
                strategy=proposed.to_dict(),
                population_size_before=pop_before,
                population_size_after=self.db.size,  # unchanged: D is never reset
                strategy_generation=gen_meta,
            )
        else:
            self.log.log(
                "strategy_switch_rejected",
                window_index=window_index,
                delta=delta,
                tau=self.tau,
                proposed_strategy=proposed.id,
                valid=ok,
                different=is_different,
                reasons=reasons,
                strategy_generation=gen_meta,
            )

    @staticmethod
    def _strategy_gen_meta(response, cap: int = 2000) -> dict:
        """Debugging metadata for one strategy-generation call: model, token
        usage, attempts/fallback/acceptance, raw output, prompt preview, and any
        parse/validation errors."""
        def _trunc(text: str) -> str:
            text = text or ""
            return text if len(text) <= cap else text[:cap] + f"...<+{len(text) - cap} chars>"

        return {
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "attempts": response.attempts,
            "used_fallback": response.used_fallback,
            "errors": response.errors,
            "proposed_strategy": response.strategy.id,
            "raw_output": _trunc(response.raw_text),
            "prompt_preview": _trunc(response.prompt),
        }
