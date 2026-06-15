# Faithfulness: EvoX paper concepts → code

This document maps the EvoX architecture onto the V0 implementation. The goal of
V0 is to be *architecturally* faithful (the two coupled loops and their data
structures) while keeping the *implementation* deliberately small (one toy task,
mock generation, no API keys).

## The two coupled loops

| Paper concept | Where in code |
| --- | --- |
| Inner loop: evolve candidate solutions under the current strategy `S` | `evox/core/engine.py` → `Engine._inner_step` |
| Outer loop: monitor window progress, adapt `S` on stagnation | `evox/core/engine.py` → `Engine._close_window` / `_maybe_switch_strategy` |
| Coupling: outer loop changes `S`, inner loop keeps using the *same* `D` | `Engine.run` (single `PopulationDB`, never re-created) |

## Core abstractions

| Paper symbol / concept | Code |
| --- | --- |
| Solution database `D` | `evox/core/population.py` → `PopulationDB` (append-only, never reset) |
| A candidate / node in `D` | `evox/core/node.py` → `Node` |
| Active search strategy `S` (schema) | `evox/core/strategy.py` → `Strategy` |
| `VALID(S)` strategy validation | `evox/core/strategy.py` → `VALID` / `validate_strategy` |
| Strategy → decisions (parent/operator/inspiration) | `evox/core/strategy_interpreter.py` → `StrategyInterpreter` |
| Operators `local_refine`, `structural_variation`, `free_form` | `evox/core/operators.py` → `OperatorRegistry`, `DEFAULT_OPERATORS` |
| Population descriptor `phi(D)` | `evox/core/descriptor.py` → `PopulationDescriptor` |
| Strategy history `H = {(S, phi, J)}` | `evox/core/strategy_history.py` → `StrategyHistory`, `HistoryEntry` |
| Window progress `delta = s_end - s_start` | `evox/core/progress.py` → `window_delta` |
| Strategy score `J = delta * log(1 + s_start) / sqrt(W)` | `evox/core/progress.py` → `strategy_score` |
| Stagnation test (`delta <= tau`) | `evox/core/progress.py` → `is_stagnant` |
| Two-level engine | `evox/core/engine.py` → `Engine` |

## Decision delegation (the key invariant)

The engine **never** hard-codes parent selection, inspiration selection, or
operator choice. Every such decision is delegated:

```
Engine._inner_step
  → StrategyInterpreter.select_parent(D, S)
  → StrategyInterpreter.select_operator(S)
  → StrategyInterpreter.select_inspirations(D, S, parent)
```

Changing the active strategy `S` therefore changes search behaviour without any
change to the engine. This is verified by `tests/test_engine_smoke.py`.

## Strategy switching does not reset the population

`_maybe_switch_strategy` only reassigns `self.current` (the active `S`). It never
touches `self.db`. The `strategy_switch` log event records
`population_size_before` and `population_size_after` to make this auditable, and
`tests/test_engine_smoke.py::test_population_never_reset_across_switches`
asserts the population size is monotonically non-decreasing across the whole run.

## Generation boundary (mock now, real later)

Both solution generation (inner) and strategy proposal (outer) go through one
interface, `evox/llm/client.py::LLMClient`:

- `generate_solution(request, prompt)` — inner loop candidate generation.
- `propose_strategy(request, prompt)` — outer loop strategy proposal.

`MockLLMClient` implements both deterministically and offline. The prompt
builders (`evox/llm/solution_prompts.py`, `evox/llm/strategy_prompts.py`) and
parsers (`evox/llm/parsers.py`) are the seam a real client plugs into — they are
built/tested in V0 even though the mock client returns structured objects
directly. See `docs/ROADMAP.md`.

## Honest simplifications in V0

- **Mock generation.** Candidate mutation is a numeric perturbation of a vector
  (the toy task's representation), not LLM output. The operator *vocabulary* and
  the decision flow are identical to what a real run would use.
- **Diversity proxy.** `PopulationDescriptor.diversity` and the interpreter's
  `diverse` inspiration mode use fitness spread instead of embedding distance.
- **Single task.** Only `toy_blackbox` is implemented; the `Task` interface and
  registry are designed for the full multi-task suite.
- **`s_start`/`s_end`** are best-fitness at window boundaries, with fitness
  normalised so higher is always better (`Engine._fitness`), keeping
  `log(1 + s_start)` well-defined.
