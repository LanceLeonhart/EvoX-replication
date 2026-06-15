# Roadmap: from V0 to real LLM strategy mutation and 5-task evaluation

V0 is intentionally a skeleton with a beating heart: the two-level loop, the data
structures, and the decision-delegation invariant are all real; only the
*generation* and the *task set* are stubbed. This document describes how each
stub becomes the real thing without re-architecting.

## 1. Real LLM solution generation (inner loop)

**Today.** `MockLLMClient.generate_solution` perturbs a numeric vector.

**Next.**
- Add `evox/llm/openai_client.py` implementing the same `LLMClient` interface.
- `generate_solution` sends the already-built prompt
  (`solution_prompts.build_solution_prompt`) to the model and parses the reply
  with `parsers.parse_solution_*`.
- Select the client in `eval/runner.py::_make_client` on `llm.mode` (`mock` →
  `openai`/etc.), reading model + keys from config/env.
- Nothing in `Engine`, `OperatorRegistry`, or `StrategyInterpreter` changes —
  operators already build prompts and call the client.

The candidate representation generalises from a vector to whatever the task
defines (code string, program AST, packing layout, …). The mock's vector
assumption lives only in `MockLLMClient._mutate_vector`; real generation is
representation-agnostic because the task supplies the prompt and the parser.

## 2. Real LLM strategy mutation (outer loop)

**Today.** `MockLLMClient.propose_strategy` picks a different strategy from a
fixed catalogue.

**Next.**
- The real client sends `strategy_prompts.build_strategy_prompt(request)` — which
  already includes the current `S`, the descriptor `phi(D)`, and the tried
  strategy signatures from history `H` — and parses the reply with
  `parsers.parse_strategy`.
- `VALID(S)` already gates the result, so malformed/unsafe proposals are rejected
  and logged (`strategy_switch_rejected`) exactly as in V0.
- Optionally condition the proposal on the best-`J` history entry
  (`StrategyHistory.best_by_J`) to bias toward what has worked.

## 3. Five-task evaluation

**Today.** One task (`toy_blackbox`) registered; suite runs one entry.

**Next.**
- Implement each benchmark task as a `Task` subclass under `evox/tasks/`
  (e.g. `symbolic_regression.py`, `packing.py`, `pde_solver.py`,
  `efficient_convolution.py`) and `register_task(...)` it.
- Each task owns its representation, objective, prompt, and validity checks;
  the engine is untouched because it only uses the `Task` interface.
- Add one config per task under `configs/`, then list them all in a suite config
  (`runs:` list). `scripts/run_suite.py` already iterates the suite and writes a
  combined `suite_summary.json`.
- `scripts/summarize_runs.py` already tabulates across all runs.

## 4. Scale-up and rigor (later)

- Real budgets (100-iteration runs), multiple seeds/trials per task, and
  cost tracking via the existing `UsageTracker` (wire real token counts/pricing
  into the client).
- Baselines: the `blackbox_static_random.yaml` pattern (a fixed, never-switched
  strategy via `tau = -inf`) generalises to an ablation control for every task.
- Richer `phi(D)` (embedding-based diversity, novelty), replacing the V0 fitness-
  spread proxy in `descriptor.py` and the interpreter's `diverse` mode.

## Invariants to preserve through all of the above

1. The engine never hard-codes parent/operator/inspiration choice — always via
   `StrategyInterpreter`.
2. Strategy switches never reset the population `D`.
3. Every proposed strategy passes `VALID(S)` before becoming active.
4. Each window records `(S, phi(D), J)` into history `H`.
