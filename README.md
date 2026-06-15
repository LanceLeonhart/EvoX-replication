# EvoX replication (V0)

A minimal but **architecture-faithful** reproduction of EvoX: a two-level
evolutionary search framework.

- **Inner loop** evolves candidate solutions in a population `D` under the
  *current* search strategy `S`.
- **Outer loop** monitors window-level progress and adapts `S` when the search
  stagnates — *without ever resetting the population*.

V0 runs one deterministic toy black-box task with **mock** generation, so it
works offline with **no API keys**. The architecture is built to extend to real
LLM-driven generation and a full multi-task suite (see `docs/ROADMAP.md`).

## Quickstart

```bash
# (optional) create an env and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # or: pip install pyyaml pytest

# 1. smallest end-to-end run (no API keys)
python scripts/run_smoke.py

# 2. full mock EvoX run (adapts the strategy on stagnation)
python scripts/run_task.py --config configs/blackbox_evox_mock.yaml

# 3. run a task suite (one task in V0, scales to many)
python scripts/run_suite.py --config configs/suite_smoke.yaml

# 4. tabulate everything under runs/
python scripts/summarize_runs.py runs

# tests
pytest -q
```

If you are not installing the package, the scripts add the repo root to
`sys.path` automatically, so the commands above work as-is.

## Real inner-loop generation (OpenAI, V1)

The inner loop can generate candidates with a real model via the OpenAI
**Responses API** (default `gpt-5-mini`). Strategy proposal is still
mock/static in V1.

```bash
pip install 'openai>=1.40'         # or: pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
python scripts/run_task.py --config configs/blackbox_openai.yaml
```

- Select the backend with `llm.mode: mock | openai` in the config. `mock` is the
  default and is what tests and smoke runs use.
- Low-cost defaults: `reasoning_effort: minimal`, `verbosity: low`, configurable
  `max_output_tokens` (see `configs/blackbox_openai.yaml`).
- If `OPENAI_API_KEY` is missing, an `openai` run **fails clearly and does not
  fall back to mock**.
- Each generation logs model, token usage, the raw model output, a prompt
  preview, and any parse error (under `generation` in the iteration events). A
  candidate that fails to parse is marked invalid and the run continues.

## Layout

```
evox/
  core/         node, population (D), strategy (S), interpreter, history (H),
                descriptor (phi), progress (delta, J), operators, engine
  tasks/        base Task interface, registry, blackbox (the only V0 task)
  eval/         EvalResult / RunSummary, runner (config -> engine -> artifacts)
  logging/      JSONL event log, usage tracker
  llm/          LLMClient + MockLLMClient, prompt builders, parsers
configs/        smoke, static-random baseline, evox-mock, suite
scripts/        run_smoke, run_task, run_suite, summarize_runs
tests/          strategy validation, descriptor, task registry, engine smoke
docs/           FAITHFULNESS.md, ROADMAP.md
```

## Key formulas (outer loop)

For a window of `W` inner iterations with best fitness `s_start` → `s_end`:

```
delta = s_end - s_start
J     = delta * log(1 + s_start) / sqrt(W)
stagnant if delta <= tau   ->   propose a new strategy, validate, switch
```

## What's real vs. mocked (V1)

- **Solution generation** — real OpenAI backend available (`llm.mode: openai`,
  default `gpt-5-mini`); a deterministic mock backend (`llm.mode: mock`, the
  default) is used for tests and smoke runs. Same operators and decision flow
  either way.
- **Strategy proposal** — still mock/static: picks a different valid strategy
  from a small catalogue. No real strategy-generation API call yet.
- **One task only** (`toy_blackbox`); the `Task` interface + registry are ready
  for the rest.

See `docs/FAITHFULNESS.md` for the concept→code map and the honest list of
simplifications, and `docs/ROADMAP.md` for the path to real LLM generation and a
5-task evaluation.

## Constraints

This repo is independent. It does **not** import from, depend on, or copy code
from `delta-evolve-replication` (used only as read-only style reference).
