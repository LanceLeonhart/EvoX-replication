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

## Real generation (OpenAI Responses API, default `gpt-5-mini`)

Solution generation (inner loop, `G_sol`) and strategy generation (outer loop,
`G_str`) each have an independently selectable backend:

```yaml
llm:
  solution_mode: mock | openai   # inner-loop candidate generation
  strategy_mode:  mock | openai   # outer-loop strategy proposal (on stagnation)
  model: gpt-5-mini
  reasoning_effort: minimal       # low-cost default
  verbosity: low                  # concise output
  max_output_tokens: 512
  strategy_retries: 1             # retries before falling back to the mock proposer
```

Three useful combinations: mock/mock (default, offline), openai/mock
(`configs/blackbox_openai.yaml`), openai/openai (`configs/blackbox_openai_strategy.yaml`).

```bash
pip install 'openai>=1.40'         # or: pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
python scripts/run_task.py --config configs/blackbox_openai_strategy.yaml
```

- `mock` is the default (and what tests / smoke runs use). The legacy `llm.mode`
  key still works: it sets `solution_mode`, leaving strategy mock unless you opt
  in with `strategy_mode: openai`.
- If `OPENAI_API_KEY` is missing, any `openai` mode **fails clearly and does not
  fall back to mock**.
- **Solution generation** logs model, tokens, raw output, prompt preview, and any
  parse error (`generation` in iteration events); an unparseable candidate is
  marked invalid and the run continues.
- **Strategy generation** is called only on stagnation. The model returns one
  `Strategy` JSON, which is parsed (`parse_strategy`) and validated (`VALID`);
  on repeated parse/validation failure it retries and then falls back to the
  mock proposer, so a malformed strategy never crashes the run. Each proposal
  logs model, tokens, attempts, fallback flag, errors, raw output, and prompt
  preview (`strategy_generation` in `strategy_switch` / `strategy_switch_rejected`
  events). Strategy switches never reset the population.

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

## What's real vs. mocked (V2)

- **Solution generation** — real OpenAI backend (`solution_mode: openai`,
  default `gpt-5-mini`) or deterministic mock (`solution_mode: mock`, default).
- **Strategy generation** — real OpenAI backend (`strategy_mode: openai`) or
  the mock/static catalogue proposer (`strategy_mode: mock`, default). Real
  proposals are parsed, validated, retried, and fall back to mock on failure.
- **One task only** (`toy_blackbox`); the `Task` interface + registry are ready
  for the rest.

See `docs/FAITHFULNESS.md` for the concept→code map and the honest list of
simplifications, and `docs/ROADMAP.md` for the path to real LLM generation and a
5-task evaluation.

## Constraints

This repo is independent. It does **not** import from, depend on, or copy code
from `delta-evolve-replication` (used only as read-only style reference).
