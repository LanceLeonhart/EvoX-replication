"""Runner: build and execute an Engine from a config dict.

This is the single entry point used by all the scripts. It wires the task (via
the registry), the initial strategy, the mock client, logging, and the engine,
runs it, and persists the artifacts (events JSONL, summary.json, usage.json).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

from ..core.engine import Engine
from ..core.strategy import Strategy, validate_strategy
from ..llm.client import CompositeLLMClient, LLMClient, MockLLMClient, OpenAIClient
from ..logging.event_log import EventLog
from ..logging.usage import UsageTracker
from ..tasks.registry import create_task
from .result import RunSummary


@dataclass
class RunResult:
    summary: RunSummary
    run_dir: str
    events_path: str
    summary_path: str
    usage_path: str


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_client(config: Dict[str, Any], usage: UsageTracker) -> LLMClient:
    """Build the LLM client from config, with independent solution/strategy modes.

    New keys ``solution_mode`` / ``strategy_mode`` select each backend. The
    legacy ``mode`` key is honoured for backward compatibility: it sets the
    solution mode, while strategy stays mock unless explicitly opted in (matching
    V1 behaviour where strategy generation was always mock/static).
    """
    llm = config.get("llm", {}) or {}
    legacy = llm.get("mode")
    solution_mode = llm.get("solution_mode", legacy or "mock")
    strategy_mode = llm.get("strategy_mode", "mock")

    # construct each backend at most once and share it / the usage tracker
    cache: Dict[str, LLMClient] = {}

    def _make(mode: str) -> LLMClient:
        if mode in cache:
            return cache[mode]
        if mode == "mock":
            client: LLMClient = MockLLMClient(usage=usage)
        elif mode == "openai":
            # A missing OPENAI_API_KEY raises inside OpenAIClient — no silent
            # fallback to mock. Constructed lazily, only when an openai mode is used.
            client = OpenAIClient(
                model=llm.get("model", "gpt-5-mini"),
                reasoning_effort=llm.get("reasoning_effort", "minimal"),
                verbosity=llm.get("verbosity", "low"),
                max_output_tokens=int(llm.get("max_output_tokens", 512)),
                strategy_retries=int(llm.get("strategy_retries", 1)),
                usage=usage,
            )
        else:
            raise ValueError(f"unknown llm mode {mode!r}; expected 'mock' or 'openai'")
        cache[mode] = client
        return client

    solution_client = _make(solution_mode)
    strategy_client = _make(strategy_mode)
    if solution_client is strategy_client:
        return solution_client  # one client already serves both
    return CompositeLLMClient(solution_client=solution_client, strategy_client=strategy_client)


def _run_dir(config: Dict[str, Any], task_name: str) -> str:
    out = config.get("output", {})
    base = out.get("dir", "runs")
    name = config.get("name") or f"{task_name}"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"{name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_from_config(
    config: Dict[str, Any],
    run_dir: Optional[str] = None,
) -> RunResult:
    task_cfg = config["task"]
    task = create_task(task_cfg["name"], task_cfg.get("params"))

    strategy = Strategy.from_dict(config["strategy"])
    ok, reasons = validate_strategy(strategy)
    if not ok:
        raise ValueError(f"initial strategy is invalid: {reasons}")

    eng_cfg = config["engine"]
    if run_dir is None:
        run_dir = _run_dir(config, task.name)

    events_path = os.path.join(run_dir, "events.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")
    usage_path = os.path.join(run_dir, "usage.json")

    usage = UsageTracker()
    client = _make_client(config, usage)
    event_log = EventLog(events_path)

    engine = Engine(
        task=task,
        initial_strategy=strategy,
        client=client,
        budget_T=eng_cfg["budget_T"],
        window_W=eng_cfg["window_W"],
        tau=eng_cfg["tau"],
        seed=eng_cfg.get("seed", 0),
        event_log=event_log,
        output_dir=run_dir,
    )
    summary = engine.run()

    with open(summary_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    with open(usage_path, "w") as f:
        json.dump(usage.to_dict(), f, indent=2)

    return RunResult(
        summary=summary,
        run_dir=run_dir,
        events_path=events_path,
        summary_path=summary_path,
        usage_path=usage_path,
    )


def run_config_file(path: str, run_dir: Optional[str] = None) -> RunResult:
    return run_from_config(load_config(path), run_dir=run_dir)
