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
from ..llm.client import MockLLMClient
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


def _make_client(config: Dict[str, Any], usage: UsageTracker) -> MockLLMClient:
    mode = config.get("llm", {}).get("mode", "mock")
    if mode != "mock":
        raise NotImplementedError(
            f"llm.mode={mode!r} not supported in V0; only 'mock' is available"
        )
    return MockLLMClient(usage=usage)


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
