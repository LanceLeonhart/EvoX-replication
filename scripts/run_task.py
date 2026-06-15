#!/usr/bin/env python3
"""Run a single task from a config file.

    python scripts/run_task.py --config configs/blackbox_evox_mock.yaml
"""

import _bootstrap  # noqa: F401

import argparse

from evox.eval.runner import run_config_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one EvoX task from a config.")
    parser.add_argument("--config", required=True, help="path to a task YAML config")
    args = parser.parse_args()

    result = run_config_file(args.config)
    s = result.summary
    print("== run complete ==")
    print(f"config                : {args.config}")
    print(f"task                  : {s.task}")
    print(f"population size       : {s.population_size}")
    print(f"best fitness          : {s.best_fitness:.6f}  (raw score {s.best_score:.6f})")
    print(f"windows               : {s.num_windows}")
    print(f"strategy switches     : {s.num_strategy_switches}")
    print(f"strategies used       : {s.strategy_ids}")
    print(f"artifacts             : {result.run_dir}")
    print(f"  events  : {result.events_path}")
    print(f"  summary : {result.summary_path}")
    print(f"  usage   : {result.usage_path}")


if __name__ == "__main__":
    main()
