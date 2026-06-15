#!/usr/bin/env python3
"""Run the tiny smoke config end-to-end. No API keys required.

    python scripts/run_smoke.py
"""

import _bootstrap  # noqa: F401  (sets up sys.path)

import os

from evox.eval.runner import run_config_file

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "smoke.yaml")


def main() -> None:
    result = run_config_file(os.path.normpath(CONFIG))
    s = result.summary
    print("== smoke run complete ==")
    print(f"task                 : {s.task}")
    print(f"population size       : {s.population_size}  (seed + {s.iterations} iters)")
    print(f"best fitness          : {s.best_fitness:.6f}")
    print(f"windows               : {s.num_windows}")
    print(f"strategy switches     : {s.num_strategy_switches}")
    print(f"strategies used       : {s.strategy_ids}")
    print(f"artifacts             : {result.run_dir}")


if __name__ == "__main__":
    main()
