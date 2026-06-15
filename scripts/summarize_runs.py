#!/usr/bin/env python3
"""Summarize runs by reading their summary.json files.

    python scripts/summarize_runs.py [runs_dir]
"""

import _bootstrap  # noqa: F401

import json
import os
import sys


def find_summaries(root: str):
    for dirpath, _dirs, files in os.walk(root):
        if "summary.json" in files:
            yield os.path.join(dirpath, "summary.json")


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    if not os.path.isdir(root):
        print(f"no such directory: {root}")
        return

    header = f"{'run':40s} {'task':14s} {'pop':>4s} {'best_fit':>10s} {'wins':>5s} {'switch':>6s}"
    print(header)
    print("-" * len(header))
    count = 0
    for path in sorted(find_summaries(root)):
        with open(path) as f:
            s = json.load(f)
        run_name = os.path.basename(os.path.dirname(path))[:40]
        print(
            f"{run_name:40s} {s.get('task', '?'):14s} "
            f"{s.get('population_size', 0):>4d} {s.get('best_fitness', 0.0):>10.6f} "
            f"{s.get('num_windows', 0):>5d} {s.get('num_strategy_switches', 0):>6d}"
        )
        count += 1
    if count == 0:
        print("(no summary.json files found)")


if __name__ == "__main__":
    main()
