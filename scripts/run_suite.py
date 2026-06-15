#!/usr/bin/env python3
"""Run an evaluation suite: several task configs through one interface.

    python scripts/run_suite.py --config configs/suite_smoke.yaml
"""

import _bootstrap  # noqa: F401

import argparse
import json
import os

import yaml

from evox.eval.runner import run_config_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EvoX evaluation suite.")
    parser.add_argument("--config", required=True, help="path to a suite YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        suite_cfg = yaml.safe_load(f)

    suite = suite_cfg["suite"]
    out_dir = suite.get("output_dir", "runs/suite")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    print(f"== suite: {suite.get('name', 'suite')} ==")
    for i, entry in enumerate(suite_cfg["runs"]):
        cfg_path = entry["config"]
        run_dir = os.path.join(out_dir, f"run{i:02d}")
        os.makedirs(run_dir, exist_ok=True)
        result = run_config_file(cfg_path, run_dir=run_dir)
        s = result.summary
        rows.append(s.to_dict())
        print(
            f"[{i}] {cfg_path} -> task={s.task} "
            f"best_fitness={s.best_fitness:.6f} pop={s.population_size} "
            f"switches={s.num_strategy_switches}"
        )

    suite_summary_path = os.path.join(out_dir, "suite_summary.json")
    with open(suite_summary_path, "w") as f:
        json.dump({"suite": suite.get("name"), "runs": rows}, f, indent=2)
    print(f"suite summary written to {suite_summary_path}")


if __name__ == "__main__":
    main()
