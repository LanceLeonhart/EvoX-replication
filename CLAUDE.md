# CLAUDE.md

This repo is an independent EvoX reproduction.

Hard constraints:
- Do not import from, symlink to, modify, or depend on `delta-evolve-replication`.
- You may inspect `delta-evolve-replication` only as read-only reference for task/evaluator/logging style.
- Keep the EvoX architecture faithful: solution population `D`, active strategy `S`, strategy history `H`, population descriptor `phi(D)`, window progress, strategy validation, and strategy switching without resetting the solution population.
- Simplify implementation, but do not simplify architecture.
- Prefer small, testable increments.
- Always run tests or smoke scripts after changes when possible.