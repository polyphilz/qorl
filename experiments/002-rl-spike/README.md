# 002 — RL spike

**Status:** completed

This one-update run verifies the end-to-end Prime-RL, QORL environment,
PostgreSQL reward, LoRA update, and checkpoint path before a pilot.

- Output: `outputs/rl/rl-spike-v1/`
- Identity: run `rl-spike-v1`; source inventory `qorl-rl-pilot-v1`
- Dependency: read-only `experiments/003-rl-pilot-v1/selection.json`; the
  pilot selection was frozen for this spike and then reused by the full pilot
