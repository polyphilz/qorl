# 004 — RL run v2

**Status:** completed

This 100-update CEB run contains its task selection, calibrated timeouts,
four-worker concurrency preflight, training configuration, checkpoint
evaluation, and reward-protocol audit in one place.

- Outputs: `outputs/rl/rl-run-v2/`,
  `outputs/rl/qorl-rl-run-v2-checkpoint-evaluation-v1/`, and
  `outputs/analysis/rl-run-v2-default-fingerprint-rescore-v1/`
- Identity: selection `qorl-rl-run-v2`; timeout manifest
  `qorl-rl-run-v2-timeouts-v1`; run `rl-run-v2`
- Dependency: checkpoint evaluation reuses the frozen validation selection at
  `experiments/003-rl-pilot-v1/selection.json`
