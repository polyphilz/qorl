# 000 — Vanilla baselines

**Status:** completed

This experiment evaluates the untrained Qwen distill policy and the frozen
random structured-action policy on held-out JOB. `run.json` and `random.json`
bind stable run prefixes to the shared policy definitions in `configs/policy/`.

- Outputs: `outputs/runs/<run-id>/`
- Identity: model revision
  `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`; JOB inventory
  `job-v1-tasks-v1`
