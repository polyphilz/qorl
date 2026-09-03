# 003 — RL pilot v1

**Status:** completed

This 12-update pilot uses 48 CEB training tasks and a fixed 16-task validation
slice to verify that the initial reward and training path can learn at all.
`build_inventory.py` produces the frozen selection, and `run.py` preserves the
pilot's experiment-specific training and preflight logic.

- Outputs: `outputs/rl/rl-pilot-v1/` and
  `outputs/rl/qorl-rl-pilot-validation-v1/{pre,post}/`
- Identity: selection inventory `qorl-rl-pilot-v1`; run `rl-pilot-v1`
- Dependency: `experiments/002-rl-spike/` reads this experiment's frozen task
  selection

The completed pre-RL report records the policy checksum from before
`run_id_prefix` moved out of `configs/policy/run-v1.json`. Re-running the legacy
`run.py` gate against the reorganized policy file will therefore reject that
historical report; the completed pilot and validation artifacts remain valid.

This experiment used `benchmark-v1`, with GEQO enabled. Results for tasks with
at least 12 relations are not directly comparable with `benchmark-v2` results.
