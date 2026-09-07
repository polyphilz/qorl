# 003 — RL pilot v1

Execution entrypoint retired on 2026-09-07 during the shared-agent migration.
The inputs and recorded results below are historical evidence, not rerun instructions.

**Status:** completed

This 12-update pilot uses 48 CEB training tasks and a fixed 16-task validation
slice to verify that the initial reward and training path can learn at all.
`build_inventory.py` produces the frozen selection, and `run.py` preserves the
pilot's experiment-specific training and preflight logic.

`run.py` uses the root QORL environment; its launch config resolves the training
plugins through the `qorl` package.

The launch config explicitly selects `000-pgconf-default` and `002-poolconf-4x8`.
Rerunning uses those configurations; completed outputs retain their recorded settings.

- Outputs: `outputs/rl/rl-pilot-v1/` and
  `outputs/rl/qorl-rl-pilot-validation-v1/{pre,post}/`
- Identity: selection inventory `qorl-rl-pilot-v1`; run `rl-pilot-v1`
- Dependency: `experiments/002-rl-spike/` reads this experiment's frozen task
  selection

The pre-RL gate reads `model/configs/000-modelconf/modelconf.json`. The completed
pre-RL report's policy checksum predates the configuration refactors, so the gate
rejects that historical report. The recorded pilot and validation results remain valid.

This experiment used `benchmark-v1`, with GEQO enabled. Results for tasks with
at least 12 relations are not directly comparable with `benchmark-v2` results.
