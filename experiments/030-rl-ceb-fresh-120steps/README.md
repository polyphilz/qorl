# 030-rl-ceb-fresh-120steps

Method: `rl`. Seed: `43`.
Created through `qorl experiment create`; training settings follow the 029 smoke,
with 120 updates and checkpoints every ten updates. `config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/030-rl-ceb-fresh-120steps/<run-number>/`.

Starts from the merged 027 step-2204 SFT model with a fresh rank-16/alpha-32
RL adapter, AdamW at `1e-6`, and anchored GRPO. The SFT model and all prior
adapters/checkpoints remain intact; this does not continue the smoke's RL weights.

- 120 optimizer updates, nominal batch eight, group four: 960 nominal training
  episodes. Native filtering, cancellation and prefetch affect actual consumption.
- Twelve episodes in flight; sixteen model requests and vLLM sequences; policy lag
  at most two updates. Five candidate attempts, thinking enabled, 49,152 context.
- Measurement-phase worker leases; four PostgreSQL workers, each four physical
  cores and 8 GiB, with 2 GiB shared buffers. Warmups and final paired scoring are
  unchanged. Evaluation retains whole-episode worker ownership.
- Checkpoints at steps 10, 20, ..., 120; `keep_last = 12` retains every scheduled
  checkpoint in this run, including earlier policies if performance deteriorates.
- Training and inference use the two H100s. PostgreSQL uses the calibrated worker
  pool. No competing evaluation or calibration runs during training.

The training pool contains **300 new query instances**, thirty from each original
CEB training template. Seed 43 selects them and separately determines the shuffled
RL task order. Each task receives four rollouts for group credit; new instances
do not mean unseen join templates or one rollout per task. The finite taskset
can cycle if dispatch exhausts it; 120 updates do not guarantee all 300 are consumed.

Both prior SFT training selections are excluded, including queries whose
demonstrations were later filtered. Prior CEB validation and smoke selections
are excluded too: 420 distinct prior IDs/SQL hashes in total. The new validation
split contains twenty fresh CEB queries from the held-out templates 5a and 8a.
The existing 113-query JOB test split stays held out. It is not new to evaluation.

`selection-audit.json` records counts, template coverage and file hashes. All
selected SQL files were checked against their catalog hashes; training has 300
distinct IDs and SQL hashes with zero overlap against the exclusions. Training,
validation and JOB retain disjoint join topologies. Validation and JOB rollouts
are separate explicit evaluation stages, not automatic work during this run.

Stage commands:

```bash
uv run --extra gpu qorl experiment run experiments/030-rl-ceb-fresh-120steps --stage train
uv run --extra gpu qorl experiment run experiments/030-rl-ceb-fresh-120steps --stage evaluate --run 000 --checkpoint /path/to/checkpoint --split validation
```

Task IDs and identical SQL hashes in the following inputs were excluded before sampling. Recreating the selection requires these inputs via `--exclude-tasks-from`, along with the recorded seed and expressions. Exclusions are already reflected in the resolved task files; execution does not resample them.

- `excluded-tasks-000.json`: frozen exclusion input from `experiments/017-sft-astra-ceb-100/training-tasks.json`.
- `excluded-tasks-001.json`: frozen exclusion input from `experiments/025-sft-astra-ceb-300/training-tasks.json`.
- `excluded-tasks-002.json`: frozen exclusion input from `experiments/017-sft-astra-ceb-100/validation-tasks.json`.
- `excluded-tasks-003.json`: frozen exclusion input from `experiments/025-sft-astra-ceb-300/validation-tasks.json`.
- `excluded-tasks-004.json`: frozen exclusion input from `experiments/028-hybrid-rl-smoke/training-tasks.json`.
- `excluded-tasks-005.json`: frozen exclusion input from `experiments/028-hybrid-rl-smoke/validation-tasks.json`.
