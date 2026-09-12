# 031-rl-ceb-fresh-600steps

Method: `rl`. Seed: `42`.
Created through `qorl experiment create` using `configs/defaults/002-rl.toml`,
then configured for the next RL run. `config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/031-rl-ceb-fresh-600steps/<run-number>/`.

Starts from the merged 027 step-2204 SFT model with a **fresh** rank-16/alpha-32
RL adapter and optimizer. The preceding SFT and RL checkpoints are preserved.

- AdamW at `1e-5`, nominal batch 16, group eight, at most 600 optimizer updates.
  Two query groups contribute to a nominal batch; native filtering can reduce
  the actual number of training episodes. Group eight targets the high fraction
  of zero-advantage episodes observed in 030; improvement is not assumed.
- Twenty episodes in flight, twenty model requests, twenty vLLM sequences;
  policy lag at most two updates. Context 49,152; five candidate attempts;
  thinking enabled. The anchored reward and `min_peers = 2` are unchanged.
- Checkpoints every 20 updates through step 600. `keep_last = 31` retains all
  30 scheduled checkpoints and leaves room for an initial checkpoint.
- Measurement-phase leases, four calibrated PostgreSQL workers, each with
  four physical cores and 8 GiB. Baseline timing uses one warmup plus three
  measurements; feedback and final pairing follow 030's protocol.
- No automatic validation during training. Evaluation uses seed 42 and one
  rollout per query. JOB contains the existing 113 held-out test queries.

The training selection contains **600 distinct new CEB queries**, sixty from
each original training template. The optional validation selection contains
twenty new queries from templates 5a and 8a. All selected SQL files passed their
catalog checksum checks, and the three splits have disjoint join topologies.
The finite training set can cycle; 600 updates do not imply 600 unique visits.

Exclusions cover **1,513 distinct prior CEB queries and SQL hashes**, including
the two SFT datasets, subsequent smoke/RL selections and validation sets. Legacy
experiment records were also checked: old filename-based IDs were mapped to
current catalog IDs. `legacy-exclusion-audit.json` records source hashes and the
mapping; `selection-audit.json` records zero overlap, quotas and file hashes.
Duplicated saved selections are represented once in the frozen inputs below.

`[evaluation] selection_policy = "best_feedback"` enables a separate selection
ablation after evaluation conversations. It selects the earliest best eligible,
non-timeout candidate with a completed feedback ratio of at least 1.05x, otherwise
default. The choice is fixed before final timing. A different choice receives
fresh paired measurements; an identical choice reuses the model's final result.
Reports retain both results and count additional executions separately. This
setting **does not change RL decisions or rewards**. Compare the same selector
on the SFT baseline when assessing changes in proposal quality.

Each execution allocates a numbered run and copies its config and task selections.
Do not run a competing evaluation or calibration against the training worker pool.

Stage commands:

```bash
uv run --extra gpu qorl experiment run experiments/031-rl-ceb-fresh-600steps --stage train
uv run --extra gpu qorl experiment run experiments/031-rl-ceb-fresh-600steps --stage evaluate --run 000 --checkpoint /path/to/checkpoint --split test
```

Task IDs and identical SQL hashes in the following inputs were excluded before sampling. Recreating the selection requires these inputs via `--exclude-tasks-from`, along with the recorded seed and expressions. Exclusions are already reflected in the resolved task files; execution does not resample them.

- `excluded-tasks-000.json`: frozen exclusion input from `experiments/017-sft-astra-ceb-100/training-tasks.json`.
- `excluded-tasks-001.json`: frozen exclusion input from `experiments/017-sft-astra-ceb-100/validation-tasks.json`.
- `excluded-tasks-002.json`: frozen exclusion input from `experiments/025-sft-astra-ceb-300/training-tasks.json`.
- `excluded-tasks-003.json`: frozen exclusion input from `experiments/028-hybrid-rl-smoke/training-tasks.json`.
- `excluded-tasks-004.json`: frozen exclusion input from `experiments/028-hybrid-rl-smoke/validation-tasks.json`.
- `excluded-tasks-005.json`: frozen exclusion input from `experiments/030-rl-ceb-fresh-120steps/training-tasks.json`.
- `excluded-tasks-006.json`: frozen exclusion input from `experiments/030-rl-ceb-fresh-120steps/validation-tasks.json`.
- `excluded-tasks-007.json`: normalized legacy exclusions; source hashes and ID mapping are in `legacy-exclusion-audit.json`.
