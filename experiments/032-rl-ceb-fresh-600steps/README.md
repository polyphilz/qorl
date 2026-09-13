# 032-rl-ceb-fresh-600steps

Method: `rl`. Seed: `42`.
Created through `qorl experiment create`, then configured from 031.
`config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/032-rl-ceb-fresh-600steps/<run-number>/`.

**Continue learning from 031's final step-600 policy on 600 fresh CEB queries.**
The verified 031 RL adapter was merged into its exact merged 027 SFT base using
`qorl model merge`. The resulting complete model is
`/lambda/nfs/qorl/models/qwen-4b-rl-031-step-600`.
Its weight identity and merge provenance are recorded in
[base-merge-manifest.json](base-merge-manifest.json). Both source artifacts remain
intact. The merge incorporated all 128 adapted weight matrices.

032 trains a **fresh rank-16/alpha-32 RL LoRA with fresh AdamW state** over that
model. The knowledge learned in 031 is carried by the merged base. The local
step counter starts at zero; step 600 in this experiment represents another
600 optimizer updates after 031.

Training and evaluation settings match 031:

- Constant learning rate `1e-5`; nominal batch 16, group eight; 600 updates.
- Twenty episodes in flight and twenty inference sequences; maximum policy
  lag two updates. Context 49,152, reply limit 8,192, thinking enabled and five
  candidate attempts.
- Anchored GRPO with the same reward parameters and `min_peers = 2`.
- Four PostgreSQL workers and measurement-phase leases during RL. Each worker
  has four physical cores and 8 GiB, with 2 GiB shared buffers. Initial defaults
  receive one warmup and three measurements; final scoring uses one warmup pair
  and three measured pairs.
- Checkpoints every 20 updates, retaining all 30 scheduled checkpoints.
  No automatic validation during training.
- JOB evaluation uses all 113 queries, one rollout each, seed 42, and the
  separate `best_feedback` selection comparison. The rule does not alter RL
  decisions or rewards.

The training allocation changes because **only six unseen 7a queries remain**
after exclusions: six from 7a and **66 from each of the other nine original
training templates**. This retains 600 fresh queries across the same ten
templates. Validation contains twenty fresh queries, ten each from 5a and 8a.
The JOB test selection is identical to 031's.

Frozen exclusions cover **2,133 distinct prior CEB query IDs and SQL hashes**:
031's inherited exclusions plus its 600 training and twenty validation queries.
The audit verifies zero overlap with those exclusions, unique SQL within each
split, disjoint SQL and join topologies across splits, all 733 selected SQL
checksums, and exact deterministic selection replay. See
[selection-audit.json](selection-audit.json). The only configuration differences
from 031 are the new merged model path and the training allocation expression.
Legacy exclusion provenance is retained in
[legacy-exclusion-audit.json](legacy-exclusion-audit.json).

Each execution allocates a numbered run and copies its config and task selections.
Do not run a competing evaluation or calibration against the training worker pool.

Stage commands:

```bash
uv run --frozen --extra gpu qorl experiment run experiments/032-rl-ceb-fresh-600steps --stage train
uv run --frozen --extra gpu qorl experiment run experiments/032-rl-ceb-fresh-600steps --stage evaluate --run 000 --checkpoint /path/to/checkpoint --split test
```

Task IDs and identical SQL hashes in the following inputs were excluded before sampling. Recreating the selection requires these inputs via `--exclude-tasks-from`, along with the recorded seed and expressions. Exclusions are already reflected in the resolved task files; execution does not resample them.

- `excluded-tasks-000.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/excluded-tasks-000.json`.
- `excluded-tasks-001.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/excluded-tasks-001.json`.
- `excluded-tasks-002.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/excluded-tasks-002.json`.
- `excluded-tasks-003.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/excluded-tasks-003.json`.
- `excluded-tasks-004.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/excluded-tasks-004.json`.
- `excluded-tasks-005.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/excluded-tasks-005.json`.
- `excluded-tasks-006.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/excluded-tasks-006.json`.
- `excluded-tasks-007.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/excluded-tasks-007.json`.
- `excluded-tasks-008.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/training-tasks.json`.
- `excluded-tasks-009.json`: frozen exclusion input from `experiments/031-rl-ceb-fresh-600steps/validation-tasks.json`.
