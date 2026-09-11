# 028 — Hybrid RL smoke

Four optimizer updates to test [044's hybrid architecture](../../.plans/044-hybrid-rl-machinery.md):
Lambda owns training and inference on its two H100s; FLOPper owns qo-agent,
rendering and all PostgreSQL work. This is an integration test, not a policy
quality comparison. Run `000` completed all four updates on September 11, 2026,
from **16:19:05 to 16:30:20 UTC (12:19–12:30 EDT)**. See
[results.md](results.md) for the live audit, checkpoint and cleanup evidence.

## Recipe

| Setting | Value |
| --- | --- |
| Starting policy | Separate merged base from 027/000, step 2204 |
| Learning | Existing anchored GRPO; fresh rank-16/alpha-32 RL LoRA; AdamW `1e-6` |
| Optimizer updates | 4 |
| Batch / group / concurrent rollouts | 4 / 4 / 4 |
| Maximum policy lag | 1 step, to exercise updated-policy rollouts during the short run |
| Agent | v7, five candidate attempts, 64 model turns, thinking enabled |
| Context / maximum reply | 49,152 / 8,192 tokens |
| Measurement | Existing three initial default measurements, one candidate feedback measurement, three final pairs; normal warmups |
| PostgreSQL | FLOPper's `001-pgconf-2gb-sb`, `002-poolconf-4x8` |
| GPUs on Lambda | GPU 0 training; GPU 1 inference |
| Checkpoints | Every update; retain all four |
| Seed | 42 |

The four training batches consume 16 rollouts. Native prefetch can produce
additional rollouts that are not used for an update; this is not a hard cap on
episode count or wall time. The usual request/SQL deadlines remain unchanged.
Four updates do not guarantee nonzero advantages or policy improvement.

Saved training queries:

- `ceb-1a-1a2461`: original 017 teacher run recorded a 360.727 ms initial
  default and a final 2.667× speedup.
- `ceb-2a-2a279`: original 017 teacher run recorded a 584.349 ms initial
  default and a final 1.141× speedup.

These familiar SFT training queries were chosen for a short integration check.
They are not a fresh generalization set, and their old teacher results do not
predict RL outcomes. Their SQL was inspected and checked against the catalog.
The saved validation selection is `ceb-5a-5a977`, from a separate topology.
JOB is not in the training selection. RL training does not automatically run
the separate rollout evaluation; `evaluation.rollouts_per_task=1` only controls
an explicitly requested evaluation.

Created through `qorl experiment create`, using seed 42 and expressions
`train=ceb[1a:1,2a:1]` and `validation=ceb[5a:1]`, then adjusted from
`configs/defaults/002-rl.toml` for this smoke. No defaults were edited.
The saved task files own the IDs and are not resampled at execution.

## Starting model, already prepared on Lambda

```text
/lambda/nfs/qorl/models/qwen-4b-sft-027-step-2204
```

Built with `qorl model merge` from the original pinned
`empero-ai/Qwen3.8-4B-Distill` base and
`outputs/027-sft-astra-ceb-300-epoch2/000/training/checkpoints/step_2204/adapter`.
The merge verified 128 LoRA updates, the complete model and its assets. The
original base and SFT adapter were preserved; the source adapter checksum was
checked again after merging. RL creates a new adapter over the merged base
and does not resume the SFT optimizer or modify its checkpoints.

| Artifact | SHA-256 |
| --- | --- |
| Source SFT adapter | `f20c171c1da2cdf45f4d84bd945877702214e4b81d3586b3a0abef201ea3ed4f` |
| Merged base weights | `a2fd098341e57376695a986454e35ab38c9594c0a41b736abff16f9683205709` |

The merge manifest is `qorl-merge.json` inside that directory. FLOPper can use
its already-cached original model directory for renderer assets: all six
config/tokenizer/template asset hashes match the merged model exactly. The
environment service reads those assets, not the original model's weights.
Later evaluation of an RL adapter will require the **merged base**, since that
is the base the new adapter was trained against.

## Execution

Both hosts use commit `45693b2` with matching source, dependency origins,
config/selection bytes, PostgreSQL/pool settings and renderer assets. The
environment service owns the database worker pool exclusively during training.
Once it is ready, start the GPU-side training stage:

```bash
uv run --frozen --extra gpu qorl experiment run \
  experiments/028-hybrid-rl-smoke --stage train
```

No SFT preparation stage is needed. PostgreSQL execution stays on the database
host throughout the run.

## What constitutes a useful smoke result

Record results in this experiment's `results.md` after the run. Check the gates
in order: matching claim, successful FLOPper-to-Lambda inference probe, a
complete native episode, then optimizer updates and weight publication.

Verify all four updates and their saved checkpoints, returned logprobs/token
masks, anchored advantages, and subsequent rollouts carrying an updated policy
version. Check actual gradient/weight changes rather than inferring learning
from step numbers. If all advantages are zero, report that limitation.
Inspect `training/report.json`, `training/remote-readiness.json`,
`training/remote-cleanup.json` and the service's `rollouts/` and `episodes/`
evidence. A working transport with invalid actions is still not policy-quality
evidence.

At exit, the trainer drains its remote work but does not terminate the
externally owned service. Stop the FLOPper service with Ctrl-C in its terminal,
then confirm its pool has shut down before another database experiment.
