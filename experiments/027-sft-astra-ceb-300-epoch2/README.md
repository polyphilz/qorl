# 027-sft-astra-ceb-300-epoch2

Continue experiment 025's completed step-1102 checkpoint for one additional
epoch on the same filtered CEB demonstrations. `training.epochs = 2` is the
cumulative target for this dataset: 1,102 additional optimizer updates, ending
at step 2,204. Restore the adapter, AdamW state, scheduler and training progress
with `--resume-from`.

Source checkpoint (read-only):
`outputs/025-sft-astra-ceb-300/002/training/checkpoints/step_1102`.
Source conversations (read-only):
`outputs/025-sft-astra-ceb-300/002/dataset`.

The new run owns its outputs under `outputs/027-sft-astra-ceb-300-epoch2/000/`.
The original native checkpoint and exported adapter at step 1,102 remain intact.
The new final adapter will be at `000/training/checkpoints/step_2204/adapter`.

Keep the source recipe: 49,152-token context, seed 42, batch/microbatch 1,
LoRA rank 16/alpha 32/dropout 0, constant AdamW learning rate 1e-4, BF16 and
GPU 0. Reuse the saved task IDs and conversation files without teacher calls or
resampling. The six excluded early `keep_default` demonstrations stay excluded;
the existing context filter leaves 293 accepted training conversations in
1,102 packed rows and 19 validation conversations in 73 rows. JOB remains the
113-query test set. See [025's filtering notes](../025-sft-astra-ceb-300/README.md)
and [results](../025-sft-astra-ceb-300/results.md).

Created with `qorl experiment create --method sft --dataset-from
outputs/025-sft-astra-ceb-300/002/dataset --tasksets test=job`, then aligned with
the source run's settings except experiment name, dataset import and total
epochs. Validation runs on the incoming weights at step 1,102 and after the
additional epoch at step 2,204.

Run `000` has been prepared on Lambda. Both packed splits match the source
byte for byte, and the full native continuation check passed. Training was
launched on September 11, 2026, at 04:40 UTC, from step 1,102 to step 2,204.
The original checkpoint checksum remains
`a91ee09bea645a93fb8316ec9d7813fe7e18557d98ffa2839aa3b12d53d219c8`.

The commands below use the same installed GPU environment and code snapshot as
025, from `~/qorl/projects/qorl`. Activate it so the trainer also resolves that
environment's `torchrun`:

```bash
source ~/.venvs/qorl-025-training/bin/activate
qorl experiment run \
  experiments/027-sft-astra-ceb-300-epoch2 --stage prepare
```

Continue from the original checkpoint (the resume flag is required):

```bash
qorl experiment run \
  experiments/027-sft-astra-ceb-300-epoch2 --stage train --run 000 \
  --resume-from outputs/025-sft-astra-ceb-300/002/training/checkpoints/step_1102
```

Follow `outputs/027-sft-astra-ceb-300-epoch2/000/training-launch.log` during
startup, then `000/training/logs/attempt_1/trainer.log` for training progress.
Launch metadata and eventual exit status are recorded in `training-launch.json`
and `training-launch.exit`; verification evidence is in
`continuation-preflight.json`.

The first launch picked up the system `torchrun` and failed before model loading
or optimizer updates. Its files are retained under `000/startup-failures/000/`.
The retry activates the GPU environment in the launch process's `PATH`.
