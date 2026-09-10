# 023-sft-astra-ceb-epoch3

Continue experiment 020's completed step-764 checkpoint for one additional epoch:
382 new optimizer updates, 1,146 cumulative. Keep the identical prepared training
and validation rows, original split assignments and seeds, 49,152-token context,
batch/microbatch 1, LoRA rank 16/alpha 32/dropout 0, constant AdamW learning rate
of 1e-4, BF16 and the existing native runtime settings. Training uses GPU 0 on
Lambda. `training.epochs = 3` is the cumulative epoch target.

Source checkpoint (read-only):
`outputs/020-sft-astra-ceb-epoch2/000/training/checkpoints/step_764`.
Source conversations (read-only):
`outputs/020-sft-astra-ceb-epoch2/000/dataset`.

The new checkpoint is owned by `outputs/023-sft-astra-ceb-epoch3/000/`.
Validation runs on the incoming weights at step 764 and after the added epoch
at step 1146. The existing measurement configuration is retained with the
source recipe; it does not execute PostgreSQL during SFT training. Reevaluate
checkpoints on FLOPper under one consistent, audited harness configuration.

Created with `qorl experiment create --method sft --dataset-from
outputs/020-sft-astra-ceb-epoch2/000/dataset --tasksets test=job`, then aligned
with the source run's complete configuration except name, dataset source and
total epochs.

Stage commands on Lambda, from the repository root:

```bash
uv run --frozen --no-sync qorl experiment run experiments/023-sft-astra-ceb-epoch3 --stage prepare
uv run --frozen --no-sync qorl experiment run experiments/023-sft-astra-ceb-epoch3 --stage train --run 000 \
  --resume-from outputs/020-sft-astra-ceb-epoch2/000/training/checkpoints/step_764
```

The commands use Lambda's already-installed GPU environment. On a fresh Linux
host install the pinned GPU dependencies with `uv sync --frozen --extra gpu`.

Run `000` was launched detached on Lambda. Follow progress from its repository root:

```bash
tail -f outputs/023-sft-astra-ceb-epoch3/000/training/logs/attempt_1/trainer.log
```

Launch metadata and exit status are recorded in `000/train-launch.json` and
`000/train-exit.json`; startup output is in `000/train-launch.log`. The final
adapter will be in `000/training/checkpoints/step_1146/adapter`.
