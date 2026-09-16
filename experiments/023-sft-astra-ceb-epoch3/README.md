# 023-sft-astra-ceb-epoch3

Continue experiment 020's completed step-764 checkpoint for one additional epoch:
382 new optimizer updates, 1,146 cumulative. Keep the identical prepared training
and validation rows, original split assignments and seeds, 49,152-token context,
batch/microbatch 1, LoRA rank 16/alpha 32/dropout 0, constant AdamW learning rate
of 1e-4, BF16 and the existing native runtime settings. Training uses GPU 0 on
the training host. `training.epochs = 3` is the cumulative epoch target.

Source checkpoint (read-only):
`outputs/020-sft-astra-ceb-epoch2/000/training/checkpoints/step_764`.
Source conversations (read-only):
`outputs/020-sft-astra-ceb-epoch2/000/dataset`.

The new checkpoint is owned by `outputs/023-sft-astra-ceb-epoch3/000/`.
Validation runs on the incoming weights at step 764 and after the added epoch
at step 1146. The existing measurement configuration is retained with the
source recipe; it does not execute PostgreSQL during SFT training. Reevaluate
checkpoints on the benchmark host under one consistent, audited harness configuration.

Created with `qorl experiment create --method sft --dataset-from
outputs/020-sft-astra-ceb-epoch2/000/dataset --tasksets test=job`, then aligned
with the source run's complete configuration except name, dataset source and
total epochs.

Stage commands on the training host, from the repository root:

```bash
uv run --frozen --no-sync qorl experiment run experiments/023-sft-astra-ceb-epoch3 --stage prepare
uv run --frozen --no-sync qorl experiment run experiments/023-sft-astra-ceb-epoch3 --stage train --run 000 \
  --resume-from outputs/020-sft-astra-ceb-epoch2/000/training/checkpoints/step_764
```

The commands require the pinned GPU environment. On a fresh Linux host, install
the dependencies with `uv sync --frozen --extra gpu`.

Follow training progress from the repository root:

```bash
tail -f outputs/023-sft-astra-ceb-epoch3/000/training/logs/attempt_1/trainer.log
```

The final adapter is in `000/training/checkpoints/step_1146/adapter`.
