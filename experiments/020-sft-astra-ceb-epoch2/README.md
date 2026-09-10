# 020-sft-astra-ceb-epoch2

Continue experiment 018's completed step-382 checkpoint for one additional epoch:
382 new updates, 764 cumulative. Keep the same prepared data, 49,152-token context,
batch/microbatch 1, LoRA 16/32/0, constant AdamW learning rate of 1e-4, BF16 and
native CPU-offload settings. Seed: 42. `config.toml` owns the active values.

Dataset input (read-only): `outputs/018-sft-astra-ceb-100-remasked/000/dataset`. Original split assignments and seeds are retained.

Training uses GPU 0 on Lambda. Source artifacts remain unchanged; the new checkpoint
goes under `outputs/020-sft-astra-ceb-epoch2/000/`. Validation runs on the incoming
weights at step 382 and after the added epoch at step 764. Run the subsequent JOB
rollout evaluation on FLOPper to retain the database setup used for 018 and 019.

Stage commands:

```bash
uv run --frozen --extra gpu qorl experiment run experiments/020-sft-astra-ceb-epoch2 --stage prepare
uv run --frozen --extra gpu qorl experiment run experiments/020-sft-astra-ceb-epoch2 --stage train --run 000 \
  --resume-from outputs/018-sft-astra-ceb-100-remasked/000/training/checkpoints/step_382
uv run --frozen --extra gpu qorl experiment run experiments/020-sft-astra-ceb-epoch2 --stage evaluate --run 000 \
  --checkpoint outputs/020-sft-astra-ceb-epoch2/000/training/checkpoints/step_764 --split test
```
