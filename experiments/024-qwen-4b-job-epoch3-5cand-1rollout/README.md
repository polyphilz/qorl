# 024-qwen-4b-job-epoch3-5cand-1rollout

Evaluate experiment 023's third-epoch SFT adapter against all 113 JOB queries.
The adapter was exported at step 1146 after three total epochs.

Created with `qorl experiment create` from `configs/defaults/002-eval.toml`,
then configured for one rollout per task, five candidate attempts, thinking enabled,
49,152-token context and 8,192 output tokens. Seed 42, task IDs, sampling settings,
GPU 0, the four-worker pool and PostgreSQL's 2 GiB shared-buffer configuration
match experiment 021. Initial baseline timing uses the newly approved three
measurements; candidate feedback and final paired measurement counts are unchanged.

Base: `empero-ai/Qwen3.8-4B-Distill` at revision
`c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`.
Adapter: `outputs/023-sft-astra-ceb-epoch3/000/training/checkpoints/step_1146/adapter`.
Adapter weights SHA-256:
`84ee7a07696fea72027588d49a1d4d2d0925220ad8790a5231bf3b4ff72dc1b8`.

The adapter is copied to FLOPper and verified against its cached base weights.
After pulling this experiment and the current harness onto FLOPper, run:

```bash
uv run --frozen --extra gpu qorl experiment run experiments/024-qwen-4b-job-epoch3-5cand-1rollout --stage evaluate --split test
```

Each invocation creates a fresh numbered run under
`outputs/024-qwen-4b-job-epoch3-5cand-1rollout/`.

The historical epoch-two evaluation used interface v6 and one initial baseline
measurement. For a controlled epoch comparison, reevaluate epoch two with the
same current harness and measurement settings used here.
