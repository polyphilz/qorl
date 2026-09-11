# 026-qwen-4b-job-epoch2-v7-5cand-1rollout

Method: `eval`. Seed: `42`.
Defaults copied from `configs/defaults/002-eval.toml`; `config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/026-qwen-4b-job-epoch2-v7-5cand-1rollout/<run-number>/`.

Each execution allocates a numbered run and copies its config and task selections. Calibration retains partial results on failure.

Stage commands:

```bash
uv run --extra gpu qorl experiment run experiments/026-qwen-4b-job-epoch2-v7-5cand-1rollout --stage evaluate --split test
```

Epoch-2 baseline under qo-agent interface v7, for comparison with the new SFT
run in experiment 025. Matches experiment 024's 113 JOB tasks and all evaluation
settings; only the adapter and experiment name differ. Five candidate attempts,
one rollout per task, seed 42, thinking enabled, 49,152-token context, and
8,192-token maximum replies. Uses three initial default measurements and the
FLOPper four-worker pool with 2 GiB PostgreSQL shared buffers.

Adapter: experiment 020, step 764 (epoch 2). Tensor SHA-256:
`d1f055e4a347fca34185a34d2f9baf22f9197e006dacfb0b018b52f178c918f4`.
The serving base and exported adapter checksums were verified on FLOPper.
Interface v7 comes from the executing checkout; the experiment config does not
pin harness code.

Completed run analysis: [results.md](results.md).
