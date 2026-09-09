# 013-qwen-4b-test-5cand-1rollout

Method: `eval`. Seed: `42`.
Defaults copied from `configs/defaults/001-eval.toml`; `config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/013-qwen-4b-test-5cand-1rollout/<run-number>/`.

Each execution allocates a numbered run and copies its config and task selections. Calibration retains partial results on failure.

Stage commands:

```bash
uv run --extra gpu qorl experiment run experiments/013-qwen-4b-test-5cand-1rollout --stage evaluate --split test
```
