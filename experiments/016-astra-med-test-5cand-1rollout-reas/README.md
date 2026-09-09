# 016-astra-med-test-5cand-1rollout-reas

Method: `eval`. Seed: `42`.
Defaults copied from `configs/defaults/001-eval.toml`; `config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/016-astra-med-test-5cand-1rollout-reas/<run-number>/`.

Each execution allocates a numbered run and copies its config and task selections. Calibration retains partial results on failure.

Stage commands:

```bash
uv run qorl experiment run experiments/016-astra-med-test-5cand-1rollout-reas --stage evaluate --split test
```
