# 008-pg-32mb-wm-noise-baseline

Method: `calibrate`. Seed: `42`.
Defaults copied from `configs/defaults/000-calibration.toml`; `config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/008-pg-32mb-wm-noise-baseline/<run-number>/`.

Each execution allocates a numbered run and copies its config and task selections. Calibration retains partial results on failure; resumption is not supported.

Stage commands:

```bash
uv run qorl experiment run experiments/008-pg-32mb-wm-noise-baseline --stage calibrate
```
