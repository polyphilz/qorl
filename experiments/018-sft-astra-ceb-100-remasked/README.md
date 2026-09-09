# 018-sft-astra-ceb-100-remasked

Method: `sft`. Seed: `42`.
Defaults copied from `configs/defaults/002-sft.toml`; `config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/018-sft-astra-ceb-100-remasked/<run-number>/`.

Dataset input (read-only): `outputs/017-sft-astra-ceb-100/000/generation/conversations`. Original split assignments and seeds are retained.

Each execution allocates a numbered run and copies its config and task selections. Calibration retains partial results on failure.

Stage commands:

```bash
uv run --extra gpu qorl experiment run experiments/018-sft-astra-ceb-100-remasked --stage prepare
uv run --extra gpu qorl experiment run experiments/018-sft-astra-ceb-100-remasked --stage train --run 000
uv run --extra gpu qorl experiment run experiments/018-sft-astra-ceb-100-remasked --stage evaluate --run 000 --checkpoint /path/to/checkpoint --split validation
```
