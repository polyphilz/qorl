# 017-sft-astra-ceb-100

Method: `sft`. Seed: `42`.
Defaults copied from `configs/defaults/001-sft.toml`; `config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/017-sft-astra-ceb-100/<run-number>/`.

Dataset generation requires `data.generation.model` and `generations_per_task`.

Each execution allocates a numbered run and copies its config and task selections. Calibration retains partial results on failure.

Stage commands:

```bash
uv run --extra gpu qorl experiment run experiments/017-sft-astra-ceb-100 --stage prepare
uv run --extra gpu qorl experiment run experiments/017-sft-astra-ceb-100 --stage train --run 000
uv run --extra gpu qorl experiment run experiments/017-sft-astra-ceb-100 --stage evaluate --run 000 --checkpoint /path/to/checkpoint --split validation
```
