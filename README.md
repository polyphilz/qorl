<h1 align="center">QORL: Query Optimization via Reinforcement Learning</h1>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/qorl-gh-banner-dark-optimized.gif">
    <source media="(prefers-color-scheme: light)" srcset="assets/qorl-gh-banner-light-optimized.gif">
    <img src="assets/qorl-gh-banner-light-optimized.gif" width="100%" alt="Animated QORL banner">
  </picture>
</p>

`qorl` is a research harness for training and evaluating an agent that steers
PostgreSQL's query optimizer toward faster physical plans.

## Requirements

- Python 3.12
- Linux x86-64 for training
- The normal install includes a `prime-rl` fork and PyTorch for CPU tests. The `gpu` extra adds vLLM, FlashAttention, and the CUDA training dependencies

## Development

For local development:

```bash
uv sync --frozen
uv run --frozen pytest
```

On the Linux GPU host:

```bash
uv sync --frozen --extra gpu
uv run --extra gpu
```

## Configuration and setup

The [worker pool guide](docker/worker_pool/README.md) lists resource allocations.
Pool selection and PostgreSQL settings are independent.

`benchmarks/` holds stable workloads, `experiments/NNN-name/` collocates each run's
inputs, and [`configs/defaults/`](configs/defaults/) supplies experiment defaults.

IMDb preparation is under [`scripts/imdb/`](scripts/imdb/README.md); raw inputs,
the prepared archive, and verification reports live in ignored `data/`.
JOB and CEB SQL and task inventories are checked in under `benchmarks/`.
Benchmark identifiers are `job` and `ceb`; the database fixture identifier is `imdb`.

## Create an experiment

Choose one of four experiment types with `--method`:

| Method | What it does |
| --- | --- |
| `calibrate` | Runs the same queries repeatedly to see how much their execution times vary |
| `eval` | Runs the agent to test whether a model can find faster plans than PostgreSQL's defaults |
| `sft` | Trains a model to optimize queries by learning from another model's examples |
| `rl` | Trains a model through trial and error, rewarding it for making queries faster |

Creation writes configuration and resolved task IDs without starting a model or
database. For example:

```bash
uv run qorl experiment create --name buffer-study --method calibrate \
  --tasksets 'test=job' \
  --postgres-config docker/postgres/configs/000-pgconf-default \
  --pool-config docker/worker_pool/configs/002-poolconf-4x8
```

The created `experiments/NNN-buffer-study/` contains `config.toml`, `test-tasks.json`,
`README.md`, and a thin `run.py`. Defaults come from the highest numeric version
in [`configs/defaults/`](configs/defaults/README.md). The seed defaults to 42.

SFT/RL require `train=...` and `validation=...`, with optional `test=...`.
Evaluation/calibration require only `test=...`. Splits must have disjoint query
topologies. Model-based methods require a complete local model directory or a
Hugging Face ID plus an immutable `--base-model-revision`. Hosted evaluation selects
`--model-provider openai --base-model-name-or-path gpt-6-astra`; local evaluation also
accepts a separate `--adapter-path` without merging it.

Local models and Astra use the same agent tools and budgets. Astra uses the
Responses API and the `OPENAI_API_KEY` environment variable. Creation copies its
connection and inference preset from `configs/defaults/models/` into the
experiment's `config.toml`.

SFT `--dataset-from` imports a QORL conversation artifact's saved train/validation
selections and original seeds. It accepts only an optional new `test=...` selection.
Without reuse, generator identity and generation count remain explicit placeholders.
Creation checks artifact metadata and file presence. Calibration and live evaluation
are executable.

For a fresh batch, `--exclude-tasks-from experiments/NNN-name/training-tasks.json`
excludes that selection's IDs and identical SQL hashes before sampling. The flag
is repeatable; requested counts must fit the remaining queries. Creation saves
frozen `excluded-tasks-*.json` inputs alongside the resolved selections. The flag
cannot be combined with `--dataset-from`, which preserves its original selections.

Set `[evaluation] selection_policy = "best_feedback"` to add an evaluation-only
selection ablation (the default is `"model"`). After the same conversation, the
rule chooses the eligible, non-timeout candidate with the highest completed
feedback ratio, requiring at least 1.05x; otherwise it keeps the default. Ties
choose the earliest attempt. The choice is fixed before final timing. Different
choices receive fresh final pairs; identical choices reuse the same final result,
including timeouts. Records and summaries retain the model outcome under
`rollout` / `performance` and the rule's result under `feedback_selection`.
Search failures remain failures in both summaries; keeping default does not
count as producing a valid candidate. Additional executions are counted separately.
Training and teacher generation always retain the model's own final decision.
