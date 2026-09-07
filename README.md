# QORL

QORL (Query Optimization with Reinforcement Learning) is a focused research
harness for training and evaluating an agent that steers PostgreSQL's query
optimizer toward faster physical plans.

`benchmarks/` holds stable workloads, `experiments/NNN-name/` collocates each run's
inputs, and [`configs/defaults/`](configs/defaults/) supplies experiment defaults.

IMDb preparation is under [`scripts/imdb/`](scripts/imdb/README.md); raw inputs,
the prepared archive, and verification reports live in ignored `data/`.
JOB and CEB SQL and task inventories are checked in under `benchmarks/`.
Benchmark identifiers are `job` and `ceb`; the database fixture identifier is `imdb`.

QORL uses Python 3.12 and one lockfile for macOS ARM64 development and Linux
x86-64 training. The normal install includes Prime-RL and PyTorch for CPU tests;
the `gpu` extra adds vLLM, FlashAttention, and the CUDA training dependencies.

For local development:

```bash
uv sync --frozen
uv run --frozen pytest
```

On the Linux GPU host, install with `uv sync --frozen --extra gpu` and use
`uv run --extra gpu` for training and serving commands. A plain `uv sync` removes
optional GPU packages, so keep `--extra gpu` when syncing that environment.

The [worker pool guide](docker/worker_pool/README.md) lists resource allocations.
Pool selection and PostgreSQL settings are independent.

## Create an experiment

Creation writes configuration and resolved task IDs without starting a model or
database. For example:

```bash
uv run qorl experiment create --name buffer-study --method calibrate \
  --tasksets 'test=job' \
  --postgres-config docker/postgres/configs/000-pgconf-default \
  --pool-config docker/worker_pool/configs/002-poolconf-4x8
```

The next `experiments/NNN-buffer-study/` contains `config.toml`, `test-tasks.json`,
`README.md`, and a thin `run.py`. Defaults come from the highest numeric version
in [`configs/defaults/`](configs/defaults/README.md). The seed defaults to 42.

SFT/RL require `train=...` and `validation=...`, with optional `test=...`.
Evaluation/calibration require only `test=...`. Splits must have disjoint query
topologies. Model-based methods require a complete local model directory or a
Hugging Face ID plus an immutable `--base-model-revision`. Hosted evaluation selects
`--model-provider openai --base-model-name-or-path gpt-6-astra`; local evaluation also
accepts a separate `--adapter-path` without merging it.

Local models and Astra use the same agent tools and budgets. Astra uses the
Responses API and the `OPENAI_API_KEY` environment variable; credentials do not
belong in experiment files. Creation copies its connection and inference preset
from `configs/defaults/models/` into the experiment's `config.toml`.

SFT `--dataset-from` imports a QORL conversation artifact's saved train/validation
selections and original seeds. It accepts only an optional new `test=...` selection.
Without reuse, generator identity and generation count remain explicit placeholders.
Creation checks artifact metadata and file presence. Calibration and live evaluation
are executable.

## Run a calibration experiment

On the database host, run the experiment directory printed by creation:

```bash
uv run qorl experiment run experiments/NNN-buffer-study --stage calibrate
```

The command executes the experiment's `run.py`, which delegates to the shared
runner. It allocates `outputs/NNN-buffer-study/000/`, prints that path, and copies
the configuration and resolved task selections there. Calibration records and
per-worker environment captures go in its `calibration/` directory. Another
execution allocates `001`, leaving the earlier results untouched.

Only the IDs in `test-tasks.json` are measured; both JOB and CEB are supported.
Set counts and the per-statement timeout in the experiment's `config.toml`:

```toml
[measurement]
max_warmup_runs = 5
num_trials = 20
default_timeout_seconds = 300.0
```

Both counts must be at least two. Warmups stop early when consecutive plans and
buffer counts stabilize. Trials exclude warmups and supply the median and sample
coefficient of variation. Every query runs under the configured timeout. Individual
query failures retain their completed observations and do not stop other tasks;
the command exits unsuccessfully if any task fails. Workers are closed on exit.

`--run 000` explicitly selects recorded inputs; changed experiment inputs require
a new run. Existing calibration outputs cannot be overwritten, and calibration
does not support `--resume`. Start a new run after an interrupted calibration.

## Training and development

Training integration lives in `src/qorl/training/`; adapter export and merge live
in `src/qorl/adapters/`. Prime-RL loads the environment, task set, and harness
through the `qorl` plugin ID. Run all commands from the repository root:

```bash
uv run --frozen --extra gpu rl --help
uv run --frozen python -m qorl.adapters.merge --help
uv run --frozen python -m qorl.training.audit.dataset --help
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pyright
```

The root test suite includes CPU-side training-plugin and adapter tests. Actual
training, vLLM serving, and benchmark-host integration checks run on Linux.
