# QORL

QORL (Query Optimization with Reinforcement Learning) is a focused research
harness for training and evaluating an agent that steers PostgreSQL's query
optimizer toward faster physical plans.

`benchmarks/` holds stable workloads, `experiments/NNN-name/` collocates each run's
inputs, and [`model/configs/`](model/README.md) holds model and sampling settings.

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

```bash
uv run --extra gpu qorl calibrate \
  --postgres-config docker/postgres/configs/000-pgconf-default \
  --pool-config docker/worker_pool/configs/002-poolconf-4x8
```

`calibrate` always measures all 113 JOB queries. It requires explicit PostgreSQL
and pool configurations, records results and environment identity under
`outputs/calibration/`, and removes its workers afterward. Select one, two, or four
containers with `--pool-config`:

```bash
uv run qorl calibrate \
  --pool-config docker/worker_pool/configs/001-poolconf-2x16 \
  --postgres-config docker/postgres/configs/000-pgconf-default
```

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
`--model-provider openai` or `anthropic`; local evaluation also accepts a separate
`--adapter-path` without merging it.

SFT `--dataset-from` imports a QORL conversation artifact's saved train/validation
selections and original seeds. It accepts only an optional new `test=...` selection.
Without reuse, generator identity and generation count remain explicit placeholders.
Creation checks artifact metadata and file presence; rendering is gated on 036 Phase 8.
Shared experiment execution is gated on 036 Phase 3; the generated entrypoint
reports that gap instead of claiming success.

## Existing execution commands

`--max-warmup-runs` defaults to 5 and `--num-trials` defaults to 20; both must be
at least 2. Warmups stop early once consecutive plans and buffer counts stabilize.
Trials are the measured executions after warmup, used for the median and CV.
The selected counts are recorded in the calibration manifest.

`run` loads `experiments/000-vanilla-baseline/run.json`, which selects the
shared policy in `model/configs/000-modelconf/modelconf.json`. It runs one JOB task per worker,
using the required `--postgres-config` and `--pool-config` selections,
then records trusted results plus the complete policy trace under
`outputs/runs/`. The default policy is the untrained
`empero-ai/Qwen3.8-4B-Distill` `qo-agent` served through a local
OpenAI-compatible vLLM endpoint.

The SFT gate and live-validation runners manage their model servers. A direct `qorl run`
requires a server matching its selected policy configuration.
The Linux GPU environment uses PyTorch 2.13.0 and vLLM 0.28.0 with CUDA 13.0,
matching `model/configs/002-modelconf/modelconf.json`. Configs 000 and 001 retain
their vLLM 0.27.1 requirement; their version check rejects this environment,
including the frozen config selected by the `qorl run` command below.

```bash
uv run --extra gpu qorl run \
  --postgres-config docker/postgres/configs/000-pgconf-default \
  --pool-config docker/worker_pool/configs/002-poolconf-4x8
```

To reproduce the frozen random baseline instead:

```bash
uv run python experiments/000-vanilla-baseline/run.py \
  --postgres-config docker/postgres/configs/000-pgconf-default \
  --pool-config docker/worker_pool/configs/002-poolconf-4x8
```

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
