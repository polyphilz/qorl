# QORL

QORL (Query Optimization with Reinforcement Learning) is a focused research
harness for training and evaluating an agent that steers PostgreSQL's query
optimizer toward faster physical plans.

`benchmarks/` holds stable workloads, `experiments/NNN-name/` collocates each run's
inputs, and [`model/configs/`](model/README.md) holds model and sampling settings.

```bash
uv sync
uv run qorl calibrate
uv run qorl calibrate ceb
uv run qorl run
```

`calibrate` measures all 113 JOB queries; `calibrate ceb` measures the complete
CEB workload. Both default to the four-worker PostgreSQL pool used by training,
record results and environment identity under `outputs/calibration/`, and remove
their workers afterward. Select one, two, or four containers with `--pool-config`:

```bash
uv run qorl calibrate job \
  --pool-config docker/worker_pool/configs/001-poolconf-2x16 \
  --postgres-config docker/postgres/configs/001-pgconf
```

The [worker pool guide](docker/worker_pool/README.md) lists resource allocations.
Pool selection and PostgreSQL settings are independent.

Pass a versioned selection manifest to calibrate only a workload slice. If the
manifest contains multiple splits, select one explicitly:

```bash
uv run qorl calibrate ceb \
  --selection experiments/004-rl-run-v2/selection.json
uv run qorl calibrate ceb \
  --selection path/to/selection.json --split validation
```

`run` loads `experiments/000-vanilla-baseline/run.json`, which selects the
shared policy in `model/configs/000-modelconf/modelconf.json`. It runs one JOB task per worker,
using the default four-worker pool or the selected `--pool-config`,
then records trusted results plus the complete policy trace under
`outputs/runs/`. The default policy is the untrained
`empero-ai/Qwen3.8-4B-Distill` `qo-agent` served through a local
OpenAI-compatible vLLM endpoint.

Keep vLLM in a separate environment so its CUDA dependencies do not enter QORL
or break non-GPU development environments:

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python 'vllm==0.27.1'
```

The SFT gate and live-validation runners manage their model servers. A direct `qorl run`
requires a server matching its selected policy configuration.

To reproduce the frozen random baseline instead:

```bash
uv run python experiments/000-vanilla-baseline/run.py
```
