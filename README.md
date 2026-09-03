# QORL

QORL (Query Optimization with Reinforcement Learning) is a focused research
harness for training and evaluating an agent that steers PostgreSQL's query
optimizer toward faster physical plans.

`data/` holds stable workloads, `experiments/NNN-name/` collocates each run's
inputs, and `configs/` holds settings reused unchanged across experiments. The
[architecture guide](docs/architecture.md) describes the database image,
fixture, worker, and runtime-profile boundary.

```bash
uv sync
uv run qorl calibrate
uv run qorl calibrate ceb
uv run qorl run
```

`calibrate` measures all 113 JOB queries; `calibrate ceb` measures the complete
CEB workload. Both use the same persistent four-worker PostgreSQL pool as
training, record results and environment identity under `outputs/calibration/`,
and remove their workers afterward.

Pass a versioned selection manifest to calibrate only a workload slice. If the
manifest contains multiple splits, select one explicitly:

```bash
uv run qorl calibrate ceb \
  --selection experiments/004-rl-run-v2/selection.json
uv run qorl calibrate ceb \
  --selection path/to/selection.json --split validation
```

`run` loads `experiments/000-vanilla-baseline/run.json`, which selects the
shared policy in `configs/policy/run-v1.json`. It runs up to four JOB tasks
concurrently on the same four-worker pool used for calibration and training,
then records trusted results plus the complete policy trace under
`outputs/runs/`. The default policy is the untrained
`empero-ai/Qwen3.8-4B-Distill` `qo-agent` served through a local
OpenAI-compatible vLLM endpoint.

Keep vLLM in a separate environment so its CUDA dependencies do not enter QORL
or break non-GPU development environments:

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python 'vllm==0.27.1'
./scripts/serve-qwen38-4b-distill.sh
```

To reproduce the frozen random baseline instead:

```bash
uv run python experiments/000-vanilla-baseline/run.py
```

## Development

The repository records mechanical formatting commits in
`.git-blame-ignore-revs`. Configure a fresh clone to honor that file with:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
