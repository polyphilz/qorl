# QORL

QORL (Query Optimization with Reinforcement Learning) is a focused research
harness for training and evaluating an agent that steers PostgreSQL's query
optimizer toward faster physical plans.

```bash
uv sync
uv run qorl calibrate
uv run qorl calibrate ceb
uv run qorl run
```

`calibrate` measures all 113 JOB queries on one isolated PostgreSQL worker.
`calibrate ceb` measures the complete CEB workload concurrently on the same
persistent four-worker pool used by training. Both record results and
environment identity under `outputs/calibration/` and remove their workers
afterward.

Pass a versioned selection manifest to calibrate only a workload slice. If the
manifest contains multiple splits, select one explicitly:

```bash
uv run qorl calibrate ceb \
  --selection data/ceb/ceb-v1/rl-run-v2.json
uv run qorl calibrate ceb \
  --selection path/to/selection.json --split validation
```

`run` loads `configs/evaluation/run-v1.json`, runs the configured five-candidate
policy over JOB, and records trusted results plus the complete policy trace
under `outputs/runs/`. The default policy is the untrained
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
QORL_RUN_CONFIG=configs/evaluation/random-v1.json uv run qorl run
```
