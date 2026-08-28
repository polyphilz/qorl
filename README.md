# QORL

QORL (Query Optimization with Reinforcement Learning) is a focused research
harness for training and evaluating an agent that steers PostgreSQL's query
optimizer toward faster physical plans.

```bash
uv sync
uv run qorl calibrate
uv run qorl run
```

`calibrate` restores the pinned JOB snapshot, starts one isolated PostgreSQL
worker, measures all 113 default queries, records results and environment
identity under `outputs/calibration/`, and removes the worker afterward.

`run` loads `configs/evaluation/run-v1.json`, runs the configured five-candidate
policy over JOB, and records trusted results plus the complete policy trace
under `outputs/runs/`. The default policy is the vanilla Qwen3.5-2B `qo-agent`
served through a local OpenAI-compatible vLLM endpoint.

Keep vLLM in a separate environment so its CUDA dependencies do not enter QORL
or break non-GPU development environments:

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python 'vllm==0.27.1'
./scripts/serve-qwen35-2b.sh
```

To reproduce the frozen random baseline instead:

```bash
QORL_RUN_CONFIG=configs/evaluation/random-v1.json uv run qorl run
```
