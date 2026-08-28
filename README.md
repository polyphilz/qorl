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

`run` currently performs the frozen five-candidate random structured-action
baseline over JOB and records its results under `outputs/runs/`.
