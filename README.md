# QPRL

QPRL is a focused research harness for training and evaluating an agent that
steers PostgreSQL query plans.

```bash
uv sync
uv run qprl calibrate
uv run qprl run
```

`calibrate` restores the pinned JOB snapshot, starts one isolated PostgreSQL
worker, measures all 113 default queries, records results and environment
identity under `outputs/calibration/`, and removes the worker afterward.

`run` currently performs the frozen five-candidate random structured-action
baseline over JOB and records its results under `outputs/runs/`.
