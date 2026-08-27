# QPRL

QPRL is a focused research harness for training and evaluating an agent that
steers PostgreSQL query plans.

```bash
uv sync
uv run qprl calibrate
```

`calibrate` restores the pinned JOB snapshot, starts one isolated PostgreSQL
worker, measures all 113 default queries, records results and environment
identity under `outputs/calibration/`, and removes the worker afterward.
