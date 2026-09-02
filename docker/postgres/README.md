# PostgreSQL image

This directory builds the database runtime shared by every QORL worker. It
contains PostgreSQL 18.6, `pg_hint_plan` 1.8.0, the first-start role bootstrap,
and the immutable [`benchmark-v2` contract](./contract/README.md). It contains
no workload data, Python code, model weights, or host-specific resource shape.

Exact upstream versions, commits, and checksums live in `versions.json`. The
Dockerfile also records them as image labels and verifies the compiled
extension before the image is accepted.

Runtime resources are explicit profiles, not Compose defaults:

- `configs/postgres/evaluation-worker-v1.json` defines one evaluation worker.
- `configs/postgres/training-pool-v1.json` defines the four-worker training pool.

The Python worker loads the applicable profile and supplies every required
Compose variable. For a manual development check, load the evaluation profile,
start the service, and run the host-side smoke test:

```bash
source scripts/docker/runtime-profile.sh
qorl_load_postgres_runtime_profile "$PWD" \
  configs/postgres/evaluation-worker-v1.json
docker compose build postgres
docker compose up --detach --wait postgres
scripts/docker/smoke-test-postgres.sh
docker compose down --volumes
```

Building an image and freezing workload data are deliberately separate. See
[`docker/README.md`](../README.md) for the full image-to-fixture-to-worker chain
and [`scripts/job/README.md`](../../scripts/job/README.md) for fixture creation.
