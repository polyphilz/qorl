# QPRL PostgreSQL image

This directory builds the shared server image used by every PostgreSQL worker.

## The Docker vocabulary

- An **image** is an immutable template. Ours contains PostgreSQL,
  `pg_hint_plan`, configuration, and first-start bootstrap scripts.
- A **container** is one running process made from that image. Later, each
  measurement worker will own one container and run at most one timed query.
- A **volume** is persistent mutable storage attached to a container. JOB data,
  indexes, and the frozen statistics snapshot belong here, not in this base
  image.
- A **registry** stores and distributes images. We do not need one to begin:
  all local workers can use the same local image tag. Before experiments move
  between machines, publish the built image to a private registry or transfer a
  `docker save` archive and record its immutable digest.

So “build the image” is the right verb. We *find* and pin the official
PostgreSQL base image, then *build* our thin project-specific image on top.

## What is pinned

- PostgreSQL `16.15` on Debian Bookworm, by Docker image index digest.
- The benchmark target platform, `linux/amd64` (FLOPper).
- `pg_hint_plan` `1.6.2`, by tag, commit, and source-archive SHA-256.

See `versions.json` for the machine-readable provenance. A successful build
produces a new project image. Its resulting image ID/digest must also be stored
in each experiment manifest; the inputs above do not substitute for recording
the actual output artifact. The image also carries the complete builder package
list and a SHA-256 for the compiled extension under `/usr/share/qprl/`.

Bookworm is explicit rather than relying on the floating default Debian suite.
It also gives us the normal glibc/PGDG build environment for a C extension,
which is less surprising here than Alpine/musl.

## What is deliberately in the image

- PostgreSQL server and standard client utilities from the official image.
- A compiled, checksum-verified `pg_hint_plan` module.
- `shared_preload_libraries = 'pg_hint_plan'`.
- `autovacuum = off`, so neither vacuum nor auto-analyze mutates the frozen
  benchmark snapshot during collection.
- UTC logging/time settings and durable PostgreSQL defaults.
- A bootstrap superuser supplied at runtime and a minimally privileged
  `qp_agent` login whose default transactions are read-only.

`CREATE EXTENSION pg_hint_plan` is run even though comment hints only require
the library to be loaded. This makes the installed extension version directly
queryable and leaves the optional hint-table schema available; the hint table
itself remains disabled.

## What is deliberately not in the image

- JOB data, indexes, or statistics. Those will become a separately checksummed
  snapshot so the same server image can host JOB, generated IMDb training
  tasks, and later transfer workloads.
- Python, the agent/orchestrator, vLLM, CUDA, or model weights. Database workers
  do not need them, and mixing those layers would make every code change rebuild
  a large database image.
- Guessed memory, cost-model, JIT, or parallelism tuning. Those settings can
  change physical plans. We will choose and freeze them after deciding worker
  count, CPU pinning, RAM allocation, and the measurement protocol.
- A fixed `statement_timeout`. The protocol requires a task-relative timeout;
  the trusted harness will apply it per transaction.

## Build and run

From the repository root:

```bash
cp .env.example .env
# Replace both placeholder passwords in .env.
docker compose build postgres
docker compose up --detach postgres
docker compose ps
./docker/postgres/smoke-test.sh
```

The development port is bound only to localhost at `127.0.0.1:55432` by
default. The smoke test checks versions and configuration, forces a real index
scan with a hint, verifies the agent can connect, and verifies that it cannot
create persistent or temporary tables.

Stop the container without deleting its volume:

```bash
docker compose down
```

`docker compose down --volumes` deletes the database volume and is therefore
intentionally not part of the normal workflow.

## Sharing the built artifact

Every worker on FLOPper can refer to the same local tag:

```text
qprl-postgres:16.15-pg_hint_plan-1.6.2
```

Record the local content-addressed image ID with:

```bash
docker image inspect qprl-postgres:16.15-pg_hint_plan-1.6.2 \
  --format '{{.Id}}'
```

For another machine, prefer a registry pinned by digest. An offline alternative
is `docker save` on the source and `docker load` on the destination; checksum
the archive during transfer. Do not rebuild independently on every worker and
assume matching tags imply matching bytes.

## Two important caveats

1. `autovacuum = off` is correct only for the immutable benchmark snapshot.
   During snapshot construction, the trusted loader must run an explicit final
   `VACUUM (FREEZE, ANALYZE)` or equivalent agreed procedure before the snapshot
   is sealed.
2. The database role is defense in depth. The primary safety boundary remains
   the typed harness: it owns the fixed SQL, starts an explicitly read-only
   transaction, applies a timeout, and never sends model-authored SQL.
