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
- The benchmark image platform, `linux/amd64`.
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
- The versioned `benchmark-v1` contract: explicit stock PostgreSQL planner,
  memory, cost, GEQO, and parallel settings; JIT disabled; and the unused
  logical-replication launcher disabled.
- UTC logging/time settings, explicit durable PostgreSQL defaults, and
  startup assertions over every experiment-critical value.
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
- Hardware-tuned memory or cost-model values. The primary baseline explicitly
  preserves PostgreSQL 16.15's stock values; any hardware-tuned comparison
  will be a separately labeled secondary baseline.
- A fixed `statement_timeout`. The protocol requires a task-relative timeout;
  the trusted harness will apply it per transaction.

## Reproducibility contract

- `benchmark-v1.conf` is the complete versioned PostgreSQL configuration.
- `benchmark-v1.expected.json` is the machine-readable identity, critical
  settings, role, background-process, and container-runtime contract.
- `qprl-assert-benchmark-config` fails if the running server, worker role, or
  container limits differ from that contract.
- `qprl-dump-postgres-state` emits the effective PostgreSQL identity, every
  setting, every value differing from PostgreSQL's compiled default, and
  `SHOW ALL` as the actual benchmark role.
- `scripts/capture-benchmark-environment.py` combines those database artifacts
  with sanitized host, CPU/cache/NUMA, storage, GPU, Docker, image, and cgroup
  identity. It deliberately never records container environment variables,
  which contain the database passwords.

The future orchestrator calls the assertion and capture utilities
automatically before and after a run. They are implementation primitives, not
a manual checklist for the person launching an experiment.

## Build and run

From the repository root:

```bash
cp .env.example .env
# Replace both placeholder passwords in .env.
docker compose build postgres
docker compose up --detach --wait postgres
docker compose ps
./docker/postgres/smoke-test.sh
```

The checked-in Compose service is the current pilot worker profile: CPUs
`8-15,24-31`, 32 GiB of memory, no swap, and 1 GiB of shared memory. It is not a
portable development profile; building the image is portable, but running this
service requires a host with that CPU and memory shape. A later experiment on
different hardware should use a separately versioned runtime profile and record
its actual topology rather than silently changing this one.

The development port is bound only to localhost at `127.0.0.1:55432` by
default. The smoke test checks versions and configuration, forces a real index
scan with a hint, verifies the agent can connect, and verifies that it cannot
create persistent or temporary tables.

Until the orchestrator owns run capture, the underlying capture primitive can
be exercised with an explicit output directory:

```bash
python3 scripts/capture-benchmark-environment.py \
  --output-dir /tmp/qprl-environment-smoke \
  --phase pre
```

Normal benchmark users will not run that command directly.

Stop the container without deleting its volume:

```bash
docker compose down
```

`docker compose down --volumes` deletes the database volume and is therefore
intentionally not part of the normal workflow.

## Sharing the built artifact

Every worker on a single Docker host can refer to the same local tag:

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

## Three important caveats

1. `autovacuum = off` is correct only for the immutable benchmark snapshot.
   During snapshot construction, the trusted loader must run an explicit final
   `VACUUM (FREEZE, ANALYZE)` or equivalent agreed procedure before the snapshot
   is sealed.
2. The database role is defense in depth. The primary safety boundary remains
   the typed harness: it owns the fixed SQL, starts an explicitly read-only
   transaction, applies a timeout, and never sends model-authored SQL.
3. PostgreSQL's normal GEQO policy remains enabled. In the pinned
   `pg_hint_plan` release, join-order and join-method hints cannot take effect
   while GEQO is active; the future `PlanAction` compiler must lower such an
   action with `Set(geqo off)` so the requested semantics are actually applied.

The separate [`scripts/job`](../../scripts/job/README.md) pipeline builds and
clean-room verifies the immutable JOB data snapshot consumed by this image.
