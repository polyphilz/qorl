# PostgreSQL worker pools

Each directory in `configs/` contains a `poolconf.json` and a short README.
The directory name is the pool ID. Each worker entry starts one PostgreSQL
container; the worker list determines the container count.

| Config | Containers | RAM each | Physical cores each | Host ports |
| --- | ---: | ---: | ---: | --- |
| `000-poolconf-1x32` | 1 | 32 GiB | 16 | 55432 |
| `001-poolconf-2x16` | 2 | 16 GiB | 8 | 56000–56001 |
| `002-poolconf-4x8` | 4 | 8 GiB | 4 | 56000–56003 |

All three allocate 32 GiB and 16 physical cores in total, including both hardware
threads of each core. `physical_core_count` checks the intended allocation against
the host topology. `cpuset_mems` selects NUMA node 0. Each container has a 1 GiB
`/dev/shm` limit within its RAM allowance and has swap disabled. `/dev/shm` is
separate from PostgreSQL's `shared_buffers` setting.

From the repository root, select container resources independently of PostgreSQL
settings:

```bash
uv run qorl calibrate job \
  --pool-config docker/worker_pool/configs/001-poolconf-2x16 \
  --postgres-config docker/postgres/configs/001-pgconf
```

`qorl run` also accepts `--pool-config`. Both commands accept either the directory
or its `poolconf.json`. An explicit option takes precedence over
`QORL_RL_WORKER_POOL_CONFIG`; otherwise that environment variable selects the pool
for all harnesses, including training and SFT. The default is `002-poolconf-4x8`.
Fixture construction and restore verification use `000-poolconf-1x32`.

Each worker gets its own Compose project and restored database volume. The loader
validates the config, rejects overlapping CPUs or duplicate ports, and checks the
physical core assignments before starting containers. Results record the pool ID,
config checksum, worker allocation, and selected PostgreSQL configuration.

The configurations reuse ports and CPU assignments; run one pool at a time.
