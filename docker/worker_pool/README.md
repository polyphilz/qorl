# PostgreSQL worker pools

Each directory in `configs/` contains a `poolconf.json` and a short README.
The directory name is the pool ID. Each worker entry starts one PostgreSQL
container; the worker list determines the container count.

| Config | Containers | RAM each | Physical cores each | Host ports |
| --- | ---: | ---: | ---: | --- |
| `000-poolconf-1x32`       | 1 | 32 GiB | 16 | 55432       |
| `001-poolconf-2x16`       | 2 | 16 GiB |  8 | 56000–56001 |
| `002-poolconf-4x8`        | 4 |  8 GiB |  4 | 56000–56003 |
| `003-poolconf-lambda-4x8` | 4 |  8 GiB |  4 | 56000–56003 |

The first three configurations target FLOPper. They allocate 32 GiB and 16 physical
cores in total, including both hardware threads of each core. `physical_core_count`
checks the intended allocation against the host topology. `cpuset_mems` selects
NUMA node 0. Each container has a 1 GiB `/dev/shm` limit within its RAM allowance
and has swap disabled.

The Lambda profile matches FLOPper's four-worker resource limits using CPU sets
`0-7`, `8-15`, `16-23`, and `24-31`. The 52-vCPU Lambda guest reports adjacent
sibling threads, unlike FLOPper's numbering; these profiles are host-specific.
Guest-reported cores do not imply exclusive physical cores on the cloud host.

Each worker gets its own Compose project and restored database volume. The loader
validates the config, rejects overlapping CPUs or duplicate ports, and checks the
physical core assignments before starting containers. Results record the pool ID,
config checksum, worker allocation, and selected PostgreSQL configuration.

The configurations reuse ports and CPU assignments; run one pool at a time.
