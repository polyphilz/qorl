# 003-poolconf-lambda-4x8

Four PostgreSQL containers on the 52-vCPU Lambda H100 instance, each with
8 GiB RAM, eight vCPUs, and a 1 GiB `/dev/shm` limit. Swap is disabled.
Host ports: 56000–56003.

The guest reports sibling threads in adjacent pairs: `0,1`, `2,3`, etc.
Each consecutive eight-vCPU block therefore maps to four reported cores.
All allocations use NUMA node 0; CPUs 32–51 remain outside the pool.
The startup topology check validates this mapping on the actual host.

This matches the container count and per-worker resources of FLOPper's
`002-poolconf-4x8`, with CPU numbering adjusted for Lambda. Guest topology
does not establish exclusive ownership of the underlying physical cores.
