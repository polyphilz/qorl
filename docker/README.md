# Database runtime architecture

QORL separates three things that change for different reasons:

1. [`postgres/`](./postgres/README.md) builds the pinned PostgreSQL image and
   immutable benchmark contract.
2. [`scripts/job/`](../scripts/job/README.md) loads IMDb once, freezes it, and
   seals the stopped data directory as a physical snapshot.
3. `src/qorl/worker.py` restores that snapshot into a fresh volume and starts a
   disposable worker from the pinned image for calibration or measurement.

The snapshot's data identity is independent of its runtime identity. Rebuilding
byte-identical image inputs changes the image ID, but it does not change the
database archive or force workload inventories to be regenerated. The restore
verifier can validate the old archive with the rebuilt image and refresh only
the snapshot manifest's image metadata.

Hardware allocation is also separate from the image. Named profiles under
`configs/postgres/` define CPU affinity, memory, shared memory, and ports; QORL
enforces no swap for every worker. Every measurement records the exact profile
and running container identity.

See [`docs/architecture.md`](../docs/architecture.md) for the complete flow.
