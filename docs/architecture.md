# Architecture

QORL keeps the database bytes, database runtime, workload definitions, and
experiment inputs independently identifiable.

```text
docker/postgres/                 data/job/manifest.json
image + benchmark-v2 contract             |
        |                                  v
        +----------------------> fixture build
                                      |
                                      v
                         checksummed stopped PGDATA
                         artifacts/job-v1/*.snapshot.*
                                      |
data/{job,ceb}/tasks.json              |     configs/postgres/*.json
workload + data identity --------------+-------------+
                                                    |
                                                    v
                                         disposable worker(s)
                                                    |
                                                    v
                              calibration / SFT sampling / RL / evaluation
                                                    |
                                                    v
                                      experiment and output manifests
```

The workload inventories bind only to the frozen data identity: fixture ID,
snapshot ID, archive checksum, and PostgreSQL system identifier. The image ID
and benchmark-contract ID form the runtime identity and are recorded by every
measurement, but an image rebuild does not rewrite stable workload data.

At startup a worker verifies the snapshot archive, exact image ID, benchmark
contract, restricted database role, system identifier, and declared runtime
profile. Before and after measurement it captures effective PostgreSQL state,
host topology, Docker identity, and the selected resource profile.

Fixture construction uses `compose.yaml` plus `compose.fixture-build.yaml`.
Normal workers use only `compose.yaml`; Python supplies all hardware-specific
values from an explicit profile.
