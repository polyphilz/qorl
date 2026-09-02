# `benchmark-v2` contract

This directory is the immutable database contract for the primary baseline.
Host filenames are concise; the `benchmark-v2` identifier and stable in-image
paths remain unchanged.

| File | Purpose | Used by |
|---|---|---|
| `benchmark.conf` | Explicit PostgreSQL server configuration | PostgreSQL startup |
| `benchmark.expected.json` | Machine-readable expected settings, database properties, role behavior, and forbidden backends | In-image assertion and prompt-setting equivalence check |
| `assert-benchmark-config.sh` | Fails on contract or declared container-resource drift | Healthcheck, worker startup, smoke test, environment capture |
| `dump-postgres-state.sh` | Emits effective identity, non-default settings, `pg_settings`, and `SHOW ALL` | Environment capture before and after measurements |

The contract contains database behavior, not host topology. CPU, memory,
shared memory, and ports live in named profiles under `configs/postgres/`; QORL
enforces zero swap for every worker.

The host-side `scripts/docker/smoke-test-postgres.sh` checks this assertion plus
real hint behavior, prompt-visible settings, and restricted-role permissions.
`python -m qorl.db.capture` records the asserted state and full
runtime identity; it does not define either one.

## Why GEQO is off

The pinned `pg_hint_plan` source says: “In the case using GEQO, only scan method
hints and Set hints have effect.” ([source](https://github.com/ossc-db/pg_hint_plan/blob/88437bde5947e7bc40719e8b82eb8be2b39d71e8/pg_hint_plan.c#L4640-L4644))
At PostgreSQL's default threshold, that made join-order, join-method, memoize,
and row-correction actions ineffective on queries with 12 or more relations.
`benchmark-v2` therefore turns GEQO off so PostgreSQL's default and every
candidate use the same deterministic join search and every exposed hint family
remains effective.

Historical `benchmark-v1` results remain valid records of those runs, but its
GEQO-eligible tasks are not directly comparable with `benchmark-v2`: 20 of 113
JOB tasks and 100 of 400 RL-v2 training tasks crossed the 12-relation threshold.

Change this directory only when intentionally defining a new benchmark
contract. A path-only reorganization still changes the Docker build output, so
the rebuilt image must pass smoke and clean-room snapshot restore verification.
