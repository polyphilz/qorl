# PostgreSQL image

Builds the PostgreSQL image used by QORL workers, including database-role setup.
Workload data is loaded separately.

PostgreSQL is pinned to 18.6; `pg_hint_plan` to PG18 commit `5af0c526b26c`
(version 1.8.1). Full source pins and checksums are in [versions.json](versions.json).

`configs/` contains selectable PostgreSQL settings and their expected values.
`scripts/` contains config creation, validation, and state-capture helpers.
Create a config by copying the default with:

```bash
docker/postgres/scripts/create-new-config.sh
```

Container counts and resources are configured separately in [worker_pool/](../worker_pool/README.md).
