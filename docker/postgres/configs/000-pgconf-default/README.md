# 000-pgconf-default

Uses PostgreSQL's stock 128 MiB `shared_buffers`. Relative to stock PostgreSQL,
it loads pg_hint_plan and disables autovacuum, logical-replication workers, GEQO,
JIT, and huge pages to keep plan steering and measurements deterministic.
