# 000-pgconf-default

Uses sane PostgreSQL defaults. Relative to stock PostgreSQL, it loads
pg_hint_plan and disables autovacuum, logical-replication workers, GEQO,
JIT, and huge pages to keep plan steering and measurements deterministic.
