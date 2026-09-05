# PostgreSQL image

This directory builds the database runtime shared by every QORL worker. It
contains PostgreSQL 18.6, `pg_hint_plan` 1.8.0, and the first-start role
bootstrap. It contains no workload data, Python code, model weights, or
host-specific resource shape.

`configs/NNN-pgconf*/` holds the selectable PostgreSQL configurations. Each
directory contains `pg.conf`, its machine-readable expectations, and a short
description of its differences from stock PostgreSQL. Reusable validation and
state-capture commands live in `scripts/`.

Create the next numbered config as a copy of the default with:

```bash
docker/postgres/scripts/create-new-config.sh
```

The script assigns the next three-digit prefix, updates the copied config ID,
and leaves the new config ready for its deliberate settings edits.

Exact upstream versions, commits, and checksums live in `versions.json`. The
Dockerfile also records them as image labels and verifies the compiled
extension before the image is accepted.

Container resources are defined in [`docker/worker_pool/configs/`](../worker_pool/README.md).
Calibration, training, and benchmark runs default to `002-poolconf-4x8`;
fixture construction and restore verification use `000-poolconf-1x32`.
The Python worker loads the selected configuration and supplies Compose's resource
variables. Pool selection is independent of the PostgreSQL config.
