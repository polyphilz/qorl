# IMDb preparation

Run these commands from the repository root on the benchmark host, with the
PostgreSQL image already built and credentials supplied through the environment or `.env`:

```bash
uv run python -m scripts.imdb.fetch
uv run python -m scripts.imdb.load_verify_archive
```

1. Fetch downloads the raw CSV archive under `data/raw/`, verifies its size and
   SHA-256 against the manifest, and extracts it. Cached archives get the same checks.
2. Load checks the extracted files against the manifest, creates a fresh
   container, imports the CSVs, builds indexes and statistics,
   verifies the database, stops PostgreSQL, and writes `data/imdb.tar.gz`.

The load command uses `000-pgconf-default` and the single-container
`000-poolconf-1x32` configuration. It does not fetch missing inputs.
It removes its temporary database container and volume on success;
failures retain them for inspection. Existing archives and verification reports are not overwritten.

`manifest.json` is the checked-in source recipe and expected database contents.
`schemas.py` defines its types; fetch validates the manifest before downloading.
`load.sql` creates the loaded database and finalizes its statistics.
It vendors the table and index definitions inline from JOB's
[`schema.sql`](https://github.com/gregrahn/join-order-benchmark/blob/a39603662e023e449cb2121997a5034df9e02ebf/schema.sql)
and [`fkindexes.sql`](https://github.com/gregrahn/join-order-benchmark/blob/a39603662e023e449cb2121997a5034df9e02ebf/fkindexes.sql),
unchanged from commit `a39603662e023e449cb2121997a5034df9e02ebf`. IMDb preparation does not download the JOB repository.

`load_verify_archive.py` owns the input and database checks and writes
`data/imdb-verification/loaded.json`; all of `data/` is ignored.
Database workers restore directly from `data/imdb.tar.gz`. Preparation verifies
the loaded database but does not perform an archive restore-and-compare test.
