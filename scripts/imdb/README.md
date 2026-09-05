# IMDb preparation

Run these commands from the repository root on the benchmark host, with the
PostgreSQL image already built and credentials supplied through the environment or `.env`:

```bash
uv run scripts/imdb/fetch_imdb.py
uv run scripts/imdb/load_verify_archive.py
uv run scripts/imdb/restore_and_verify.py
```

1. Fetch downloads and checks the raw CSVs under `data/raw/`. Schema/index sources
   share the JOB repository cache under `benchmarks/raw/job/`.
2. Load creates a fresh container, imports the CSVs, builds indexes and statistics,
   verifies the database, stops PostgreSQL, and writes `data/imdb.tar.gz`.
3. Restore unpacks that archive into another fresh container and compares schema,
   index, statistics, row-count, and representative-query fingerprints with the loaded database.

The last two commands use `000-pgconf-default` and the single-container
`000-poolconf-1x32` configuration. Neither fetches missing inputs or invokes another stage.
Successful commands remove their temporary database containers and volumes;
failures retain them for inspection. Existing archives and verification reports are not overwritten.

`imdb-metadata.json` is the checked-in source recipe and expected database contents.
`load.sql` creates the loaded database and finalizes its statistics.
`verify.py` provides the shared checks. Generated reports are
`data/imdb-verification/loaded.json` and `restored.json`; all of `data/` is ignored.
Database workers restore directly from `data/imdb.tar.gz`.
