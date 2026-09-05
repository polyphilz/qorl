# IMDb fixture scripts

These scripts prepare the shared IMDb database. Inputs and archive metadata live
under [imdb/](../../imdb/README.md).

- `fetch_imdb.py`: download/check IMDb rows and the pinned schema/index definitions.
- `load_imdb.sh`: create a fresh database, load rows, create indexes, and finalize statistics.
- `verify_imdb.py`: check tables, rows, schema, indexes, statistics, and representative query results.
- `archive_imdb.py`: archive a cleanly stopped PostgreSQL volume and record its checksum.
- `restore_verify_imdb.sh`: restore into a fresh volume and compare the database fingerprints.
- `update_fixture_runtime.py`: update image metadata after successful restore verification.
- `build_imdb.sh`: run the complete build, archive, and restore-verification sequence.

The build and restore use the single-worker pool configuration
`docker/worker_pool/configs/000-poolconf-1x32`. The database image must already
exist; startup does not rebuild it. Runtime credentials come from the environment
or the ignored `.env` file.

All containers use the root `compose.yaml`. It mounts the IMDb CSV and schema
directories read-only; the loader sets their paths. Normal workers do not use
these inputs, and Docker creates empty default directories when they are absent.

```bash
uv run python -m scripts.fixtures.fetch_imdb
./scripts/fixtures/build_imdb.sh --output-dir /path/to/new-imdb-build
```

The default output directory is `imdb/`. Existing archives and verification
records are never overwritten. A successful build removes its temporary database
volumes; failed resources are retained for diagnosis.

Verify the checked-in fixture against the current image with:

```bash
./scripts/fixtures/restore_verify_imdb.sh \
  --archive-manifest imdb/archive.json \
  --build-verification imdb/verification/build.json \
  --output-dir imdb/verification/restore \
  --refresh-runtime-identity
```

The verifier uses representative SQL from `benchmarks/job/queries/` as integrity
checks, not performance measurements. It reads their selection from the JOB
manifest. It does not require the raw CSVs or downloaded repository.

The IMDb fetcher and `scripts.benchmarks.fetch_job` share the cached upstream
repository under `benchmarks/raw/job/`, using `scripts/shared/source_archive.py` for
checksummed downloads and extraction. The fixture fetcher validates schema/index
files; the JOB fetcher validates query files and never downloads IMDb table data.
