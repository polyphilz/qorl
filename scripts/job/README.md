# JOB fixture pipeline

This directory builds the immutable `job-v1` database fixture. The pipeline
downloads checksum-pinned inputs, loads them once, freezes and analyzes every
JOB table, seals the cleanly stopped PostgreSQL data directory, restores that
archive into a new Docker volume, and refuses success unless the restored
database matches the original build.

## Inputs and trust boundary

`data/manifests/job-v1.json` pins:

- The May 2013 IMDb archive URL, byte length, SHA-256, and every extracted
  member's byte length and SHA-256.
- The exact JOB repository commit and archive SHA-256.
- The schema, foreign-key index definitions, and aggregate identity of all 113
  query files.
- Every expected table row count and the intended final maintenance procedure.

The files under `data/raw/job-v1/` are downloaded inputs and remain ignored by
git. The loader uses the trusted PostgreSQL administrator connection; the model
and `qprl_runner` never receives loading, indexing, vacuuming, or snapshot tools.

## One-command build

The PostgreSQL image must already exist, and the two passwords required by
`compose.yaml` must be present in the process environment or the repository's
ignored `.env` file. From the repository root, run:

```bash
./scripts/job/build-job-v1.sh
```

That command owns the complete workflow, including the clean-room restore
test. Individual scripts exist for orchestration and diagnosis; benchmark users
should not need to remember or invoke the internal steps.

The build and restore use distinct Compose project names and refuse to reuse an
existing data volume. If a step fails, its resources are retained for diagnosis
instead of being silently destroyed; after full success, both temporary
database volumes are removed.

## Outputs

Generated files are written to the ignored `artifacts/job-v1/` directory:

- `job-v1.snapshot.tar.gz`: the stopped physical PostgreSQL data directory.
- `job-v1.snapshot.json`: archive checksum, snapshot ID, source manifest,
  PostgreSQL system identity, and exact image identity.
- `job-v1.database.build.json` and `job-v1.database.restore.json`: table,
  schema, constraint, index, planner-statistics, and representative-query
  fingerprints from both sides of the restore.
- Pre-build and post-restore PostgreSQL settings plus sanitized host, Docker,
  image, topology, and runtime manifests.

The physical snapshot contains PostgreSQL's password hashes. Treat it as a
private benchmark artifact, and supply the same runtime credentials used when
it was built; neither plaintext password is written to the manifests.

## Checked-in JOB task inventory

`data/job/job-v1/queries/` contains the 113 exact, human-readable JOB SQL
files. `data/job/job-v1/tasks.json` is the single machine-readable inventory:
it declares the entire collection held-out test data and records each query's
template, SQL checksum, relation instances, table set, join predicates, and
alias-independent join-graph fingerprint.

The inventory is generated from the pinned query archive and the validated
snapshot manifest. Verify that neither the SQL nor its derived metadata has
drifted with:

```bash
./scripts/job/build-job-task-inventory.py --check
```

That clean-clone check rebuilds the inventory metadata from the checked-in SQL
and verifies the query-set checksum against the pinned source manifest. To also
compare every file with a separately fetched upstream source directory, run:

```bash
./scripts/job/build-job-task-inventory.py \
  --check \
  --source-dir data/raw/job-v1/source
```

The normal fixture pipeline fetches and verifies that raw source directory.

## Exact construction order

1. Verify every downloaded and extracted byte against `job-v1.json`.
2. Initialize a fresh database with the pinned QPRL PostgreSQL image.
3. Apply upstream `schema.sql`, import all 21 CSV files, and apply
   `fkindexes.sql`.
4. Grant the restricted agent read access, run the explicit
   `VACUUM (FREEZE, ANALYZE)` table list, and checkpoint.
5. Verify row counts, schema, constraints, 44 indexes, statistics, and three
   representative JOB queries as `qprl_runner`.
6. Capture the environment, stop PostgreSQL cleanly, and create the normalized,
   checksummed physical archive.
7. Restore into a fresh volume and require every logical and statistical
   fingerprint plus representative query output to match.
