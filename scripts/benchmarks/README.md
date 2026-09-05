# Benchmark preparation

JOB and CEB are query workloads over the same [IMDb fixture](../imdb/README.md).
These scripts prepare queries and inventories.

Both workload manifests use the same sections: `source` pins the download,
`queries` defines the file selection and integrity checks, and `preparation`
describes how to produce the SQL and inventory. `fixture_id` names the shared
database; neither manifest contains PostgreSQL settings or calibration results.

## JOB

`benchmarks/job/manifest.json` pins the query repository and SQL checksums.

```bash
uv run python -m scripts.benchmarks.fetch_job
uv run python -m scripts.benchmarks.build_job_task_inventory --check
```

Building an inventory reads the pinned queries from `benchmarks/raw/job/source/`
and their source manifest. The existing query directory
and inventory must be absent; the source manifest is preserved.

## CEB

`benchmarks/ceb/manifest.json` pins the recovered CEB query archive. Fetching it
extracts query definitions, not another IMDb database.

The `queries` section describes the full collection and its `unique_plans` subset.
Historical recovery evidence is kept separately in
[`provenance/recovery.json`](../../benchmarks/ceb/provenance/recovery.json); the
preparation scripts do not read that report.

```bash
uv run python -m scripts.benchmarks.fetch_ceb
uv run python -m scripts.benchmarks.extract_sql_from_qreps --check
uv run python -m scripts.benchmarks.build_ceb_task_inventory --check
```

The CEB builder reads its own source manifests and SQL. It also deduplicates exact SQL
within each template and records query structure/provenance.

`extract_sql_from_qreps.py` extracts length-framed UTF-8 SQL without executing
pickle contents.

## Optional cross-workload audit

The standalone audit compares CEB and JOB join structures. Its report is an
on-demand output, not an input to either inventory builder or SFT assembly.

```bash
uv run python -m scripts.benchmarks.audit_ceb_job_overlap \
  --ceb-source-dir benchmarks/ceb/queries \
  --source-kind sql \
  --source-repository https://github.com/ycy-YYYY/CEB \
  --source-commit e3862c9ab6a8210f52927ada424b2e21bad1dab1 \
  --output outputs/audits/ceb-job-overlap.json
```

Raw downloads remain ignored under `benchmarks/raw/job/` and `benchmarks/raw/ceb/`.
