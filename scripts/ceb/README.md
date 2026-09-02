# CEB workload pipeline

QORL uses the recovered full CEB-IMDb workload for training and validation and
keeps JOB held out for testing. The original author-hosted archives are gone;
`data/ceb/manifest.json` pins the high-confidence community recovery, its
immutable commit archive, and the independently corroborated branch archive.

The import trust boundary is intentionally narrow:

1. `fetch_ceb_v1.py` verifies the archive byte length and SHA-256, rejects
   traversal paths and links, and extracts only `queries/imdb` and
   `queries/imdb-unique-plans` into ignored `data/raw/`.
2. `extract_sql_from_qreps.py` never imports or executes `pickle`; it extracts
   the single length-framed UTF-8 SQL string from each qrep and verifies the
   checked-in SQL and source manifests with `--check`.
3. `audit_ceb_job_overlap.py` compares every CEB template with every JOB
   template using alias-independent, table-colored join topology.
4. `build_ceb_task_inventory.py` removes excluded templates, deduplicates exact
   SQL, and freezes a template-level train/validation split.

The checked-in result is `data/ceb/`. Verify it without downloading the
raw pickles:

```bash
uv run python -m scripts.ceb.extract_sql_from_qreps --check
uv run python -m scripts.ceb.audit_ceb_job_overlap \
  --ceb-source-dir data/ceb/queries \
  --source-kind sql \
  --source-repository https://github.com/ycy-YYYY/CEB \
  --source-commit e3862c9ab6a8210f52927ada424b2e21bad1dab1 \
  --output data/ceb/provenance/job-overlap.json \
  --check
uv run python -m scripts.ceb.build_ceb_task_inventory --check
```

To independently recover and verify the ignored source bytes first:

```bash
uv run python -m scripts.ceb.fetch_ceb_v1
```
