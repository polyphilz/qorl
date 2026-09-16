# 022-pg-2gb-sb-noise-baseline-lambda

Repeat experiment 007's calibration on the training host: all 113 JOB queries,
seed 42, up to five warmups, 20 measured trials, five-minute statement timeout,
and the same `001-pgconf-2gb-sb` PostgreSQL settings. Only the experiment name
and worker-pool reference differ from 007's config.

`003-poolconf-lambda-4x8` preserves four containers with 8 GiB RAM and four
guest-reported cores each, using the training host's CPU numbering. Docker
volumes and temporary database files live on the instance's local disk.

The PostgreSQL image matches the benchmark host's image. IMDb was built on the
training host with the pinned `scripts/imdb/` recipe. Row counts, schema, indexes,
and sample query outputs were verified. Fresh ANALYZE samples can differ from
the benchmark host's samples; compare plan hashes and inspect the same-plan
subset when interpreting host differences. The load report is
`data/imdb-verification/loaded.json`.

Run from the repository root on the training host:

```bash
uv run --frozen --extra gpu qorl experiment run experiments/022-pg-2gb-sb-noise-baseline-lambda --stage calibrate
```

Each invocation creates a fresh numbered run under
`outputs/022-pg-2gb-sb-noise-baseline-lambda/`. Run `000` stopped before query
measurement. The completed calibration runs are `001` and `002`; compare:

```bash
uv run --frozen --no-sync python -m scripts.analysis.analyze_calibration \
  outputs/022-pg-2gb-sb-noise-baseline-lambda/001/calibration \
  --compare outputs/022-pg-2gb-sb-noise-baseline-lambda/002/calibration \
  --paired-measurements 3 --tau 0.05 \
  --output-dir outputs/022-pg-2gb-sb-noise-baseline-lambda/noise-comparison
```
