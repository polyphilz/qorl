# 022-pg-2gb-sb-noise-baseline-lambda

Lambda counterpart of experiment 007: all 113 JOB queries, seed 42, up to five
warmups, 20 measured trials, five-minute statement timeout, and the same
`001-pgconf-2gb-sb` PostgreSQL settings. Only the experiment name and worker-pool
reference differ from 007's config.

`003-poolconf-lambda-4x8` preserves four containers with 8 GiB RAM and four
guest-reported cores each, using Lambda's CPU numbering. Docker volumes and
temporary database files live on the instance's local disk.

The PostgreSQL image is copied from FLOPper. IMDb is freshly built on Lambda
with the pinned `scripts/imdb/` recipe. Row counts, schema, indexes, and sample
query outputs are verified. Fresh ANALYZE samples can differ from FLOPper;
compare plan hashes and inspect the same-plan subset when interpreting host
differences. The load report is `data/imdb-verification/loaded.json` on Lambda.

Run from the repository root on Lambda:

```bash
uv run --frozen --extra gpu qorl experiment run experiments/022-pg-2gb-sb-noise-baseline-lambda --stage calibrate
```

Each invocation creates a fresh numbered run under
`outputs/022-pg-2gb-sb-noise-baseline-lambda/`. Initial setup attempt `000` stopped
before measuring queries because `lscpu` emitted invalid JSON for unavailable
CPU frequency limits. The host-inventory fix leaves query measurement unchanged;
the first measurement run is `001`. After a successful second run, compare:

```bash
uv run --frozen --no-sync python -m scripts.analysis.analyze_calibration \
  outputs/022-pg-2gb-sb-noise-baseline-lambda/001/calibration \
  --compare outputs/022-pg-2gb-sb-noise-baseline-lambda/002/calibration \
  --paired-measurements 3 --tau 0.05 \
  --output-dir outputs/022-pg-2gb-sb-noise-baseline-lambda/noise-comparison
```
