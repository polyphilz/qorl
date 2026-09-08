# Calibration noise analysis

Analyze an existing calibration without running PostgreSQL, contacting a model,
or changing the measurement/reward code. Run from the repository root in the
existing project environment:

```sh
uv run --frozen --no-sync python -m scripts.analysis.analyze_calibration \
  outputs/NNN-buffer-study/000/calibration \
  --paired-measurements 3 --tau 0.05 \
  --output-dir outputs/NNN-buffer-study/000/noise-analysis
```

Optionally add `--compare /path/to/second/calibration`. Inputs must contain
`calibration.json` and `tasks/*.json` in the current saved-record format. The
script accepts finished calibrations, including those with failed tasks. It
rejects incomplete runs, malformed records, missing tasks, and inconsistent
counts or trial order. It does not support historical formats.

The output directory receives `noise.json` and `noise.md`; the Markdown report
is also printed. Re-running replaces those derived reports. Input records are
unchanged. Existing project schemas are used only to read saved artifacts.

## Estimator

For each query, use measured executions only, in recorded order. With `k` pairs,
slide a window of `2k` consecutive trials by one trial at a time. Within each
window, pair positions `(0,1), (2,3), ...`. Enumerate all `2^k` assignments of
one execution in each pair to “default” and the other to “candidate.” Each
window and assignment has equal weight. Do not deduplicate equal trial values.

For each assignment, calculate:

```text
error = log(median(default executions)) - log(median(candidate executions))
false_signal = abs(error) > tau
```

This is a ratio of medians, matching the current final timing statistic; it is
not a median of per-pair ratios. An error exactly at the threshold is inside
the dead zone. Even pair counts use the ordinary arithmetic median.

The estimated no-op error rate is the fraction of assignments crossing the
threshold. The report also gives the 95th percentile of `abs(error)` in log
units, median execution time, sample CV (`stdev / mean`), worker slot, and plan
stability. Percentiles use linear interpolation at `(n - 1) * p`.

Twenty trials with three pairs produce **15 windows and 120 comparisons**.
Those remain only **20 real observations**. There is no random sampling or
seed. Requests exceeding one million comparisons per task fail explicitly.
This defined estimator need not reproduce 038's historical resampling tables.

Failed tasks, nonpositive timings, changing full plan hashes, and tasks with
fewer than `2k` measurements are excluded from noise estimates and listed with
a reason. Summary rates weight eligible tasks equally and show their mean and
90th percentile. Zero eligible tasks produce `null`/`n/a`, never a zero rate.
Warmup stability is retained as a diagnostic, not an exclusion rule. Basic
latency/CV values on excluded tasks describe only their available observations.

## Comparisons and interpretation

Comparisons require the same benchmark and fingerprint version. Match queries
by task ID; list unmatched IDs and queries excluded on either side. Per-query
changes and aggregate deltas use only the common eligible queries. Error/CV
changes are percentage points (comparison minus primary), and latency ratios
are comparison divided by primary. JSON rates and CVs are fractions; Markdown
displays percentages. Changed-plan rows are flagged, and a separate mean error
delta is reported for queries whose full plan hash matches across configs.

Keep pair count and tau fixed when comparing database configurations. The
estimate concerns threshold crossings of raw measurement log ratios, before
native clipping, shared speedups, peer references, or protocol costs. It does
not estimate how often final training advantages are wrong.

Overlapping windows are dependent; enumerating their labelings adds no fresh
evidence. Symmetric errors are built into the label assignments and do not
demonstrate unbiased live measurements. Default-only trials cannot establish
the noise of other candidate plans or of the host under inference/training
load. Use this report to screen configurations, then validate promising ones
with fresh repeated runs and the actual paired measurement protocol. The
script does not implement a live no-op probe, adaptive sampling, or thresholds
chosen separately for each task.
