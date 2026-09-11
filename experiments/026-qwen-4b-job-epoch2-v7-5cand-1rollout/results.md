# Results: epoch-2 adapter on JOB under harness v7

Completed September 10, 2026, on FLOPper. Run `000`, evaluation `test/000`.

**The epoch-2 adapter remains the stronger checkpoint.** This run produced a
valid candidate on **85/113 queries (75.2%)** and achieved **1.075× geometric-mean
speedup across 108 scored outcomes**. Harness v7 substantially reduced terminal
loops and allowed useful returns to PostgreSQL's default. Performance gains are
concentrated in a few queries, and the model still sometimes chooses a candidate
that its own feedback shows is slower than the default.

This is the baseline for evaluating experiment 025's additional SFT data. It uses
the original epoch-2 adapter; it does **not** include training on the new
293 demonstrations.

## Setup and provenance

| Setting | Value |
| --- | --- |
| Workload | All 113 JOB queries; one rollout per query; seed 42 |
| Base | `empero-ai/Qwen3.8-4B-Distill` |
| Base revision | `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e` |
| Adapter | Experiment 020, step 764: two epochs on the original SFT dataset |
| Serving | vLLM 0.28.0, `qorl-adapter` over `qorl-base`, FLOPper GPU 0 |
| Harness / fingerprints | qo-agent v7 / plan fingerprint v4 |
| Agent budget | Up to five candidate attempts; 64 model turns |
| Model settings | Thinking enabled; 49,152-token context; 8,192-token maximum reply |
| Sampling | Temperature 1.0; top-p 1.0; top-k 20 |
| Initial default timing | One warmup, then three measured executions |
| Candidate feedback | One warmup, then one measured execution when timing cannot be reused |
| Final candidate timing | One warmup pair, then three fresh measured pairs when required |
| PostgreSQL configuration | `001-pgconf-2gb-sb` |
| Worker pool | `002-poolconf-4x8`: four workers, four physical cores and 8 GiB RAM each |
| Wall time | **32m 19s**, 20:55:46–21:28:05 America/New_York |

Adapter path:
`outputs/020-sft-astra-ceb-epoch2/000/training/checkpoints/step_764/adapter`.
Verified adapter tensor SHA-256:
`d1f055e4a347fca34185a34d2f9baf22f9197e006dacfb0b018b52f178c918f4`.

The serving record identifies this adapter and the pinned base; all 915 model
responses identify `qorl-adapter`. All 113 recorded trajectories use interface v7.
Compared with experiment 024, the saved configs
differ only in experiment name and adapter path; system prompts, tool definitions,
task IDs, model seeds, measurement seeds, and default timing-reuse keys match
for every query. The default timing-reuse keys also match experiment 021 for
all 113 queries.

## Outcomes and what the headline means

| Metric | Result |
| --- | ---: |
| Recorded rollouts | 113/113 |
| Rollouts with a valid candidate | 85/113 (75.2%) |
| Rollouts with a structurally novel candidate | 76/113 (67.3%) |
| Distinct novel query/plan pairs | 115 |
| Candidate attempts | 547 |
| Scored outcomes | 108/113 (95.6%) |
| Geometric-mean speedup, scored outcomes | **1.075×** |
| Ratio of summed default/candidate latencies, scored outcomes | **1.036×** |
| Recorded infrastructure failures | 0 |

The 113 final outcomes break down as follows:

| Outcome | Count | Treatment in speedup statistics |
| --- | ---: | --- |
| Measured candidate | 40 | Fresh final paired timing |
| Kept default | 53 | Exactly 1.0× |
| Candidate with the default's timing-reuse key | 15 | Exactly 1.0× |
| No valid candidate | 4 | Excluded from speedup aggregate |
| Selection failed | 1 | Excluded from speedup aggregate |
| Selected-candidate timeout | 0 | None |

Thus, **108 scored outcomes do not mean 108 improvements or 108 measured
alternative plans**. Sixty-eight outcomes contribute exactly 1.0×. Among the
40 measured outcomes, 18 were faster and 22 slower. Using the descriptive band
`abs(log(speedup)) <= 0.05`, there were **12 improvements, 8 regressions, and
20 near-neutral measurements**. This band is not a statistical significance test.

Summed scored latencies were **55.254 seconds for the default versus 53.311
seconds for the selected behavior**, a 1.942-second reduction (3.5%). These are
sums of per-query medians, with the initial default median used on both sides
for retained defaults and duplicates. They exclude inference, search,
inspection, and warmup costs; they are not end-to-end application speedups.

## Where the performance comes from

Selected examples below use final paired medians, in milliseconds.

| Query | Default | Candidate | Speedup | Selected intervention |
| --- | ---: | ---: | ---: | --- |
| job-01b | 17.288 | 0.159 | **108.730×** | `BitmapScan(mi_idx)` |
| job-01a | 17.160 | 2.381 | **7.207×** | `BitmapScan(mi_idx)` |
| job-26c | 1,498.281 | 358.580 | **4.178×** | `BitmapScan(mi_idx)` |
| job-11c | 456.100 | 162.195 | **2.812×** | Bitmap scan on `mc` using `company_type_id_movie_companies` |
| job-06d | 1,604.918 | 639.857 | **2.508×** | Hash join on `k,mk`, bitmap scan on `t`, disable sort |
| job-06b | 79.523 | 33.567 | **2.369×** | Bitmap scan on `mk` using `movie_id_movie_keyword` |
| job-19a | 100.343 | 141.867 | **0.707×** | Bitmap scan on `cn`, sequential scan on `t` |
| job-05a | 44.220 | 74.500 | **0.594×** | Leading tree, hash join, bitmap scan, lower random-page cost |
| job-01c | 18.029 | 32.124 | **0.561×** | Leading tree and nested-loop join |
| job-31b | 141.058 | 555.942 | **0.254×** | Leading tree |
| job-21c | 30.607 | 131.261 | **0.233×** | Bitmap scan on `mc` using `company_type_id_movie_companies` |
| job-04c | 40.525 | 226.036 | **0.179×** | Hash join on `it,mi_idx`, index scan on `mi_idx` |

The extreme job-01b gain appears consistently in all three final pairs:
default `17.356, 17.283, 17.288` ms; candidate `0.164, 0.152, 0.159` ms.
It saves about 17 ms per execution, despite its very large ratio.

The geometric mean is sensitive to these wins: excluding job-01b gives
**1.030×**; excluding the three largest wins gives **0.997×**. This is a modest,
concentrated positive result rather than broad dominance across JOB.

Ten of the twelve improvements outside the log-space band came from actions
containing only scan constraints. One combined scan, join, and planner settings;
one used parallelism and planner settings. None supplied a Leading tree. These
results support the usefulness of simple scan interventions, while providing
little evidence yet of beneficial learned join-order selection.

## Candidate selection: ranking candidates works better than choosing default

All **55 selected candidates tied for the lowest observed feedback latency
among eligible candidates in their rollout**, resolving reused feedback to its
source. Forty-seven selections chose an earlier attempt. This includes rollouts
with only one eligible candidate and ties, so it does not establish sophisticated
ranking in all 55 cases.

The remaining mistake is often whether to choose any candidate at all:

- Nine selected candidates looked slower than the initial default by more than
  0.05 in log space. Seven remained outside that regression band in final timing.
  For example, job-04c showed **225.497 ms versus a 40.077 ms default**, yet the
  model selected the candidate. Returning to default was available.
- Conversely, job-12a and job-17e retained the default despite available
  preliminary improvements of **2.12× and 2.46×**. Those alternatives were not
  finally selected and paired, so these are missed apparent opportunities,
  not verified lost speedups.
- The eighth final regression outside the band, job-12c, looked slightly faster
  during feedback and slower during final timing. That case illustrates why
  preliminary feedback and final measurements must remain distinct.

Of the 53 kept-default endings, **52 followed candidate attempts** and one
occurred before any attempt. Of those 52 searches, 29 produced an eligible
candidate and 23 produced none. The latter still score 1.0× under the explicitly
accepted fallback policy, but do not count as producing valid candidates.

The new default option can work well: on job-06e, the model explicitly compared
the approximately **6.1 ms default** with a **3.6-second candidate**, then chose
`finish(selected_candidate_id="default")`. Another candidate had timed out.

## Comparison with earlier epoch evaluations

| Metric | 021: epoch 2, v6 | 024: epoch 3, v7 | 026: epoch 2, v7 |
| --- | ---: | ---: | ---: |
| Valid-candidate rollouts /113 | 89 | 57 | **85** |
| Novel-candidate rollouts /113 | 84 | 50 | **76** |
| Distinct novel query/plan pairs | 135 | 92 | **115** |
| Scored outcomes /113 | 88 | 99 | **108** |
| Geometric-mean speedup, scored subset | 0.901× | 0.817× | **1.075×** |
| Kept-default outcomes | 0 | 66 | **53** |
| No-valid-candidate outcomes | 7 | 8 | **4** |
| Selection failures | 17 | 0 | **1** |
| Selected-candidate timeouts | 1 | 6 | **0** |
| Rejected selection calls | 389 | 10 | **6** |
| Context-budget endings | 16 | 0 | **1** |
| Prompt tokens | 34,509,285 | 10,082,244 | **11,756,852** |
| Completion tokens | 328,135 | 237,297 | **165,262** |
| Evaluation wall time | 53m 18s | 37m 28s | **32m 19s** |

**026 versus 021 tests the same weights under changed harness behavior and
initial timing counts.** No additional learning occurred between these two
evaluations. Lower selection failure and better default handling explain much
of the practical improvement; candidate validity and novelty did not improve.
Wall time fell 39%, prompt tokens 66%, and completion tokens 50%. There were
915 model replies, versus 1,798 previously.

The speedup columns use different scored subsets. Restricting to the **84 queries
scored by both 021 and 026** gives geometric means of **0.888× and 1.110×**,
respectively. This remains a harness comparison with changed trajectories and
fresh measurements, not a controlled replay of identical candidates.

**026 versus 024 is the cleaner checkpoint comparison:** the same v7 prompts,
tools, settings, and tasks, with epoch 2 versus epoch 3. On the **95 queries scored
by both**, geometric means are **1.015× for epoch 2 and 0.807× for epoch 3**.
Together with 85 versus 57 valid-candidate rollouts, this supports retaining
epoch 2 as the starting checkpoint. One rollout per query does not establish
that a third epoch invariably hurts or isolate overfitting as its cause.

## Harness checks and remaining failures

The principal fixes are visible in the saved behavior:

- **Default fallback:** searches can now end by explicitly selecting default;
  valid-candidate statistics remain separate from fallback outcomes.
- **Terminal guidance:** unavailable-tool calls fell from **513 to 35**, including
  out-of-budget `evaluate_candidate` calls falling from **473 to 32**. Their
  feedback names the available tools and explains how to finish.
- **Invalid selection handling:** six selection rejections occurred across six
  rollouts. Four with eligible candidates corrected their selection; two with
  none ended as `no_valid_candidate`. Repeated finish loops no longer dominate.
- **Initial timing:** every rollout used one warmup plus three measurements,
  totaling 452 initial-default executions. Candidate feedback remains one
  measured execution when fresh feedback is required.
- **Memoize diagnostics:** unhonored memoization requirements still produce
  explicit feedback explaining that the hint does not guarantee a Memoize node.
  The verifier has not silently relaxed the constraint.

There were **two exploratory candidate execution timeouts**, on job-06e and
job-17b. Neither timed-out candidate was finally selected. The reported zero
timeouts therefore describes final outcomes, not every attempted execution.

The five unscored queries were:

| Query | Outcome | Explanation |
| --- | --- | --- |
| job-02a | No valid candidate | Invalid actions; terminated through `finish` |
| job-08c | No valid candidate | No eligible plan; finish named an ineligible candidate and closed cleanly |
| job-24a | No valid candidate | Invalid actions; terminated after three attempts |
| job-30a | No valid candidate | Invalid actions; finish named an ineligible candidate and closed cleanly |
| job-33a | Selection failed | Context budget exhausted after candidate 5, before a terminal selection |

job-33a had produced a candidate with **5.221 ms feedback versus an 11.819 ms
initial default**, but never selected it. This was a conversation-size failure
after execution feedback, not another rejected-finish loop.

Action reliability remains the larger training target. Of 547 attempts:
**217 were action-invalid**, **141 were action-valid but failed plan constraints**,
and **189 passed both checks** (including the two execution timeouts). Among the
invalid actions, 72 encoded `action` as a string, 87 had disconnected join-tree
or join-relation errors, and 21 had unknown-field errors. These diagnostic
categories are not an exhaustive partition. Frequent plan-constraint failures
involved requested scan methods/indexes not appearing in the resulting plan.

## Implications for the next experiment

Use **026 as the v7 epoch-2 baseline for experiment 025's trained adapter**.
Keep the evaluation settings fixed and compare valid-plan coverage, action and
constraint failures, correct default/candidate selection, terminal failures,
token use, and final paired performance. Additional learning should improve
the 75.2% valid-plan rate and the decisions identified above, rather than merely
increase the number of 1.0× fallback outcomes.

This run supports continuing from epoch 2 on the additional demonstrations.
It does not yet establish a broad optimizer advantage, a statistically reliable
7.5% gain, or an advantage over a mechanical scan/flag sweep. JOB has also been
used repeatedly during iteration; it should be described that way in the write-up.

## Evidence and metric definitions

Primary report on FLOPper:
`/home/rohan/projects/qorl/outputs/026-qwen-4b-job-epoch2-v7-5cand-1rollout/000/evaluation/test/000/evaluation.json`.

Report SHA-256:
`4974825b610d87b70a36b6ca51638d56fa5f67d1711ee6e7151b2449920c82a1`.

Per-query records are under the same evaluation directory at
`rollouts/<task-id>/000.json`. Comparisons use the corresponding saved reports
and all 113 per-query records from experiments 021 and 024, each at
`000/evaluation/test/000`. Configuration values come from each run's saved
`config.toml`, rather than current defaults.

The geometric mean is `exp(mean(log(default_ms / candidate_ms)))` over outcomes
with recorded speedups. Defaults and timing-reuse duplicates contribute 1.0;
unscored outcomes are reported separately. A valid-plan rollout requires at least
one action-valid candidate whose plan satisfies its constraints. Structural
novelty is recorded independently from timing reuse. Reasoning tokens are a
subset of completion tokens: this run recorded **93,243 reasoning tokens**
within its 165,262 completion tokens.
