# 030 — Fresh CEB RL results

**All 120 optimizer updates completed successfully in 3 hours 27 minutes.**
The run needed no restart, configuration adjustment or code repair. This verifies
the longer RL run and measurement-phase worker leasing.

**The final checkpoint did not demonstrate a clear JOB improvement over its
starting SFT checkpoint.** All 113 held-out rollouts completed successfully in
35m 14s. Reported geometric-mean speedup was **1.1353×**, versus **1.1553×** for
027, with different scored subsets. On the 100 queries scored in both, the
comparison is **1.1355× → 1.1441×**, a 0.75% increase. Valid-plan coverage stayed
at **71/113**. New substantial wins coexist with severe, avoidable selection
errors. One rollout per query, different evaluation seeds, and the model-merge
difference described below prevent attributing small changes confidently to RL.

## Run and data

- Run: `outputs/030-rl-ceb-fresh-120steps/000`.
- September 11, 2026: launch **17:47:23 UTC**, successful exit **21:14:35 UTC**;
  **12,431.76 seconds**, including startup, checkpoints and shutdown.
- Starting policy: merged 027 step-2204 SFT base, with a fresh rank-16/alpha-32
  RL adapter. Earlier SFT and smoke checkpoints were preserved.
- AdamW at `1e-6`, anchored GRPO, nominal batch eight, group four, maximum policy
  lag two updates, twelve episodes in flight, sixteen inference sequences.
- Five candidate attempts; thinking enabled; 49,152-token context and 8,192-token
  reply limit. Harness interface v7 and the recorded renderer remained unchanged.
- Two H100s handled training and inference. PostgreSQL used four calibrated
  workers on FLOPper, each with four physical cores, 8 GiB and 2 GiB shared buffers.
- Prime-RL/Verifiers/renderers source pin: `d4636a532d668ffbe727eb9e7730e7ea24ca384c`.
- Matched QORL source contract hash:
  `4334425cd1906976880da7b30463b0abd98ac81b17adc96f8373de00a647c638`.

Seed 43 selected **300 distinct training queries**, thirty from each of the ten
original CEB training templates. IDs and SQL hashes were checked against both
prior SFT training selections, prior CEB validation, and smoke selections: zero
overlap with 420 distinct prior queries. These are new query instances within
familiar training topologies. The twenty fresh CEB validation queries and 113
JOB test queries remained held out. See `selection-audit.json` for the frozen
selection checks and hashes.

## Actual training consumption

| Quantity | Result |
| --- | --- |
| Optimizer updates | 120 |
| Native episode records | 1,197 |
| Scored final outcomes | 1,196 |
| Distinct tasks in recorded episodes | 297 of 300 |
| Episodes shipped to training | 556 |
| Distinct tasks shipped to training | 180 |
| Training branches | 1,492 |
| Training token positions, including masked context | 24,829,361 |
| Sampled positions receiving nonzero advantage | 1,063,949 |
| Shipped episodes with positive / negative advantage | 174 / 382 |
| Recorded model completion tokens, including unused episodes | 2,265,562 |
| Recorded prompt tokens | 120,967,228 |

**The nominal 960 episodes were not 960 training examples.** Native filtering
removes episodes with no learning signal and episodes exceeding the policy-lag
limit; empty batches trigger further collection. Prefetch also creates work that
is not ultimately trained on. The trainer uses variable batch sizes after these
filters. Recorded and shipped counts above come from the native evidence and
ship annotations, not from multiplying configured steps by batch size.

The ordinary console `Reward 0.0000` is the unused scalar reward hook. Anchored
credit supplied positive and negative advantages, as verified in the saved
annotations. This run's RL loss is not SFT cross-entropy and is not a quality score.

## Mechanical audit and monitoring

Foreground 20-minute sleeps alternated with checks of progress, numerical values,
rollout failures, worker leases and retained checkpoints. Checkpoints at every
scheduled step **10, 20, ..., 120** remain present. Every optimizer update had a
finite gradient norm. Peak reported trainer memory was **22.4 GiB**.

The final audit checked all 556 shipped traces: sampled positions belong to
sampled assistant messages; token/mask lengths and sampling log-probability counts
align; trainer log-probabilities are finite and align with those positions; each
branch receives its conversation's recorded scalar advantage. All **1,063,949**
sampled positions match the nonzero advantage positions. No audit errors were found.

One episode failed because the model emitted an empty tool-function name. The
response validator rejected it. Its four-episode group was marked
`discarded=true`, reason `errored_group`; the failed trace had no ship annotation
and no supervised advantages. This isolated malformed reply did not invalidate
the run. No failed/discarded trace was shipped to training.

Occasional `loss/std` warnings come from computing the sample standard deviation
of a singleton. Training losses and gradients remained finite. Other warnings
concerned waiting for rollout batches, stale-policy rejection and routine
inference startup/shutdown. They did not require restarting the experiment.

The final model export contains **256 finite LoRA tensors**, with nonzero LoRA B
weights. Export checked the recorded trainer configuration, base-weight identity
and checkpoint provenance. Remote drain was acknowledged without error, the
trainer exited 0, and both GPUs became idle. The PostgreSQL service subsequently
exited 0 and removed its four owned containers at **21:24:47 UTC**.

## Worker leasing and throughput

The evidence contains **9,018 leases**, with no invalid worker slots or lease
records. Baseline warmups/measurements, each candidate phase and each final paired
protocol retain one worker for the complete phase; model turns release it.

Summed worker ownership was **12,484.56 seconds**. Summed queue waiting was
**2,826.08 seconds**; median wait was about **7 microseconds**, maximum **138.21
seconds**. Longer waits occurred in bursts, so the near-zero median alone does
not describe the tail. These sums span concurrent episodes and are not wall time.

The practical result is a completed 120-update run in 3h27m. This is not a
controlled measurement of the claimed threefold gain: the earlier smoke used
different tasks, batching and policy-lag settings. Same-worker final pairs
preserve the measurement protocol, but cache interference, cross-worker
preliminary comparisons and timeout decisions can still change timing noise.

## Training outcomes — descriptive only

| Outcome | Count |
| --- | --- |
| `kept_default` | 720 |
| `measured` | 399 |
| `default_duplicate` | 29 |
| `no_valid_candidate` | 30 |
| `selection_failed` | 3 |
| `timed_out` | 15 |
| Unscored malformed-reply failure | 1 |

The 1,148 speedup-bearing outcomes have a geometric mean of **1.1195×**; 749 of
them contribute 1.0× through default selection or duplication. There were 95
reported regressions. These combine different queries, changing policy versions
and unused episodes. They are **not the final checkpoint's held-out score or a
before/after comparison**. The separate JOB evaluation below evaluates only the
final step-120 checkpoint.

## Held-out JOB evaluation: method and verification

The final RL adapter was evaluated within this experiment at
`outputs/030-rl-ceb-fresh-120steps/000/evaluation/test/000`. Evaluation started
**September 11, 2026, at 21:50:35 UTC** and completed at **22:25:50 UTC**
(18:25:50 America/New_York), taking **2,114.38 seconds**. The launcher exited 0;
the four owned PostgreSQL containers were removed. The requested 75-minute
foreground wait completed before inspection. Detailed analysis used downloaded
records locally after confirming evaluation had stopped.

- **113 JOB queries, one rollout each, seed 43**, five candidate attempts,
  thinking enabled, 49,152-token context, 8,192-token maximum reply, temperature
  1.0, top-p 1.0 and top-k 20.
- The exported **step-120 RL adapter** ran on its exact **merged 027 step-2204
  base**. Transfer checks matched the model artifacts, recorded training-base
  hash, adapter checksum and the source checkpoint's verified export manifest.
  Applying this RL adapter directly to the original pretrained base would be
  incorrect; that was not done.
- vLLM 0.28.0 served the model on one RTX 3090. All **810 recorded requests and
  responses** identify `qorl-adapter`, and the server records that adapter's
  parent as the transferred merged base. Local serving used four sequences.
- PostgreSQL used the usual FLOPper configurations, `001-pgconf-2gb-sb` and
  `002-poolconf-4x8`: four workers, four physical cores and 8 GiB per worker.
  **Standalone evaluation retained a worker for each complete rollout**;
  measurement-phase leasing applied to RL collection, not this evaluation.
- All 113 defaults have one warmup and three measurements. All 37 measured
  final selections have one warmup pair and three measured pairs. Candidate
  feedback retains the configured one warmup and one measurement when fresh
  timing is needed. Execution accounting reports 452 initial-default, 235
  candidate-feedback and 296 final-paired executions, including warmups.
- There were **zero runtime failures, recorded model-call failures, truncated
  replies, context-budget stops or model-turn-limit stops**. The largest request
  contained 37,055 prompt tokens; the largest reply used 785 completion tokens.
  Two selected-candidate timeouts are policy outcomes, reported separately from
  runtime failures.

The existing evaluation library was invoked with recorded local serving
overrides. Original experiment/run inputs and native training configuration were
preserved. `deployment/eval-launch-000.json` records the effective evaluation
settings and checkpoint identity; the evaluation directory holds the normal
QORL report, traces and worker evidence.

### Comparability with the starting checkpoint

The reference is [027's JOB evaluation](../027-sft-astra-ceb-300-epoch2/results.md),
which evaluated the step-2204 SFT adapter before this RL run. All 113 queries
match on **interface v7, fingerprint v4, system prompt, initial tool-schema
hash, default timing-reuse key, default physical-plan hash, planner settings
and measurement protocol**. Physical worker assignment varies: 73 queries
used a different CPU set, while the resource capacities were unchanged. Initial
default latencies have a new/old geometric-mean ratio of **0.9969**, and a median
ratio of **0.9922**; there is no large overall baseline-latency shift.

Two comparison limitations matter. **027 used seed 42; 030 used seed 43**, so
all per-query model and measurement seeds differ. Also, 027 served an SFT LoRA
over the original base, whereas 030 served an RL LoRA over the merged SFT base.
The merge was verified, but finite-precision merging need not reproduce the
unmerged inference path exactly. This is therefore not a comparison with matched
sampling seeds and identical model representation. An evaluation of the merged
base without the RL adapter under seed 43 would be the cleaner step-zero control.

## JOB results: starting SFT policy versus final RL policy

| Metric | 027: SFT step 2204 | 030: RL step 120 |
| --- | ---: | ---: |
| Queries with a valid candidate /113 | 71 (62.8%) | **71 (62.8%)** |
| Queries with a novel candidate /113 | 69 | **67** |
| Distinct novel query/plan pairs | 113 | **114** |
| Candidate attempts | 507 | **513** |
| Attempts passing action validation | 246 (48.5%) | **240 (46.8%)** |
| Attempts passing action and plan constraints | 138 (27.2%) | **134 (26.1%)** |
| Scored outcomes /113 | 107 | **106** |
| Geometric-mean speedup, respective scored subsets | 1.1553× | **1.1353×** |
| Geometric-mean speedup, common 100-query subset | 1.1355× | **1.1441×** |
| Ratio of summed scored default/selected latencies | 1.0567× | **0.9906×** |
| Measured improvements outside the 0.05 log band | 20 | **21** |
| Measured regressions outside the same band | 5 | **7** |
| Kept-default outcomes | 72 | **67** |
| No-valid-candidate outcomes | 4 | **5** |
| Selection-failed outcomes | 1 | **0** |
| Selected-candidate timeouts | 1 | **2** |
| Model replies | 817 | **810** |
| Prompt tokens | 9,760,337 | **9,708,427** |
| Completion tokens | 208,848 | **209,959** |
| Reasoning tokens, included in completion tokens | 81,878 | **81,840** |
| Evaluation wall time | 34m 22s | **35m 14s** |

The 030 outcomes are **37 measured, 67 kept default, two default duplicates,
five no-valid-candidate and two timeouts**. The first three groups produce the
106 scored outcomes. **The seven failures/timeouts are excluded from the
geometric mean**, not silently assigned 1.0×. Kept-default outcomes and
duplicates explicitly contribute 1.0×, including default selections after an
unsuccessful search.

Among the 37 measured selections, 27 were faster and ten slower than default.
Using `abs(log(speedup)) <= 0.05` as a descriptive band gives **21 improvements,
seven regressions and nine near-neutral measurements**. The band is not a
per-query significance test. The report's ten regressions simply count every
speedup below 1.0, including the three small slowdowns inside the band.

Summed scored default and selected latencies were **53.706 s and 54.214 s**:
selected plans consumed **0.508 s more**, or about **0.95%**. These sums use
query medians on the same 106 scored queries, with the initial default on both
sides for kept-default and duplicate outcomes. They exclude unscored outcomes,
inference and candidate-search overhead. The positive geometric mean and
negative total time saving are compatible: one weights multiplicative changes
equally by query, while the other emphasizes absolute milliseconds.

### Coverage changed even though the valid-query total did not

**23 queries gained a valid candidate and 23 lost one.** The unchanged 71/113
total does not mean the same queries succeeded.

Six formerly unscored queries became scored: job-01c, job-19d, job-25a,
job-26b and job-28a became kept defaults; job-32a measured **0.990×**. Seven
formerly scored queries became unscored: job-03c, job-13c, job-19b, job-28b and
job-33b ended with no valid candidate; **job-17c and job-18a timed out**, after
scoring **2.752× and 5.645×** in 027. These exclusions explain why the headline
comparison and the common-query comparison move in different directions.

The largest win is job-01b at **97.017×**, saving only **17.19 ms**. Excluding
030's largest scored win leaves **1.0882×**; excluding its three largest leaves
**1.0375×**. The benefit is not exclusively one query, but the headline is
sensitive to a few large ratios.

## Analysis: successful training mechanics, unresolved decision quality

### There are useful new plans, but also expensive lost wins

| Query | 027 speedup | 030 speedup | 030 final default → candidate ms |
| --- | ---: | ---: | ---: |
| job-06d | 2.842× | **17.620×** | 1,708.354 → 96.954 |
| job-30b | 1.000× | **9.105×** | 792.795 → 87.072 |
| job-02c | 1.004× | **7.667×** | 163.650 → 21.345 |
| job-11b | 1.111× | **6.730×** | 15.910 → 2.364 |
| job-25c | 0.416× | **1.764×** | — |
| job-26c | 4.338× | **0.401×** | 1,715.347 → 4,279.557 |

Other earlier wins became defaults, including job-01a (**7.660× → 1.000×**)
and job-24a (**2.379× → 1.000×**). These are actual differences between the
recorded searches, not proof that RL caused each change. Sampling changed too.

### The model still sometimes picks the best bad candidate over default

| Selection diagnostic | 027 | 030 |
| --- | ---: | ---: |
| Default before any candidate attempt | 4 | **3** |
| Default after search, with an eligible candidate | 34 | **30** |
| Default after search, with no eligible candidate | 34 | **34** |
| Chose default when every numerically timed eligible option was materially slower | 23/28 | **19/25** |
| Chose a best-feedback candidate or exact tie, among selections with numerical feedback | 34/35 | **37/39** |
| Selected a candidate already materially slower in feedback | 5 | **6** |
| Selected an already timed-out candidate | 1 | **2** |

The default fallback worked: it was selected after search 64 times. It did not
prevent the model from making bad choices when faster default execution was
available. **37 of the 67 kept defaults produced no valid candidate**: 34 after
failed search and three before submitting anything. The early defaults were
job-10c, job-16d and job-21b.

| Query | Selected feedback ratio | Final default ms | Final candidate ms | Final speedup |
| --- | ---: | ---: | ---: | ---: |
| job-06c | 0.061× | 3.430 | 54.311 | **0.063×** |
| job-06e | 0.080× | 6.078 | 76.258 | **0.080×** |
| job-17d | 0.632× | 2,205.730 | 3,643.478 | **0.605×** |
| job-18c | 0.414× | 1,463.094 | 3,622.064 | **0.404×** |
| job-22d | 0.691× | 307.591 | 447.441 | **0.687×** |
| job-26c | 0.366× | 1,715.347 | 4,279.557 | **0.401×** |

All six were visibly poor choices before final scoring. This is not a case
where an apparently strong speedup disappeared only during fresh measurement.
The seventh material regression, job-15c, is near the descriptive threshold:
**0.961× feedback, 0.949× final**.

The two non-best numerical selections need different interpretations. On
job-30a, the model explicitly treated the small feedback difference as negligible;
the selected plan still measured **2.847×**. On job-22d it picked a slower
candidate despite another being faster, but both were slower than default.
Improving candidate ranking alone would not fix that loss.

Both timeout selections had already timed out during execution feedback:
job-17c selected candidate-01 with a **6,827 ms** cutoff; job-18a selected
candidate-04 with a **5,000 ms** cutoff. Neither had a numerical feedback ratio.
Their terminal replies still favored finishing with a candidate; job-18a's
reply even asserted an approximately 0.2-second runtime absent from that
candidate's measured feedback. The recorded tool results and default option
were available. These are remaining evidence-use errors, not missing harness
capabilities or new infrastructure failures.

A narrow diagnostic replaces only the six selected candidates whose recorded
feedback has `log(ratio) < -0.05` with default, leaving every other scored choice
unchanged. It raises 030's geometric mean from **1.1353× to 1.2241×** on the
same 106 queries; the corresponding 027 diagnostic is **1.1824×**. This invents
no timing for unselected candidates and leaves all unscored outcomes excluded.
It is **not an executed policy evaluation**, but illustrates the available gain
from using the evidence already shown to the model.

### Formatting improved in one respect, not overall

| Action diagnostic | 027 | 030 |
| --- | ---: | ---: |
| Malformed string-valued actions | 170 | **139** |
| Those strings that parse as JSON objects | 0 | **0** |
| Other action-validation failures | 91 | **134** |
| Action-valid but constraint-unsatisfied attempts | 108 | **106** |
| Object-valued actions containing `leading` | 197 | **217** |
| Those passing action and plan constraints | 83 (42.1%) | **72 (33.2%)** |
| Material measured wins using a selected `leading` action | 10 | **9** |

Fewer actions were wrapped in malformed strings, but other validation failures
increased. Every one of 030's 139 malformed strings contains `leading`, and none
is rescued by simply parsing a valid embedded JSON object. Overall action-valid
and constraint-valid attempt rates did not improve. This run therefore does not
show that RL repaired PlanAction conformance.

Terminal handling remained bounded: six selection rejections occurred on six
queries, at most one per query, and there were ten unavailable-tool calls, all
to `evaluate_candidate`. There were no context-exhaustion loops or
selection-failed outcomes. The model successfully finished rather than getting
stuck, while the five no-valid-candidate outcomes retained their intended meaning.

### What this run supports

The hybrid RL path, worker leasing, weight updates, scalar credit masks,
checkpoint retention, export and held-out evaluation all worked. The final
policy remains usable and sometimes finds much faster plans. **This particular
120-update pass has not earned replacement of 027 as a demonstrated better
policy.** Matched-query performance is close, valid coverage is unchanged, and
avoidable bad selections remain.

Do not infer from this one pass that RL cannot help or that a larger run will
necessarily help. Only 180 distinct training queries contributed shipped
episodes. A useful next control is the identically merged step-zero policy at
the same evaluation seed; checkpoint comparisons can then use the held-out CEB
validation set and repeated seeds before spending further JOB evaluations on
model selection. The clearest behavioral target in these records is selecting
default when all measured alternatives are slower or timed out, alongside
continued work on valid action generation. No new harness bug or timing-reuse
failure was found that warrants changing the measurement protocol here.

## Saved model and evidence

Final exported adapter on Lambda:

```text
/lambda/nfs/qorl/projects/qorl/outputs/030-rl-ceb-fresh-120steps/000/training/checkpoints/step_120/adapter
```

Use it with the merged base `/lambda/nfs/qorl/models/qwen-4b-sft-027-step-2204`.
Adapter weights are **42,500,760 bytes (40.53 MiB)**, SHA-256
`1b5f449a22e68881d6eddf3a9c470366b3cde5acff522d5641be7ad5a31628c4`.

Primary evidence under the experiment's output directory:

- `000/training/report.json`, native trace/annotation/metric streams, and all
  twelve saved checkpoints.
- `000/training/remote-readiness.json` and `remote-cleanup.json`.
- `deployment/monitor/` for the periodic audit snapshots.
- `deployment/completion-summary.json`, `credit-mask-audit.json` and
  `final-adapter-audit.json`.
- Service episode/lease records and the service exit record on the database host.
- `000/evaluation/test/000/evaluation.json`, the 113 rollout records, worker
  snapshots and server log for the final checkpoint's JOB evaluation.
- `deployment/eval-launch-000.json` and `eval-exit-000.json` for the effective
  evaluation settings, source export identity and successful exit.

The offline task-level comparison and its analysis script are retained locally
under `outputs/analysis/030-final-checkpoint-job/`. All reported comparison
figures were recomputed from completed 027 and 030 rollout records.
