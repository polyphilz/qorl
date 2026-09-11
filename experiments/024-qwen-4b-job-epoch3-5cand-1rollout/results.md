# Results: third SFT epoch on the original Astra dataset, evaluated on JOB

Evaluation completed on **September 10, 2026**, on FLOPper: run `000`, split
`test`, evaluation `000`.

**The third epoch was a regression, and epoch 2 was the better checkpoint to
retain.** This run produced valid candidates on **57/113 queries (50.4%)** and
achieved **0.817× geometric-mean speedup across 99 scored outcomes**. The later
epoch-2 evaluation under matching v7 settings produced **85/113 valid-candidate
rollouts (75.2%)** and **1.075× across 108 scored outcomes**. Held-out teacher
validation loss also worsened during the third epoch, from **0.306212 to
0.322112**.

The harness fixes did work: terminal loops and context exhaustion largely
disappeared. That operational improvement should be distinguished from the
checkpoint's worse candidate generation and final decisions.

## What was evaluated

Experiment **023 performed the training**; experiment **024 evaluated its
exported adapter**. This was another pass over the original dataset, not training
on the additional demonstrations later collected in 025.

| Setting | Value |
| --- | --- |
| Base | `empero-ai/Qwen3.8-4B-Distill` |
| Base revision | `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e` |
| Adapter | Experiment 023, step 1146: three total epochs |
| Training lineage | 018 step 382 → 020 step 764 → 023 step 1146 |
| Training data | 99 accepted Astra conversations from 100 selected CEB tasks |
| Held-out teacher data | 19 accepted conversations from 20 selected validation tasks |
| Additional training in 023 | 382 updates on the same 382 packed training rows |
| Training settings | Batch/microbatch 1; LoRA rank 16, alpha 32, dropout 0; constant AdamW LR `1e-4`; BF16 |
| Evaluation workload | All 113 JOB queries, one rollout per query, seed 42 |
| Agent budget | Five candidate attempts; 64 model turns |
| Model settings | Thinking enabled; 49,152-token context; 8,192-token maximum reply |
| Sampling | Temperature 1.0, top-p 1.0, top-k 20 |
| Serving | vLLM 0.28.0, `qorl-adapter` over `qorl-base`, FLOPper GPU 0 |
| Harness / fingerprints | qo-agent v7 / plan fingerprint v4 |
| PostgreSQL / pool | `001-pgconf-2gb-sb` / `002-poolconf-4x8` |
| Evaluation wall time | **37m 28s**, 13:49:41–14:27:09 America/New_York |

Initial default timing used one warmup followed by three measurements. Fresh
candidate feedback used one warmup and one measurement. Final paired scoring,
when required, used one warmup pair and three measured pairs.

The serving record identifies
`outputs/023-sft-astra-ceb-epoch3/000/training/checkpoints/step_1146/adapter`.
All **858 model responses** identify `qorl-adapter`; all 113 trajectories record
interface v7. The archived adapter's tensor checksum matches its export manifest:
`84ee7a07696fea72027588d49a1d4d2d0925220ad8790a5231bf3b4ff72dc1b8`.

## Outcomes and performance

| Metric | Result |
| --- | ---: |
| Recorded rollouts | 113/113 |
| Rollouts with a valid candidate | 57/113 (50.4%) |
| Rollouts with a structurally novel candidate | 50/113 (44.2%) |
| Distinct novel query/plan pairs | 92 |
| Issued candidate attempts | 541 |
| Scored outcomes | 99/113 (87.6%) |
| Geometric-mean speedup, scored outcomes | **0.816540×** |
| Ratio of summed default/selected latencies, scored outcomes | **0.891780×** |
| Recorded infrastructure failures | 0 |

The final outcomes were:

| Outcome | Count | Speedup treatment |
| --- | ---: | --- |
| Measured candidate | 28 | Final paired medians |
| Kept default | 66 | Exactly 1.0× |
| Default timing-reuse duplicate | 5 | Exactly 1.0× |
| No valid candidate | 8 | Unscored |
| Selected-candidate timeout | 6 | Unscored |
| Selection failed | 0 | None |

Thus **71 of the 99 scored outcomes contribute exactly 1.0×**. Of the 28 measured
candidates, 10 were faster and 18 slower. Using the descriptive log-space band
`abs(log(speedup)) <= 0.05`, there were **9 improvements, 15 regressions, and
4 near-neutral results**. This band is not a statistical significance test.

Summed scored default latencies were **47.971 seconds**, versus **53.793 seconds**
for the selected behavior: **5.821 seconds more, or 12.1% higher latency**. These
are sums of per-query medians, using the initial default median on both sides
for retained defaults and duplicates. They exclude search, inference, warmups,
and unscored outcomes. In particular, the six selected timeouts do not enter
either speedup aggregate; the 0.817× headline does not capture all failures.

Selected examples, using final paired medians in milliseconds:

| Query | Default | Candidate | Speedup | Intervention |
| --- | ---: | ---: | ---: | --- |
| job-11c | 450.485 | 81.449 | **5.531×** | Leading tree with explicit join methods |
| job-17b | 2,566.156 | 1,107.726 | **2.317×** | Nested-loop constraint, index scan on `mk`, sequential scan on `t` |
| job-17e | 3,744.741 | 1,834.437 | **2.041×** | Leading tree, join/scan constraints, parallelism |
| job-11d | 56.536 | 48.012 | **1.178×** | Leading tree alone |
| job-03a | 55.572 | 1,081.207 | **0.051×** | Leading tree, hash joins, parallel override |
| job-01a | 18.276 | 285.512 | **0.064×** | Leading tree, hash joins, scan constraints |
| job-12b | 17.283 | 237.345 | **0.073×** | Leading tree, hash-join constraint, sequential scan |
| job-25a | 775.204 | 3,884.561 | **0.200×** | Leading tree alone |

The overall regression is not explained by one bad query: removing the worst
scored result, job-03a, still leaves **0.840×** over the remaining 98 outcomes.

## The clean checkpoint comparison came later, in 026

The first comparison available was with **021: epoch 2 under harness v6**.
024 also introduced v7 and increased initial default measurements from one to
three, so 021 versus 024 changed both weights and evaluation behavior.

Experiment **026 reran epoch 2 under 024's settings**, providing the better
checkpoint comparison. The saved configs differ only in experiment name and
adapter path. All 113 task IDs, model seeds, measurement seeds, system prompts,
initial tool definitions, and default timing-reuse keys match.

| Metric | 026: epoch 2, v7 | 024: epoch 3, v7 |
| --- | ---: | ---: |
| Valid-candidate rollouts /113 | **85** | 57 |
| Novel-candidate rollouts /113 | **76** | 50 |
| Distinct novel query/plan pairs | **115** | 92 |
| Scored outcomes /113 | **108** | 99 |
| Geometric-mean speedup, each scored subset | **1.075×** | 0.817× |
| Geometric mean on the same 95 scored queries | **1.015×** | 0.807× |
| Measured improvements / regressions outside the log band | **12 / 8** | 9 / 15 |
| Selected-candidate timeouts | **0** | 6 |
| No-valid-candidate outcomes | **4** | 8 |
| Kept-default outcomes | 53 | 66 |
| Prompt tokens | 11,756,852 | 10,082,244 |
| Completion tokens | 165,262 | 237,297 |
| Wall time | **32m 19s** | 37m 28s |

On individual queries, **40 lost valid-candidate coverage** going from epoch 2 to
epoch 3, while **12 gained it**; 45 had valid candidates in both runs and 16 in
neither. The aggregate decline therefore appears across many queries.

This remains one rollout per query at temperature 1.0, with fresh execution
measurements and potentially different GPU batching. Matching settings and seeds
make it a useful checkpoint comparison, not proof that every additional epoch
would hurt or that overfitting is the sole cause.

## What got worse

### More elaborate actions did not produce better overall reliability

Of the 541 issued attempts, **314 failed action validation**, **115 had valid
actions but failed plan constraints**, and **112 satisfied the constraints**.
The last category includes **24 candidates that timed out during execution**:
validity establishes a permitted physical plan, not successful or fast execution.
Epoch 2 under v7 had 189 constraint-satisfying attempts out of 547, compared with
112/541 here: **34.6% versus 20.7%**.

The action distribution changed substantially. Object-valued actions containing
a Leading tree increased from **142 in 026 to 340 here**. The number of these
that satisfied plan constraints also increased, from **18 to 79**. The model
therefore showed some increased ability to express accepted trees, alongside a
much stronger tendency to propose them and worse overall results.

Of the 18 measured selections containing Leading, **5 improved and 13 regressed**
outside the log band. Several actions also changed join methods, scans, or
parallelism, so the effect cannot always be assigned to the tree alone. The
job-11d and job-25a examples show that even tree-only interventions went both
ways. These results do not support declaring Leading universally useless.

Other concrete failure patterns:

- **108 recorded candidate actions were strings instead of objects**; all 108
  failed inner JSON parsing when checked. Simple unwrapping would not repair
  these particular records.
- **93 attempts** received disconnected-subtree diagnostics.
- **37 attempts** received unmet Memoize diagnostics, versus 6 in 026. The v7
  explanation that a Memoize hint does not guarantee a Memoize node was present;
  constraint verification remained strict.
- No model response was recorded as truncated, and no rollout ended at its
  context or model-turn limit. Context capacity was not the observed failure mode.

These diagnostic categories can overlap. Counts concern issued candidate
attempts, not every attempted tool call after the budget was exhausted.

### Final decisions frequently ignored the execution feedback

**Fourteen selected candidates already looked slower than the initial default
by more than 0.05 in log space; all fourteen remained regressions beyond that
band in final timing.** This accounts for 14 of the 15 final measured regressions.
The remaining one, job-17f, moved from a near-neutral preliminary 0.983× to a final
0.894×, illustrating the limits of single-measurement candidate feedback.

For job-03a, the agent saw **1,065.945 ms for its candidate versus a 51.579 ms
default**, yet selected that candidate. The final paired result was 0.051×.
Returning to default was explicitly available under v7.

All **six selected timeouts** chose candidates already known to have timed out
during feedback. On job-20a and job-26a, there was also an eligible candidate
with a completed feedback measurement; in every case the default was available.
These were not six new slowdowns discovered only after selection.

Ranking candidates was better than comparing them with the default, but imperfect:
**30 of 33 selections with numeric feedback tied for the lowest eligible feedback
latency**, including five default duplicates. The three exceptions were job-01a,
job-01c, and job-21a. A further missed apparent opportunity was job-17c: the model
retained default despite a candidate with a **24.41× preliminary ratio**. That
candidate was not selected for final pairing, so this is not a verified lost
24× speedup.

### More default endings did not mean the model simply refused to search

Of the **66 kept-default outcomes**, only **two** occurred before any candidate
attempt: job-18c and job-19a. The other **64 followed search**: 18 had at least one
eligible alternative and 46 had none. Overall, 104/113 rollouts used all five
candidate attempts.

Those 46 fallbacks after unsuccessful search correctly contribute 1.0× under the
accepted v7 policy, but they do not count as producing valid candidates. This is
why scored coverage, valid-plan coverage, and optimization gains must be reported
separately. The main failure was unsuccessful search and poor decisions after
feedback, not widespread early `keep_default` usage.

## What the harness fixes accomplished

Compared with the earlier 021 evaluation, the operational changes were visible:

| Metric | 021: epoch 2, v6 | 024: epoch 3, v7 |
| --- | ---: | ---: |
| Rejected selection calls | 389 | **10** |
| Unavailable-tool calls | 513 | **19** |
| Out-of-budget `evaluate_candidate` calls | 473 | **15** |
| Selection-failed outcomes | 17 | **0** |
| Context-budget endings | 16 | **0** |
| Initial default executions, including warmups | 226 | **452** |

Each of 024's ten selection rejections occurred in a different rollout; none
became a repeated selection-rejection loop. All eight `no_valid_candidate`
outcomes followed five attempts. Their queries were job-01b, job-01d, job-02b,
job-08c, job-13b, job-13c, job-15a, and job-18a.

Final timeout queries were job-06d, job-13d, job-19d, job-20a, job-20b, and job-26a.
All 113 rollouts were recorded, with no recorded infrastructure failure.

Because the weights also changed, the 021→024 comparison alone cannot attribute
every operational improvement to the harness. The later **021→026 comparison
holds the epoch-2 weights fixed** and corroborates the reduction in terminal
failures; see [026's results](../026-qwen-4b-job-epoch2-v7-5cand-1rollout/results.md).

## Training evidence and the decision it supported

The teacher-validation history on the original held-out conversations was:

| Completed epoch | Training experiment / step | Final validation loss |
| --- | --- | ---: |
| 1 | 018 / 382 | 0.305326 |
| 2 | 020 / 764 | 0.306212 |
| 3 | 023 / 1146 | 0.322112 |

023's incoming validation at step 764 was exactly **0.306212**, matching 020's
final value, before rising **5.2%** during the additional epoch. The native
continuation record identifies 020's completed step-764 checkpoint, and the
training/validation row counts and supervised-token counts stayed unchanged.

The worsening held-out loss and weaker matched-harness rollout result are
consistent with overtraining on this small dataset, but they do not isolate its
cause. The shift toward more elaborate actions and poor feedback use are direct
behavioral observations. Validation loss alone would also have missed the earlier
benefit of the second epoch; autonomous rollout evaluation remains essential.

The supported decision was to **retain epoch 2 as the starting adapter and expand
demonstration coverage**, which is the branch subsequently used in experiment
025. Future training needs to be judged on valid candidate coverage, execution
feedback use, default-versus-candidate selection, and final measured performance,
alongside teacher-token validation loss. This result is a warning against assuming
another pass automatically helps; it does not settle the value of a second pass
over 025's larger, different dataset.

## Evidence and definitions

Primary evaluation artifacts are on FLOPper:

- `outputs/024-qwen-4b-job-epoch3-5cand-1rollout/000/evaluation/test/000/evaluation.json`
- Per-query records in the same directory: `rollouts/<task-id>/000.json`.
- Comparisons use the corresponding 021 and 026 reports and all 113 per-query
  records from each run, plus their saved `000/config.toml` files.

024 report SHA-256:
`e28817c2f5e764a5cbc96bfd3d5066caad44a45f07eed384f306c85ac85cab21`.

Training evidence is in Lambda's
`outputs/023-sft-astra-ceb-epoch3/000/training/{report.json,configs/identity.json}`
and the corresponding 020 files. Epoch-1 validation comes from 018's training
report on FLOPper. See [023's README](../023-sft-astra-ceb-epoch3/README.md) for
the continuation setup.

Geometric mean is `exp(mean(log(default_ms / candidate_ms)))` over outcomes with
recorded speedups. Defaults and timing-reuse duplicates contribute 1.0; invalid,
selection-failed, and timed-out outcomes are reported separately, without an
invented speedup. Candidate feedback and final paired measurements are distinct.
Reasoning tokens are a subset of completion tokens: 024 recorded **84,637
reasoning tokens within 237,297 completion tokens**.

JOB had already been used repeatedly during project iteration. Describe this
as a checkpoint-selection experiment on that benchmark, not a pristine final
held-out evaluation or a comparison against a mechanical scan/flag sweep.
