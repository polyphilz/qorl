# Results: second SFT epoch on the original Astra dataset, evaluated on JOB

Evaluation completed on **September 10, 2026**, on FLOPper: run `000`, split
`test`, evaluation `000`.

**The second epoch substantially improved candidate generation, but this run did
not beat PostgreSQL overall.** Valid-candidate coverage rose from **48/113 queries
(42.5%) after epoch 1 to 89/113 (78.8%) after epoch 2**, compared with **14/113
(12.4%) for the vanilla checkpoint**. Geometric-mean speedup was **0.901× across
88 scored outcomes**; the ratio of summed default/selected latencies was
**0.995×**, approximately neutral.

Two findings mattered for the next experiments. First, another epoch helped
autonomous behavior even though held-out teacher loss barely changed. Second,
the v6 harness prevented returning to default after search and allowed long
terminal loops. Those limitations obscured the checkpoint's usefulness, while
its remaining invalid and slow proposals still required better learning.

## What was evaluated

Experiment **020 performed the continuation training**; experiment **021
evaluated its exported adapter**. This was a second pass over the original
demonstrations, before the additional data collected in 025.

| Setting | Value |
| --- | --- |
| Base | `empero-ai/Qwen3.8-4B-Distill` |
| Base revision | `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e` |
| Adapter | Experiment 020, step 764: two total epochs |
| Training lineage | 018 step 382 → 020 step 764 |
| Training data | 99 accepted Astra conversations from 100 selected CEB tasks |
| Held-out teacher data | 19 accepted conversations from 20 selected validation tasks |
| Additional training in 020 | 382 updates on the same 382 packed training rows |
| Training settings | Batch/microbatch 1; LoRA rank 16, alpha 32, dropout 0; constant AdamW LR `1e-4`; BF16 |
| Evaluation workload | All 113 JOB queries, one rollout per query, seed 42 |
| Agent budget | Five candidate attempts; 64 model turns |
| Model settings | Thinking enabled; 49,152-token context; 8,192-token maximum reply |
| Sampling | Temperature 1.0, top-p 1.0, top-k 20 |
| Serving | vLLM 0.28.0, `qorl-adapter` over `qorl-base`, FLOPper GPU 0 |
| Harness / fingerprints | qo-agent v6 / plan fingerprint v4 |
| PostgreSQL / pool | `001-pgconf-2gb-sb` / `002-poolconf-4x8` |
| Evaluation wall time | **53m 18s**, 01:07:15–02:00:33 America/New_York |

The pool had four workers, each assigned four physical cores and 8 GiB RAM.
Initial default timing used one warmup followed by **one measurement**. Fresh
candidate feedback also used one warmup and one measurement. Final paired
scoring, when required, used one warmup pair and three measured pairs.

The serving record identifies
`outputs/020-sft-astra-ceb-epoch2/000/training/checkpoints/step_764/adapter`.
All **1,798 model responses** identify `qorl-adapter`; all 113 trajectories record
interface v6. The adapter tensor SHA-256 is
`d1f055e4a347fca34185a34d2f9baf22f9197e006dacfb0b018b52f178c918f4`.
The later 026 evaluation used this same exported adapter.
All 113 rollouts were recorded, with no recorded infrastructure failures or
truncated model responses.

## Comparison with vanilla and epoch 1

| Metric | 019: vanilla | 018: epoch 1 | 021: epoch 2 |
| --- | ---: | ---: | ---: |
| Queries with a valid candidate | 14/113 (12.4%) | 48/113 (42.5%) | **89/113 (78.8%)** |
| Queries with a novel candidate | 7/113 (6.2%) | 34/113 (30.1%) | **84/113 (74.3%)** |
| Distinct novel query/plan pairs | 8 | 51 | **135** |
| Issued candidate attempts | 493 | 535 | 558 |
| Valid actions, before plan satisfaction | 23/493 (4.7%) | 182/535 (34.0%) | **332/558 (59.5%)** |
| Candidates satisfying plan constraints | 15/493 (3.0%) | 90/535 (16.8%) | **225/558 (40.3%)** |
| Scored outcomes | 15 | 44 | **88** |
| Geometric-mean speedup, scored outcomes | 0.851× | 0.722× | **0.901×** |
| Ratio of summed default/selected latencies | 0.851× | 0.761× | **0.995×** |
| Final selection failures | 16 | 40 | **17** |
| Selected-candidate timeouts | 1 | 4 | **1** |
| Context-budget endings | 12 | 37 | **16** |
| Rejected selection calls | 326 | 1,330 | **389** |
| Model response count | 2,333 | 2,932 | **1,798** |
| Completion tokens | 1,305,579 | 584,102 | **328,135** |
| Evaluation wall time | 2h 02m 46s | 1h 18m 05s | **53m 18s** |

The saved PostgreSQL, pool, agent, measurement, evaluation, and inference
settings match across these three runs. All 113 task IDs, per-query model and
measurement seeds, initial system messages, initial tool definitions, and
default timing-reuse keys match. The serving records identify the intended
vanilla, step-382, and step-764 weights respectively.

This is a meaningful improvement in valid-candidate coverage under the same
harness: **54 queries gained a valid candidate relative to epoch 1, 13 lost one,
35 retained one, and 11 had none in either run**. The net gain was 41 queries,
or 36.3 percentage points. It was not uniform improvement on every query.

The speedup columns have different scored-query denominators. Vanilla's 0.851×
over just 15 outcomes cannot establish better performance than epoch 1's 0.722×
over 44. Restricting epoch 1 and epoch 2 to their **31 commonly scored queries**
gives **0.732× versus 0.965×**. That supports improvement on that subset, but
still excludes failures and is not an all-query performance estimate.

The reduced inference cost was also useful. Mean reasoning tokens per response
were approximately **406 for vanilla, 106 after epoch 1, and 102 after epoch 2**.
The second epoch's additional savings came largely from fewer responses, with
similar reasoning length per response. These observations are consistent with
learning a more concise interaction style from the demonstrations; there was
no ablation isolating reasoning summaries as the cause.

## Outcomes and measured performance

| Final outcome | Count | Speedup treatment |
| --- | ---: | --- |
| Measured candidate | 64 | Final paired medians |
| Default timing-reuse duplicate | 24 | Exactly 1.0× |
| Kept default | 0 | None |
| No valid candidate | 7 | Unscored |
| Selection failed | 17 | Unscored |
| Selected-candidate timeout | 1 | Unscored |

Thus **24 of the 88 scored outcomes contribute exactly 1.0×**. Among the 64
measured candidates, 23 were faster and 41 slower. Using the descriptive
log-space band `abs(log(speedup)) <= 0.05`, the breakdown is **15 improvements,
21 regressions, and 28 near-neutral results**. This band is not a statistical
significance test.

Summed scored default latencies were **42.011 seconds**, versus **42.203 seconds**
for selected candidates: **0.192 seconds more, or 0.46% higher latency**. These
are sums of per-query medians, using the initial default median on both sides
for timing-reuse duplicates. They exclude inference, search, warmups, and all
25 unscored outcomes. The 0.901× geometric mean and near-neutral summed latency
describe different aspects of this heterogeneous workload; neither includes
the cost of failures.

Selected examples, using final paired medians in milliseconds:

| Query | Default | Candidate | Speedup | Intervention |
| --- | ---: | ---: | ---: | --- |
| job-02c | 162.561 | 6.732 | **24.148×** | Bitmap scans on `k` and `mk` with specified indexes |
| job-01a | 17.768 | 2.379 | **7.469×** | Bitmap scan on `mi_idx` |
| job-13b | 193.588 | 33.898 | **5.711×** | Leading tree plus an index scan on `t` |
| job-16b | 5,784.934 | 2,178.956 | **2.655×** | Index scan on `mk` |
| job-05a | 41.584 | 1,626.772 | **0.0256×** | Bitmap scan on `mi` |
| job-06a | 6.483 | 249.699 | **0.0260×** | Bitmap scan on `k` |
| job-04c | 39.257 | 162.782 | **0.241×** | Leading tree plus a hash join |

The run found substantial improvements on individual queries and severe
regressions on others. Scans and indexes appear in many large gains, but this
run does **not** support saying that every Leading intervention lost: job-13b
and job-06b improved beyond the log-space band. Combined constraints also
prevent attributing a gain to one action field without further measurement.

## What the v6 harness obscured

### Returning to default was unavailable after search

After a submission, `keep_default` was unavailable. If eligible candidates
existed, v6 required choosing one of them at `finish`. It did not support
selecting PostgreSQL's default after discovering that the proposed alternatives
were worse.

In **21 rollouts**, every eligible candidate had either timed out or had a
preliminary ratio below `exp(-0.05)` relative to the initial default. Twenty
ended in measured regressions; the remaining one, **job-13a**, selected its only
eligible candidate, which had already timed out at **5,000 ms** against an
initial default of **385.501 ms**.

For **job-05a**, the agent had used all five attempts and only candidate 1 was
eligible. Its feedback was **1,645.611 ms**, versus approximately **40 ms** for
default, before final scoring confirmed **1,626.772 versus 41.584 ms**. Selecting
another eligible candidate could not repair this outcome. Candidate generation
had failed to find an improvement, and the interface prevented a safe final
choice of default.

Selection among the available candidates was comparatively strong: **85 of 88
selections with numeric feedback tied for the lowest observed candidate
latency**, and **73 selections used an earlier attempt**. These counts include
ties and sole eligible choices, so they do not establish sophisticated ranking
or that the selected candidate was truly fastest. They do show why a poor
aggregate speedup should not automatically be interpreted as ignoring feedback.

Single-measurement feedback was another limitation. For example, **job-20a**
looked roughly **1.051×** faster in preliminary feedback but scored **0.864×**
under final paired measurement. Preliminary ratios were observations for
decision-making, not guaranteed final speedups.

### No-eligible terminal states still wasted substantial inference

All **17 selection-failed rollouts had zero eligible candidates**. Sixteen
exhausted context and one hit the model-turn limit. Together those 17 rollouts
consumed **928 of the run's 1,798 model responses**.

Across the full run there were **389 rejected selection calls**, including
**296 asking for `selected_candidate_id="default"`**, which v6 rejected. There
were also **513 unavailable-tool calls**, including **473 calls to
`evaluate_candidate` when it was unavailable**. These calls do not count as
additional issued candidate attempts. Prompt usage reached **34.5 million
tokens**, counting repeated conversation context across requests.

The model could terminate with no valid candidate by using the correct empty
`finish` call, and seven rollouts eventually did so. Rejection guidance and the
training distribution were insufficient to make that behavior reliable. The
terminal failure was therefore a model/interface interaction, not an inference
server crash or evidence that a longer context window was the necessary fix.

## What still needed to improve in the policy

Of the **558 issued candidate attempts**:

- **226** failed action validation.
- **107** had valid actions but did not satisfy the requested plan constraints.
- **225** satisfied the plan constraints; this includes **nine execution-timeout
  attempts**, so plan validity does not guarantee acceptable runtime.

Recorded string-valued actions fell from **170 in epoch 1 to 57 in epoch 2**, but
the model still frequently produced invalid or unsupported interventions.
Grounding a complete Leading tree remained difficult: only **29 of 170
object-valued actions containing Leading** satisfied the plan constraints,
compared with **35 of 135** after epoch 1. Overall candidate coverage improved
while this particular action family did not.

The appropriate interpretation was that the model could learn substantially
more of the harness from the existing demonstrations, but had not mastered
PlanAction grounding or reliable optimization. Allowing default fallback would
address harmful final choices; it would not turn failed proposals into valid
plans or teach the model to find better ones.

## Training evidence and later clarification

020's validation measured the incoming and outgoing weights on the same held-out
teacher data on Lambda:

| Checkpoint | Held-out teacher loss |
| --- | ---: |
| Incoming epoch 1, step 382 | 0.305408 |
| Outgoing epoch 2, step 764 | 0.306212 |

Loss rose by only **0.000804**, approximately **0.26%**, while JOB valid-candidate
coverage rose from 42.5% to 78.8%. Teacher-token validation and autonomous
rollouts measured different things. A nearly flat validation curve did not
mean that another epoch could not improve useful behavior. Conversely, this
result did not imply that additional epochs would keep helping.

The later [026 evaluation](../026-qwen-4b-job-epoch2-v7-5cand-1rollout/results.md)
held the **epoch-2 adapter fixed** while applying harness v7, including default
fallback, clearer terminal handling, and three initial default measurements:

| Metric | 021: epoch 2, v6 | 026: same epoch 2, v7 |
| --- | ---: | ---: |
| Queries with a valid candidate | 89 | 85 |
| Scored outcomes | 88 | 108 |
| Geometric-mean speedup, scored outcomes | 0.901× | 1.075× |
| Kept-default outcomes | 0 | 53 |
| Rejected selection calls | 389 | 6 |
| Selection failures | 17 | 1 |
| Context-budget endings | 16 | 1 |
| Evaluation wall time | 53m 18s | 32m 19s |

On the **84 commonly scored queries**, geometric means were **0.888× versus
1.110×**. The change therefore was not solely adding more neutral outcomes to
the scored denominator. Nevertheless, these were new stochastic rollouts with
changed prompts and feedback, not a replay that only replaced final selections.
The comparison supports the value of the combined harness changes, without
isolating the effect of any single fix. It is **not additional model learning**.

The later [024/026 comparison](../024-qwen-4b-job-epoch3-5cand-1rollout/results.md)
also established that a third epoch on the original dataset regressed under
matching v7 settings. In retrospect, the useful next branch was to **retain
epoch 2, repair the interface, and expand demonstration coverage**, which became
the starting point for experiment 025. This run showed meaningful learning from
Astra demonstrations containing reasoning summaries; it did not establish that
summaries were optimal or compare them with raw teacher reasoning.

## Evidence and definitions

Primary evaluation artifacts are on FLOPper:

- `outputs/021-qwen-4b-job-epoch2-5cand-1rollout/000/evaluation/test/000/evaluation.json`
- Per-query records in the same directory: `rollouts/<task-id>/000.json`.
- Comparisons use the corresponding 018, 019, and 026 reports, saved run configs,
  and all 113 per-query records from each run.

021 report SHA-256:
`96ded2c4c945b087f15ef5ef59fa9218b6349ae048a27e71e7fd5d09a3e37fcd`.

Training counts and paired incoming/final validation losses come from Lambda's
`outputs/020-sft-astra-ceb-epoch2/000/training/report.json`.
See [020's README](../020-sft-astra-ceb-epoch2/README.md) for the continuation setup.

Valid-candidate coverage means at least one submitted candidate satisfied the
plan constraints; it does not mean every reply in that trajectory was valid.
Geometric mean is `exp(mean(log(default_ms / candidate_ms)))` over outcomes with
recorded speedups. Timing-reuse duplicates contribute 1.0; invalid,
selection-failed, and timed-out outcomes remain separate, without an invented
speedup. The summed-latency ratio uses the same scored cohort. Candidate
feedback is distinct from final paired timing. Reasoning tokens are a subset
of completion tokens: **183,658 within 328,135** in this run.

This was one rollout per query at temperature 1.0, and JOB was used repeatedly
for checkpoint selection. It provides evidence about this training trajectory
and harness, not a pristine final held-out result, a guarantee of repeatability,
or a comparison against a mechanical scan/flag sweep.
