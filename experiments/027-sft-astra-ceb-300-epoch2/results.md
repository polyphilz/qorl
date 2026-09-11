# Results: second epoch on the additional Astra demonstrations, run 000

Training completed on Lambda on **September 11, 2026, at 02:51:57
America/New_York** (06:51:57 UTC). Evaluation of the resulting step-2204 adapter
completed on FLOPper at **03:34:03 America/New_York** (07:34:03 UTC), after
**34m 22s**. All 113 JOB rollouts were recorded; there were no recorded
infrastructure or model-call failures.

**The extra epoch improved conservative selection, but did not produce a clear
overall upgrade over step 1102.** Reported geometric-mean speedup rose from
**1.103× to 1.155×**, while valid-plan coverage fell from **77/113 to 71/113**.
The scored subsets differ: on the **95 queries scored in both evaluations**,
speedup was essentially unchanged, **1.0918× versus 1.0940×**. The model made
fewer visibly bad final choices, generated more successful `Leading` actions,
and also produced substantially more malformed string actions.

## Training and checkpoint lineage

| Setting | Value |
| --- | --- |
| Starting checkpoint | `025-sft-astra-ceb-300/002`, step 1102 |
| Earlier lineage | Original 99-demonstration epoch-2 adapter, step 764, followed by 025's first epoch on the new demonstrations |
| Continuation | Native adapter, AdamW state, scheduler and progress restored |
| Additional training | 1,102 updates, steps 1103–2204; second epoch on this dataset |
| Base | `empero-ai/Qwen3.8-4B-Distill`, revision `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e` |
| LoRA | Rank 16, alpha 32, dropout 0 |
| Learning rate / batch | Constant `1e-4`; batch and microbatch size 1 |
| Context / precision | 49,152 tokens; BF16 |
| Hardware | One H100 80 GB on Lambda |
| Training data | 293 conversations; 2,180 supervised requests; 1,102 packed rows |
| Validation data | 19 conversations; 131 supervised requests; 73 packed rows |
| Supervised tokens per epoch | 376,847 training; 24,459 validation |

The prepared data was reused from 025. The six early `keep_default`
demonstrations stayed excluded, and one of the remaining 294 conversations
exceeded the context limit, leaving 293. The 197 malformed or action-invalid
training requests remained excluded from supervision while their messages
remained available as context. No new teacher calls or additional outcome
filtering were introduced. See [025's filtering notes](../025-sft-astra-ceb-300/README.md)
and [results](../025-sft-astra-ceb-300/results.md).

This was a continuation with optimizer state, unlike 025's initialization from
an exported adapter with a fresh optimizer. The source step-1102 checkpoint was
read-only and remains intact. Continuation records explicitly say
`historical_rng_restored=false`; this is not a claim of bitwise equivalence to
an uninterrupted two-epoch job. The initial launcher failed before model
loading or updates because it selected the system `torchrun`; activating the
intended GPU environment resolved that, as recorded in [README.md](README.md).

| Training measurement | Result |
| --- | ---: |
| Incoming validation loss, step 1102 | 0.336020 |
| Final validation loss, step 2204 | 0.331642 |
| Relative validation-loss reduction | 1.30% |
| Mean training loss, steps 1103–1202 | 0.224383 |
| Mean training loss, steps 2105–2204 | 0.242547 |
| Mean training loss, all 1,102 new updates | 0.239941 |
| Missing update-loss records | 0 |
| Nonfinite losses / recorded NaN counts | 0 / 0 |
| Trainer-reported peak GPU memory | 21.65 GiB |
| Launcher exit code | 0 |

Validation still improved, but much less than 025's 7.3% reduction. The
training-window means are unweighted averages over different batches, not a
fixed-example learning curve. Neither their fluctuations nor the small
validation improvement proves overfitting. Validation measures prediction of
held-out teacher messages; autonomous JOB behavior must be assessed separately.

## Evaluation conditions and verification

The evaluation used **all 113 JOB queries, one rollout per query, seed 42,
five candidate attempts, thinking enabled, 49,152-token context and an
8,192-token maximum reply**. Temperature was 1.0. PostgreSQL ran on FLOPper
with `001-pgconf-2gb-sb` and `002-poolconf-4x8`: four workers, four physical
cores and 8 GiB per worker. Candidate selection remained the model's decision.

Each initial default received one warmup and three measured executions.
Candidate feedback used one warmup and one measurement when fresh timing was
needed. Final measured selections used one warmup pair and three measured
default/candidate pairs. Feedback ratios are preliminary observations, not the
final paired score.

All **817 model responses identify `qorl-adapter`**. The completed serving
report identifies its parent as `qorl-base` and its adapter path as this run's
`training/checkpoints/step_2204/adapter`; it reports vLLM 0.28.0. The exported
adapter and native checkpoint passed the transfer/evaluation verification.

All 113 traces record **interface v7 and plan fingerprint v4**. Compared
query by query with both 025 and 026, the system prompts, tool-definition
hashes, model seeds, measurement seeds and default timing-reuse keys match
for all 113 queries. Relevant evaluation, inference, measurement, PostgreSQL
and pool settings match. The 026 experiment supplied its adapter explicitly;
025 and 027 evaluated their own training checkpoint. This comparison is not
confounded by another harness revision or a different default-plan key.

## JOB results across the three checkpoints

026 is the original epoch-2 adapter evaluated under v7. 025 is the first epoch
on the additional 293 demonstrations. 027 is the second epoch on those same
demonstrations.

| Metric | 026: step 764 | 025: step 1102 | 027: step 2204 |
| --- | ---: | ---: | ---: |
| Queries with a valid candidate /113 | 85 (75.2%) | 77 (68.1%) | **71 (62.8%)** |
| Queries with a novel candidate /113 | 76 | 72 | **69** |
| Distinct novel query/plan pairs | 115 | 123 | **113** |
| Candidate attempts | 547 | 521 | **507** |
| Attempts passing action validation | 330 | 246 | **246** |
| Attempts passing action and plan constraints | 189 (34.6%) | 148 (28.4%) | **138 (27.2%)** |
| Scored outcomes /113 | 108 | 101 | **107** |
| Geometric-mean speedup, scored subset | 1.075× | 1.103× | **1.155×** |
| Ratio of summed scored default/selected latencies | 1.036× | 1.047× | **1.057×** |
| Measured improvements outside the 0.05 log band | 12 | 29 | **20** |
| Measured regressions outside the same band | 8 | 13 | **5** |
| Kept-default outcomes | 53 | 43 | **72** |
| No-valid-candidate outcomes | 4 | 12 | **4** |
| Selection failures | 1 | 0 | **1** |
| Selected-candidate timeouts | 0 | 0 | **1** |
| Rejected selections | 6 | 16 | **6** |
| Model replies | 915 | 803 | **817** |
| Prompt tokens | 11,756,852 | 9,448,730 | **9,760,337** |
| Completion tokens | 165,262 | 191,726 | **208,848** |
| Reasoning tokens, included in completion tokens | 93,243 | 78,583 | **81,878** |
| Evaluation wall time | 32m 19s | 32m 35s | **34m 22s** |

The 027 outcomes were **33 measured, 72 kept default, two default duplicates,
four no-valid-candidate, one selection failure and one timeout**. The first
three groups make up the 107 scored outcomes; the other six are excluded from
the geometric mean. A kept default contributes exactly 1.0×, including when
the preceding search produced no valid candidate. Scored completion and
valid-plan production therefore measure different things.

Among the 33 measured selections, 27 were faster and six slower than default.
Using the descriptive band `abs(log(speedup)) <= 0.05` gives **20 improvements,
five regressions and eight near-neutral measurements**. The 74 defaults and
duplicates also contribute 1.0×. This band is not a per-query significance test
or a calibrated false-reward guarantee.

Summed scored default and selected latencies were **54.960 s and 52.012 s**:
**2.948 s saved, or 5.36%**. These are sums of query medians on the same 107
queries, using the initial default on both sides for defaults and duplicates.
They exclude the six unscored outcomes, inference and candidate-search overhead;
they do not mean that optimizing and executing JOB took 52 seconds.

### Why the headline increase needs qualification

| Matched comparison | Queries scored in both | Earlier checkpoint GM | 027 GM |
| --- | ---: | ---: | ---: |
| 025 versus 027 | 95 | 1.0918× | **1.0940×** |
| 026 versus 027 | 102 | 1.0860× | **1.1537×** |

On the matched 025/027 subset, the relative increase is just **0.21%**. With
one stochastic rollout per query and changing candidates, this is not evidence
of a reliable performance improvement. It also does not establish equivalence.

Twelve queries gained a scored outcome relative to 025. They include
**job-01a at 7.660× and job-01b at 109.596×**, both previously no-valid-candidate.
Conversely, six formerly scored queries became unscored: **job-01c**
(previously 10.089×), **job-19d** (0.962×), **job-25a** (1.0×), **job-26b**
(1.0×), **job-28a** (0.505×) and **job-32a** (1.0×). These changes affect both
the numerator and denominator of the headline comparison.

Valid-plan coverage also turned over substantially: **23 queries gained a
valid candidate and 29 lost one**, for the net decline of six. This is more
than a uniform improvement or deterioration on the same queries.

The largest win is a short query: job-01b saves about **16.94 ms**, despite its
109.6× ratio. Excluding the largest scored win leaves 027 at **1.107×**;
excluding its three largest wins leaves **1.069×**. The gains are not entirely
one outlier, but the geometric mean and total time saved tell different stories.

## Analysis: selection improved, exploration became more conservative

### Default selection is now much more common

| Kept-default outcome | 026 | 025 | 027 |
| --- | ---: | ---: | ---: |
| Before any candidate attempt | 1 | 1 | **4** |
| After search, with at least one eligible candidate | 29 | 19 | **34** |
| After search, with no eligible candidate | 23 | 23 | **34** |
| Total | 53 | 43 | **72** |

The model selected a candidate whose feedback was clearly slower than the
initial default on **five queries**, down from **16 in 025**. Conditional on
every numerically timed eligible option being slower than default outside the
0.05 log band, it chose default in **23/28 cases**, versus **15/31 in 025**. This is direct evidence
that the extra epoch changed the decision behavior we were concerned about.

However, **38 of the 72 kept defaults produced no valid candidate**: 34 after
failed search and four without trying. The four early defaults were job-04c,
job-06e, job-21b and job-29a. They are legitimate terminal decisions under v7,
but not demonstrations of successful plan generation. Some old regressions
disappeared because the model gave up, rather than because it repaired them.

The final candidate histories contained a feedback improvement outside the
band on **22 queries**, down from 29 in 025. There were fewer opportunities
to choose a clear win, as well as fewer bad choices. For example, 025's large
wins on job-01d, job-07a, job-13c, job-28c and job-30b became defaults in 027.
That does not mean the model ignored the same good candidate: those earlier
winning options generally were not available in its new search.

Among the **35 selections with numerical feedback**, 34 selected the best
feedback candidate or a tie. The exception, job-13d, selected an option only
1.614 ms slower than the best at roughly 2,005 ms. The much larger mistake was
choosing any of them over the roughly 599 ms default. Selecting among candidates
is largely working; comparing the search result with default is still imperfect.
The remaining selected candidate, job-19d, had timed out and had no numerical
feedback ratio.

### The remaining regressions were visible before final scoring

| Query | Selected feedback ratio | Final default ms | Final selected ms | Final speedup |
| --- | ---: | ---: | ---: | ---: |
| job-13d | 0.298× | 595.066 | 1,997.729 | **0.298×** |
| job-25c | 0.424× | 2,123.286 | 5,102.919 | **0.416×** |
| job-07b | 0.927× | 76.724 | 88.739 | **0.865×** |
| job-15a | 0.873× | 94.560 | 108.030 | **0.875×** |
| job-17b | 0.948× | 2,471.955 | 2,756.663 | **0.897×** |

All five material measured regressions were already on the slower side of
the same log-space band in the feedback. They were not apparent wins that
became losses only under fresh final measurement. Job-19d adds a separate
failure: the model selected an already timed-out candidate instead of default.

A narrow counterfactual illustrates the remaining selection cost: keep every
actual choice and final paired score, but replace a selected candidate with
default when its recorded feedback has `log(ratio) < -0.05`. This gives
**1.182× for 027** versus **1.285× for 025**, each on its original scored subset.
On the common 95-query subset, the corresponding values are **1.123× and
1.274×**. This diagnostic uses no invented timings for unselected candidates
and does not assign scores to the six failures/timeouts. It is not an executed
policy evaluation or a recommendation to override the agent's choice.

The result explains the tradeoff: **025's recorded choices leave more gain
available through better selection; 027 already makes more of those conservative
decisions, but retains fewer measured wins.** It does not prove which checkpoint
will learn better under RL.

### `Leading` is becoming usable, while formatting remains fragile

| Action diagnostic | 026 | 025 | 027 |
| --- | ---: | ---: | ---: |
| Object-valued actions containing `leading` | 142 | 161 | **197** |
| Those passing action and plan constraints | 18 (12.7%) | 41 (25.5%) | **83 (42.1%)** |
| Measured wins outside the band whose selected action contains `leading` | 0 | 3 | **10** |
| String-valued actions rejected as not an object | 72 | 93 | **170** |
| String-valued actions that parse into a JSON object | 0 | 0 | **0** |

The object-valued `Leading` success rate omits malformed string submissions.
In 027, **167 of the 170 rejected strings contain the text `leading`**. All
170 strings also fail JSON parsing; simply unwrapping a valid JSON object would
not rescue these recorded attempts. The run had no model-output-limit stops,
so recorded reply truncation does not explain this increase.

There is useful learning here: ten measured improvements now use actions
containing a complete leading tree, including job-18a at **5.645×**, job-17c at
**2.752×** and job-24a at **2.379×**. Most combine the tree with join, scan,
parallel or row-correction constraints. The result supports learning to use the
representation; it does not isolate join order as the cause of each speedup.

Across all 507 attempts, **246 passed action validation and 138 also passed
plan constraints**. The remaining 261 failed action validation; 108 valid
actions did not satisfy their requested plan constraints. Common latter
diagnostics concern PostgreSQL choosing a different scan or index than requested.
Ten constraint-valid attempts timed out during execution feedback. Successful
tree construction has therefore improved without solving overall reliability.

### What changed relative to the hypothesis in 025

The earlier analysis proposed that another epoch would do little for default
selection. **That prediction was not borne out:** slower-option selections
fell from 16 to five, including a clear improvement within all-slower search
states. At the same time, validity did not broadly improve, despite more
successful `Leading` actions.

The evidence supports a change in both action mix and decision behavior. It
does not establish a single cause, prove general overfitting, or show that
reasoning summaries are inherently inadequate. The held-out teacher loss still
fell slightly, while autonomous behavior changed in opposing directions.

## Harness behavior and the six unscored queries

The v7 behavior remained intact: default could be selected after searching,
earlier candidates could be selected, and rejected or absent candidates did
not become measured outcomes. There were **six rejected selections across six
queries**, with no repeated rejection loop. Three cases with eligible choices
were corrected: job-13a selected candidate-02, job-20a selected candidate-03,
and job-23a chose default after naming `candidate-6`.

Of the 36 final candidate selections, **32 selected an attempt earlier than
the last submission**. The selected positions were 1/2/3/4/5 on 12/9/8/6/1
rollouts, respectively. The multi-candidate selection machinery is being used;
the harness is not simply selecting the last proposal.

| Unscored query | Outcome | Recorded behavior |
| --- | --- | --- |
| job-01c | `no_valid_candidate` | No eligible option; named rejected candidate-04. Search ended without a score. |
| job-25a | `no_valid_candidate` | Four action-invalid attempts, followed by `finish({})`. |
| job-26b | `no_valid_candidate` | Named candidate-05, which failed plan constraints; no eligible option. |
| job-32a | `no_valid_candidate` | Named candidate-05, which failed plan constraints; no eligible option. |
| job-28a | `selection_failed` | Reached the context budget after five valid candidates, before a terminal selection. |
| job-19d | `timed_out` | Selected candidate-02 after its recorded execution timeout at 6,647 ms. |

Job-28a is a context-capacity failure, not an endless finish loop: it had only
six model replies and never called `finish`. Candidate-04's preliminary ratio
was 1.082×, but it was never selected or freshly scored. Job-19d's two eligible
options were both recorded execution timeouts; the other three failed plan
constraints. The default was still available. This is a poor model decision
within the intended timeout-selection contract, not an infrastructure error.

There were **seven unavailable-tool calls**: five `evaluate_candidate` calls,
one invented `evaluation_candidate` and one `get_plan`. Thus budget/tool
conformance is not perfect, but the earlier large terminal loops did not
reappear. Stops were 108 `model_finish`, four `model_keep_default` and one
`context_budget`; there were no output-limit or turn-limit stops.

Execution accounting is complete: **452 initial-default executions, 247
candidate-feedback executions and 264 final-paired executions**, including
warmups. The 264 final executions equal 33 measured selections times eight
executions each. No timing score was substituted for a failed selection or
timeout.

## Representative measured gains

| Query | Final default ms | Final selected ms | Speedup | Selected intervention |
| --- | ---: | ---: | ---: | --- |
| job-01b | 17.097 | 0.156 | **109.596×** | Bitmap `mi_idx` scan and nested-loop `it`/`mi_idx` join |
| job-01a | 18.331 | 2.393 | **7.660×** | Bitmap `mi_idx`, hash join constraint and `enable_sort=false` |
| job-18a | 1,220.073 | 216.145 | **5.645×** | Leading tree, nested loops and `ci` index scan |
| job-26c | 1,488.330 | 343.059 | **4.338×** | `random_page_cost=1`, with `enable_memoize=true` |
| job-06d | 1,622.031 | 570.688 | **2.842×** | Hash join, row corrections, parallel `k` and `mk` index scan |
| job-17c | 2,189.657 | 795.617 | **2.752×** | Leading tree, hash joins and row corrections |
| job-24b | 90.754 | 36.918 | **2.458×** | Parallel `k` and bitmap `cn` scan |
| job-24a | 206.296 | 86.728 | **2.379×** | Leading tree, nested loops, `mk` index and parallel-cost settings |

These describe complete selected actions, not ablations attributing the
improvement to an individual field. The five material measured regressions are
listed above; the per-query appendix also preserves neutral and failed outcomes.

## Implications for the next experiment

**There is enough evidence to stop adding blind SFT epochs and test the RL
learning signal.** The second epoch improved default decisions and some
structured actions, but has not delivered a clean improvement in both validity
and performance. Another pass over the same demonstrations is not clearly the
best use of the next training window.

Step 2204 is a reasonable, more conservative RL starting candidate, with
fewer measured regressions and more scored completions. Step 1102 should remain
available: it produced more valid queries, more distinct novel plans and more
measured wins. Neither this single evaluation nor the counterfactual establishes
which will respond better to RL. Track valid-plan production, actual speedups,
post-search defaults with no valid candidate, and timeout/selection failures
separately; completion alone can improve through defaulting.

JOB has now been used repeatedly to guide development and checkpoint choices.
It is a useful comparison workload, but should not be described as an untouched
final test set. These results also do not establish superiority over a mechanical
hint sweep or end-to-end savings after the search cost. No new training,
evaluation or RL smoke was launched during this analysis.

## Evidence and reproduction

The completed run exists on both hosts:

```text
Lambda:  /lambda/nfs/qorl/projects/qorl/outputs/027-sft-astra-ceb-300-epoch2/000
FLOPper: /home/rohan/projects/qorl/outputs/027-sft-astra-ceb-300-epoch2/000
```

Training evidence originated on Lambda and was copied to FLOPper. Evaluation
evidence is on FLOPper. Paths below are relative to the run directory:

- `training/report.json`: preparation counts, continuation progress and validation losses.
- `training/configs/identity.json`: source checkpoint, base hash and trainer provenance.
- `training/monitors/file/metrics.jsonl`: every new optimizer update and the validation measurements.
- `training-launch.exit`: successful training exit.
- `training/checkpoints/step_2204/adapter/qorl-manifest.json`: exported tensor and native checkpoint hashes.
- `transfer-from-lambda.json` and `evaluation-preflight.json`: transfer and evaluation verification.
- `evaluation/test/000/evaluation.json`: completed report and served model identity.
- `evaluation/test/000/rollouts/<task-id>/000.json`: raw actions, requests, feedback, selection and paired timings.

The exported `adapter_model.safetensors` is **42,500,760 bytes (40.5 MiB)**.
Its SHA-256 is:

```text
f20c171c1da2cdf45f4d84bd945877702214e4b81d3586b3a0abef201ea3ed4f
```

The native step-2204 checkpoint hash is:

```text
d0a6587998ad68b040d2ce6dbad29405c4d06324192c9848d643cad4518254ea
```

The untouched source step-1102 checkpoint hash is:

```text
a91ee09bea645a93fb8316ec9d7813fe7e18557d98ffa2839aa3b12d53d219c8
```

The evaluated report SHA-256 is:

```text
7ff72d86bdd0b557dace22bd1d4a719f82d67f213e3018b719f0d68b22e9130b
```

Counts and geometric means were independently recomputed from all 113
per-query records in each of 026, 025 and 027. The geometric mean is
`exp(mean(log(final.speedup)))` over non-null scores. Matched comparisons
intersect task IDs with non-null scores in both runs; they do not impute
default for failures. Validity counts use `action_valid && constraints_satisfied`;
eligibility and feedback comparisons use the final candidate history shown to
the model. Candidate actions can contain multiple intervention families, so
their family counts overlap.

## Mean training loss in 20-update windows

Windows cover the new updates only, using cumulative native step numbers.
Each value is the arithmetic mean of per-update losses; the last window has
only two updates. These averages are not weighted by supervised-token count.

| Updates | Mean training loss |
| --- | ---: |
| 1103–1122 | 0.237951 |
| 1123–1142 | 0.261814 |
| 1143–1162 | 0.297086 |
| 1163–1182 | 0.148493 |
| 1183–1202 | 0.176569 |
| 1203–1222 | 0.263353 |
| 1223–1242 | 0.287755 |
| 1243–1262 | 0.202944 |
| 1263–1282 | 0.219145 |
| 1283–1302 | 0.334119 |
| 1303–1322 | 0.148080 |
| 1323–1342 | 0.166992 |
| 1343–1362 | 0.219380 |
| 1363–1382 | 0.253892 |
| 1383–1402 | 0.230031 |
| 1403–1422 | 0.245844 |
| 1423–1442 | 0.192113 |
| 1443–1462 | 0.193300 |
| 1463–1482 | 0.379108 |
| 1483–1502 | 0.148627 |
| 1503–1522 | 0.157285 |
| 1523–1542 | 0.369759 |
| 1543–1562 | 0.165613 |
| 1563–1582 | 0.264659 |
| 1583–1602 | 0.166799 |
| 1603–1622 | 0.226652 |
| 1623–1642 | 0.246578 |
| 1643–1662 | 0.260455 |
| 1663–1682 | 0.276712 |
| 1683–1702 | 0.289849 |
| 1703–1722 | 0.253719 |
| 1723–1742 | 0.170203 |
| 1743–1762 | 0.368041 |
| 1763–1782 | 0.069641 |
| 1783–1802 | 0.299675 |
| 1803–1822 | 0.186365 |
| 1823–1842 | 0.334239 |
| 1843–1862 | 0.289894 |
| 1863–1882 | 0.117308 |
| 1883–1902 | 0.323252 |
| 1903–1922 | 0.196573 |
| 1923–1942 | 0.242000 |
| 1943–1962 | 0.127285 |
| 1963–1982 | 0.204232 |
| 1983–2002 | 0.172196 |
| 2003–2022 | 0.313421 |
| 2023–2042 | 0.343670 |
| 2043–2062 | 0.256621 |
| 2063–2082 | 0.391364 |
| 2083–2102 | 0.291827 |
| 2103–2122 | 0.234613 |
| 2123–2142 | 0.334811 |
| 2143–2162 | 0.240013 |
| 2163–2182 | 0.244518 |
| 2183–2202 | 0.179107 |
| 2203–2204 | 0.051783 |

## Per-query JOB comparison

Speedups are final reported scores, rounded to three decimals. An em dash means
the outcome has no numerical speedup; it is not a zero or a retained default.
The last columns give 027's outcome and number of attempts passing both action
validation and plan constraints. A valid timed-out candidate still counts as
constraint-valid, although it supplies no speedup. See the linked
[025 results](../025-sft-astra-ceb-300/results.md) and
[026 results](../026-qwen-4b-job-epoch2-v7-5cand-1rollout/results.md) for the
earlier runs' outcome details.

| Query | 026 speedup | 025 speedup | 027 speedup | 027 outcome | 027 valid attempts / tried |
| --- | ---: | ---: | ---: | --- | ---: |
| job-01a | 7.207× | — | 7.660× | `measured` | 2/5 |
| job-01b | 108.730× | — | 109.596× | `measured` | 1/3 |
| job-01c | 0.561× | 10.089× | — | `no_valid_candidate` | 0/4 |
| job-01d | 1.000× | 122.465× | 1.000× | `kept_default` | 0/5 |
| job-02a | — | — | 1.000× | `kept_default` | 0/5 |
| job-02b | 1.000× | 1.000× | 1.000× | `kept_default` | 1/4 |
| job-02c | 1.000× | 23.955× | 1.004× | `measured` | 2/5 |
| job-02d | 1.000× | 1.638× | 1.000× | `kept_default` | 0/5 |
| job-03a | 0.947× | 0.712× | 1.000× | `kept_default` | 1/5 |
| job-03b | 1.000× | 1.000× | 1.534× | `measured` | 2/4 |
| job-03c | 1.000× | 1.000× | 1.000× | `default_duplicate` | 2/4 |
| job-04a | 0.967× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-04b | 1.000× | 1.186× | 1.024× | `measured` | 2/4 |
| job-04c | 0.179× | 0.198× | 1.000× | `kept_default` | 0/0 |
| job-05a | 0.594× | 1.000× | 1.000× | `kept_default` | 3/5 |
| job-05b | 0.998× | 1.007× | 1.000× | `kept_default` | 0/5 |
| job-05c | 1.000× | — | 1.000× | `kept_default` | 1/4 |
| job-06a | 1.000× | 0.107× | 1.000× | `kept_default` | 1/5 |
| job-06b | 2.369× | 1.000× | 1.000× | `kept_default` | 1/5 |
| job-06c | 1.024× | 1.178× | 1.000× | `kept_default` | 0/5 |
| job-06d | 2.508× | 1.000× | 2.842× | `measured` | 4/4 |
| job-06e | 1.000× | 0.145× | 1.000× | `kept_default` | 0/0 |
| job-06f | 1.000× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-07a | 0.993× | 4.857× | 1.000× | `kept_default` | 1/4 |
| job-07b | 1.737× | 1.000× | 0.865× | `measured` | 3/3 |
| job-07c | 1.005× | — | 1.040× | `measured` | 2/5 |
| job-08a | 1.000× | 0.958× | 1.000× | `kept_default` | 0/5 |
| job-08b | 1.000× | 1.000× | 1.057× | `measured` | 4/5 |
| job-08c | — | 0.704× | 1.000× | `kept_default` | 3/4 |
| job-08d | 1.000× | 0.952× | 1.000× | `kept_default` | 0/5 |
| job-09a | 1.000× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-09b | 0.989× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-09c | 0.987× | 0.193× | 1.049× | `measured` | 2/5 |
| job-09d | 1.000× | 1.411× | 1.400× | `measured` | 2/4 |
| job-10a | 1.000× | — | 1.000× | `kept_default` | 0/5 |
| job-10b | 1.543× | 1.000× | 1.935× | `measured` | 4/4 |
| job-10c | 1.000× | 1.098× | 1.000× | `kept_default` | 2/5 |
| job-11a | 0.987× | 1.078× | 1.000× | `kept_default` | 0/4 |
| job-11b | 0.993× | 1.228× | 1.111× | `measured` | 2/5 |
| job-11c | 2.812× | — | 1.000× | `kept_default` | 0/5 |
| job-11d | 0.989× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-12a | 1.000× | 0.999× | 1.000× | `kept_default` | 0/5 |
| job-12b | 1.000× | 1.005× | 1.000× | `kept_default` | 0/5 |
| job-12c | 0.943× | 0.816× | 1.000× | `kept_default` | 1/5 |
| job-13a | 1.000× | 1.000× | 1.214× | `measured` | 2/5 |
| job-13b | 2.293× | 1.000× | 1.000× | `kept_default` | 0/1 |
| job-13c | 1.000× | 3.500× | 1.000× | `kept_default` | 1/5 |
| job-13d | 1.000× | 1.000× | 0.298× | `measured` | 3/3 |
| job-14a | 1.000× | 1.000× | 1.000× | `kept_default` | 3/5 |
| job-14b | 1.000× | 1.147× | 1.000× | `default_duplicate` | 1/3 |
| job-14c | 1.000× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-15a | 0.977× | 1.000× | 0.875× | `measured` | 1/5 |
| job-15b | 1.000× | 1.000× | 1.029× | `measured` | 3/5 |
| job-15c | 1.000× | 1.066× | 1.000× | `kept_default` | 1/5 |
| job-15d | 1.000× | 0.961× | 1.000× | `kept_default` | 0/5 |
| job-16a | 1.000× | 1.000× | 1.034× | `measured` | 1/5 |
| job-16b | 1.000× | 1.000× | 1.000× | `kept_default` | 1/4 |
| job-16c | 1.000× | 1.001× | 1.000× | `kept_default` | 2/5 |
| job-16d | 0.997× | 1.000× | 1.000× | `kept_default` | 1/5 |
| job-17a | 1.003× | 1.000× | 1.306× | `measured` | 2/5 |
| job-17b | 0.994× | 0.916× | 0.897× | `measured` | 2/5 |
| job-17c | 0.987× | 1.000× | 2.752× | `measured` | 2/4 |
| job-17d | 1.000× | 1.000× | 1.814× | `measured` | 2/5 |
| job-17e | 1.000× | 1.000× | 1.000× | `kept_default` | 1/5 |
| job-17f | 1.000× | 2.554× | 1.000× | `kept_default` | 1/5 |
| job-18a | 1.000× | 1.000× | 5.645× | `measured` | 1/5 |
| job-18b | 1.000× | 1.010× | 1.000× | `kept_default` | 3/4 |
| job-18c | 1.000× | 1.000× | 1.000× | `kept_default` | 0/4 |
| job-19a | 0.707× | 0.083× | 1.000× | `kept_default` | 0/5 |
| job-19b | 1.000× | — | 1.000× | `kept_default` | 1/5 |
| job-19c | 1.000× | — | 1.000× | `kept_default` | 4/5 |
| job-19d | 1.000× | 0.962× | — | `timed_out` | 2/5 |
| job-20a | 1.000× | 1.000× | 1.261× | `measured` | 3/5 |
| job-20b | 1.000× | — | 1.153× | `measured` | 2/5 |
| job-20c | 1.000× | 2.166× | 1.000× | `kept_default` | 0/5 |
| job-21a | 1.078× | 1.000× | 1.000× | `kept_default` | 4/4 |
| job-21b | 1.000× | 2.601× | 1.000× | `kept_default` | 0/0 |
| job-21c | 0.233× | 1.756× | 1.720× | `measured` | 1/5 |
| job-22a | 1.000× | 0.322× | 1.000× | `kept_default` | 2/3 |
| job-22b | 1.000× | 1.000× | 0.999× | `measured` | 3/5 |
| job-22c | 1.000× | 1.481× | 1.000× | `kept_default` | 0/5 |
| job-22d | 1.000× | 1.368× | 1.000× | `kept_default` | 0/3 |
| job-23a | 1.000× | 1.013× | 1.000× | `kept_default` | 3/5 |
| job-23b | 1.001× | 1.000× | 1.000× | `kept_default` | 2/4 |
| job-23c | 1.000× | 0.334× | 1.000× | `kept_default` | 2/5 |
| job-24a | — | 1.013× | 2.379× | `measured` | 1/5 |
| job-24b | 2.014× | 1.980× | 2.458× | `measured` | 1/5 |
| job-25a | 1.000× | 1.000× | — | `no_valid_candidate` | 0/4 |
| job-25b | 1.000× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-25c | 1.007× | 1.000× | 0.416× | `measured` | 4/5 |
| job-26a | 1.000× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-26b | 1.000× | 1.000× | — | `no_valid_candidate` | 0/5 |
| job-26c | 4.178× | 4.860× | 4.338× | `measured` | 1/5 |
| job-27a | 1.000× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-27b | 1.000× | — | 1.000× | `kept_default` | 0/5 |
| job-27c | 1.000× | 1.966× | 1.000× | `kept_default` | 1/5 |
| job-28a | 1.000× | 0.505× | — | `selection_failed` | 5/5 |
| job-28b | 1.722× | 1.000× | 1.000× | `kept_default` | 1/5 |
| job-28c | 1.000× | 3.158× | 1.000× | `kept_default` | 0/5 |
| job-29a | 1.000× | 1.112× | 1.000× | `kept_default` | 0/0 |
| job-29b | 1.000× | 1.000× | 1.000× | `kept_default` | 1/5 |
| job-29c | 1.000× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-30a | — | 2.253× | 1.000× | `kept_default` | 0/5 |
| job-30b | 1.000× | 2.969× | 1.000× | `kept_default` | 2/5 |
| job-30c | 1.000× | 1.371× | 1.015× | `measured` | 1/5 |
| job-31a | 1.000× | 0.972× | 1.000× | `kept_default` | 1/5 |
| job-31b | 0.254× | 1.124× | 1.526× | `measured` | 2/5 |
| job-31c | 1.000× | — | 1.000× | `kept_default` | 1/5 |
| job-32a | 0.987× | 1.000× | — | `no_valid_candidate` | 0/5 |
| job-32b | 1.000× | 1.000× | 1.000× | `kept_default` | 2/5 |
| job-33a | — | 0.233× | 1.000× | `kept_default` | 0/5 |
| job-33b | 1.000× | 1.000× | 1.000× | `kept_default` | 0/5 |
| job-33c | 1.000× | 1.000× | 1.000× | `kept_default` | 2/2 |
