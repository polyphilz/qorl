# 033 — Three searches per JOB query with the final 032 policy

**Three searches expose substantially more useful plans from the same model.**
Selecting across them using preliminary execution feedback achieves **1.8054×
geometric-mean speedup over PostgreSQL**, with **44.67% less summed query
execution time**. There are **68 material improvements, three measured-neutral
results and 42 kept defaults across all 113 JOB queries**, with no measured
regressions in the selected set.

This best-of-three result is an **offline replay of a feedback-only selector**
over the completed searches. Each selected candidate already has a separate
final paired measurement in the saved evaluation. Final timings are used to
score the choice, never to make it. The native evaluation report instead
aggregates all 339 searches: **1.4048× for the model's choices** and **1.4355×
for the configured within-search `best_feedback` rule**.

There was **no additional training**. This is the same 032 step-600 adapter over
the merged 031 step-600 base, with three rollouts per query instead of one.
Run `000` completed **339/339 rollouts** and exited successfully. Evaluation ran
from **2026-09-13 23:43:17 UTC to 2026-09-14 01:27:08 UTC**, taking
**1h 43m 51s**.

## Results and metric definitions

| Metric | Model selection, all searches | Best feedback, within each search | Best feedback, across three searches |
| --- | ---: | ---: | ---: |
| Scored outcomes | 339/339 | 339/339 | 113/113 |
| Geometric-mean speedup | 1.4048× | **1.4355×** | **1.8054×** |
| Ratio of summed default/selected latencies | 1.2366× | 1.2949× | **1.8075×** |
| Reduction in summed latency versus default | 19.13% | 22.77% | **44.67%** |
| Measured candidate selections | 159 | 138 | 71 |
| Kept-default outcomes | 177 | 201 | 42 |
| Default-duplicate outcomes | 3 | 0 | 0 |
| Material improvements | 119 | 129 | **68** |
| Material regressions | 7 | 1 | **0** |
| Any measured regression below 1.0× | 17 | 3 | **0** |
| Selection failures / selected-candidate timeouts | 0 / 0 | 0 / 0 | 0 / 0 |

The first two columns give every query three equally weighted observations.
They describe single-search performance across the three repetitions; they do
not select the best repetition. The last column gives each query one outcome
after searching three times. All candidates in both feedback columns came from
the model's own trajectories.

A material change is outside `abs(log(speedup)) <= 0.05`, approximately
**0.9512×–1.0513×**. This is a descriptive neutral band, not a per-query
significance test. The selector's **1.05× preliminary threshold** is a separate
rule. Kept defaults and default duplicates contribute exactly 1.0×, and keeping
the default does not count as generating a valid plan.

Geometric means weight queries equally. The summed-latency metric weights the
longer queries more heavily: for best-of-three, selected medians sum to
**31.756 s**, compared with **57.398 s** for their corresponding default
medians, saving **25.642 s**. These are sums of individual query medians,
excluding model inference, inspection and candidate search. They are not an
end-to-end workload benchmark or the wall time needed to optimize the workload.
Each measured result uses its own final paired default; default outcomes use
their recorded initial default median on both sides of the sum.

## How best-of-three was computed

The within-search rule was enabled during evaluation. The analysis extends the
same rule across searches without changing its threshold:

1. Read the recorded candidate feedback from all three searches for a query.
2. Keep candidates that are selection-eligible, have no recorded timeout and
   have a finite numerical preliminary speedup of at least **1.05×**.
3. Select the largest preliminary speedup. Ties go to the earlier rollout
   index, then the earlier issued candidate.
4. If none qualifies, retain PostgreSQL's default. For accounting, use rollout
   index zero's initial default median as both default and selected latency.
5. Only after choosing, read that candidate's saved final paired score.

Preliminary speedup is the source search's initial default median divided by
the candidate's feedback execution time. It is not the final paired speedup.
Because the maximum across searches must also be the maximum within its source
search, **every pooled candidate winner matches that search's recorded
`feedback_selection` choice**. This was checked for every query, including all
three two-search subsets.

The selected candidate's final score uses one warmup pair and three measured
default/candidate pairs. Where the model had chosen another result, the native
evaluation already measured the rule's choice separately. No additional SQL
executions or model calls were made for the pooled analysis, and the original
model choices remain intact.

This supports a reproducible feedback-selected system result, but it is still
an offline replay. A production best-of-three path would run three searches,
select using their feedback, then measure or deploy the winner. This evaluation
measured the per-search choices before the offline pooling. Its final paired
timings were withheld from the selector, rather than obtained in a new
post-pooling confirmation run. Selecting the largest noisy preliminary ratio
can still favor an overestimate; the paired scoring checks that risk here.

### Benefit of the extra search budget

| Search budget / subset | GM speedup | Summed-latency reduction | Material wins /113 |
| --- | ---: | ---: | ---: |
| One search: all three repetitions aggregated | 1.4355× | 22.77% | 43 on average |
| Best of indices 0 and 1 | 1.6546× | 41.66% | 62 |
| Best of indices 0 and 2 | 1.6641× | 36.18% | 59 |
| Best of indices 1 and 2 | 1.6642× | 31.61% | 59 |
| Best of indices 0, 1 and 2 | **1.8054×** | **44.67%** | **68** |

Best-of-three improves geometric-mean speedup by **25.77% relative to the
single-search feedback aggregate**. All three two-search combinations improve
on the single-search result, although their absolute latency savings differ
considerably. These subsets share searches and are not independent confirmation
runs. The best-of-three candidate winners come from all three indices:
**27 from index zero, 26 from index one and 18 from index two**.

For a deliberately non-deployable upper bound, choose using the final scores
of the three rule-selected outcomes, also allowing default. This hindsight
oracle reaches **1.8089×**. Feedback selection matches its score on
**107/113 queries**, including the 42 defaults, and is only **0.19% below its
geometric mean**. The six misses are job-01c, job-07a, job-20a, job-28c,
job-31b and job-32a; the feedback-selected result is still a material win in
each case. This oracle covers the three scored within-search winners, not all
possible plans or every candidate proposed during the searches.

## Variation between searches

Each row below contains the same 113 queries and the same checkpoint.

| Rollout index | Model GM | Within-search rule GM | Rule latency reduction | Model wins | Rule wins | Valid / novel rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.4651× | 1.4817× | 31.82% | 43 | 46 | 101 / 101 |
| 1 | 1.4256× | 1.4713× | 24.19% | 43 | 47 | 109 / 104 |
| 2 | 1.3272× | 1.3569× | 12.16% | 33 | 36 | 101 / 98 |

The model's geometric mean spans **1.3272×–1.4651×**, and the rule's spans
**1.3569×–1.4817×**, without any intervening update to the weights. This is
direct evidence that a single JOB search per query is an unstable basis for
judging modest checkpoint differences. Three panels describe the observed
variation; their range is not a 95% confidence interval.

The useful plans are often different. Among **103 same-query pairs of searches
where both model choices were measured**, only **nine** selected the same
structural plan; only five had an identical full timing-reuse key. Among 288
pairs where both searches generated a valid plan, 61 shared any valid plan
structure. These comparisons distinguish changed proposals from merely
remeasuring the same plan, though they do not assign an exact fraction of
outcome variance to sampling versus database feedback.

| Number of searches with a material rule-selected win | Queries |
| --- | ---: |
| 0 of 3 | 45 |
| 1 of 3 | 24 |
| 2 of 3 | 27 |
| 3 of 3 | 17 |

Thus **68 queries have a win somewhere**, but only **17 win in every search**.
Repeated search increases the chance of finding a useful proposal. It does not
make every query consistently easy for the policy.

### Relationship to the preceding 032 evaluation

Index zero reuses 032's per-query model and measurement seeds. On the **112
queries scored in both evaluations**, model-selection GM changes from
**1.4132× to 1.4581×**, or **+3.18%**, despite identical weights and fixed
request settings. The geometric mean of new/old initial default latency ratios
is **0.9966**, so there is no comparable overall shift in database baseline
speed.

Reusing a seed does not make these identical-input reruns: **none of the 113
full first message payloads is byte-identical**, because the prompt includes
freshly measured database feedback and worker information. System prompts,
tool definitions, seeds, default plan fingerprints and measurement settings do
match. Four of the 34 queries measured in both index-zero runs select the same
structural plan. The comparison cannot isolate an inference-server determinism
issue, nor should its aggregate change be described as further learning.

The previously unfinished **job-30a** completes in all three new searches; its
index-zero result is **2.5055×**. This run has full model-score coverage.

## Which queries benefit?

These are within-search rule scores, followed by the feedback-selected pooled
score and the time it saves against its own paired default.

| Query | Index 0 | Index 1 | Index 2 | Pooled | Saved execution time |
| --- | ---: | ---: | ---: | ---: | ---: |
| job-01d | 1.000× | 90.160× | 1.000× | **90.160×** | 16.762 ms |
| job-02c | 1.000× | 1.000× | 23.898× | **23.898×** | 157.310 ms |
| job-12b | 49.627× | 35.327× | 35.598× | **49.627×** | 20.083 ms |
| job-16b | 1.000× | 2.349× | 1.000× | **2.349×** | **3,341.602 ms** |
| job-17a | 2.165× | 1.000× | 1.000× | **2.165×** | **2,057.429 ms** |
| job-17b | 15.365× | 0.987× | 1.723× | **15.365×** | **2,124.745 ms** |
| job-17e | 2.552× | 2.123× | 1.000× | **2.552×** | **2,268.083 ms** |
| job-17f | 2.528× | 0.904× | 1.193× | **2.528×** | **1,909.027 ms** |
| job-26c | 5.946× | 4.090× | 1.589× | **5.946×** | **1,227.167 ms** |

The 90× job-01d result contributes much more to the geometric mean than its
absolute saving suggests. Conversely, job-16b saves more than three seconds
with a less dramatic ratio. Both metrics are necessary to describe the
workload benefit.

Repeated search also has limits. **job-31a remains at default in all three
searches**, although 032's preceding single search found 7.70×. **job-17d**
reaches only 1.69× here, compared with the previous 13.15× result. Three
searches improve aggregate coverage; they do not recover every previously
observed strong plan.

### Is the gain just a few outliers?

Remove the queries with the largest relative pooled improvement over their
own three-search geometric mean, from both sides:

| Retained queries | Single-search rule GM | Pooled GM | Relative increase |
| --- | ---: | ---: | ---: |
| All 113 | 1.4355× | **1.8054×** | **25.77%** |
| Remove job-01d | 1.4210× | **1.7434×** | **22.69%** |
| Remove the largest three gains | 1.4024× | **1.6691×** | **19.01%** |
| Remove the largest five gains | 1.3775× | **1.6108×** | **16.94%** |

The largest three are job-01d, job-02c and job-17b; the next two are job-11c
and job-06d. This intentionally removes favorable observations and is not a
significance test. It shows that the extra-search benefit is broader than the
concentrated incremental 031-to-032 gain described in [032's report](../032-rl-ceb-fresh-600steps/results.md).

## Candidate generation and remaining harness errors

| Diagnostic | All three searches |
| --- | ---: |
| Candidate attempts | 1,549 |
| Action-valid attempts | 1,079 /1,549 (69.7%) |
| Attempts passing action and plan constraints | 855 /1,549 (55.2%) |
| Invalid actions | 470 |
| Action-valid attempts failing plan constraints | 224 |
| Object actions containing `leading` | 917 |
| Those attempts passing action and plan constraints | 534 /917 (58.2%) |
| Malformed string actions | 202 |
| Candidate attempts reporting execution timeout | 32 |
| Rollouts generating a valid plan | 311 /339 |
| Rollouts generating a novel plan | 303 /339 |
| Distinct novel query/plan pairs across all searches | 648 |

Across the three searches, **every query generates at least one valid
candidate**, and **112/113 generate a novel candidate**; job-29b is the
exception for novelty. Ninety queries generate valid plans in all three
searches, 18 in two, and five in only one. Default outcomes therefore do not
necessarily indicate an inability to form a plan: many searches correctly
decline their available alternatives.

The per-index action-and-plan pass rates are **53.9%, 57.1% and 54.6%**,
consistent with the previous 032 result of 57.1%. This repeats the improved
proposal behavior seen after RL, while making its remaining failure rate
clear. The `leading` statistic measures success of the entire submitted action
and all its plan constraints; it is not an isolated test of Leading-tree
correctness. Attempts within a query and trajectory are correlated, so these
1,549 attempts should not be treated as independent observations when claiming
statistical confidence about policy improvement.

None of the 202 string actions parses as a JSON object; 201 mention Leading.
These are still substantive formatting errors rather than a simple case of
valid JSON wrapped in a string. Other recurring failures include duplicated
or disconnected Leading relations, missing tree fields, and requested scans
or indexes not appearing in the plan. The 32 timeout records include one reuse
of a previous timed-out result, corresponding to **31 fresh timed-out SQL
executions**. None becomes a selected-candidate timeout.

### Final selection is good, but not solved

The model selects the highest numerical preliminary feedback in **160/162
candidate selections**. That count alone misses the decision between a
candidate and default. It also selects **five visibly slower candidates** and
keeps default in **nine searches with a material feedback opportunity**.

The rule changes **43/339 outcomes**: 33 to default and ten to a candidate.
It reuses the other 296 results exactly. Its aggregate GM gain over the model
is **2.19%**; raw regressions fall from 17 to three, and material regressions
from seven to one.

Examples explain why the rule still matters:

- **job-02d, index 2:** the model selects a candidate with approximately
  0.143× feedback and scores **0.141×**. The rule retains default.
- **job-33c, index 1:** the model selects an approximately 0.161× feedback
  candidate and scores **0.161×**. The rule retains default.
- **job-17a, index 0:** the model scores **0.939×**, despite another candidate
  showing **2.153×** feedback. The rule selects that candidate and scores
  **2.165×**.
- **job-17f, index 1:** feedback predicts **1.125×**, but paired scoring gives
  **0.904×**. This is the rule's remaining material regression. Another search
  supplies the **2.528×** candidate selected by best-of-three.

There is one rejected final selection and five calls to unavailable tools,
but all searches terminate normally. The fixes prevent the earlier terminal
loops from dominating this evaluation; they do not eliminate every malformed
proposal or mistaken choice. Zero regressions in the pooled sample is an
observed result, not a guarantee from the feedback rule.

## Search cost, configuration and audit

The run takes **103.8 minutes**, versus **35.0 minutes** for 032's single-search
evaluation, approximately three times the wall time on the same evaluation
setup. It makes **2,543 model requests**, with **35,852,748 prompt tokens** and
**551,319 completion tokens**, including **246,261 reasoning tokens**. Cached
token usage is not reported. These are local inference counts, not a hosted API
bill or a count of unique prompt content.

The native execution counters account for **4,183 SQL executions**: 1,356 for
initial defaults, 1,475 for candidate feedback, 1,272 for the model's final
paired measurements, and 80 extra executions for the ten alternative
rule-selected candidates. Reused measurements are counted once. Planning-only
EXPLAINs and catalog inspections are outside these execution counts. The
evaluation includes both model and rule outcomes for comparison; an optimized
deployment need not measure an unused model selection separately.

Key settings match 032: seed 42; five candidate attempts per search; thinking
enabled; temperature 1.0, top-p 1.0 and top-k 20; **49,152-token context and
8,192-token reply limit**; one RTX 3090 serving GPU with four sequences; four
PostgreSQL workers with four physical cores and 8 GiB each; and **2 GiB shared
buffers**. Standalone evaluation holds a worker for the whole rollout.
Initial defaults use one warmup plus three measurements; fresh candidate
feedback uses one warmup plus one measurement; final scoring uses one warmup
pair plus three measured pairs. See [the config](config.toml) and
[experiment README](README.md) for the pinned paths and weight hashes.

The audit checked all 339 rollout records against the native schema and
reconstructed the complete native summary. It verified query/index coverage,
derived model and measurement seeds, interface v7 and fingerprint v4, baseline
and final medians, final speedup calculations, the within-search selection rule,
reuse of unchanged results, additional measurements for changed choices, and
the cross-search selection/score joins. **No audit discrepancies were found.**

All saved model requests and responses identify **`qorl-adapter`**. The launch
manifest identifies the same merged 031 base and 032 final adapter as the
previous evaluation. The largest sent prompt is **39,432 tokens**, and the
largest completion is **796 tokens**. There are no recorded inference failures,
truncated responses, context-limit stops or turn-limit stops. The evaluation
status is completed and its launcher exit code is zero.

## Interpretation for the write-up

**The model contains more useful search behavior than one rollout reliably
reveals.** Additional searches improve coverage and expose different fast plans,
while the existing preliminary feedback is effective at choosing among them.
The principal gain in 033 comes from spending more inference and database search
budget on the fixed final RL policy.

A precise result statement is:

> On all 113 JOB queries, the final RL policy achieves 1.435× geometric-mean
> execution speedup with one feedback-selected search, averaged over three
> repetitions. An offline feedback-selected best-of-three replay achieves
> 1.805× and reduces summed query execution time by 44.7%, using approximately
> three times the search budget. The model's own single-search selection
> achieves 1.405×.

The 1.805× ratio is not an 80.5% latency reduction. The measured summed-latency
reduction is 44.7%, and both measures exclude search cost. Likewise, comparing
this three-search result directly with a one-search SFT baseline would mix
policy improvement with additional search compute. A matched three-search
evaluation of the SFT checkpoint would be needed to isolate the RL contribution
at this budget. It was not part of 033.

The three repetitions strengthen the case that search variance matters, but
do not establish an exact confidence interval or a universal no-regression
guarantee. JOB queries also share templates, and JOB has been inspected during
development; this is a repeatedly used held-out evaluation workload, not a
fresh blind confirmation benchmark.

## Evidence and per-query results

Native evidence is under
`outputs/033-qwen-4b-job-rl032-5cand-3rollouts/000/evaluation/test/000/`, including
`evaluation.json` and the 339 `rollouts/<task>/<index>.json` files. It remains
unchanged. The offline audit and extracted evidence are retained locally under
`outputs/analysis/033-job-three-rollouts/`.

[Per-query results](results-per-query.csv) record all three model and rule
scores, the pooled source index and candidate, preliminary selection ratio,
scoring medians, final pooled score and restricted oracle score. For defaults,
the two scoring medians are the same initial baseline; no extra final execution
is implied. These values make the pooled result and its distinction from the
native 339-search aggregate explicit.
