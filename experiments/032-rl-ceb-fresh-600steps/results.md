# 032 — Continuing RL for another 600 updates on fresh CEB queries

**032 improved on 031's aggregate JOB result, with stronger candidate generation
and some substantial new query optimizations.** On the **112 queries scored by
both models**, geometric-mean speedup increased from **1.3410× to 1.4132×**, a
**5.39% relative increase**. Reduction in summed query latency versus PostgreSQL's
default increased from **13.02% to 22.53%** on that same subset.

The improvement is uneven. Some earlier wins disappeared, and removing the
three largest relative improvements reverses the incremental geometric-mean
gain. One trajectory, **job-30a**, exhausted its context before selecting a
candidate and is unscored in the model panel. The separate `best_feedback` rule
rescued that query and achieved **1.4178× across all 113**; on the same 112 scored
queries, the rule adds no aggregate improvement over the model's choices.

Run `000` completed all **600 additional optimizer updates**, approximately
**21 hours of training**, and exited successfully on **September 13, 2026, at
22:19:10 UTC**. JOB evaluation ran from **22:26:41 to 23:01:43 UTC**, taking
**35m 02s**. This evaluates **032 step 600 over the merged 031 step-600 base**:
the lineage has now received **1,200 RL updates** after the SFT checkpoint.

## JOB results

Both evaluations use all 113 JOB queries, one rollout per query, seed 42, five
candidate attempts, thinking enabled, and interface v7. The rule column uses
032's existing trajectories and changes only final selection and any required
measurement. It does not generate additional candidates.

| Metric | 031: model selection | 032: model selection | 032: best feedback |
| --- | ---: | ---: | ---: |
| Queries with a valid candidate /113 | 99 (87.6%) | **101 (89.4%)** | Same proposals |
| Queries with a novel candidate /113 | 96 | **100** | Same proposals |
| Distinct novel query/plan pairs | 182 | **244** | Same proposals |
| Candidate attempts | 516 | 489 | Same proposals |
| Attempts passing action validation | 326 (63.2%) | **360 (73.6%)** | Same proposals |
| Attempts passing action and plan constraints | 232 (45.0%) | **279 (57.1%)** | Same proposals |
| Scored outcomes /113 | 113 | **112** | 113 |
| Geometric-mean speedup, scored subset | 1.3521× | **1.4132×** | **1.4178×** |
| Ratio of summed default/selected latencies | 1.1601× | **1.2909×** | **1.2972×** |
| Measured selections | 40 | 49 | 41 |
| Kept-default outcomes | 69 | 63 | 72 |
| Default-duplicate outcomes | 4 | 0 | 0 |
| No-valid-candidate outcomes | 0 | 0 | 0 |
| Selection failures | 0 | **1** | 0 |
| Selected-candidate timeouts | 0 | 0 | 0 |
| Raw measured regressions below 1.0× | 3 | 5 | 2 |
| Measured improvements outside the 0.05 log band | 34 | **38** | 39 |
| Measured regressions outside the same band | 0 | **2** | 1 |

The descriptive neutral band is `abs(log(speedup)) <= 0.05`, approximately
0.9512×–1.0513×. It is not a per-query significance test. Kept defaults contribute
exactly 1.0×; they do not count as having produced a valid candidate. A selection
failure contributes no speedup, so coverage must accompany the geometric mean.

### Comparison on identical scored queries

Exclude job-30a from **both** model panels to make the primary comparison:

| Matched metric, 112 queries | 031 | 032 |
| --- | ---: | ---: |
| Geometric-mean speedup | 1.3410× | **1.4132×** |
| Ratio of summed default/selected latencies | 1.1498× | **1.2909×** |
| Summed default medians | 56.948 s | 57.130 s |
| Summed selected medians | 49.531 s | **44.256 s** |
| Time saved against each run's default measurements | 7.417 s | **12.874 s** |
| Reduction versus default | 13.02% | **22.53%** |

The selected-latency sum is **10.65% lower** than 031's on these same queries.
These are sums of per-query medians, excluding inference, inspection and
candidate search; they are not end-to-end optimization latency. Each evaluation
measures its own defaults. The geometric mean of the 113 initial-default
latency ratios, 032 divided by 031, is **1.0019**, with median **1.0012**. There
is no comparable overall shift in baseline database speed explaining the gain.

For a separate sensitivity check, assigning the failed model rollout a
hypothetical default score of 1.0× would give **1.4089× over 113**, still above
031's 1.3521×. This is an explicitly imputed result, not the observed model score
and not what the native report records.

Against the original **027 step-2204 SFT** policy, the **106 queries scored by
both 027 and 032** improve from **1.1569× to 1.4379×**, or **24.29% relatively**.
Their reduction in summed latency versus default improves from **5.45% to
23.71%**. That comparison describes the cumulative two-run RL lineage; the
112-query comparison above isolates the observed increment after 031 as closely
as these evaluations allow.

## What improved: candidate generation

The clearest broad change is **more useful proposals per attempt**. Action
validation improves by **10.4 percentage points**, and passing both action and
plan constraints improves by **12.1 points**, despite fewer attempts overall.
Distinct novel query/plan pairs increase **34.1%**, from 182 to 244.

| Candidate-generation diagnostic | 031 | 032 |
| --- | ---: | ---: |
| Object actions containing `leading` | 211 | 292 |
| Those attempts passing action and plan constraints | 95 (45.0%) | **179 (61.3%)** |
| Measured improvements using a selected `leading` action | 12 | **24** |
| Actions emitted as strings | 82 | **53** |
| String actions that parse as a JSON object | 0 | 0 |
| Action-valid attempts with unsatisfied plan constraints | 94 | **81** |
| Candidate attempts reporting an execution timeout | 4 | **16** |
| Rejected final selections | 2 | 1 |
| Calls to unavailable tools | 5 | 4 |

The improvement in `Leading` construction matters: 032 both attempts more join
orders and successfully realizes more of them. Its new wins also use index,
parallelism and planner-setting changes, so the improvement is not confined to
join ordering.

Remaining conformance errors are material: **129/489 actions are invalid**, and
another **81** pass action validation but fail plan constraints. Recorded
diagnostics include string actions, malformed or duplicate relations in Leading
trees, requested bitmap scans becoming sequential scans, unexpected indexes,
and requested parallel scans not appearing. Diagnostic counts can overlap
within an attempt and should not be summed as independent failures. The 53
string actions are not simply valid JSON objects wrapped in a string.

Overall query coverage grows only slightly: **11 queries gain a valid candidate
and nine lose one**. The main improvement at this stage is proposal success and
diversity within searches. Search timeouts increase, but neither the model nor
the rule selects a timed-out candidate in the final evaluation.

## Which queries account for the change?

These examples show both the magnitude of new wins and their absolute value.
The final column is the change in time saved against each evaluation's own
default, rather than a ratio of ratios.

| Query | 031 speedup | 032 speedup | Change in time saved |
| --- | ---: | ---: | ---: |
| job-12b | 1.000× | **45.632×** | +21.825 ms |
| job-17d | 0.992× | **13.148×** | **+2,197.246 ms** |
| job-31a | 1.000× | **7.698×** | +358.776 ms |
| job-20a | 1.000× | **6.319×** | +612.934 ms |
| job-20b | 2.119× | **6.650×** | +199.969 ms |
| job-16b | 1.000× | **1.544×** | **+2,156.880 ms** |
| job-26c | 1.000× | **2.406×** | +863.515 ms |
| job-17a | **1.940×** | 1.023× | **−1,824.889 ms** |
| job-28c | **3.123×** | 1.000× | −238.533 ms |
| job-01d | **102.083×** | 1.000× | −16.982 ms |
| job-01c | **9.751×** | 1.000× | −16.040 ms |

On **job-12b**, an `IndexScan` on `mi_idx` reduces the final median from
22.314 ms to 0.489 ms. That is a very large multiplicative win on a short query.
On **job-17d**, a realized Leading order with scan/parallel hints reduces the
median from 2,357.156 ms to 179.282 ms. **job-16b** saves more than two seconds
through a Leading order, join/scan hints and parallel cost settings, despite its
less spectacular 1.54× ratio. Both geometric means and absolute sums are needed
to describe the result.

The lost wins are primarily missing proposals in this sampled search, not a
failure to choose an already-found fast candidate. **job-01d** produces no
eligible candidate; **job-01c** and **job-28c** offer only slower eligible
options and correctly default. **job-17a** stops after two attempts with a best
feedback ratio of only 1.016×; the stronger combination found by 031 is absent.
These examples establish observed inconsistency, not permanent forgetting from
a single rollout per query.

Across the 112 matched queries, **27 improve and 25 worsen** by more than the
0.05 log band relative to their previous speedup; 60 stay within it. Twenty
previously neutral queries become wins against PostgreSQL, while fifteen prior
wins become neutral. This is substantial movement in both directions.

### Sensitivity to the largest improvements

Remove the queries with the largest **relative gains over 031** from both sides
of the matched comparison:

| Retained comparison | 031 GM | 032 GM | Relative change |
| --- | ---: | ---: | ---: |
| All 112 matched queries | 1.3410× | **1.4132×** | **+5.39%** |
| Remove job-12b | 1.3445× | **1.3697×** | **+1.87%** |
| Remove the three largest improvements | **1.3520×** | 1.3205× | **−2.33%** |
| Remove the five largest improvements | **1.3501×** | 1.2818× | **−5.06%** |

The top three are job-12b, job-17d and job-31a; the next two are job-20a and
job-20b. This deliberately removes favorable observations and is not a
significance test. It shows that **the incremental aggregate gain is
concentrated**, unlike the broader 031-versus-SFT gain documented in
[031's results](../031-rl-ceb-fresh-600steps/results.md).

## Selection behavior and the mechanical rule

The model selects the highest recorded numerical feedback ratio in **49/49
selections**, compared with 41/44 in 031. **Thirty-nine of those 49 selections
return to an earlier candidate**, using results accumulated through the search.
All **27** searches whose best numerical feedback is below the lower log-band
boundary end by keeping default. There are **41** queries with a visible
feedback improvement above the upper boundary,
versus 39 in 031.

Of the 63 model defaults, **two occur before any candidate**, **ten follow a
search with no eligible candidate**, and **51 follow a search with eligible
candidates**. These are explicit model decisions. Only **job-06f** defaults
despite having a visible material feedback improvement; final remeasurement
actually favors that default decision.

The `best_feedback` rule chooses the eligible, non-timeout candidate with the
highest finite numerical preliminary ratio if it is at least **1.05×**, otherwise
default; ties go to the earliest candidate. Its 1.05 threshold is slightly
different from the descriptive `exp(0.05)` boundary. The recorded model choice
is preserved, and the rule uses no unseen final timing when choosing.

| Selection comparison | Model | Rule |
| --- | ---: | ---: |
| 032 GM on the same 112 model-scored queries | **1.4132×** | 1.4123× |
| Reduction versus default on those 112 | **22.53%** | 22.47% |
| 032 GM on each panel's scored set | 1.4132× /112 | **1.4178× /113** |
| Rule-only comparison across all 113: 031 → 032 | 1.3695× | **1.4178×** |

The rule's cross-run increase is **3.52%** with identical 113-query coverage.
Within 032, however, its matched-panel change is **−0.07%**, effectively no gain
in this single evaluation. Comparing 1.4178× directly with 1.4132× would mix
selection effects with the additional scored query.

The rule agrees with the model on **101/113** trajectories. Ten changes replace
a selected candidate with default. The other two measure a candidate for
job-06f and rescue the unselected candidate on job-30a. These two measurements
add **16 PostgreSQL executions** and no model calls.

Two residual model regressions lie outside the log band:

- **job-15b: 0.8823×**, despite preliminary feedback of **1.2151×**. Feedback
  measured 9.042 ms; the final candidate median is 12.486 ms against an
  11.017 ms default. The rule picks the same candidate. Ranking feedback
  correctly cannot eliminate discrepancies between preliminary and final timing.
- **job-16d: 0.9441×**, with preliminary feedback already at **0.9539×**. The
  model picks its best candidate even though default is available; the rule
  defaults and avoids the regression. The other three raw model regressions
  are inside the neutral band.

On **job-06f**, the rule follows 1.1370× feedback and obtains **0.9583×** in final
pairs; the model's default was better. Taken together, these outcomes leave
little evidence that a more elaborate final-ranking mechanism is the main
remaining opportunity. Proposal coverage, the choice to keep default, and
feedback reliability matter more in this run.

### job-30a: context budget prevents final selection

This trajectory makes seven model calls and produces **four eligible
candidates**, with best preliminary feedback **2.4309×**. It has no rejected
finish calls. After another full-plan inspection and candidate feedback, its
next rendered prompt reaches **43,486 tokens**. The client reserves the full
**8,192-token** maximum reply: `43,486 + 8,192 > 49,152`, so it stops before
sending another completion request. The model never calls `finish`.

The rule can still select **candidate-04** from the saved history. Fresh final
pairs give **2.1867×**, saving **432.680 ms**. This recovers a usable plan from
an unfinished search; it does not make the model trajectory a successful
termination. The record has neither a truncated model reply nor an OOM.

## Training setup and actual consumption

032 continues the learned policy from **031 step 600**, merged into its exact
merged **027 step-2204 SFT base** with `qorl model merge`. A **fresh LoRA and fresh
AdamW state** are then trained over that complete model. It is not a restart
from the pre-RL SFT policy or a continuation of 031's optimizer state. Source
checkpoints remain intact; merge provenance is in
[base-merge-manifest.json](base-merge-manifest.json).

| Setting | Value |
| --- | --- |
| New LoRA | Rank 16, alpha 32, dropout 0 |
| Optimizer | AdamW; constant learning rate 1e-5; weight decay 0; max gradient norm 1 |
| Batch / group | Nominal 16 episodes; eight rollouts per query group |
| Updates / maximum policy lag | 600 additional updates; lag two |
| Concurrency | 20 episodes in flight and 20 inference sequences |
| Anchored GRPO | `tau=0.05`, `c=0.1`, `d=0.02`, `t=0.1`, `min_peers=2` |
| Agent limits | Five attempts; 64 model turns; thinking enabled |
| Context / maximum reply | 49,152 / 8,192 tokens |
| Sampling | Temperature 1.0; top-p 1.0; top-k 20; seed 42 |
| Hardware | Two H100 80 GB GPUs for training/inference; PostgreSQL on the benchmark host |
| PostgreSQL pool | Four workers, four physical cores and 8 GiB each; 2 GiB shared buffers |
| Worker ownership | Measurement-phase leases during RL; whole-rollout leases during JOB evaluation |
| Checkpoints / validation | Every 20 updates, all 30 retained; no in-run validation |

The training pool has **600 fresh CEB queries**: six from template 7a and
66 from each of the other nine familiar training templates. The 7a allocation
changes because only six unseen queries remain after exclusions. Twenty fresh
CEB validation queries are reserved, ten each from 5a and 8a.

The frozen [selection audit](selection-audit.json) records **zero ID or SQL
overlap with 2,133 excluded prior CEB queries**, unique SQL within each split,
disjoint SQL and join topologies across splits, and deterministic selection
replay. JOB's frozen 113-query selection is identical to 031's. All other
training settings match 031; see [config.toml](config.toml) and
[README.md](README.md).

| Training quantity | 031 | 032 |
| --- | ---: | ---: |
| Completed optimizer updates | 600 | 600 |
| Returned episode records | 11,682 | 11,589 |
| Failed episodes | 5 | 3 |
| Queries represented in returned episodes | 600 | 600 |
| Groups with returned episodes | 1,507 | 1,494 |
| Episodes with non-discarded anchored credit | 10,944 | 10,936 |
| Positive / negative / zero advantages in those episodes | 2,577 / 5,519 / 2,848 | **3,448 / 5,093 / 2,395** |
| Episodes shipped to the trainer | 7,116 | **7,589** |
| Positive / negative advantages among shipped episodes | 2,234 / 4,882 | **3,051 / 4,538** |
| Groups represented in shipped episodes | 1,050 | 1,058 |
| Distinct queries represented in shipped episodes | 540 | 540 |
| Shipped episodes per update | 1–16; mean 11.86 | 1–16; mean **12.65** |

Returned work spans three groups for **297 queries**, two groups for **300**,
and one group for **three**. Groups may return partially, and native signal,
policy-lag and prefetch filtering separate collection from shipment. Every
shipped episode has nonzero, non-discarded anchored credit, and its assigned
advantage matches that credit. **Sixty sampled queries contribute no shipped
episode**; the task-list size should not be presented as 600 distinct queries
that all produced gradient updates.

The 11,586 non-failure training outcomes are **6,308 measured**, **4,926 kept
default**, **151 default duplicates**, **nine no-valid-candidate**, **183
selection failures**, and **nine selected timeouts**. These aggregate changing
policies on training queries and are not final-checkpoint test results.

### Training signal over the run

Windows use the **policy version when each rollout starts**. Rates use episodes
with non-discarded anchored credit, including some subsequently not shipped.

| Starting policy versions | Credited episodes | Positive advantage | Zero advantage | Positive measured quality |
| --- | ---: | ---: | ---: | ---: |
| 0–99 | 1,832 | 28.5% | 24.1% | 36.1% |
| 100–199 | 1,872 | 28.5% | 24.0% | 35.1% |
| 200–299 | 1,848 | 33.0% | 19.3% | 41.7% |
| 300–399 | 1,808 | 29.4% | 25.2% | 38.8% |
| 400–499 | 1,816 | 35.5% | 19.1% | 49.7% |
| 500–599 | 1,760 | 34.5% | 19.6% | 45.5% |

Positive-quality episodes become more common overall, and the zero-advantage
fraction falls, though neither trend is monotonic. Query mixtures and repeated
visits change across windows, so these are supporting training diagnostics,
not a held-out learning curve. The differing template allocation also prevents
interpreting 031-versus-032 training rates as a fixed-task comparison.

All 600 trainer metric records are present and finite. Mean token entropy is
approximately **0.81–0.83** across the six 100-update windows, with no aggregate
entropy collapse. Maximum recorded gradient norm is **0.1621**; trainer peak
memory is **22.36 GiB**. Inference logs record policy loads **0 through 600**.
The RL surrogate loss is not an SFT imitation loss or an independent quality
metric. The launcher exits 0, and the final report records an acknowledged
service drain without error.

## Evaluation integrity and evidence

The preflight records no competing QORL processes or worker containers before
evaluation. vLLM **0.28.0** serves the final exported **032 step-600 adapter** over
the exact **merged 031 step-600 model**, on one RTX 3090 with four serving
sequences. All **822 saved model requests and responses identify
`qorl-adapter`**. Deployment evidence verifies the base and adapter identities:

- Merged base SHA-256:
  `5de0270bf5329657364c9f86c344527f6a0af5aac989195fd9e8a6aa2df2619d`.
- Exported adapter SHA-256:
  `7a29b32f23c5efb8b663494beb616a2e6b29281f07854b3f9a2ecae4cfd9adbc`.
- The adapter has **256 tensors**, approximately **40.5 MiB** of weights, and
  nonzero trained LoRA B weights. The original pretrained model revision is
  `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`; the Prime-RL fork pin is
  `d4636a532d668ffbe727eb9e7730e7ea24ca384c`.

The offline audit reports **no discrepancies** in the checked invariants:

- All 113 records validate against the native schema, and rebuilding the entire
  evaluation summary reproduces the saved summary exactly. Independent
  arithmetic reproduces geometric means, latency sums and paired speedups.
- Initial defaults have **one warmup and three measurements**. Fresh completed
  candidate feedback has **one warmup and one measurement**. Each measured final
  model or rule outcome has **one warmup pair and three measured pairs**.
- All records use interface **v7** and plan fingerprint **v4**. All 113 initial
  system prompts, tool hashes, per-query model and measurement seeds, default
  full/structural plan hashes, timing-reuse keys, planner settings and
  measurement configurations match 031. Worker assignments differ on 85 queries;
  resource limits differ only in CPU assignment, not memory or core counts.
- Replaying the rule from recorded candidate histories reproduces all 113
  choices. All **101 reused outcomes** match exactly; changed selections retain
  the model decision and use the expected derived measurement seed.
- There are **no recorded evaluation model-call/infrastructure failures or
  truncated replies**. The one context-budget stop is documented above. The
  largest sent request has **38,751 prompt tokens**, and the largest completion
  has **934 tokens**. The evaluation launcher exits 0 and removes its owned
  database containers.

Evaluation consumes **11,418,460 prompt tokens and 181,938 completion tokens**,
including **79,851 reasoning tokens**. Recorded query-execution counts are
**452 initial default executions**, **497 candidate-feedback executions**, and
**392 final model-path executions**: **1,341**, plus the rule's **16**, for
**1,357 total**.
The candidate accounting includes 241 fresh completed feedback protocols,
15 fresh timeout executions and reused results. The rule panel contains reused
execution histories and must not be added wholesale to the model panel. These
counts exclude planning-only EXPLAINs and catalog inspection.

Primary evidence is under
`outputs/032-rl-ceb-fresh-600steps/000/`: `training/report.json`, the step-600
export manifest, `evaluation/test/000/evaluation.json`, all 113 saved rollouts,
and deployment launch/exit records. The local offline audit is preserved under
`outputs/analysis/032-final-checkpoint-job/`, including `audit.py`, `details.py`,
`analysis.json`, `details.json` and `per-task.csv`.

## Interpretation and limits

**Further RL produced a promising additional result:** proposals are more often
valid, more distinct plans appear, and the aggregate query-time savings improve.
Final selection is already closely aligned with observed feedback, so most of
the useful policy work now lies in finding good plans consistently. A default
or near-neutral finish can be the correct choice among a weak sampled set even
when a previous checkpoint found a much better plan for the same query.

The evidence supports retaining **032 as the current preferred aggregate
result**, while preserving 031 as a serious comparison. It does not establish
that another 600 updates will improve reliably, that 032 dominates on individual
queries, or that the changed 7a allocation caused any particular gain or loss.
The most informative next check is repeated evaluation of both checkpoints with
matched additional seeds, or the reserved CEB validation cohort, before treating
the incremental 5.39% as a stable effect.

These are **single-rollout, single-seed** evaluations at temperature 1.0. Matching
seeds and measurement settings improves comparability but does not quantify
generation or timing variance. There is no separate step-zero evaluation of the
newly merged 031 base, so this comparison cannot isolate merge rounding from
subsequent RL. Finally, JOB remains excluded from training, but its results have
informed iterative experiment choices; it is not an untouched final blind test.
