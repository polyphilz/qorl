# 031 — Fresh CEB RL, 600 updates

**The final checkpoint improved both plan generation and selection in this JOB
evaluation.** The model's own choices achieved **1.3521× geometric-mean speedup
across all 113 queries**, with valid candidates on **99/113**. The starting SFT
checkpoint, 027 step 2204, achieved 1.1553× on 107 scored queries and valid
candidates on 71/113. On those **same 107 scored queries**, the comparison is
**1.1553× → 1.3447×**, a **16.39% relative increase**.

The separate `best_feedback` selection rule achieved **1.3695×** on the same
031 trajectories. Most of the observed improvement therefore exists with the
model making its own final decision. The rule provides a smaller additional
gain and is reported separately throughout.

Run `000` completed all **600 optimizer updates**, with approximately **22h 9m
in the native training loop**. The launcher exited successfully on **September
12, 2026, at 23:06:36 UTC**. JOB evaluation ran from **23:20:13 to 23:54:58 UTC**
(19:20–19:54 America/New_York), taking **34m 45s**. All 113 rollouts completed,
with no recorded inference/infrastructure failures or context-limit stops.

## Training setup and lineage

| Setting | Value |
| --- | --- |
| Starting policy | Merged **027 step-2204 SFT** checkpoint |
| New trainable weights | Fresh RL LoRA, rank 16, alpha 32, dropout 0 |
| Optimizer | Fresh AdamW state; constant learning rate **1e-5**; weight decay 0 |
| Batch / group | Nominal 16 episodes, eight rollouts per query group |
| Updates / policy lag | 600 updates; maximum lag **two** updates |
| Concurrency | 20 episodes in flight, 20 inference requests/sequences |
| Algorithm | Anchored GRPO: `tau=0.05`, `c=0.1`, `d=0.02`, `t=0.1`, `min_peers=2` |
| Agent | Five candidate attempts; thinking enabled; at most 64 model turns |
| Token limits | 49,152 context; 8,192 maximum reply |
| Hardware | Two H100 80 GB GPUs for training/inference; PostgreSQL on FLOPper |
| Database workers | Four workers, each with four physical cores and 8 GiB; 2 GiB shared buffers |
| RL worker ownership | A worker is held for each complete measurement phase and released during inference |
| Checkpoints | Every 20 updates; all 30 scheduled checkpoints retained through step 600 |
| In-run validation | None; step 600 was the prescribed final checkpoint |

031 starts independently from the merged SFT policy; it does not continue 030's
RL adapter. The original pretrained model in this lineage is
`empero-ai/Qwen3.8-4B-Distill`, revision
`c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`. The Prime-RL fork pin is
`d4636a532d668ffbe727eb9e7730e7ea24ca384c`.

The training set contains **600 distinct fresh CEB queries**, sixty from each
of the ten familiar training templates. Selection excluded **1,513 prior CEB
query IDs and SQL hashes**, covering earlier SFT, RL, smoke and validation
selections. The frozen selection audits record zero overlap. Twenty additional
CEB validation queries were reserved; JOB's 113 queries stayed outside training.
See [README.md](README.md), [selection-audit.json](selection-audit.json) and
[legacy-exclusion-audit.json](legacy-exclusion-audit.json).

## Actual training consumption

| Quantity | Result |
| --- | ---: |
| Completed optimizer updates | 600 |
| Returned episode records | 11,682 |
| Failed episodes | 5 |
| Queries represented in returned episodes | 600/600 |
| Groups with returned episodes | 1,507 |
| Episodes with non-discarded anchored credit | 10,944 |
| Positive / negative / zero advantages in those episodes | 2,577 / 5,519 / 2,848 |
| Episodes shipped to the trainer | **7,116** |
| Positive / negative advantages among shipped episodes | **2,234 / 4,882** |
| Query groups represented in shipped episodes | 1,050 |
| Distinct queries represented in shipped episodes | 540/600 |
| Shipped episodes per update | 1–16; mean 11.86 |

The finite task pool was revisited. Saved returns span three groups for 309
queries, two groups for 289 queries and one group for two queries. A group can
have incomplete returns, and returned episodes need not reach the trainer.
These counts describe the actual records, rather than an assumed
`600 updates × 16 episodes` training set. Native zero-signal filtering,
policy-lag filtering and prefetched work separate collection from training.
No shipped episode had a discarded anchored-credit annotation, and every
shipped episode had a nonzero advantage.

The completed report lists every optimizer step from 1 through 600. Runtime
checks found no nonfinite training/optimizer metrics or fatal infrastructure
error. The final launcher exited 0 and acknowledged a clean service drain.
The five failed episodes are distinct from the 11,677 returned final outcomes:
6,501 kept default, 4,707 measured, 222 default duplicates, 77 no-valid-candidate,
116 selection failures and 54 timeouts. Those outcomes combine changing
policies and training queries; they are not the final checkpoint's test score.

### Training signal over the run

These windows group episodes by the **policy version at rollout start**. Rates
use episodes with non-discarded anchored credit, including those subsequently
filtered before shipment to the trainer.

| Starting policy versions | Credited episodes | Positive advantage | Zero advantage | Positive measured quality |
| --- | ---: | ---: | ---: | ---: |
| 0–99 | 1,816 | 18.2% | 35.2% | 21.3% |
| 100–199 | 1,872 | 21.5% | 29.1% | 24.3% |
| 200–299 | 1,800 | 23.9% | 21.3% | 28.2% |
| 300–399 | 1,752 | 23.8% | 25.4% | 28.1% |
| 400–499 | 1,832 | 25.8% | 23.4% | 31.3% |
| 500–599 | 1,872 | 28.0% | 21.7% | 33.0% |

Positive-quality episodes became more common and the zero-advantage fraction
fell overall. This is consistent with useful policy change. The windows contain
different query mixtures and repeated visits, so they are not a fixed-cohort
validation curve. They also cannot isolate group size from learning rate,
training duration or the other changes relative to 030. The console's unused
scalar reward hook and the RL surrogate loss should not be read as SFT loss or
held-out quality metrics.

## Evaluation conditions and mechanical audit

Evaluation used **all 113 JOB queries, one rollout per query, seed 42**, five
candidate attempts, thinking enabled, temperature 1.0, top-p 1.0 and top-k 20.
The context and reply limits matched training. vLLM 0.28.0 served the exported
**step-600 RL adapter over its exact merged 027 base** on one RTX 3090, with four
serving sequences. All **836 recorded requests and responses identify
`qorl-adapter`**. The exported adapter contains 256 tensors and approximately
40.5 MiB of weights; its checksum and base identity matched the export manifest.

The RL database service had stopped before evaluation. Standalone evaluation
held one worker for each complete rollout, using `001-pgconf-2gb-sb` and
`002-poolconf-4x8`. Each initial default received **one warmup plus three measured
executions**. Fresh candidate feedback received **one warmup plus one measured
execution**. Final comparisons received **one warmup pair plus three measured
default/candidate pairs**, eight executions in total.

The offline audit checked all 113 saved records and found **no discrepancies**:

- Every record validates against the native evaluation schema. Rebuilding the
  entire evaluation summary exactly reproduces the saved summary.
- Independent arithmetic reproduces both geometric means and summed latencies.
  Every baseline and every measured model/rule result has the expected sample
  counts, medians and paired speedup.
- All records use **interface v7 and plan fingerprint v4**. Relative to 027,
  all 113 system prompts, initial tool-definition hashes, per-query model and
  measurement seeds, default plan hashes, timing-reuse keys, planner settings
  and measurement configurations match. Worker assignment differs on 85 queries;
  the only resource-limit difference is the assigned CPU set, with identical
  memory and physical-core limits.
- Replaying the selection rule from the recorded model-visible candidate history
  reproduces all 113 rule choices. Original model choices remain intact. All 99
  reused results match exactly; changed selections have the expected derived
  measurement seed and additional execution accounting.
- There are no model-call failures, truncated replies, context-budget endings
  or turn-limit endings. The largest request contained **40,606 prompt tokens**;
  the largest reply contained **791 completion tokens**. The evaluation exited 0
  and removed its owned database containers.

## JOB results

027 is the starting SFT policy. 030 is the earlier 120-update RL experiment,
which used seed 43 for evaluation; 027 and 031 use seed 42. The rule column
reuses 031's proposals and changes only final selection and any required scoring.

| Metric | 027: SFT step 2204 | 030: RL step 120 | 031: model selection | 031: best feedback |
| --- | ---: | ---: | ---: | ---: |
| Queries with a valid candidate /113 | 71 (62.8%) | 71 (62.8%) | **99 (87.6%)** | Same proposals |
| Queries with a novel candidate /113 | 69 | 67 | **96** | Same proposals |
| Distinct novel query/plan pairs | 113 | 114 | **182** | Same proposals |
| Candidate attempts | 507 | 513 | **516** | Same proposals |
| Attempts passing action validation | 246 (48.5%) | 240 (46.8%) | **326 (63.2%)** | Same proposals |
| Attempts passing action and plan constraints | 138 (27.2%) | 134 (26.1%) | **232 (45.0%)** | Same proposals |
| Scored outcomes /113 | 107 | 106 | **113** | **113** |
| Geometric-mean speedup, scored subset | 1.1553× | 1.1353× | **1.3521×** | **1.3695×** |
| Ratio of summed default/selected latencies | 1.0567× | 0.9906× | **1.1601×** | **1.1845×** |
| Measured selections | 33 | 37 | **40** | **39** |
| Kept-default outcomes | 72 | 67 | **69** | **74** |
| Default-duplicate outcomes | 2 | 2 | **4** | **0** |
| No-valid-candidate outcomes | 4 | 5 | **0** | **0** |
| Selection failures | 1 | 0 | **0** | **0** |
| Selected-candidate timeouts | 1 | 2 | **0** | **0** |
| Raw measured regressions below 1.0× | 6 | 10 | **3** | **1** |
| Measured improvements outside the 0.05 log band | 20 | 21 | **34** | **36** |
| Measured regressions outside the same band | 5 | 7 | **0** | **0** |

The descriptive band is `abs(log(speedup)) <= 0.05`, approximately
0.9512×–1.0513×. It is not a per-query significance test or a guarantee about
false rewards. The model's three raw regressions are **job-08a at 0.9660×,
job-08b at 0.9862× and job-17d at 0.9924×**. All lie inside that band. The rule
defaults on the first two and retains the third.

For model selection, summed default and selected query medians are **57.727 s
and 49.759 s**: **7.968 s saved, or 13.80%**. For the rule they are **57.641 s
and 48.663 s**: **8.978 s saved, or 15.58%**. Defaults and duplicates contribute
their initial default median to both sums and exactly 1.0× to the geometric
mean. Changed rule selections use fresh paired timings, so the two panels have
slightly different default sums. These sums exclude inference and candidate
search; they do not describe end-to-end optimization latency.

Evaluation consumed **11,311,146 prompt tokens and 182,715 completion tokens**,
including **79,194 reasoning tokens**. The selection rule required no additional
model calls. Its six newly measured choices added **48 database executions** to
the model path's **1,170**, for **1,218 actual executions**. The rule's performance
panel includes reused execution histories; it must not be added wholesale to
the model panel when counting actual work.

## Comparison with the starting SFT checkpoint

The most useful comparison uses the **107 queries scored by both 027 and 031**:

| Matched metric | 027 | 031 model selection |
| --- | ---: | ---: |
| Geometric-mean speedup | 1.1553× | **1.3447×** |
| Ratio of summed default/selected latencies | 1.0567× | **1.1705×** |
| Reduction in summed query latency versus default | 5.36% | **14.56%** |

The matched geometric-mean increase is **16.39%**. All six previously unscored
queries also receive scores in 031, but they are excluded from this matched
comparison. Across the full 113, **36 queries gained a valid candidate and eight
lost one**, producing the net increase of 28.

This comparison has no evaluation-seed or harness-version change. The geometric
mean of per-query initial-default latency ratios, 031 divided by 027, is
**0.9979**, and the median ratio is **1.0004**. There is no comparable overall
shift in baseline database speed that explains the policy gain.

Some wins have large ratios on short queries. **job-01d** changed from default
selection to **102.08×**, saving about **17 ms**; **job-01b** achieved **124.50×**,
also saving about **17 ms**. Larger absolute savings came from **job-17a
(1.94×, 1.913 s saved)** and **job-06d (14.63×, 1.469 s saved)**. Both geometric
means and summed latencies are needed to describe this workload.

Removing the queries with the largest relative improvements from **both sides
of the matched comparison** gives:

| Matched sensitivity check | 027 GM | 031 GM | Relative increase |
| --- | ---: | ---: | ---: |
| Remove the largest improvement, job-01d | 1.1569× | 1.2908× | **11.58%** |
| Remove the three largest improvements | 1.1486× | 1.2441× | **8.32%** |
| Remove the five largest improvements | 1.1416× | 1.2082× | **5.83%** |

The three removed queries are job-01d, job-07a and job-06d; removing five also
excludes job-24b and job-30a. The gain survives these descriptive checks, although
the largest wins clearly influence its size.

The new policy also lost some substantial SFT wins: **job-18a fell from 5.64×
to default, job-26c from 4.34× to default, and job-17c from 2.75× to default**.
In this evaluation, the first two searches offered only slower eligible plans;
the third produced no eligible plan. The mechanical selector also defaults on
all three. Recovering those wins requires better proposals, rather than a
different final choice among the recorded candidates.

Relative to 030, the 106 commonly scored queries improve from **1.1353× to
1.3650×**. That comparison additionally changes evaluation seed. Learning rate,
group size, nominal batch size, training data, concurrency and update count all
changed between the two RL experiments, so this run does not isolate which
change was responsible.

## Selection behavior and the rule ablation

| Model decision evidence | 027 | 031 |
| --- | ---: | ---: |
| Default before attempting a candidate | 4 | **3** |
| Default after search with no eligible candidate | 34 | **11** |
| Default after search with an eligible alternative | 34 | **55** |
| Default chosen when the best numerical feedback was below the 0.05 log band | 23/28 | **31/31** |
| Final candidate choices with materially negative feedback | 5 | **0** |
| Candidate choices matching the highest numerical feedback, including ties | 34/35 | **41/44** |
| Queries offering a feedback win above the band | 22 | **39** |

The similar total number of defaults, 72 versus 69, hides an important change.
Far fewer searches ended without an eligible candidate, more searches offered
a useful improvement, and the model avoided every clearly slower candidate set.
This is evidence of better proposals and safer selection together. A scored
default still does not mean the policy produced a valid plan: 14 of 031's
69 defaults had no eligible candidate, including the three early defaults.

The evaluation-only rule selects the highest numerical preliminary ratio among
eligible, non-timeout candidates **if that ratio is at least 1.05×**; otherwise
it selects default. Ties go to the earliest candidate. Its threshold is the
configured 1.05 ratio, distinct from the descriptive 0.05 log band above.
The decision uses feedback available before final timing and has no access to
the subsequent paired result. RL continued to train the model's own selection.

The rule agreed with the model on **99/113 queries**. Of the 14 changes, eight
selected default, including four cosmetic changes from a default-duplicate
candidate. Six selected a different candidate and required new final pairs.
Examples show both its value and its limits:

| Query | Model speedup | Rule speedup | Interpretation |
| --- | ---: | ---: | --- |
| job-06f | 1.000× | **2.180×** | Model defaulted despite a good recorded candidate; rule saves about 844 ms |
| job-22c | 1.155× | **2.069×** | Model selected a weaker candidate; rule's measured result saves about 236 ms more |
| job-14b | 1.000× | **1.060×** | Modest missed opportunity recovered |
| job-32b | 1.000× | **1.064×** | Modest missed opportunity recovered |
| job-18c | **1.067×** | 1.000× | Preliminary ratio was only 1.035×, so the threshold rejects a final measured win |
| job-17e | 1.001× | 1.003× | Better preliminary feedback did not become a meaningful final gain |

The aggregate increase from **1.3521× to 1.3695× is 1.29% relative**. The rule
helps, but the model has already corrected much of the severe selection problem
seen in earlier runs. This rule was not executed on 027's trajectories, so
1.3695× versus 027's model-selected score is not a comparison that holds the
selector constant.

## Remaining conformance problems

Invalid actions remain common even after the improvement. Of **516 attempts**,
**190 failed action validation** and another **94 passed action validation but
failed plan constraints**. The remaining **232 passed both**, up from 138 in
027. Four candidate attempts timed out during search; none was finally selected.

The largest single formatting defect is still an action supplied as a string:
**82 attempts**, down from 170 in 027. Eighty of those strings mention `Leading`;
none of the 82 strings parses directly into a JSON object. These are not cases
that a simple extra JSON decode would repair. Other failures include malformed
or disconnected join trees and requested scans/indexes that the resulting plan
does not satisfy. Error-message counts can overlap within one rejected attempt.

Among object-form `Leading` attempts, **95/211 passed constraints**, compared
with 83/197 in 027. The model used `Leading` in twelve measured wins outside
the band. There were **two rejected final selections and five unavailable-tool
calls**, but all affected conversations recovered and terminated normally.
The harness now supports successful completion reliably; proposal conformance
still leaves substantial room for improvement.

## Interpretation and limits

**This is a positive observed RL result on JOB:** more valid and novel plans,
more useful candidate sets, fewer visibly bad selections, higher matched-query
speedup and more total execution time saved. Reporting the model-selected
result as primary preserves that finding independently of the rule ablation.
Step 600 and its full training evidence should remain the reference checkpoint
for this experiment.

The evidence does not establish that 600 updates is optimal, that group eight
is superior to four, or that learning rate alone explains the result. There was
no held-out validation curve or checkpoint sweep. Training windows support the
direction of change but cannot replace those controls.

Each JOB query has one stochastic rollout. Identical seeds do not eliminate
sampling variability across changed policies, and the measurement band is not
a statistical confidence interval. The 027 baseline used an SFT adapter over
the original base, while 031 uses an RL adapter over the BF16-merged SFT base;
there is no merged-only step-zero control. JOB was excluded from training but
has been inspected repeatedly during project development. These limitations
belong with any write-up claim about generalization or causal attribution.

If further evaluation time is available, the most informative confirmation is
repeated, matched evaluation of the starting merged SFT policy and this final
checkpoint, with model and rule results reported separately. The current
result already supports a much stronger conclusion than 030's near-flat run;
it does not by itself justify assuming further RL will continue improving.

## Evidence

Canonical artifacts are under `outputs/031-rl-ceb-fresh-600steps/`:

- `000/training/report.json`: completed steps, checkpoints, outcomes and
  per-episode anchored-credit/shipment evidence.
- `000/training/checkpoints/step_600/adapter/`: verified adapter and manifest.
  Adapter SHA-256:
  `96d195b4187645769621307552b84f9a8c830602b7a01f751e1fc5f3aa14715f`.
- `000/evaluation/test/000/evaluation.json`: complete evaluation summary.
- `000/evaluation/test/000/rollouts/<task>/000.json`: original conversations,
  candidate histories, model results and rule-comparison measurements.
- `deployment/eval-launch-000.json` and `deployment/eval-exit-000.json`:
  checkpoint identity, resolved local evaluation settings and successful exit.

The local offline audit is saved under
`outputs/analysis/031-final-checkpoint-job/`: `audit.py`, `helpers.py`,
`analysis.json` and `per-task.csv`. It validates the records, rebuilds the native
summary, independently recomputes timings, replays the rule and compares the
downloaded 027/030 records. Audit result: **zero discrepancies**.

Prior analyses: [027 SFT results](../027-sft-astra-ceb-300-epoch2/results.md) and
[030 RL results](../030-rl-ceb-fresh-120steps/results.md).
