# Results: additional Astra SFT and JOB evaluation, run 002

Training completed successfully on Lambda on **September 10, 2026, at 22:34
America/New_York** (September 11, 02:34 UTC). The resulting adapter's JOB
evaluation completed on FLOPper on **September 11 at 00:00:54 America/New_York**.

**The rollout result is mixed.** Compared with the epoch-2 baseline in 026,
the new adapter found more improvements and used a broader range of
interventions, but produced valid candidates on fewer queries: **77/113 versus
85/113**. Its reported geometric-mean speedup was **1.103× over 101 scored
outcomes**, versus **1.075× over 108**. It is not an unqualified checkpoint
upgrade; validity and choosing default when appropriate remain weaknesses.

## Training outcome

**One epoch completed: 1,102 optimizer updates on 293 training conversations.**
Held-out validation loss improved from **0.362505 to 0.336020**, a **7.3% reduction**.
Both measurements use the same 19 held-out validation conversations: the initial
measurement evaluates the loaded epoch-2 weights, and the final measurement
evaluates the weights after the new training epoch.

The launcher exited with status 0, the final native checkpoint and exported
adapter exist, and the exported tensor checksum matches its manifest. All 1,102
updates have recorded losses, with no nonfinite losses or recorded NaN counts.
Both Lambda GPUs were idle when checked after completion.

The validation improvement measures prediction of held-out teacher
conversations. As the JOB results below show, that improvement did not translate
into uniformly better autonomous behavior.

## What was trained

| Setting | Value |
| --- | --- |
| Starting weights | Experiment 020's epoch-2 adapter, step 764 |
| Base | `empero-ai/Qwen3.8-4B-Distill`, revision `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e` |
| Initialization | Existing LoRA weights loaded; fresh optimizer state and progress |
| New training | One epoch; 1,102 updates; batch size 1 |
| LoRA | Rank 16, alpha 32, dropout 0 |
| Learning rate | Constant `1e-4` |
| Context / packed row length | 49,152 tokens |
| Hardware | One H100 80 GB on Lambda; second GPU unused |
| Training data | 293 conversations; 2,180 supervised requests; 1,102 packed rows |
| Validation data | 19 conversations; 131 supervised requests; 73 packed rows |
| Supervised tokens | 376,847 training; 24,459 validation |

The six early `keep_default` conversations were excluded before preparation, as
recorded in [README.md](README.md#training-exclusions). Of the remaining 294,
one exceeded the context limit, leaving 293. Preparation skipped 197 malformed
or action-invalid training requests while retaining their messages as context.
No additional outcome-based filtering was applied.

The trainer explicitly logged loading the initial adapter from this run's
`training/configs/initial_adapter`. The final step number is **1102 within this
new training run**; it does not replace the source checkpoint's step numbering.

## Training behavior and runtime

| Measurement | Result |
| --- | ---: |
| Initial held-out validation loss | 0.362505 |
| Final held-out validation loss | 0.336020 |
| Mean training loss, updates 1–100 | 0.370049 |
| Mean training loss, updates 1003–1102 | 0.235897 |
| Mean update time excluding the final validation-bearing step | 6.65 seconds |
| Launch through export/exit | Approximately 2h 8m |
| Trainer-reported peak GPU memory | 21.5 GiB |

Training-window means are ordinary arithmetic averages of per-update losses.
Different rows contain different targets and amounts of supervised text, so
these means describe the observed run rather than a fixed-batch learning curve.
The held-out before/after comparison is the more useful evidence of improvement.
The last update's very small loss alone is not informative: it supervised only
35 tokens. Its 143-second duration included final validation.

## Final adapter and evaluation provenance

On Lambda, the exported adapter is at:

```text
/lambda/nfs/qorl/projects/qorl/outputs/025-sft-astra-ceb-300/002/training/checkpoints/step_1102/adapter
```

`adapter_model.safetensors` is **42,500,760 bytes (40.5 MiB)**. Its computed
SHA-256 matches the export manifest:

```text
5ab99dece40b5aaecda0aff4e34c4ed0c5d4e1a50740b224df8e2d82480bafad
```

The completed JOB evaluation used the same v7 settings as
[experiment 026](../026-qwen-4b-job-epoch2-v7-5cand-1rollout/results.md): all 113
JOB queries, one rollout each, seed 42, five candidate attempts, thinking enabled,
49,152-token context, and an 8,192-token maximum reply. PostgreSQL and worker-pool
settings, three initial default measurements, one candidate feedback
measurement, and three final measured pairs also match.

The checkpoint, exported adapter, and run metadata have also been transferred
to `/home/rohan/projects/qorl/outputs/025-sft-astra-ceb-300/002` on FLOPper.
The native checkpoint, exported tensor checksum, LoRA settings, and local base
passed the evaluation entrypoint's verification. Experiment/run inputs agree,
and JOB selections and evaluation settings match 026. The source epoch-2
adapter's checksum is unchanged. See the
[evaluation command](README.md#evaluate-the-trained-adapter-on-job).

The completed serving report identifies the new step-1102 adapter over the
same pinned base. All **803 model responses identify `qorl-adapter`**, and all
113 traces record interface v7 and fingerprint v4. Compared query by query with
026, system prompts, tool definitions, model seeds, measurement seeds, and
default timing-reuse keys match for all 113 queries. This comparison is not
confounded by another harness revision or different default plans.

Evidence lives under the Lambda run directory above: `training/report.json`,
`training/monitors/file/metrics.jsonl`, `training/logs/attempt_1/trainer.log`,
`training-launch.exit`, and the checkpoint's adapter manifest.

## JOB evaluation: comparison with the starting adapter

| Metric | 026: epoch-2 baseline | 025: after the additional SFT epoch |
| --- | ---: | ---: |
| Queries with a valid candidate /113 | 85 (75.2%) | **77 (68.1%)** |
| Queries with a novel candidate /113 | 76 | **72** |
| Distinct novel query/plan pairs | 115 | **123** |
| Candidate attempts | 547 | **521** |
| Attempts passing action and plan constraints | 189 (34.6%) | **148 (28.4%)** |
| Scored outcomes /113 | 108 | **101** |
| Geometric-mean speedup, scored subset | 1.075× | **1.103×** |
| Ratio of summed scored default/candidate latencies | 1.036× | **1.047×** |
| Improvements outside the 0.05 log-space band | 12 | **29** |
| Regressions outside the same band | 8 | **13** |
| No-valid-candidate outcomes | 4 | **12** |
| Selection failures / context-budget endings | 1 / 1 | **0 / 0** |
| Selected-candidate timeouts | 0 | **0** |
| Model replies | 915 | **803** |
| Prompt tokens | 11,756,852 | **9,448,730** |
| Completion tokens | 165,262 | **191,726** |
| Reasoning tokens, included in completion tokens | 93,243 | **78,583** |
| Wall time | 32m 19s | **32m 35s** |

The final outcomes were **54 measured candidates, 43 kept defaults, four
default timing-reuse duplicates, and 12 no-valid-candidate outcomes**. The first
three categories produce 101 scored outcomes; the 12 failures are excluded from
the speedup aggregate. Forty-seven outcomes contribute exactly 1.0×.

Among the 54 measured outcomes, 35 were faster and 19 slower. With the descriptive
band `abs(log(speedup)) <= 0.05`, these become **29 improvements, 13 regressions,
and 12 near-neutral measurements**. This band does not establish statistical
significance. There were no recorded infrastructure or model-call failures.

Summed scored default and selected latencies were **53.697 s and 51.290 s**,
respectively: 2.407 seconds saved, or 4.5%. These are sums of query medians,
using the initial baseline on both sides for defaults/duplicates. They exclude
inference and search overhead.

Because the scored subsets differ, also compare the **97 queries scored in both
runs**: their geometric means are **1.119× for 025 versus 1.001× for 026**.
This supports a performance improvement within that shared subset, but does
not erase failures on excluded queries. In particular, 025 failed to produce an
eligible candidate on job-01a, job-01b, and job-11c, which had achieved 7.207×,
108.730×, and 2.812× in 026.

The aggregate gain remains concentrated. Excluding 025's largest win gives
**1.053×**; excluding its three largest wins gives **0.996×**. One sampled rollout
per query, changing candidate trajectories, and fresh timings are insufficient
to establish a statistically reliable overall checkpoint ranking.

### More varied interventions, including some useful Leading trees

The new policy shifted away from predominantly scan interventions. Counts below
refer to recorded object-form actions containing each field; fields can overlap
within an action, and string-wrapped actions are excluded from this breakdown.

| Action field | 026 attempts | 025 attempts |
| --- | ---: | ---: |
| Scans | 360 | 145 |
| Join constraints | 208 | 291 |
| Row corrections | 13 | 166 |
| Parallelism | 31 | 244 |
| Planner settings | 53 | 156 |
| Leading tree | 142 | 161 |

Of the Leading-containing attempts, **41/161 passed action and plan constraints**,
versus **18/142** previously. Selected actions containing Leading exceeded the
positive log-space band on **job-13c (3.500×), job-09d (1.411×), and job-15c
(1.066×)**; there were none in 026. These actions also contain other constraints,
so their gains cannot be attributed to the join tree alone.

Fewer duplicate plans accompanied this broader search: structural duplicates
fell **74 → 25**, and timing-reuse duplicates fell **66 → 9**, across all attempts.
That explains how distinct novel plans increased despite lower valid-query
coverage and fewer valid attempts.

Selected examples use final paired medians, in milliseconds:

| Query | Default | Selected | Speedup | Selected intervention |
| --- | ---: | ---: | ---: | --- |
| job-01d | 17.390 | 0.142 | **122.465×** | Row correction on `it,mi_idx`; lower random-page cost |
| job-02c | 167.973 | 7.012 | **23.955×** | Hash join, row correction, parallelism |
| job-01c | 18.120 | 1.796 | **10.089×** | Index scan on `mi_idx` |
| job-26c | 1,466.910 | 301.861 | **4.860×** | Two row corrections |
| job-07a | 227.837 | 46.911 | **4.857×** | Parallelism on `pi` |
| job-13c | 173.904 | 49.687 | **3.500×** | Leading, hash join, row corrections, parallelism |
| job-17f | 3,238.569 | 1,267.850 | **2.554×** | Hash join on `k,mk` |
| job-09d | 1,233.310 | 873.982 | **1.411×** | Leading tree and join constraints |
| job-22a | 165.381 | 513.836 | **0.322×** | Leading, join, scans, row correction |
| job-04c | 41.304 | 209.096 | **0.198×** | Row correction and parallelism |
| job-09c | 140.021 | 723.831 | **0.193×** | Bitmap/index scans |
| job-06e | 7.685 | 52.908 | **0.145×** | Hash join, row corrections, parallelism |
| job-06a | 6.224 | 57.913 | **0.107×** | Hash join and row correction |
| job-19a | 105.055 | 1,264.639 | **0.083×** | Leading tree and hash-join constraints |

The large job-01d result is consistent across its three pairs: default
`17.505, 17.321, 17.390` ms; candidate `0.142, 0.142, 0.150` ms. It is a large
ratio on a small query, saving approximately 17 ms. Larger absolute savings
include job-17f (approximately 1.97 s) and job-26c (1.17 s).

### Choosing default remains a specific failure

Of 58 selected candidates, **56 tied for the fastest available candidate
feedback**. One exception was a negligible 0.026 ms difference on job-13c;
the other was job-19a, where the selected candidate was slower than another
eligible candidate as well as much slower than the default. These counts
include sole eligible candidates and ties.

More importantly, **16 selected candidates already looked worse than the
initial default by more than 0.05 in log space**, versus nine in 026.
**All 13 final regressions outside that band were among these 16.** For example,
job-19a displayed 1,240.841 ms candidate feedback against a 107.075 ms default,
then selected that candidate. The `finish("default")` option was present.
These large regressions were visible before the final measurement; they do not
require a timing-noise explanation.

Of the 43 defaults, one was chosen before attempting a candidate and 42 after
search. Those 42 comprise **19 searches with an eligible candidate and 23 with
none**. There was only one apparent missed improvement among defaults:
job-17e at 1.053× preliminary feedback, close to the descriptive band. It was
not finally paired, so it is not a verified missed win.

The reasoning text is not consistently grounded in the observed numbers.
For example, the job-01d terminal explanation calls its 0.142 ms candidate
"142ms", although the selected ID is correct. On job-19a it describes a
candidate as promising despite the displayed slowdown. These are useful
counterexamples to treating fluent reasoning as evidence of correct decisions;
they do not isolate reasoning-summary distillation as the cause.

### Cleaner termination, weaker action validity

There were **zero unavailable `evaluate_candidate` calls**, down from 32 in 026.
Only five unavailable-tool calls remained, all using invented tool names.
All 113 rollouts terminated through a model terminal call, with no context,
output-token, or model-turn exhaustion. Thus the earlier terminal loops remain
fixed. Five exploratory candidate execution timeouts occurred, but none of
those candidates was selected as the final outcome.

However, **275/521 attempts were action-invalid**, up from 217/547. Another
98 passed action validation but failed plan constraints; 148 passed both.
Action-validity rate fell **60.3% → 47.2%**, and the rate passing both checks
fell **34.6% → 28.4%**. String-wrapped `action` errors increased **72 → 93**.
Disconnected-tree/relation errors declined **87 → 57**, while unknown-field
errors increased **21 → 26**. These error categories are not exhaustive.

The 12 no-valid-candidate queries were:
`job-01a`, `job-01b`, `job-02a`, `job-05c`, `job-07c`, `job-10a`, `job-11c`,
`job-19b`, `job-19c`, `job-20b`, `job-27b`, and `job-31c`.
Every one tried to finish with an ineligible candidate ID; v7 ended the
conversation as `no_valid_candidate`. These account for 12 of the 16 recorded
selection rejections; the remaining four were corrected in other rollouts.

Five of these failed rollouts finished before using all five candidate attempts.
For example, job-05c stopped after **one rejected candidate**, with four attempts
remaining and `evaluate_candidate` still offered in the actual model request.
Its explanation claimed tool limitations that did not prevent further attempts.
This is premature stopping by the policy, not a missing-tool restriction in
the harness.

Validity changed on many individual queries: **29 lost a valid candidate that
026 had produced, while 21 gained one**. The net loss of eight therefore hides
a substantial change in search behavior.

### Interpretation and next decision

The additional SFT produced **broader exploration and more measured wins,
alongside worse action reliability and more avoidable final regressions**.
It improved some Leading usage and learned to respect exhausted attempt
budgets, but it did not satisfy a simple "better harness conformance" criterion.
The 7.3% validation-loss reduction should not be used alone to promote this
checkpoint.

Retain both adapters and keep 026 as the comparison baseline. The clearest
remaining training targets are grounded, valid PlanActions; continuing after
rejection when budget remains; and comparing the best eligible candidate with
the default before finishing. This run gives a reason to retain 025 as a more
varied search policy, but does not by itself justify another unfiltered SFT
epoch or establish an overall improvement over the starting adapter.

Evaluation report on FLOPper:
`/home/rohan/projects/qorl/outputs/025-sft-astra-ceb-300/002/evaluation/test/000/evaluation.json`.
Per-query evidence is under that evaluation directory at
`rollouts/<task-id>/000.json`. Report SHA-256:
`b42115ce79aabd67208938466a1141288607a3611ef23d35e3e0b76eb646bd49`.
Counts, aggregate speedups, and summed latencies were independently reconciled
against all per-query records in 025 and 026. JOB has been used repeatedly
during iteration and should be described accordingly in the write-up.

## Mean training loss in 20-update windows

The final row contains only two updates. These are unweighted per-update means,
matching the earlier check-ins.

| Updates | Mean training loss |
| --- | ---: |
| 1–20 | 0.574533 |
| 21–40 | 0.298582 |
| 41–60 | 0.300555 |
| 61–80 | 0.346591 |
| 81–100 | 0.329986 |
| 101–120 | 0.274839 |
| 121–140 | 0.392750 |
| 141–160 | 0.232154 |
| 161–180 | 0.282917 |
| 181–200 | 0.166009 |
| 201–220 | 0.400689 |
| 221–240 | 0.390578 |
| 241–260 | 0.166097 |
| 261–280 | 0.251561 |
| 281–300 | 0.443388 |
| 301–320 | 0.211648 |
| 321–340 | 0.287217 |
| 341–360 | 0.318189 |
| 361–380 | 0.181288 |
| 381–400 | 0.099459 |
| 401–420 | 0.290238 |
| 421–440 | 0.226535 |
| 441–460 | 0.227357 |
| 461–480 | 0.241677 |
| 481–500 | 0.293284 |
| 501–520 | 0.402555 |
| 521–540 | 0.297583 |
| 541–560 | 0.182963 |
| 561–580 | 0.476928 |
| 581–600 | 0.155071 |
| 601–620 | 0.320009 |
| 621–640 | 0.323364 |
| 641–660 | 0.276442 |
| 661–680 | 0.317257 |
| 681–700 | 0.270249 |
| 701–720 | 0.318320 |
| 721–740 | 0.479577 |
| 741–760 | 0.506734 |
| 761–780 | 0.265677 |
| 781–800 | 0.181382 |
| 801–820 | 0.427440 |
| 821–840 | 0.319988 |
| 841–860 | 0.326488 |
| 861–880 | 0.312434 |
| 881–900 | 0.451025 |
| 901–920 | 0.194848 |
| 921–940 | 0.205402 |
| 941–960 | 0.355760 |
| 961–980 | 0.187912 |
| 981–1000 | 0.211163 |
| 1001–1020 | 0.255956 |
| 1021–1040 | 0.362756 |
| 1041–1060 | 0.131941 |
| 1061–1080 | 0.245760 |
| 1081–1100 | 0.234828 |
| 1101–1102 | 0.187891 |

## Analysis

Added September 11, 2026, from the per-query records of this run
(`outputs/025-sft-astra-ceb-300/002/evaluation/test/000/rollouts/<task-id>/000.json`),
the 026 baseline (`outputs/026-qwen-4b-job-epoch2-v7-5cand-1rollout/000/evaluation/test/000`),
the Astra generation attempts behind the training data
(`outputs/025-sft-astra-ceb-300/001/generation/attempts/`), and Astra's own JOB
run (`outputs/011-astra-med-test-5cand-1rollout/000/evaluation/test/000`), all
on FLOPper.

Definitions used throughout: an *eligible* candidate is one the harness marked
`selection_eligible` in the candidate history shown to the model. That includes
candidates whose timing was reused from the default or an earlier candidate, and
candidates that timed out. Its *feedback ratio* is the
`preliminary_ratio_to_initial_default` from that history: the initial default
median divided by the candidate's feedback execution time, or exactly 1.0 for a
default-plan duplicate. Timed-out candidates have no ratio. The *best* candidate
in a rollout is the eligible candidate with the highest ratio. The band is
`abs(log(ratio)) <= 0.05`, as elsewhere in this document.

The mixed result has a single explanation. The additional demonstrations
improved what teacher-state SFT teaches directly, which actions to try, and did
little for decisions in states the teacher rarely visits, which it reaches only
indirectly. The validity drop is the price of the more ambitious actions.

### The selection failure is covariate shift, not a data-quantity problem

The state each policy is in when it finishes, classified by the best eligible
candidate in the candidate history it was shown. The teacher column covers the
294 conversations actually trained on; the six excluded conversations all fall
in the no-eligible row. The Astra-on-JOB column is experiment 011, the only
teacher run on JOB with the same five-attempt budget.

| Finish state | Astra on CEB, 294 trained-on demonstrations | Astra on JOB, 011 | 026 student | 025 student |
| --- | ---: | ---: | ---: | ---: |
| Best candidate faster than 1.05× | 258 (88%) | 8 | 18 (16%) | 29 (26%) |
| Best candidate within the band | 27 (9%) | 2 | 46 (41%) | 15 (13%) |
| Best candidate slower than 0.95× | 8 (2.7%) | 0 | 21 (19%) | 31 (27%) |
| Only timed-out candidates | 0 | 0 | 0 | 2 (1.8%) |
| No eligible candidate | 1 (0.3%) | 0 | 28 (25%) | 36 (32%) |

An earlier version of this table counted only freshly measured feedback and
therefore placed default-plan duplicates, which are eligible at exactly 1.0×,
in the slower and no-eligible rows. The corrected counts are within one
boundary case of an independent recount.

The teacher almost never holds a losing hand: the training data demonstrates
"choose default over a slower candidate" eight times in 294 conversations, and
on JOB itself Astra never faced that state in ten queries. The student's best
option fails to beat 1.05× in 84% of 026 rollouts and 74% of 025 rollouts, and
it learned the majority behavior, finish by selecting the best candidate. That
is the 16 slower-than-default selections, up from nine in 026, despite three
times the demonstrations. More passes over the same teacher-state data, or more
unselected teacher demonstrations, address this only indirectly, through
repetition of the eight examples. The direct fixes label the student's own
states: teacher corrections on student-generated histories, which is also SFT,
RL, or a harness rule. Experiment 027, a second epoch over these demonstrations,
tests the indirect route.

### A mechanical selection rule removes the regressions, and 025 wins under it

Counterfactual replay of both runs with the rule "select the best candidate if
its feedback ratio is at least 1.05×, otherwise default". Where the rule's pick
equals the actual measured selection, the final paired speedup is used;
otherwise the candidate's feedback ratio stands in. Rollouts with neither a
score nor an eligible candidate are excluded, giving n = 109 for 026 (108 scored
plus job-33a, which had a 2.26× candidate but failed selection) and n = 101 for
025 (its 12 unscored rollouts had no eligible candidate).

| | 026 | 025 |
| --- | ---: | ---: |
| Actual geometric mean | 1.075× | 1.103× |
| Actual choices, default substituted where the selected candidate's ratio was below 1.05× | 1.138× | 1.284× |
| Mechanical rule | 1.163× | 1.285× |
| Rule, excluding the largest win | 1.115× | 1.228× |
| Rule, excluding the three largest wins | 1.082× | 1.166× |
| Improvements / regressions outside the 0.05 log band under the rule | 14 / 0 | 29 / 0 |

The second row is the narrower diagnostic: it keeps every choice the model made
and replaces only selections that the model's own feedback already showed as
slower, so it uses final paired measurements for every retained candidate. It
reaches almost the full-rule value in 025, so nearly the entire counterfactual
gain comes from avoiding visibly bad choices rather than from candidates the
model passed over. The full rule differs from the model's choices on unmeasured
candidates in only one 025 rollout (job-17e at 1.053×) and three 026 rollouts
(job-12a, job-17e, job-33a).

Thresholds of 1.0×, 1.05× and 1.10× give the same geometric means to within
0.002 for both runs; at 1.0× the 026 replay has one regression. At 1.25× the
025 estimate falls to 1.274× and 026 stays at 1.163×.

The feedback ratios show little winner's-curse inflation. For candidates the
students actually selected with feedback at or above 1.05×, one feedback
execution predicted the final paired result almost exactly: in 025, 28 such
selections had a feedback geometric mean of 2.492× and a final one of 2.466×; in
026, 15 had 2.602× and 2.537×. None regressed in final timing. Over all selected
and measured candidates, feedback and final geometric means were 1.202× and
1.202× in 025, and 1.225× and 1.216× in 026. Zero regressions in 43 selections
bounds the rule's false-positive rate at roughly 7% (rule of three); it does not
establish zero future regressions, and the replay remains an estimate rather
than a completed evaluation.

Under the rule, 025 is the better candidate generator by a margin that survives
dropping its three largest wins. Its worse final decisions hide that in the
headline comparison.

### The validity drop is a reliability problem, not truncation

All 93 string-wrapped `action` values in this run are invalid JSON, with no
recorded output truncation, and 69 contain a Leading tree. A repair that closes
unbalanced brackets and quotes was tested: 71 of the 93 parse afterward, but 70
of those still fail the recorded `evaluate_candidate` parameter schema; one
passes. In 026, the transcripts contain 104 string-wrapped actions, of which 25
were valid JSON strings, 79 were invalid, 33 parse after repair, and all 33 fail
the schema. The model is not forgetting closing brackets; it emits the nested
Leading structure unreliably.

That reliability is improving with training rather than fixed: Leading-containing
attempts that passed action and plan constraints rose from 18/142 in 026 to
41/161 here, even as overall validity fell. The demonstrations pulled the policy
toward Astra's richer repertoire (row corrections 13 → 166 attempts, parallelism
31 → 244, planner settings 53 → 156, Leading trees 142 → 161). That produced the
29 improvements and the first useful Leading trees, and a third more invalid
attempts. Bracket repair recovers nothing useful. Further training can help;
the most direct signal is a penalty on the student's own invalid outputs,
meaning RL, and a simpler action grammar remains an option.

### Five of the twelve no-valid-candidate endings were premature

Candidates tried before the terminal call, for the 12 `no_valid_candidate`
rollouts: job-05c 1; job-01a 3; job-10a 3; job-01b 4; job-31c 4; the remaining
seven (job-02a, 07c, 11c, 19b, 19c, 20b, 27b) used all five. job-01a and job-01b
were the 7.207× and 108.730× wins in 026. The v7 rule that ends a conversation
immediately on an ineligible finish with no eligible candidate makes quitting
free at evaluation time; under the anchored algorithm a `no_valid_candidate`
ending costs c, which is the signal RL would use against it. When attempts
remain, that finish should return one corrective error naming the remaining
budget, with bounded retries so the earlier loops do not return. Deliberate
early stopping through `keep_default` or `finish("default")` stays available.

### What the validation loss does and does not show

The initialization record and adapter checksums are the direct evidence that the
epoch-2 weights loaded. The starting value of 0.362505 is consistent with that:
it is far below the 0.4851 the base model scored at step 0 in experiment 018 on
the earlier validation conversations. It says nothing separable about interface
drift, because the validation trajectories were also refreshed. The 7.3%
reduction measures fit to teacher tokens, which this run shows is decoupled from
student decisions. It remains a useful training-health signal alongside
autonomous evaluation, but it should not be used to rank checkpoints.

### Recommended order of work

1. Run experiment 027, a second epoch over these demonstrations by native resume
   from step 1102. It is the direct test of whether more training on the same
   data helps, at about two hours plus a 32-minute evaluation. Prediction on
   record: validity and Leading usage may keep improving; the
   slower-than-default selections will not move much.
2. Add mechanical final selection to the harness as a switchable rule: the best
   eligible candidate by feedback at 1.05× or above, otherwise default. In the
   replay it deletes the entire regression class in both runs. Keep the model's
   own choice recorded and report both numbers. Which one is the headline
   depends on whether the agent's final decision remains part of the research
   question; it should remain the agent's if RL follows, since the reward
   trains that decision directly.
3. Return one corrective error, with bounded retries, on an ineligible finish
   while candidate attempts remain.
4. Re-evaluate both adapters with three rollouts per query under those rules,
   about 1.6 hours each on FLOPper. One rollout cannot rank them: 29 queries lost
   validity and 21 gained it between 026 and this run.
5. If the same failures persist, prefer teacher corrections on student-generated
   CEB histories or RL from this run's adapter over another large, unselected
   batch of teacher demonstrations. At roughly one minute per rollout per
   worker, 100 RL steps at batch 16 is an overnight run.

Not worth doing: JSON repair of string-wrapped actions, or another large
unselected batch of teacher demonstrations before the above.
