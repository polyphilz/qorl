# Results: first Astra-demonstration adapter, one epoch, run 000

Training completed on FLOPper on **September 9, 2026, at 18:21:13
America/New_York** (22:21:13 UTC) after **13m 17s**. Evaluation of the resulting
step-382 adapter on all 113 JOB queries completed at **19:44:49
America/New_York** (23:44:49 UTC), after **1h 18m 05s**. All 113 rollouts were
recorded; there were no infrastructure or model-call failures.

**The adapter learned the shape of the harness, not the shape of a good plan.**
Against the untrained base in [019](../019-qwen-4b-job-vanilla-5cand-1rollout/config.toml),
scored tasks rose from **15/113 to 44/113** and valid candidates from **15 to
90**. The plans it chose were mostly worse than Postgres's: **5 wins against 16
regressions** among the 27 measured queries, for a geometric-mean speedup of
**0.722×** (0.739× clipped to [0.1, 10]) and a total workload speedup of
**0.761×**. Another 40 rollouts never produced a valid `finish` at all.

## Training and checkpoint lineage

| Setting | Value |
| --- | --- |
| Base | `empero-ai/Qwen3.8-4B-Distill`, revision `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e` |
| Adapter | LoRA rank 16, alpha 32, dropout 0, on all attention and MLP projections |
| Learning rate / batch | Constant `1e-4`; batch and microbatch size 1; AdamW |
| Context / precision | 49,152 tokens; BF16; flash attention 2 |
| Updates | 382, one epoch, one packed row per update |
| Hardware | One RTX 3090 on FLOPper; peak memory 21.5 GiB |
| Training data | 99 Astra conversations on 100 CEB training tasks (ten from each of ten templates); 781 supervised requests; 55 requests excluded from supervision; 382 packed rows; 132,706 supervised tokens |
| Validation data | 19 conversations on 20 CEB tasks from two held-out templates; 136 supervised requests; 71 packed rows; 26,463 supervised tokens |

The conversations were the ones generated for 017 (`data.dataset_from` points at
017's generation directory) and were re-prepared here, which is what the
`remasked` suffix records; the config also lowers the context length from
65,536 to 49,152 and imports 017's split seeds. One of the 100 selected training
tasks had no conversation.

| Training measurement | Result |
| --- | ---: |
| Validation loss, step 0 | 0.4851 |
| Validation loss, step 382 | 0.3053 |
| Validation perplexity, step 0 → 382 | 1.624 → 1.357 |
| Training loss, step 382 | 0.0895 |
| Nonfinite losses | 0 |

## Evaluation

Test split: all 113 JOB queries, one rollout each, five candidate attempts per
trajectory, execution feedback on (one warmup and one measurement per
candidate), and the final protocol of one warmup pair and three measured pairs.
PostgreSQL at 2 GB `shared_buffers` on the 4×8 pool.

### How the 113 rollouts ended

| Outcome | 018 (this adapter) | 019 (untrained base) |
| --- | ---: | ---: |
| Measured against the default | 27 | 6 |
| Candidate duplicated the default plan | 16 | 7 |
| Kept the default | 1 | 2 |
| Selection failed | 40 | 16 |
| No valid candidate | 25 | 81 |
| Timed out | 4 | 1 |
| Scored | 44 | 15 |

### Performance on the 44 scored rollouts

| Measure | Value |
| --- | ---: |
| Geometric-mean speedup | 0.722× |
| Geometric-mean speedup, clipped to [0.1, 10] | 0.739× |
| Total workload speedup | 0.761× (19.27 s default, 25.32 s candidate) |
| Wins above 1.05× | 5 |
| Regressions below 0.95× | 16 |
| Rollouts that chose an earlier candidate over the last one | 36 |

Seventeen of the 44 scores are 1.00× by construction, from duplicated or kept
default plans. Four of the five wins are the four queries of template 2
(`job-02c` 2.36×, `job-02b` 1.44×, `job-02d` 1.28×, `job-02a` 1.13×); the fifth
is `job-21b` at 1.14×. The worst regressions were `job-06c` (3 ms → 67 ms,
0.05×), `job-03c` (98 ms → 1,322 ms, 0.07×), `job-32b` (54 ms → 381 ms, 0.14×)
and `job-31a` (454 ms → 2,391 ms, 0.19×). The four timeouts were `job-01c`,
`job-03a`, `job-06f` and `job-17b`, each on a candidate the model had chosen.

### Where the trajectories went wrong

**Candidates.** The model submitted 535 candidates: 445 invalid, 39 duplicates of
the default plan, 51 novel. The most common rejection, 170 times, was an action
that was not an object at all. After that came a `leading` tree with a missing
`right` side (27), join constraints on relation sets not connected in the query
(23), duplicated join constraints (10), and plans that did not realise the
requested scan or join method once explained.

**Selection.** `finish` was called 1,402 times and rejected 1,309 times, almost
always for naming a candidate id the server never issued. All 40 selection
failures had used their full five attempts; 37 then ran out of context while
retrying and 3 hit the turn limit. This is the single largest loss of scored
tasks in the run.

**Protocol.** `evaluate_candidate` was refused 512 times because it was called
after the candidate budget was spent. `get_plan` was misused 91 times (wrong
turn, unknown candidate id, missing node, or a candidate with no plan). The
model called tools that do not exist 146 times, under 37 different names,
most often `evaluator` (27), `submit_query` (26), `evaluate` (12) and
`execute` (11).

## Reading

- The demonstrations taught the model to produce well-formed actions often
  enough to get measured: valid candidates went from 15 to 90 and measured
  queries from 6 to 27. They did not teach it which plans are faster. Sixteen
  of its 27 measured choices were regressions, and in 36 rollouts it chose an
  earlier candidate over its last one, so it was choosing among candidates it
  had feedback on and still choosing badly.
- The `finish` failure mode is a protocol problem, not a planning one: the
  model refers to candidates by ids it made up. Fixing it recovers up to 40
  rollouts before any question of plan quality.
- Template 2 accounts for four of five wins, and training covered ten CEB
  templates, so the gains are narrow. More demonstrations and further epochs
  are the obvious next step, which is what
  [020](../020-sft-astra-ceb-epoch2/config.toml) and
  [021](../021-qwen-4b-job-epoch2-5cand-1rollout/config.toml) go on to do.
