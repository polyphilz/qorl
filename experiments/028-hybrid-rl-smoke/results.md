# 028 — Hybrid RL smoke results

**The first live FLOPper/Lambda RL smoke passed.** Four optimizer updates trained
on 16 episodes, changed adapter weights, refreshed inference and retained all
four checkpoints. Both network directions, the actual native client/renderer
contract, anchored credit, token masks and cancellation/draining worked.
This establishes integration; it does not establish an improvement in policy quality.

## Run

- Date: September 11, 2026.
- Run: `outputs/028-hybrid-rl-smoke/000` on Lambda.
- Launch: **16:19:05 UTC / 12:19:05 EDT**; successful exit:
  **16:30:20 UTC / 12:30:20 EDT**, approximately **11 min 15 sec** including startup,
  checkpoint writes and shutdown.
- Lambda: training on H100 GPU 0, inference on H100 GPU 1.
- FLOPper: shared agent, renderer and PostgreSQL; four workers with four physical
  cores and 8 GiB each, existing 2 GiB shared-buffers configuration.
- Source on both hosts: `45693b2168c2057b95b61857830d66e165d52f1d`.
- Prime-RL, vendored Verifiers and renderers origin:
  `polyphilz/prime-rl`, commit `d4636a532d668ffbe727eb9e7730e7ea24ca384c`.
- Starting policy: merged 027/000 step-2204 base, with a new rank-16/alpha-32 RL
  adapter, AdamW at `1e-6`, anchored GRPO, five candidate attempts, thinking on,
  49,152-token context and 8,192-token maximum reply.
- Original base, source SFT adapter and SFT checkpoints were preserved.

## Integration evidence

| Check | Observed result |
| --- | --- |
| Remote claim | Accepted with matching execution contract |
| FLOPper → Lambda inference | Model probe and native generation requests returned successfully |
| Actual native episode path | 21 recorded episodes, zero episode failures |
| Training consumption | Four groups of four episodes: 16 used, five completed but unused |
| Native token data | 147 model calls; 36,624 sampled tokens across all recorded episodes |
| Masks and log-probabilities | Every node's token/mask lengths agree; each masked position has a finite log-probability; supervised nodes are sampled assistant messages |
| Shipped credit | 44 branches; 29,271 nonzero advantage positions, exactly matching sampled-token counts in the 16 shipped traces |
| Anchored credit | One consistent scalar per shipped conversation; positive and negative advantages both exercised |
| Optimizer updates | Steps 1, 2, 3 and 4, all with finite, nonzero gradient norms |
| Weight changes | All 256 adapter tensors changed between steps 2 and 3; all checked weights finite |
| Inference refresh | Adapters 1–4 loaded; completed episodes recorded updated policy versions through version 3 |
| Checkpoints | Complete native checkpoints retained for steps 1–4; step 4 exported successfully |
| Remote cleanup | `cancellation_acknowledged=true`, no cleanup error |
| Resource cleanup | Both Lambda GPUs idle; FLOPper service exited successfully and its four containers were removed |

The recorded policy start/end versions were `(0,0)` for seven episodes, `(0,1)`
for four, `(1,2)` for four, `(2,2)` for two and `(2,3)` for four. This exercises
weight changes during a multi-turn episode as well as later episodes using
updated weights. Version 4 loaded at the end; no separate evaluation of the final
adapter was run.

The 44 shipped branches contain 777,953 token positions including masked context.
Their 29,271 nonzero advantage positions match the sampled tokens in the shipped
traces individually. The scalar assignments were checked against the recorded
anchored-credit decisions. The ordinary scalar reward hook prints zero because
it is unused by this algorithm; it is not evidence of zero learning signal.

A concrete first-group example: a measured **1.492×** candidate received advantage
`+0.349894`; the no-valid-candidate outcome received `-0.1`; two kept-default
outcomes each received `-0.174947` relative to their peer reference.

## Updates and runtime

| Optimizer step | Logged loss | Gradient norm |
| --- | --- | --- |
| 1 | 0.0152 | 0.033936 |
| 2 | -0.0064 | 0.066895 |
| 3 | 0.0071 | 0.070312 |
| 4 | -0.0007 | 0.029419 |

RL loss here is not comparable to SFT cross-entropy. The evidence for functioning
updates is the finite gradients, recorded credit, actual weight changes and
subsequent inference refresh.

Peak trainer memory was **22.3 GiB**. The first update's elapsed time included
startup and waiting for rollouts. Later logged step times were approximately
63, 174 and 103 seconds; they include waiting and checkpoint work, not just GPU
computation. Step 3 waited about 114 seconds for a batch versus about 60 seconds
of active step time. This smoke did not tune utilization.

Recorded episode durations were **38.9–125.3 seconds**,
with a **95.7-second median**. Summed agent timing attributed
1771.6 seconds to model calls and 158.1 seconds to the harness,
about 91.8% in model calls. These sums cover concurrent episodes;
model-call time includes transport and is not a measurement of GPU compute alone.

Saved full-episode MessagePack payloads ranged from **0.43 to 2.26 MiB**,
with a **1.30 MiB median**. These are persisted episode sizes, not a separate
measurement of wire-transfer latency.

## Rollout outcomes — descriptive only

| Outcome | All 21 completed episodes | 16 used for training |
| --- | --- | --- |
| `default_duplicate` | 1 | 1 |
| `kept_default` | 10 | 6 |
| `measured` | 8 | 8 |
| `no_valid_candidate` | 2 | 1 |
| `selection_failed` | 0 | 0 |
| `timed_out` | 0 | 0 |

Across all completed episodes, 19 carried a speedup, giving a geometric mean of
**1.273×**. The two no-valid-candidate outcomes are excluded from that mean.
Ten default selections and one default duplicate contribute 1.0×. There were no
final execution timeouts or selection failures. Two rejected selection attempts
were recorded, without a final selection-failed outcome.

These are repeated rollouts on only two familiar training queries, with changing
weights and five completed episodes not used for an optimizer update. The number
is **not a before/after comparison, held-out evaluation or evidence that four RL
updates improved the SFT policy**. JOB was not used for training; the saved
validation task was not evaluated in this smoke.

## Cancellations and log caveats

The configured maximum policy lag was one update. Logs recorded stale-episode
cancellation as weights advanced, plus cancellation of remaining work when the
final batch had been shipped. All recorded completed episodes had `ok=true`;
that does not mean every prefetched episode was trained on or none was cancelled.
The strict lag exercised the refresh/protection path and generated some unused
work. No redispatch or RL resume was added.

vLLM printed an `EngineDeadError` during its **16:30:16 SIGTERM shutdown**, after
training and final checkpoint completion, with zero active inference requests.
The log shows the engine being force-stopped as part of teardown; this was not an
inference failure during the rollouts. The launcher exited 0, the remote drain
was acknowledged and GPU processes stopped. A leaked-semaphore warning also
appeared during interpreter shutdown. No code or dependency fixes were needed
to complete the training run.

## Saved checkpoint and evidence

Final exported adapter, on Lambda:

```text
/lambda/nfs/qorl/projects/qorl/outputs/028-hybrid-rl-smoke/000/training/checkpoints/step_4/adapter
```

Adapter weights: **42,500,760 bytes (40.53 MiB)**.
SHA-256: `933c3d5786f6a85472d5f315a944afaf18da7f6b2059d1d34f3714fa0f0e7b02`.
It must be used with `/lambda/nfs/qorl/models/qwen-4b-sft-027-step-2204`, the
merged base used in training, rather than the original pre-SFT base.

Primary artifacts on Lambda:

- `outputs/028-hybrid-rl-smoke/000/training/report.json`
- `000/training/configs/remote-claim.json` and `remote-environment.json`
- `000/training/remote-readiness.json` and `remote-cleanup.json`
- `000/training/monitors/file/` for metrics, native episodes and credit annotations
- `000/training/checkpoints/step_1` through `step_4`
- `outputs/028-hybrid-rl-smoke/deployment/` for launch/exit records,
  `smoke-audit.json`, `shipped-mask-audit.json`, `weight-change-audit.json` and
  `final-adapter-export.json`

Primary artifacts on FLOPper:

- `outputs/028-hybrid-rl-smoke/service-000/` for identity, claim, service log,
  full native episodes and QORL rollout records
- `outputs/028-hybrid-rl-smoke/deployment/service-audit.json` and service exit record

Service cleanup finished at **16:31:21 UTC**. Another run needs a fresh service
instance/output and a fresh training run.
