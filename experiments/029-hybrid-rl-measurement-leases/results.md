# 029 — Measurement leases smoke: intentionally stopped

The smoke exercised measurement-phase worker leasing successfully before being
stopped at the operator's request to start experiment 030. It completed three
optimizer updates, not the planned four; this is partial integration evidence.

| Check | Observed result |
| --- | --- |
| Active episode setting | Twelve, with four PostgreSQL workers |
| Completed native episode records | 37, all `ok=true` |
| Episodes using multiple workers | 37 of 37 |
| Recorded leases | 278 across worker slots 0–3 |
| Worker queue waiting | 4.027 seconds total; 0.0000072-second median; 1.174-second maximum |
| Worker ownership time | 328.692 seconds summed across completed episodes |
| Completed optimizer updates | 1, 2, 3; finite nonzero gradient norms |
| Gradient norms | 0.11328125, 0.02307129, 0.03686523 |
| Updated inference weights | Completed episodes used policy versions through 3 |
| Cancellation | Acknowledged with no cleanup error |
| Final resource cleanup | Both GPUs idle; service exited 0 and removed its four containers |

All completed rollout records use `database_worker = null` and carry explicit
per-phase worker leases. Each baseline, candidate evaluation and final paired
protocol holds one worker for its entire phase. Tests additionally check that
workers are available during model requests, simultaneous calls never share a
worker, and waiting claims respond to cancellation.

The recorded outcomes were 22 kept-default, 13 measured, one timed-out and one
no-valid-candidate. These are repeated samples of two familiar tasks, with changing
weights and prefetched work; they are not a policy-quality result.

The trainer wrapper exited 1 following the requested interruption, at
17:43:14 UTC on September 11, 2026. The remote drain was acknowledged. Requests
rejected after the service closed to new work belong to shutdown; the 37 saved
completed episodes contain no episode failures. Service cleanup finished at
17:44:22 UTC. Checkpoints from the three completed updates were retained.

The larger inference concurrency and small database queue support trying this
configuration on a longer run. They do not establish the predicted threefold
throughput gain. Experiment 028 also used a different batch size and policy-lag
limit. No controlled comparison of timing noise or feedback/final agreement was
completed before interruption.

Same-worker final pairs preserve the measurement protocol, but changing worker
ownership can change cache interference and timing distributions. Preliminary
candidate/default comparisons may cross workers; initial timing also determines
candidate timeouts. Unchanged scoring code does not imply identical observed
rewards.

Local verification: **1,824 tests passed, 47 skipped**, with changed-file Ruff,
formatting and focused typing checks clean. The opt-in setting is
`rl.worker_lease_scope = "measurement"`; existing experiments and standalone
evaluation retain their previous ownership behavior.

Evidence is under `outputs/029-hybrid-rl-measurement-leases/`: the native trace and
metric streams in `000/training/monitors/file/`, `000/training/remote-cleanup.json`,
and `deployment/partial-smoke-audit.json`. The service's phase records and exit
record are retained separately under the same experiment's output directory.
