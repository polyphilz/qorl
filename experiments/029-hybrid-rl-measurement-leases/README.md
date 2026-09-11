# 029 — RL measurement leases smoke

Test releasing PostgreSQL workers during model turns while increasing active
RL episodes from four to twelve. This measures integration and throughput;
it is not a policy-quality experiment or a promise of threefold speedup.

Created through `qorl experiment create` with seed 42 and the same two CEB
training queries and one validation query as 028. It starts with the same
merged 027 step-2204 base and a fresh rank-16/alpha-32 adapter at `1e-6`.

- Four optimizer updates, batch eight, group four: 32 nominal training episodes.
- Twelve episodes in flight; sixteen allowed model requests and vLLM sequences.
- Maximum policy lag two updates; checkpoints every update, retaining all four.
- Five candidate attempts, thinking enabled, 49,152 context and 8,192 maximum reply.
- Four database workers, four physical cores and 8 GiB each, with 2 GiB shared buffers.
- Initial baseline, candidate feedback and final pairs retain their existing
  warmups and sample counts. Each complete measurement phase uses one worker.
- SQL sessions, plan identity checks, timeout handling and anchored credit are unchanged.

`rl.worker_lease_scope = "measurement"` is explicit here. Existing experiments,
standalone evaluations and calibrations retain their original worker ownership.
The initial observation reports the baseline worker's verified resource limits;
the pool enforces equal worker capacities. Saved RL evidence records each phase's
actual worker slot, queue wait and ownership time instead of claiming that the
whole episode ran on one worker.

Compare completed and trained episodes per minute, generated tokens, trainer
waiting, unused/cancelled work and checkpoint overhead. Check every final paired
sequence's worker affinity, pool capacity, cancellation/drain, gradient updates
and refreshed inference weights. Compare candidate feedback with final paired
measurements, while keeping the small sample and changed batching in view.
No JOB evaluation runs concurrently with this smoke.

The 028 smoke used a different batch size and policy-lag limit. Its wall time is
a descriptive reference; a controlled throughput estimate requires matching
those settings and running long enough to amortize startup and prefetch cleanup.
