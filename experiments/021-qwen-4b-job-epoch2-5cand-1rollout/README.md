# 021-qwen-4b-job-epoch2-5cand-1rollout

Evaluate the second-epoch SFT adapter from experiment 020 against all 113 JOB
queries. Compare harness conformance and query speedups with first-epoch SFT
experiment 018 and vanilla experiment 019.

Settings match those evaluations: seed 42, one rollout per task, up to five
candidate attempts, thinking enabled, 49,152-token context, 8,192 output tokens
per turn, and FLOPper's existing four-worker pool with 2 GiB shared buffers.
Resolved test task IDs are identical to 018 and 019.

The pinned base is `empero-ai/Qwen3.8-4B-Distill` at revision
`c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`. Its separate LoRA adapter is
`outputs/020-sft-astra-ceb-epoch2/000/training/checkpoints/step_764/adapter`,
exported after two total SFT epochs (764 cumulative optimizer updates).
Adapter weights SHA-256:
`d1f055e4a347fca34185a34d2f9baf22f9197e006dacfb0b018b52f178c918f4`.

Run from the repository root on FLOPper, where the adapter has been copied:

```bash
uv run --frozen --extra gpu qorl experiment run experiments/021-qwen-4b-job-epoch2-5cand-1rollout --stage evaluate --split test
```

Each invocation creates a fresh numbered run under
`outputs/021-qwen-4b-job-epoch2-5cand-1rollout/`.
