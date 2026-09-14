# 033-qwen-4b-job-rl032-5cand-3rollouts

Completed run `000`: see [results and analysis](results.md), including the
three-search comparison and the offline feedback-selected best-of-three result.

Method: `eval`. Seed: `42`.
Created through `qorl experiment create`, then matched to 032's completed JOB
evaluation with three rollouts per query. `config.toml` owns the active values.
Task files contain resolved IDs and are not resampled at execution.
Outputs: `outputs/033-qwen-4b-job-rl032-5cand-3rollouts/<run-number>/`.

Evaluate **032's final step-600 adapter over the merged 031 step-600 base** on
all **113 JOB queries**, with **three rollouts each: 339 total searches**.
Each search has up to five candidate attempts. The query selection is identical
to 031 and 032. Per-query seeds incorporate the rollout index; index zero uses
the same model and measurement seeds as the previous evaluation.

Settings match 032's actual evaluation: thinking enabled, temperature 1.0,
top-p 1.0, top-k 20, **49,152-token context and 8,192-token reply limit**, one
serving GPU with four sequences, and the four-worker PostgreSQL pool with 2 GiB
shared buffers. Initial defaults use one warmup and three measurements;
candidate feedback uses one warmup and one measurement; final scoring uses one
warmup pair and three measured pairs. Database workers are held for complete
rollouts during standalone evaluation.

The configured `best_feedback` rule records both the model's choice and the
feedback-based choice **within each search**. The native summary aggregates all
339 rollouts. Analysis should also report the three 113-query panels separately
and variation within queries. A cross-search **best-of-three** score is a
separate analysis: select using preliminary feedback, keep final paired timings
out of the selection rule, and report the three-search cost explicitly.

Adapter: `outputs/032-rl-ceb-fresh-600steps/000/training/checkpoints/step_600/adapter`.
Its tensor SHA-256 is
`7a29b32f23c5efb8b663494beb616a2e6b29281f07854b3f9a2ecae4cfd9adbc`.
The merged base's weight SHA-256 is
`5de0270bf5329657364c9f86c344527f6a0af5aac989195fd9e8a6aa2df2619d`.
Interface v7 comes from the executing checkout; the config does not pin harness
code. See [032's results](../032-rl-ceb-fresh-600steps/results.md) for the prior
single-search evaluation and checkpoint lineage.

Each execution allocates a new numbered run and saves its config and selection.
Launch command:

```bash
uv run --frozen --extra gpu qorl experiment run experiments/033-qwen-4b-job-rl032-5cand-3rollouts --stage evaluate --split test
```
