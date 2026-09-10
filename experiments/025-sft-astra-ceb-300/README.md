# 025-sft-astra-ceb-300

Generate 300 additional Astra training conversations and refresh demonstrations
for the same 20 held-out CEB validation queries used in experiment 017. JOB remains
the unchanged 113-query test split and is not used for demonstration generation.

Seed 42 selects 30 queries per original training template:
`1a, 2a, 3a, 4a, 6a, 7a, 9a, 10a, 11a, 11b`. All 100 original training task IDs
and their SQL hashes are excluded before sampling. The original input is frozen
in `excluded-tasks-000.json`; resolved task files are not resampled during runs.
Within-training topology repetition is intentional. Training, validation and JOB
test topologies remain disjoint.

[Selection audit](selection-audit.md): all selected SQL and metadata checked;
ten Astra/medium reviewers examined five inputs per training template. No
selection was replaced after review. Runtime and trajectory quality still need
measurement and post-generation review.

Configuration is copied from `configs/defaults/003-sft.toml`, with the student
context set to the demonstrated 49,152-token limit. It uses Astra medium with
`reasoning_summary = "auto"`, one rollout per query, up to five candidate
attempts, execution feedback and three initial default measurements. Astra has
a 262,144-token context and 32,768 output-token limit per request. The PostgreSQL
profile has 2 GiB shared buffers; the pool is FLOPper's four-worker CPU layout.
Generation and database measurements stay on FLOPper after calibration 022 found
higher estimated no-op error rates on Lambda. Lambda remains suitable for the
subsequent GPU training on the prepared dataset.

After these changes are available on FLOPper, from its QORL checkout with
`OPENAI_API_KEY` set and the existing model/IMDB assets present:

```bash
uv run --frozen --extra gpu qorl experiment run experiments/025-sft-astra-ceb-300 --stage prepare
```

This starts 320 generation attempts (300 training + 20 validation), then renders
and packs the usable conversations. It does not train weights or generate JOB
trajectories. Run outputs are under `outputs/025-sft-astra-ceb-300/<run-number>/`.
Failed or over-context trajectories can reduce the usable count; raw generation
evidence is retained. Inspect that evidence before deciding what to curate or
train. The generated one-epoch training defaults are not a decision about the
subsequent training comparison.

Before training, audit accepted Leading trees, correction after rejection,
explicit default fallback and earlier-candidate selection, avoidance of known
timeouts, final choices supported by feedback, and rendered student lengths.
Keep useful failed/slow exploration when followed by sound correction; do not
filter solely for novelty or speedup. Existing preparation masks malformed or
`action_valid=false` assistant replies while preserving them as later context;
it does not automatically assess the quality of the final choice. Keep all
validation task IDs held out, including when their refreshed trajectories fail
quality checks. Do not move them into training or choose replacements based on
validation results.

Reproduce the selections with the frozen exclusion input (the command originally
used the identical `experiments/017-sft-astra-ceb-100/training-tasks.json`):

```bash
uv run --frozen --no-sync qorl experiment create \
  --name sft-astra-ceb-300 --method sft --seed 42 \
  --base-model-name-or-path empero-ai/Qwen3.8-4B-Distill \
  --base-model-revision c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e \
  --tasksets 'train=ceb[1a:30,2a:30,3a:30,4a:30,6a:30,7a:30,9a:30,10a:30,11a:30,11b:30]' \
    'validation=ceb[5a:10,8a:10]' 'test=job' \
  --exclude-tasks-from experiments/025-sft-astra-ceb-300/excluded-tasks-000.json \
  --postgres-config docker/postgres/configs/001-pgconf-2gb-sb \
  --pool-config docker/worker_pool/configs/002-poolconf-4x8
```

Creation copies the current defaults; set the resulting student context to
49,152 to match this experiment. Exclusion flags are only needed when creating
another selection, not when running these saved task files. Future disjoint
batches can repeat `--exclude-tasks-from` for both 017 and 025 training files.
