# 005 — Tool-use SFT v2

**Status:** teacher generation and filtering complete; measurement pending

This experiment will build expert-iteration SFT data by rejection-sampling
real one-candidate agent trajectories, then train a fresh LoRA from the pinned
base model.

The selected sampler is the frozen adapter from experiment 001. A controlled
100-task comparison produced a 19.75% distinct-novel-fingerprint yield versus
18.75% for the correctly composed RL v2 step-70 adapter. Both step-70 runs are
retained as diagnostics; the first used the wrong base and is invalid.

`selection.json` reserves 300 CEB train tasks for sampling, 64 different train
tasks for the post-SFT live gate, and the same 64 validation-template tasks
used by SFT v1. RL v3 selections exclude the first two splits.
CEB templates fix their join graph and relation count, so exact template
balance also fixes the relation-count distribution recorded in the manifest.

`dataset.json` pins the sampler policy, seeds, rollout budget, fallback gate,
label margins, and final dataset sizes. Regenerate or verify the task selection
with:

```bash
uv run python experiments/005-protocol-sft-v2/build_inventory.py --write
uv run python experiments/005-protocol-sft-v2/build_inventory.py --check
```

Calibrate the sampling split on the four-worker pool, then seal its timeouts:

```bash
uv run qorl calibrate ceb \
  --selection experiments/005-protocol-sft-v2/selection.json \
  --split sampling
uv run python experiments/005-protocol-sft-v2/build_timeouts.py \
  --write --calibration outputs/calibration/<calibration-id>
uv run python experiments/005-protocol-sft-v2/build_timeouts.py --check
```

- Planned output: `outputs/sft/protocol-sft-v2/`
- Selection identity: `qorl-protocol-sft-v2-selection-v1`

## Host sequence

Merge and verify the selected pilot-SFT sampler. The runner derives the base from the
adapter's recorded training base and rejects mismatched weights; this is
separate from the raw base used to train SFT v2:

```bash
uv run python experiments/005-protocol-sft-v2/run.py --merge-sampler
```

Serve the merge as `qorl-sft-v2-sampler`, then run
`scripts/docker/check-prompt-settings.py` against one active database worker:

```bash
experiments/005-protocol-sft-v2/serve-sampler.sh
```

Sample the template-balanced 100-task gate first:

```bash
uv run python -m qorl.sft.sample \
  --split sampling --limit 100 \
  --model qorl-sft-v2-sampler \
  --sampler-manifest outputs/sft/protocol-sft-v2-sampler-pilot-sft/qorl-merge.json
```

The selected pilot-SFT sampler cleared the gate at 19.75%; the correctly
composed RL v2 step-70 sampler reached 18.75% on the same tasks and seeds.
With the selected sampler still served, complete the sampling split and draw
six samples per validation task:

```bash
uv run python -m qorl.sft.sample \
  --split sampling \
  --model qorl-sft-v2-sampler \
  --sampler-manifest outputs/sft/protocol-sft-v2-sampler-pilot-sft/qorl-merge.json
uv run python -m qorl.sft.sample \
  --split validation --sample-count 6 \
  --model qorl-sft-v2-sampler \
  --sampler-manifest outputs/sft/protocol-sft-v2-sampler-pilot-sft/qorl-merge.json
```

Filter the sampling split, then generate the missing action-family coverage
with Fable 5.1. The API uses automatic tool choice; every accepted action is
replayed through the ordinary agent loop and plan validator before it can enter
the dataset. The smoke accepted two examples each for Leading, join, and
parallel. Memoize accepted none in six attempts and is excluded from the main
teacher batch while its pg_hint_plan behavior is investigated separately.
Accepted records and all retry provenance are stored under this
experiment directory, so later dataset builds do not call the API. Teacher
documents remain syntax examples even if measurement later finds them faster;
measured-win labels are reserved for student samples:

The completed run accepted all 40 requested examples: 28 Leading, eight join,
and four parallel. It used 64 API attempts across 44 tasks. Every accepted
record passed replay validation and is stored in this experiment directory.
The subsequent filter accepted all 40 as distinct novel syntax candidates.

```bash
uv run python -m qorl.sft.filter --split sampling
uv run python -m qorl.sft.teacher --smoke
uv run python -m qorl.sft.teacher --check
uv run python -m qorl.sft.teacher
uv run python -m qorl.sft.teacher --check
uv run python -m qorl.sft.filter --split teacher
```

The normal command targets parallel, Leading, and join. A Memoize-only
diagnostic must use both an explicit family and a separate output directory:

```bash
uv run python -m qorl.sft.teacher \
  --families memoize \
  --output experiments/005-protocol-sft-v2/teacher-memoize
```

Memoize remains in the student action space and final coverage gate for now.
Before assembly, the isolated investigation must either produce a verified
example or support an explicit decision to remove that family from this
experiment's gate.

Fable 5.1 uses provider-default decoding because its API rejects a temperature
parameter. The teacher process reads its temporary credential only from
`ANTHROPIC_API_KEY`. After teacher generation, filter the validation split and
measure the initial accepted training candidates. Measurement writes the
recorded default-best search cohort using the pinned two-fingerprint and 1.10x
selection rule:

```bash
uv run python -m qorl.sft.filter --split validation
uv run python -m qorl.sft.measure
```

The normal syntax pass stops at six samples per task. Searching for defensible
`keep_default` labels is separate: a small recorded task subset may receive up
to 30 samples so that six distinct valid novel fingerprints can actually be
measured. Pass `--default-best` for that search; without it the sampler rejects
any range above six. Measurement still caps each task at six distinct
candidates.

```bash
uv run python -m qorl.sft.sample \
  --split sampling --default-best \
  --task-ids outputs/sft/protocol-sft-v2/default-best-tasks.json \
  --sample-start 7 --sample-count 24 \
  --model qorl-sft-v2-sampler \
  --sampler-manifest outputs/sft/protocol-sft-v2-sampler-pilot-sft/qorl-merge.json
uv run python -m qorl.sft.filter --split default_best
uv run python -m qorl.sft.measure
uv run python -m qorl.sft.build_protocol_dataset
uv run python experiments/005-protocol-sft-v2/run.py --prepare
```

The exceptional default-best search has its own sampling and filter manifests,
so it does not mutate the four-sample student pass or the teacher's source-filter
identity.

Training is a separate, explicit action:

```bash
uv run python experiments/005-protocol-sft-v2/run.py --train
```

Run the exported adapter's acceptance gate against the frozen validation,
fresh unlabeled, and measured-label cohorts. The experiment runner owns the
temporary vLLM server and shuts it down when the gate completes:

```bash
uv run python experiments/005-protocol-sft-v2/run.py --gate
```

The gate records constraint validity, default duplication, novelty, action-family
coverage, intervention/abstention behavior, and distinct physical plans per task.
