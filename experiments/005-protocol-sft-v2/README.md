# 005 — Tool-use SFT v2

**Status:** inventory and pipeline frozen; host generation pending

This experiment will build expert-iteration SFT data by rejection-sampling
real one-candidate agent trajectories, then train a fresh LoRA from the pinned
base model.

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

Merge and verify the step-70 sampler. The runner derives the base from the
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
  --sampler-manifest outputs/sft/protocol-sft-v2-sampler-step70/qorl-merge.json
```

Continue only if `sampling/sampling-manifest.json` reports a distinct novel
fingerprint yield of at least 11%. Otherwise run the same 100 tasks with the
pilot SFT sampler into a separate output directory and compare that exact
metric; the sampler guard prevents accidentally mixing the two policies.
With the chosen sampler still served, complete the sampling split and draw six
samples per validation task:

```bash
uv run python -m qorl.sft.sample \
  --split sampling \
  --model qorl-sft-v2-sampler \
  --sampler-manifest outputs/sft/protocol-sft-v2-sampler-step70/qorl-merge.json
uv run python -m qorl.sft.sample \
  --split validation --sample-count 6 \
  --model qorl-sft-v2-sampler \
  --sampler-manifest outputs/sft/protocol-sft-v2-sampler-step70/qorl-merge.json
```

Filter both splits, measure the accepted training candidates, assemble the
dataset, and audit Prime-RL's rendering without beginning training:

```bash
uv run python -m qorl.sft.filter --split sampling
uv run python -m qorl.sft.filter --split validation
uv run python -m qorl.sft.measure
uv run python -m qorl.sft.build_protocol_dataset
uv run python experiments/005-protocol-sft-v2/run.py --prepare
```

If yield or family coverage is short, use `qorl.sft.sample --task-ids
<tasks.json> --sample-start 5 --sample-count 2` for the affected tasks, with
an optional task-to-guidance JSON passed through `--guidance`, then rerun the
downstream stages. Guided traces retain both the original and
guidance-stripped transcripts.

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
  --sampler-manifest outputs/sft/protocol-sft-v2-sampler-step70/qorl-merge.json
```

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
