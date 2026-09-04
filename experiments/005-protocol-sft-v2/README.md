# 005 — Tool-use SFT v2

**Status:** sampling and teacher generation complete; ready to assemble

This experiment teaches the one-candidate agent language, not query performance.
It trains a fresh LoRA from the pinned base model on 258 real, validated
trajectories:

- 206 novel student-plan trajectories;
- 40 novel Fable 5.1 plan trajectories (28 Leading, eight join, four parallel);
- 12 natural `keep_default` trajectories from the student sampler.

Every query contributes at most two examples. No query is timed, no
default-best label is inferred, and no teacher-forced validation loss is run.
RL v3 owns the performance objective and the decision of when abstaining is
correct. Memoize remains in PlanAction but is not required in this dataset.

The selected sampler is experiment 001's frozen adapter. On the same 100 tasks,
it produced a 19.75% distinct-novel-fingerprint yield versus 18.75% for the
correctly composed RL v2 step-70 adapter. The initial step-70 comparison is
invalid because that adapter was merged onto the wrong base.

`selection.json` reserves 300 CEB training tasks for sampling and 64 disjoint
CEB training tasks for the post-SFT live gate. RL v3 must exclude both splits.
The checked-in teacher records make dataset assembly independent of the teacher
API and its credential.

## Build and inspect

On the benchmark host, assemble the dataset and render it with the exact
training tokenizer:

```bash
uv run python -m qorl.sft.build_protocol_dataset
uv run python experiments/005-protocol-sft-v2/run.py --prepare
```

Inspect several files under
`outputs/sft/protocol-sft-v2/demonstrations/train/`. A plan example should read
as system prompt, query observation, inspection and result, PlanAction and
result, then `finish`. A keep-default example ends in its natural
`keep_default` call and result.

## Train and evaluate

Train one epoch from the raw base with the same LoRA, optimizer, renderer, and
loss-mask settings as protocol SFT v1:

```bash
uv run python experiments/005-protocol-sft-v2/run.py --train
```

Then merge and serve the resulting adapter and run four attempts on each of the
64 live-gate queries:

```bash
uv run python experiments/005-protocol-sft-v2/run.py --gate
```

## Frozen provenance

- Selection identity: `qorl-protocol-sft-v2-selection-v1`
- Dataset output: `outputs/sft/protocol-sft-v2/`
- Training output: `outputs/sft/protocol-sft-train-v2/`
- Teacher records: `experiments/005-protocol-sft-v2/teacher/`
- Policy config: `configs/policy/run-v2.json`

The `memoize` value retained in `teacher.json` records the completed smoke-test
budget. Editing it would invalidate the checked-in teacher records' config
checksum.
