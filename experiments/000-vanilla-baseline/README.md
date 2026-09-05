# 000 — Vanilla baselines

**Status:** completed

This experiment evaluates the untrained Qwen distill policy and the frozen
random structured-action policy on held-out JOB. `run.json` selects
`model/configs/000-modelconf/modelconf.json`; `random.json` selects the local
`random-policy.json`. Both wrappers retain their recorded run prefixes.
`run.py` reproduces the random structured-action run; the regular `qorl run`
command uses `run.json` for the untrained model run.

The model config's relocation and removal of its unused schema field change its
file checksum. Historical output manifests retain their original checksums;
the model and sampling settings are unchanged.

- Outputs: `outputs/runs/<run-id>/`
- Identity: model revision
  `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`; JOB inventory
  `job-v1-tasks-v1`

This experiment used `benchmark-v1`, with GEQO enabled. Its results for the 20
JOB tasks containing at least 12 relations are not directly comparable with
results collected under `benchmark-v2`.

The frozen runs also predate the current protocol: the random sampler is now
version 3, and GEQO controls and workload-inapplicable actions are no longer
present in the observation or tool schema.
