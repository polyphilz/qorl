# 001 — Protocol SFT v1

**Status:** completed

This experiment teaches the base model the QORL message and tool-call protocol.
`spike.toml` checks compatibility; `train.toml` defines the full one-epoch LoRA
run.

- Outputs: `outputs/sft/protocol-sft-spike/`, the prepared dataset under
  `outputs/sft/protocol-sft-v1/`, and training under
  `outputs/sft/protocol-sft-train-v1/`
- Identity: dataset `protocol-sft-v1`; base-model revision
  `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`
