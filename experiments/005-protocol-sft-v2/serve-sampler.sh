#!/usr/bin/env bash
set -Eeuo pipefail

repository="$(cd "$(dirname "$0")/../.." && pwd)"
model="${1:-$repository/outputs/sft/protocol-sft-v2-sampler-pilot-sft}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$repository/.venv-vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec vllm serve "$model" \
  --served-model-name qorl-sft-v2-sampler \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 20480 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --language-model-only \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
