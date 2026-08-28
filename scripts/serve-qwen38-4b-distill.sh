#!/usr/bin/env bash
set -Eeuo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$repository/.venv-vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec vllm serve empero-ai/Qwen3.8-4B-Distill \
  --revision c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e \
  --served-model-name empero-ai/Qwen3.8-4B-Distill \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --enable-prefix-caching \
  --language-model-only \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
