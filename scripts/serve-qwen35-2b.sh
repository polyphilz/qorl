#!/usr/bin/env bash
set -Eeuo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$repository/.venv-vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec vllm serve Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --served-model-name Qwen/Qwen3.5-2B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --language-model-only \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
