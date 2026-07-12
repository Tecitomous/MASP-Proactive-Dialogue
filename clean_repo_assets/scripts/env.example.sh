#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MASP_ROOT="${MASP_ROOT:-${ROOT}}"
export DATA_ROOT="${DATA_ROOT:-${MASP_ROOT}/data}"
export MODEL_PATH="${MODEL_PATH:-/path/to/base-model}"
export RUN_ROOT="${RUN_ROOT:-${MASP_ROOT}/runs}"
export DTYPE="${DTYPE:-bf16}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export SEED="${SEED:-42}"

# Public provider settings. Credentials must be supplied outside the repository.
export OBU_BACKEND="${OBU_BACKEND:-openai}"
export JUDGE_BACKEND="${JUDGE_BACKEND:-openai}"
export OBU_MODEL="${OBU_MODEL:-gpt-4o-mini}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "${DATA_ROOT}" "${RUN_ROOT}"
