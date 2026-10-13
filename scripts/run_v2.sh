#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RAGSCOPE_HOME="${RAGSCOPE_HOME:-/mnt/d/RAGScope_Runtime}"
export HF_HOME="$RAGSCOPE_HOME/cache/hf" HF_HUB_CACHE="$RAGSCOPE_HOME/cache/hf/hub" HF_DATASETS_CACHE="$RAGSCOPE_HOME/cache/hf/datasets"
export TORCH_HOME="$RAGSCOPE_HOME/cache/torch" SENTENCE_TRANSFORMERS_HOME="$RAGSCOPE_HOME/cache/hf/sentence_transformers"
export PIP_CACHE_DIR="$RAGSCOPE_HOME/cache/pip" UV_CACHE_DIR="$RAGSCOPE_HOME/cache/uv" UV_PYTHON_INSTALL_DIR="$RAGSCOPE_HOME/tools/python-linux"
export TIKTOKEN_CACHE_DIR="$RAGSCOPE_HOME/cache/tiktoken" NLTK_DATA="$RAGSCOPE_HOME/cache/nltk" XDG_CACHE_HOME="$RAGSCOPE_HOME/cache"
export TRITON_CACHE_DIR="$RAGSCOPE_HOME/cache/triton" CUDA_CACHE_PATH="$RAGSCOPE_HOME/cache/cuda" RAGSCOPE_CACHE_DIR="$RAGSCOPE_HOME/cache/ragscope"
export TORCH_EXTENSIONS_DIR="$RAGSCOPE_HOME/cache/torch_extensions" TORCHINDUCTOR_CACHE_DIR="$RAGSCOPE_HOME/cache/torchinductor"
export TMPDIR="$RAGSCOPE_HOME/tmp" CUDA_VISIBLE_DEVICES='' PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd -- "$project_root"
exec "$RAGSCOPE_HOME/envs/core-linux/bin/python" -u -m ragscope.experiments.run "$@"
