#!/usr/bin/env bash
set -uo pipefail
cd /home/golympie/ai-toolbox/models/sglang-official
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc
export CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 TORCH_CUDA_ARCH_LIST=12.0
export PATH=/home/golympie/.cargo/bin:/usr/local/cuda/bin:$PATH
export MAX_JOBS=24 CMAKE_BUILD_PARALLEL_LEVEL=24 CARGO_BUILD_JOBS=24
echo "=== install start $(date) ==="
/home/golympie/miniconda3/bin/uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  -e python
echo "=== install exit=$? $(date) ==="
python -c "import sglang; print('sglang', sglang.__version__)" 2>&1 | tail -2
