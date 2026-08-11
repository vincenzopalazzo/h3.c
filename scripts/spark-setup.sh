#!/usr/bin/env bash
set -euo pipefail
# Install deps + build h3 CUDA on DGX Spark
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
echo "[h3] building CUDA backend for sm_121"
if ! pkg-config --exists icu-uc; then
  echo "missing libicu-dev — run: sudo apt install -y libicu-dev ffmpeg pkg-config" >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null; then
  echo "missing ffmpeg — run: sudo apt install -y ffmpeg" >&2
  exit 2
fi
export PATH=/usr/local/cuda/bin:$PATH
make clean-cuda || true
make cuda-spark -j"$(nproc)"
echo "[h3] smoke"
make h3-cuda-smoke
./h3-cuda-smoke
echo "[h3] build ok: $ROOT/h3-cuda"
