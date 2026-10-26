#!/bin/bash
# usage: kernel/build.sh <name>   (builds kernel/<name>.cu -> kernel/lib<name>.so on the node)
set -e
cd "$(dirname "$0")"
CUTLASS=/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include
NAME=$1
/usr/local/cuda/bin/nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr \
  -DCUTE_ARCH_MMA_SM100A_ENABLED $EXTRA -I$CUTLASS -Xcompiler -fPIC -shared -o lib$NAME.so $NAME.cu -lcuda -Xptxas -v 2>&1 | grep -E "error|spill|registers|fa4_bwd_kernel" | grep -v "^$" | head -40
ls -la lib$NAME.so
